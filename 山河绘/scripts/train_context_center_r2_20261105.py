# -*- coding: utf-8 -*-
"""Second objective: target centerline vs background/known rendered band flank.
The earlier A05 test is retired into development training and disclosed.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,time,hashlib
import cv2,numpy as np,torch
from torch.nn import functional as F
from context_curve_model_20261105 import ContextCurveNet,predict,pack,raster
from context_curve_data_20261105 import ROOT,source_records,synthetic,real_examples,write
from context_curve_apply_20261105 import select
R2=ROOT/'center_round2'

def reserve_record():
    root=LAB/'user_review_v11_20261024';label=root/'labels/A06.json';d=json.loads(label.read_text('utf8'))
    assert d['review_state']=='checked_partial';image=root/'samples/A06.png';rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
    bg=np.zeros(rgb.shape[:2],np.uint8)
    for r in d['regions']:
        if r['kind']=='background':cv2.fillPoly(bg,[np.rint(r['points']).astype(np.int32)],1)
    return dict(id='A06',image=str(image),labels=str(label),rgb=rgb,refs=[np.array(s['points'],float) for s in d['strokes'] if s['id'].startswith('s_')],
        paths=[],background=bg,split='test',prediction_file=None,source_family='wang_ximeng_qianli_jiangshan')

def main():
    R2.mkdir(exist_ok=True)
    if (R2/'model.pt').exists():raise FileExistsError('Preserve completed center round')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(110521)
    records=source_records()
    for r in records:
        if r['id']=='A05':r['split']='train'
    records.append(reserve_record());sets={};manifest=[]
    for split,count,seed in [('train',3300,110521),('calibration',600,110522),('test',600,110523)]:
        x,y=synthetic(count,seed,target_center=True)
        for j,r in enumerate(records):
            if r['split']!=split:continue
            a,b,o=real_examples(r,80 if split=='train' else 36,seed+100+j);x+=a;y+=b
            # Full checked paths are always included, not only random subcurves.
            neighbors=raster(r['rgb'].shape[:2],r['paths'] or r['refs'])
            for p in r['refs']:x.append(pack(r['rgb'],p,neighbors));y.append(0)
            manifest.append(dict(id=r['id'],split=split,image=r['image'],labels=r['labels'],examples=len(b),source_family=r['source_family']))
        sets[split]=(np.stack(x),np.array(y,np.int64));print('CENTER_R2_DATA',split,len(y),flush=True)
    np.savez_compressed(R2/'train_data.npz',x=sets['train'][0],y=sets['train'][1])
    write(R2/'protocol.json',dict(records=manifest,objective='target centerline or explicit user target vs declared background and rendered off-center flank',
        off_center_negatives_only_from_known_rendered_centers=True,unmarked_real_regions_not_negative=True,
        A05_prior_test_retired_into_training=True,A06_reserved_for_context_module_test=True,
        source_families_overlap=True,test_teacher_curves_not_used_for_selection=True,robot_connected=False))
    model=ContextCurveNet().cuda();model.load_state_dict(torch.load(ROOT/'model.pt',map_location='cpu',weights_only=False)['model'])
    tx=torch.from_numpy(sets['train'][0]).cuda();ty=torch.from_numpy(sets['train'][1]).cuda();cx,cy=sets['calibration']
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00025,weight_decay=.0005)
    w=torch.tensor([1.,float((ty==0).sum()/max(1,int((ty==1).sum())))],device='cuda')
    best=1e9;state=None;retained=0;logs=[];start=time.monotonic()
    for step in range(1,1001):
        model.train();ix=torch.randint(len(tx),(32,),device='cuda');optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(model(tx[ix].float()/255),ty[ix],weight=w)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if step%100==0:
            p=predict(model,cx);ce=float(-np.log(np.maximum(p[np.arange(len(cy)),cy],1e-8)).mean());logs.append(dict(step=step,calibration_ce=ce))
            if ce<best:best=ce;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};retained=step
            print('CENTER_R2_TRAIN',step,round(ce,4),flush=True)
    model.load_state_dict(state);torch.save(dict(model=state,completed_updates=1000,retained_step=retained),R2/'model.pt')
    # Calibrate the full source-guard policy on checked calibration paths.
    calrows=[]
    for r in records:
        if r['split']!='calibration':continue
        rr=select(model,0.,r['rgb'],[dict(points=p,module=0,source_curve_ids=[i]) for i,p in enumerate(r['refs'])]);calrows+=rr['decisions']
    cp=predict(model,cx);options=[]
    for t in [.5,.6,.7,.8,.9,.95,.97,.99,.995,.999]:
        bad=sum(d['noise_probability']>=t and not d['source_contour_or_clear_texture_guard'] for d in calrows)
        raw_positive_loss=int(((cp[:,1]>=t)&(cy==0)).sum());neg=int(((cp[:,1]>=t)&(cy==1)).sum())
        if bad==0 and raw_positive_loss<=max(1,int((cy==0).sum()*.005)):
            options.append(dict(threshold=t,teacher_removed=bad,raw_positive_removed=raw_positive_loss,negative_removed=neg))
    threshold=max(options,key=lambda r:r['negative_removed'])['threshold'] if options else 1.01
    test=[]
    for r in records:
        if r['split']!='test':continue
        rr=select(model,threshold,r['rgb'],[dict(points=p,module=0,source_curve_ids=[i]) for i,p in enumerate(r['refs'])])
        test.append(dict(id=r['id'],positive_curves=len(r['refs']),removed=len(rr['rejected'])))
    p=predict(model,sets['test'][0]);y=sets['test'][1];rejected=p[:,1]>=threshold
    summary=dict(completed_updates=1000,retained_step=retained,gpu=torch.cuda.get_device_name(),seconds=time.monotonic()-start,
        threshold=threshold,calibration_options=options,test_teacher_curves=test,
        teacher_preservation_passed=all(r['removed']==0 for r in test),
        mixed_test=dict(samples=len(y),positive_removed=int((rejected&(y==0)).sum()),negative_removed=int((rejected&(y==1)).sum()),
            known_positive=int((y==0).sum()),known_negative=int((y==1).sum())),
        labels_not_read_at_inference=True,source_families_overlap=True,history=logs,
        model_sha256=hashlib.sha256((R2/'model.pt').read_bytes()).hexdigest(),robot_connected=False)
    write(R2/'training_summary.json',summary);print('CENTER_R2_DONE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
