# -*- coding: utf-8 -*-
"""Small GPU candidate classifier. Partial assistant labels, never expert GT."""
import sys
sys.dont_write_bytecode=True
from mountain_recovery_catalog_20261101 import ROOT,FEATURES,read,jsave,digest
from pathlib import Path
import numpy as np,json,time
import torch
from torch import nn

# Recorded only AFTER looking at RGB/overlay review cards. Reject is a whole
# trajectory decision (e.g. a loop round a blot), NOT a background ink label.
LABELS={
 'G01_01':dict(keep=[858,1206,1228,1208,1225,165,504,1212,121,125,128,149,152,153,160,1626,526],reject=[151,262,598,850,183]),
 'G05_01':dict(keep=[1485,590,1486,1086,1082,1090,577,233,307,311,312,349,360,361,363,380,682],reject=[437,390,598,227,400]),
 'G02_01':dict(keep=[436,1328,646,0,3,4,7,14,17],reject=[1339,39,511,147,435,277,374,364,76,522]),
 'G04_01':dict(keep=[725,730,724,722,15,731,0,5,12,14,16,22,776,793,4,861,830,795,835,805,884,789],reject=[9,931,788]),
}
TRAIN=['G01_01','G05_01'];CAL=['G02_01'];TEST=['G04_01']
USE_FEATURES=[n for n in FEATURES if n not in ['novel_fraction','inside_fraction','role_long']]
PROTOCOL=dict(version='whole_candidate_mlp_r1',architecture=[len(USE_FEATURES),16,1],seeds=[131,271,419],updates_per_seed=300,lr=.008,weight_decay=.02,features=USE_FEATURES,train=TRAIN,calibration=CAL,regression=TEST,minimum_accept_probability=.8,maximum_ensemble_std=.15,maximum_calibration_false_accepts=0,minimum_reject_probability=.2,group_split='Entire parent sheet group, not random patches',source_scope='Reused generated art, assistant local RGB review, NOT independent expert or blind artwork test',unlabeled='excluded from all losses; not negative',early_stopping=False)

def model():return nn.Sequential(nn.Linear(len(USE_FEATURES),16),nn.SiLU(),nn.Linear(16,1))

def vector(row):
    a=np.array([row['features'][n] for n in USE_FEATURES],np.float32);a[0]=np.log1p(a[0]);a[USE_FEATURES.index('extent')]=np.log1p(a[USE_FEATURES.index('extent')]);return a

def labels():
    refs=[]
    for ident,labels in LABELS.items():
        c=read(ROOT/'catalog'/f'{ident}.json');byid={r['id']:r for r in c['candidates']};review=next(r['review_ids'] for r in read(ROOT/'review_manifest.json')['records'] if r['id']==ident)
        assert set(labels['keep']).isdisjoint(labels['reject'])
        for i in review:
            r=byid[i];label=1 if i in labels['keep'] else 0 if i in labels['reject'] else None
            refs.append(dict(image=ident,candidate=i,label=label,points=r['points'],source_sha256=c['source_sha256'],basis='Visible RGB stroke follows candidate' if label==1 else 'Small closed blot perimeter or transverse spur is not a single centerline trajectory' if label==0 else 'Ambiguous pigment, foliage, boundary, or stroke identity; no supervision',annotation_author='assistant_nonexpert_local_review',expert_verified=False,scope='candidate trajectory only, not pixel background or confirmed artist stroke',split='train' if ident in TRAIN else 'calibration' if ident in CAL else 'regression'))
    return refs

def predict(artifact,rows,device='cpu'):
    if not rows:return np.zeros(0),np.zeros(0)
    x=np.stack([vector(r) for r in rows]);x=(x-np.array(artifact['mean'],np.float32))/np.array(artifact['std'],np.float32);x=torch.from_numpy(x).to(device);ps=[]
    with torch.no_grad():
        for state in artifact['states']:
            m=model().to(device);m.load_state_dict(state);m.eval();ps.append(torch.sigmoid(m(x)).flatten().cpu().numpy())
    ps=np.stack(ps);return ps.mean(0),ps.std(0)

def main():
    if (ROOT/'fit/model.pt').exists():raise RuntimeError('Training already finished; use new version directory for another fit')
    refs=labels();jsave(ROOT/'partial_references.json',dict(references=refs,expert_verified=False,unlabeled_are_negative=False,scope=PROTOCOL['source_scope']))
    jsave(ROOT/'training_protocol.json',dict(PROTOCOL,reference_sha256=digest(ROOT/'partial_references.json'),training_code_sha256=digest(__file__),catalog_code_sha256=digest(Path(__file__).with_name('mountain_recovery_catalog_20261101.py'))))
    data={ident:read(ROOT/'catalog'/f'{ident}.json')['candidates'] for ident in LABELS};train=[r for r in refs if r['image'] in TRAIN and r['label'] is not None]
    x=np.stack([vector(data[r['image']][r['candidate']]) for r in train]);mean=x.mean(0);std=np.maximum(x.std(0),.03);x=(x-mean)/std;y=np.array([r['label'] for r in train],np.float32)
    assert torch.cuda.is_available(),'Use compatible GPU environment';device='cuda';torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    xx=torch.from_numpy(x).to(device);yy=torch.from_numpy(y).to(device);weights=torch.where(yy>0,.5/max(1,int(y.sum())),.5/max(1,int((1-y).sum())));states=[];log=[];started=time.time()
    for seed in PROTOCOL['seeds']:
        torch.manual_seed(seed);m=model().to(device);optimizer=torch.optim.AdamW(m.parameters(),lr=PROTOCOL['lr'],weight_decay=PROTOCOL['weight_decay'])
        for step in range(1,PROTOCOL['updates_per_seed']+1):
            logits=m(xx).flatten();loss=(nn.functional.binary_cross_entropy_with_logits(logits,yy,reduction='none')*weights).sum();optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step in [1,100,200,300]:log.append(dict(seed=seed,update=step,weighted_loss=float(loss.detach().cpu())))
        states.append({k:v.detach().cpu() for k,v in m.state_dict().items()});print('CANDIDATE_FIT',seed,log[-1],flush=True)
    artifact=dict(states=states,mean=mean.tolist(),std=std.tolist(),features=USE_FEATURES,protocol=PROTOCOL,threshold=.8,uncertainty=.15,device=torch.cuda.get_device_name(0),updates_executed=900)
    cal=[r for r in refs if r['image'] in CAL and r['label'] is not None];cr=[data[r['image']][r['candidate']] for r in cal];cp,cs=predict(artifact,cr,device)
    thresholds=[.8,.85,.9,.95,.98,.995,1.01];selected=1.01
    for t in thresholds:
        accepted=(cp>=t)&(cs<=PROTOCOL['maximum_ensemble_std'])
        if sum(accepted[i] and r['label']==0 for i,r in enumerate(cal))==0:selected=t;break
    artifact['threshold']=selected;(ROOT/'fit').mkdir(exist_ok=True);torch.save(artifact,ROOT/'fit/model.pt');jsave(ROOT/'fit/training_log.json',log)
    evaluation=[]
    for ident in LABELS:
        subset=[r for r in refs if r['image']==ident and r['label'] is not None];p,s=predict(artifact,[data[ident][r['candidate']] for r in subset],device);pred=[]
        for r,prob,unc in zip(subset,p,s):pred.append(dict(candidate=r['candidate'],label=r['label'],probability=float(prob),ensemble_std=float(unc),decision='keep' if prob>=selected and unc<=.15 else 'reject_candidate' if prob<=.2 and unc<=.15 else 'pending'))
        evaluation.append(dict(image=ident,split=subset[0]['split'],positive=sum(r['label']==1 for r in subset),negative=sum(r['label']==0 for r in subset),true_accept=sum(r['label']==1 and r['decision']=='keep' for r in pred),false_accept=sum(r['label']==0 and r['decision']=='keep' for r in pred),abstained=sum(r['decision']=='pending' for r in pred),predictions=pred))
    jsave(ROOT/'fit/evaluation.json',dict(records=evaluation,threshold=selected,updates_executed=900,device=artifact['device'],seconds=time.time()-started,training_examples=len(train),reviewed_candidates=len(refs),unknown_excluded=sum(r['label'] is None for r in refs),reference_scope=PROTOCOL['source_scope'],full_image_accuracy=None,automatic_adoption=False))
    print(json.dumps(read(ROOT/'fit/evaluation.json'),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
