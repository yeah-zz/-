# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,time
import numpy as np,cv2,torch
from torch.nn import functional as F
from ribbon_center_learned_20261104 import CenterNet,OFFSETS,features,extract,predict,centers,analytic_center
ROOT=LAB/'ribbon_topology_refinement_20261104'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def synthetic(count,seed):
    rng=np.random.default_rng(seed);raw=[];targets=[];gold=[]
    yy=np.arange(-4,5,dtype=np.float32)[:,None]
    for i in range(count):
        paper=rng.uniform(.62,.98)+rng.uniform(-.05,.05)*OFFSETS[None]/24+np.zeros((9,97))
        image=paper.copy();center=np.zeros(97);band=np.zeros(97);positions=[]
        if i%6!=0:
            c=rng.uniform(-6,6);w=rng.uniform(1.,17.);positions=[c]
            specs=[(c,w)]
            if i%3==0:
                cw=rng.uniform(1,6);cc=c+rng.choice([-1,1])*(w/2+cw/2+rng.uniform(1.5,5.))
                if abs(cc)<19:specs.append((cc,cw));positions.append(cc)
            for c,w in specs:
                slope=rng.normal(0,.10);x=OFFSETS[None]-c-slope*yy
                alpha=rng.uniform(.02,.80)*np.exp(-.693147*(abs(x)/(w/2))**rng.uniform(2.2,5.))
                grain=np.clip(1+rng.normal(0,.13,image.shape),.1,1.5)
                image-=alpha*grain
                center=np.maximum(center,np.exp(-.5*((OFFSETS-c)/.65)**2))
                band=np.maximum(band,(abs(OFFSETS-c)<=w/2).astype(float))
        image+=rng.normal(0,rng.uniform(.001,.012),image.shape)
        image=cv2.GaussianBlur(image.astype(np.float32),(0,0),rng.uniform(.2,.6))
        raw.append(np.clip(image,0,1));targets.append([center,band]);gold.append(positions)
    return np.array(raw,np.float32),np.array(targets,np.float32),gold

def reviewed():
    root=LAB/'cross_sections_v18_20261025'
    saved=json.loads((root/'expert_cross_sections/reviewed.json').read_text('utf8'))
    candidates={r['id']:r for r in json.loads((root/'review_candidates.json').read_text('utf8'))}
    sources={r['id']:r for r in json.loads((root/'manifest.json').read_text('utf8'))['records']}
    sets={k:[] for k in ['train','calibration','test']};excluded=[]
    split={'R01':'train','R02':'train','N04':'train','N05':'calibration','N06':'test','N09':'test'}
    for r in saved['annotations']:
        if not r.get('reviewed') or r['label']=='uncertain':excluded.append(r['id']);continue
        c=candidates[r['id']];image=Path(sources[r['sample_id']]['image']);rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
        raw=extract(rgb,[c['p']],[c['normal']])[0]
        target=np.zeros((2,97),np.float32);weight=np.zeros_like(target)
        if r['label'] in ['ink','faint']:
            target[0]=np.exp(-.5*((OFFSETS-r['center'])/.65)**2)
            inside=(OFFSETS>=r['left'])&(OFFSETS<=r['right'])
            target[1]=inside;weight[:,inside]=1
            weight[0,abs(OFFSETS-r['center'])<=1.5]=1
        else:weight[:,abs(OFFSETS)<=4]=1
        sets[split[r['sample_id']]].append(dict(raw=raw,target=target,weight=weight,center=r['center'],
            label=r['label'],id=r['id'],sample_id=r['sample_id'],image=str(image),image_sha256=sha(image)))
    return sets,dict(annotation_file=str(root/'expert_cross_sections/reviewed.json'),annotation_sha256=sha(root/'expert_cross_sections/reviewed.json'),
        excluded_uncertain_or_unreviewed=excluded,split_by_source_image=split,
        same_source_family_possible=True,unmarked_offsets_not_negative=True,
        data_source='explicitly user-reviewed cross sections; no automatic extraction labels')

def objective(out,target,weight):
    b=F.binary_cross_entropy_with_logits(out,target,reduction='none')
    # Balance center positives against known non-centers, preserving faint lines.
    pos=weight[:,0]*(target[:,0]>.2);neg=weight[:,0]*(target[:,0]<.01)
    lc=(b[:,0]*pos).sum()/pos.sum().clamp_min(1)+.35*(b[:,0]*neg).sum()/neg.sum().clamp_min(1)
    li=(b[:,1]*weight[:,1]).sum()/weight[:,1].sum().clamp_min(1)
    return lc+.3*li

def human_arrays(rows):
    return [np.stack([r[k] for r in rows]) for k in ['raw','target','weight']]

@torch.no_grad()
def score(model,raw,target,weight):
    p=predict(model,raw);loss=objective(torch.from_numpy(np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7)))),torch.from_numpy(target),torch.from_numpy(weight))
    return float(loss)

def evaluate(model,raw,gold):
    p=predict(model,raw);v,confidence=centers(p)
    yes=[i for i,a in enumerate(gold) if len(a)];no=[i for i,a in enumerate(gold) if not len(a)]
    error=np.array([min(abs(v[i]-np.array(gold[i]))) for i in yes]);baseline=analytic_center(raw)
    e0=np.array([min(abs(baseline[i]-np.array(gold[i]))) for i in yes])
    return dict(samples=len(raw),stroke_sections=len(yes),paper_sections=len(no),
        learned_center_mae_px=float(error.mean()),analytic_center_mae_px=float(e0.mean()),
        learned_center_median_error_px=float(np.median(error)),analytic_center_median_error_px=float(np.median(e0)),
        paper_probability_over_p5=int((confidence[no]>.5).sum()),paper_peak_mean=float(confidence[no].mean()) if no else None)

def eval_human(model,rows):
    raw=np.stack([r['raw'] for r in rows]);p=predict(model,raw);v,c=centers(p)
    baseline=analytic_center(raw);out=[]
    for i,r in enumerate(rows):out.append(dict(id=r['id'],sample_id=r['sample_id'],label=r['label'],gold_center_px=float(r['center']),
        learned_center_px=float(v[i]),analytic_center_px=float(baseline[i]),learned_peak=float(c[i]),
        learned_error_px=float(abs(v[i]-r['center'])) if r['label']!='background' else None,
        analytic_error_px=float(abs(baseline[i]-r['center'])) if r['label']!='background' else None))
    errors=[r for r in out if r['label']!='background']
    return dict(rows=out,stroke_sections=len(errors),learned_center_mae_px=float(np.mean([r['learned_error_px'] for r in errors])),
        analytic_center_mae_px=float(np.mean([r['analytic_error_px'] for r in errors])),
        independent_painting_accuracy_established=False)

def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(2)
    if (ROOT/'model.pt').exists():raise FileExistsError('Preserve this completed experiment')
    if not torch.cuda.is_available():raise RuntimeError('GPU required for this authorized training experiment')
    torch.manual_seed(110404);rng=np.random.default_rng(110404)
    sets,protocol=reviewed();train,tt,train_gold=synthetic(4200,11040401);cal,ct,cal_gold=synthetic(600,11040402)
    hr,ht,hw=human_arrays(sets['train']);cr,cgt,cw=human_arrays(sets['calibration'])
    write(ROOT/'protocol.json',dict(**protocol,reviewed_counts={k:len(v) for k,v in sets.items()},
        synthetic_train_sections=len(train),synthetic_calibration_sections=len(cal),synthetic_not_real_paintings=True,
        test_not_read_for_checkpoint_selection=True,gpu=torch.cuda.get_device_name(),new_model_role='cross-section center proposal only',
        physical_pressure_mm=None,robot_connected=False))
    np.savez_compressed(ROOT/'training_data.npz',synthetic_raw=train,synthetic_targets=tt,human_raw=hr,human_targets=ht,human_weights=hw)
    tx=torch.from_numpy(features(train)).cuda();ty=torch.from_numpy(tt).cuda();hx=features(hr)
    model=CenterNet().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0002)
    best=1e9;state=None;retained=0;history=[];start=time.monotonic()
    for step in range(1,601):
        model.train();ix=torch.randint(len(tx),(48,),device='cuda');j=rng.integers(len(hr),size=16)
        x=torch.cat([tx[ix],torch.from_numpy(hx[j]).cuda()]);y=torch.cat([ty[ix],torch.from_numpy(ht[j]).cuda()])
        w=torch.cat([torch.ones_like(ty[ix]),torch.from_numpy(hw[j]).cuda()])
        loss=objective(model(x),y,w);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if step%100==0:
            validation=score(model,cal,ct,np.ones_like(ct))+.5*score(model,cr,cgt,cw)
            row=dict(step=step,loss=float(loss.detach()),selection_loss=validation,seconds=time.monotonic()-start);history.append(row);print('TRAIN',json.dumps(row),flush=True)
            if validation<best:best=validation;retained=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state);torch.save(dict(model=state,retained_step=retained,completed_updates=600,protocol=protocol),ROOT/'model.pt')
    # First use of held-out sections occurs AFTER model selection is sealed.
    test,_,gold=synthetic(1000,11040403)
    evaluation=dict(synthetic_test=evaluate(model,test,gold),reviewed_calibration=eval_human(model,sets['calibration']),reviewed_test=eval_human(model,sets['test']))
    gate=evaluation['reviewed_calibration']['learned_center_mae_px']<=evaluation['reviewed_calibration']['analytic_center_mae_px']
    write(ROOT/'evaluation.json',evaluation)
    summary=dict(completed_updates=600,retained_step=retained,parameters=sum(p.numel() for p in model.parameters()),
        gpu=torch.cuda.get_device_name(),tensor_device=str(next(model.parameters()).device),seconds=time.monotonic()-start,
        model_sha256=sha(ROOT/'model.pt'),annotation_unchanged=sha(protocol['annotation_file'])==protocol['annotation_sha256'],
        center_proposal_calibration_gate_passed=gate,deploy_to_robot=False,history=history)
    write(ROOT/'training_summary.json',summary);print('DONE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
