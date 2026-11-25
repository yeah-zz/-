# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,time,hashlib
import numpy as np,torch
from torch.nn import functional as F
from context_curve_model_20261105 import ContextCurveNet,predict
ROOT=LAB/'context_curve_selection_20261105'
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
    if (ROOT/'model.pt').exists():raise FileExistsError('Keep completed context experiment')
    torch.set_num_threads(4);torch.manual_seed(110505)
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no silent CPU training')
    train=np.load(ROOT/'train_data.npz');cal=np.load(ROOT/'calibration_data.npz')
    tx=torch.from_numpy(train['x']).cuda();ty=torch.from_numpy(train['y']).cuda();cx=cal['x'];cy=cal['y']
    weights=torch.tensor([1.,float((train['y']==0).sum()/max(1,(train['y']==1).sum()))],device='cuda')
    model=ContextCurveNet().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.0005)
    best=1e9;state=None;history=[];start=time.monotonic();retained=0;torch.cuda.reset_peak_memory_stats()
    print('GPU',torch.cuda.get_device_name(),'examples',len(ty),'parameters',sum(p.numel() for p in model.parameters()),flush=True)
    for step in range(1,1001):
        model.train();ix=torch.randint(len(tx),(32,),device='cuda');x=tx[ix].float()/255
        # Photometry augmentation leaves candidate/neighbor masks unchanged.
        gain=torch.empty((len(x),1,3,1,1),device='cuda').uniform_(.97,1.03)
        x[:,:,:3]=(x[:,:,:3]*gain).clamp(0,1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(model(x),ty[ix],weight=weights)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite context training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if step%100==0:
            pred=predict(model,cx);ce=float(-np.log(np.maximum(pred[np.arange(len(cy)),cy],1e-8)).mean())
            row=dict(step=step,loss=float(loss.detach()),calibration_ce=ce,seconds=time.monotonic()-start);history.append(row)
            if ce<best:best=ce;retained=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('CONTEXT_TRAIN',json.dumps(row),flush=True)
    model.load_state_dict(state);torch.save(dict(model=state,completed_updates=1000,retained_step=retained,
        classes=['visible_ink_candidate','explicit_background_or_declared_noise']),ROOT/'model.pt')
    cp=predict(model,cx);choices=[]
    for t in [.8,.9,.95,.97,.98,.99,.995,.999,.9999]:
        remove=cp[:,1]>=t;fp=int((remove&(cy==0)).sum());tp=int((remove&(cy==1)).sum())
        if fp==0 and tp>0:choices.append(dict(threshold=t,positive_removed=fp,negative_removed=tp))
    threshold=max(choices,key=lambda r:r['negative_removed'])['threshold'] if choices else 1.01
    # Test arrays are first opened after checkpoint and threshold decisions.
    test=np.load(ROOT/'test_data.npz');p=predict(model,test['x']);y=test['y'];remove=p[:,1]>=threshold
    evaluation=dict(samples=len(y),known_positive=int((y==0).sum()),known_negative=int((y==1).sum()),
        threshold=threshold,positive_removed=int((remove&(y==0)).sum()),negative_removed=int((remove&(y==1)).sum()),
        argmax_accuracy=float((p.argmax(1)==y).mean()),source_families_overlap=True,
        sample_test_contains_declared_synthetic_and_partial_real_references=True,
        independent_painting_accuracy_established=False)
    write(ROOT/'training_summary.json',dict(completed_updates=1000,retained_step=retained,gpu=torch.cuda.get_device_name(),
        tensor_device=str(next(model.parameters()).device),parameters=sum(q.numel() for q in model.parameters()),
        seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        model_sha256=hashlib.sha256((ROOT/'model.pt').read_bytes()).hexdigest(),
        threshold=threshold,calibration_choices=choices,evaluation=evaluation,history=history,robot_connected=False))
    print('CONTEXT_DONE',json.dumps(evaluation),flush=True)

if __name__=='__main__':main()
