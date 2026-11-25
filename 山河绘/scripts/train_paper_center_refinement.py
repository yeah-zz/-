"""Hard-paper and measured-band refinement; full-image photometry matches inference."""
import sys
sys.dont_write_bytecode=True
from paper_center_refinement_data import ROOT,PARENT
from rock_v15_common import read,jsave,digest
from train_joint_ink_structure import JointModel,loss_fn as base_loss,prior
from evaluate_joint_ink_structure import predict
import numpy as np,cv2,torch,time,json,functools,math
from torch.nn import functional as F
from PIL import Image
INIT=PARENT/'joint.pt'

@functools.lru_cache(maxsize=24)
def arrays(image,data):
    rgb=np.asarray(Image.open(image).convert('RGB'));z=np.load(data);d={k:z[k].astype(np.int64 if k=='role' else np.float32) for k in z.files}
    if 'paper_mask' not in d:
        margin=cv2.dilate((d['target'][0]>.01).astype(np.uint8),np.ones((9,9),np.uint8))>0
        d['paper_mask']=((d['target'][1]==0)&~margin).astype(np.float32)
    return rgb,d

def objective(o,t,m,role,rel,rm,paper):
    total,parts=base_loss(o,t,m,role,rel,rm);o=o.float();hard=[]
    values=F.softplus(o[:,0])+.7*F.softplus(o[:,1])
    for i in range(len(o)):
        selected=values[i][paper[i]>.5]
        if selected.numel():hard.append(selected.topk(min(64,selected.numel())).values.mean())
    paper_loss=torch.stack(hard).mean() if hard else total*0
    off_center=(t[:,1]>.5)&(t[:,0]<.01)&(m[:,0]>0)
    edge_values=F.softplus(o[:,0])[off_center];edge=edge_values.topk(min(128,edge_values.numel())).values.mean() if edge_values.numel() else total*0
    return total+paper_loss+.3*edge,dict(**parts,paper_hard=paper_loss,band_edge=edge)

def crop(r,rng,mode='positive',augment=True,mined=None):
    rgb,z=arrays(r['image'],r['data']);rgb=rgb.copy();h,w=rgb.shape[:2]
    if augment:
        rgb=np.clip(rgb.astype(np.float32)*rng.uniform(.97,1.02)+rng.normal(0,.2,(h,w,1)),0,255).astype(np.uint8)
    padded=cv2.copyMakeBorder(rgb,20,20,20,20,cv2.BORDER_REFLECT_101);features,contrast,_=prior.photometry(padded)
    if mode=='paper':
        coords=mined.get(r['id']) if mined else None
        if coords is None or not len(coords):coords=np.argwhere(z['paper_mask']>.5)
    elif mode=='faint':
        coords=np.argwhere((z['target'][0]>.3)&(z['weight'][0]>0)&(contrast[20:20+h,20:20+w]<.12))
        if not len(coords):coords=np.argwhere((z['target'][0]>.3)&(z['weight'][0]>0))
    else:coords=np.argwhere((z['target'][0]>.3)&(z['weight'][0]>0))
    if len(coords):cy,cx=coords[int(rng.integers(len(coords)))];y=int(np.clip(cy+20-128+rng.uniform(-24,24),0,h+40-256));x=int(np.clip(cx+20-128+rng.uniform(-24,24),0,w+40-256))
    else:y=int(rng.integers(h+40-255));x=int(rng.integers(w+40-255))
    def patch(a,role=False):
        if a.ndim==2:q=np.pad(a,((20,20),(20,20)),constant_values=255 if role else 0);return q[y:y+256,x:x+256].copy()
        q=np.pad(a,((0,0),(20,20),(20,20)));return q[:,y:y+256,x:x+256].copy()
    t=patch(z['target']);m=patch(z['weight']);role=patch(z['role'],True);rel=patch(z['relation']);rm=patch(z['relation_weight']);paper=patch(z['paper_mask']);rm[:,:4]=0;rm[:,-4:]=0;rm[:,:,:4]=0;rm[:,:,-4:]=0
    faint=(t[0]>.3)&(m[0]>0)&(contrast[y:y+256,x:x+256]<.12)
    return features[:,y:y+256,x:x+256].copy(),t,m,role,rel,rm,paper,faint.astype(np.float32)

def tensors(batch):return [torch.from_numpy(np.stack([r[k] for r in batch])).cuda() for k in range(8)]

@torch.no_grad()
def validation(model,bank):
    model.eval();parts=[];positive_hits=positive_count=faint_hits=faint_count=0
    for start in range(0,len(bank),4):
        a=tensors(bank[start:start+4])
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model(a[0]);loss,values=objective(o,*a[1:7])
        p=o[:,0].float().sigmoid();mask=(a[1][:,0]>.3)&(a[2][:,0]>0);faint=a[7]>0
        positive_hits+=int(((p>=.35)&mask).sum());positive_count+=int(mask.sum());faint_hits+=int(((p>=.35)&faint).sum());faint_count+=int(faint.sum())
        parts.append({k:float(v) for k,v in values.items()})
    result={k:float(np.mean([r[k] for r in parts])) for k in parts[0]};result.update(center_recall=positive_hits/max(1,positive_count),faint_recall=faint_hits/max(1,faint_count),positive_pixels=positive_count,faint_pixels=faint_count)
    return result

def train():
    if (ROOT/'refined.pt').exists():raise FileExistsError('Preserve completed refinement')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(291103);rng=np.random.default_rng(291103);manifest=read(ROOT/'manifest.json');records=manifest['records'];rows=[r for r in records if r['split']=='train'];cal=[r for r in records if r['split']=='calibration']
    model=JointModel().cuda();model.load_state_dict(torch.load(INIT,map_location='cpu',weights_only=False)['model']);mined={}
    for r in [r for r in rows if r['kind']=='actual_refined']:
        rgb,z=arrays(r['image'],r['data']);p=predict(model.eval(),rgb)[0];coords=np.argwhere(z['paper_mask']>0)
        if len(coords):
            values=p[coords[:,0],coords[:,1]];order=np.argsort(values)[-min(256,len(values)):];mined[r['id']]=coords[order]
    jsave(ROOT/'hard_paper_mining.json',dict(source_checkpoint_sha256=digest(INIT),training_source_only=True,points={k:v.tolist() for k,v in mined.items()}))
    old=[r for r in rows if r['kind']=='existing_declared'];hard=[r for r in rows if r['kind']=='hard_declared'];actual=[r for r in rows if r['kind']=='actual_refined' and np.load(r['data'])['target'][0].max()>.3];paper=[r for r in rows if r['kind'] in ['paper_stress','actual_refined'] and arrays(r['image'],r['data'])[1]['paper_mask'].any()]
    selected_cal=[r for r in cal if r['kind'] in ['actual_refined','paper_stress']]+[r for r in cal if r['kind']=='existing_declared'][:8]+[r for r in cal if r['kind']=='hard_declared'][:8]
    bank=[]
    for i,r in enumerate(selected_cal):
        bank.append(crop(r,np.random.default_rng(291100+i),'paper' if r['kind']=='paper_stress' else 'positive',False))
        if r['kind']=='actual_refined':bank.append(crop(r,np.random.default_rng(291900+i),'paper',False))
    baseline=validation(model,bank);jsave(ROOT/'calibration_initial.json',baseline)
    steps=600;config=dict(steps=steps,batch=6,sampling='one existing declared, two hard declared, one actual center, two paper-focused',init=str(INIT),init_sha256=digest(INIT),manifest_sha256=digest(ROOT/'manifest.json'),full_image_photometry_matches_inference=True,paper_ohem_pixels_per_sample=64,hard_paper_loss_weight=1.,band_edge_weight=.3,center_guard_drop=.05,faint_guard_drop=.05,calibration_ids=[r['id'] for r in selected_cal],actual_training_groups=['G01','G03','G04','R04'],actual_calibration_groups=['G02','N11'],test_used=False,robot_connected=False,gpu=torch.cuda.get_device_name())
    jsave(ROOT/'training_config.json',config);best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_step=0;best=baseline['paper_hard']+.3*baseline['center']+.2*baseline['relation'];optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4);logs=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(1,steps+1):
        chosen=[(old[int(rng.integers(len(old)))],'positive')]+[(hard[int(rng.integers(len(hard)))],'positive') for _ in range(2)]+[(actual[int(rng.integers(len(actual)))],'positive')]+[(paper[int(rng.integers(len(paper)))],'paper') for _ in range(2)]
        batch=[crop(r,rng,mode,True,mined) for r,mode in chosen];a=tensors(batch);model.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=1e-5*(.35+.65*.5*(1+math.cos(math.pi*step/steps)))
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=objective(model(a[0]),*a[1:7])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        if step%100==0 or step%150==0:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-start)
            if step%150==0:
                score=validation(model,bank);guard=score['center_recall']>=baseline['center_recall']-.05 and score['faint_recall']>=baseline['faint_recall']-.05;value=score['paper_hard']+.3*score['center']+.2*score['relation'];row.update(calibration=score,passes_preservation_guard=bool(guard),selection_score=value)
                if guard and value<best:best=value;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            logs.append(row)
            with (ROOT/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('PAPER_CENTER_TRAIN',json.dumps(row),flush=True)
    torch.cuda.synchronize();torch.save(dict(model=best_state,retained_step=best_step,completed_updates=steps,config=config),ROOT/'refined.pt')
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},completed_updates=steps,config=config),ROOT/'last.pt')
    jsave(ROOT/'training_summary.json',dict(completed_updates=steps,retained_step=best_step,calibration_score=best,gpu=torch.cuda.get_device_name(),tensor_device=str(next(model.parameters()).device),elapsed_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=digest(ROOT/'refined.pt'),initial_checkpoint_unchanged=digest(INIT)==config['init_sha256'],logs=logs))
    print('PAPER_CENTER_TRAINED',best_step,flush=True)
if __name__=='__main__':train()
