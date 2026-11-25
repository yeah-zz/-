# -*- coding: utf-8 -*-
from faint_joint_model_20261101 import *
import functools,time,math,json,argparse

@functools.lru_cache(maxsize=48)
def load(file):
    z=np.load(file);return z['rgb'],z['target'].astype(np.float32),z['known'].astype(np.float32)

def crop(r,rng,augment=True):
    rgb,t,k=load(r['file']);h,w=rgb.shape[:2];x=int(rng.integers(w-192+1));y=int(rng.integers(h-192+1));rgb=rgb[y:y+192,x:x+192].copy();t=t[:,y:y+192,x:x+192].copy();k=k[:,y:y+192,x:x+192].copy()
    if augment:
        if rng.random()<.5:rgb=rgb[:,::-1].copy();t=t[:,:,::-1].copy();k=k[:,:,::-1].copy();t[3]*=-1
        if rng.random()<.25:rgb=rgb[::-1].copy();t=t[:,::-1].copy();k=k[:,::-1].copy();t[3]*=-1
    return features(rgb),t,k

def balanced(logits,target,mask):
    b=F.binary_cross_entropy_with_logits(logits,target,reduction='none');pos=(target>.15)*mask;neg=(target<.02)*mask;return (b*pos).sum()/pos.sum().clamp_min(1)+(b*neg).sum()/neg.sum().clamp_min(1)

def loss_fn(o,t,k):
    o=o.float();lc=balanced(o[:,0],t[:,0],k[:,0]);p=o[:,0].sigmoid();dice=1-(2*(p*t[:,0]*k[:,0]).sum()+1)/(((p+t[:,0])*k[:,0]).sum()+1);li=balanced(o[:,1],t[:,1],k[:,1]);le=balanced(o[:,4],t[:,4],k[:,3]);direction=F.normalize(o[:,2:4],dim=1);ld=((1-(direction*t[:,2:4]).sum(1))*k[:,2]).sum()/k[:,2].sum().clamp_min(1);weak=(t[:,0]>.5)*(k[:,0]==0)*(k[:,1]>0);near=F.max_pool2d(p[:,None],9,1,4)[:,0];presence=(-near.clamp_min(1e-5).log()*weak).sum()/weak.sum().clamp_min(1);return lc+.45*dice+.65*li+.13*ld+.3*le+.2*presence

@torch.no_grad()
def validation(model,rows):
    model.eval();values=[]
    for i in range(0,len(rows),8):
        batch=[crop(r,np.random.default_rng(1101+j),False) for j,r in enumerate(rows[i:i+8])];x,t,k=[torch.from_numpy(np.stack([a[j] for a in batch])).cuda() for j in range(3)]
        with torch.autocast('cuda',dtype=torch.bfloat16):value=loss_fn(model(x),t,k)
        values.append(float(value))
    return float(np.mean(values))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',default='r2');args=parser.parse_args();assert args.revision=='r2';folder=ROOT/'fit_r2';folder.mkdir(exist_ok=True);checkpoint=folder/'model.pt'
    if checkpoint.exists():raise FileExistsError('Preserve completed checkpoint')
    manifest=read(ROOT/'paired_manifest_r2.json');train=[r for r in manifest['synthetic'] if r['split']=='train'];real=[r for r in manifest['real'] if r['split']=='train'];cal=[r for r in manifest['synthetic'] if r['split']=='calibration'][:24]+[r for r in manifest['real'] if r['split']=='calibration'];assert {r['parent'] for r in real}.isdisjoint({r['parent'] for r in manifest['real'] if r['split']!='train'})
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(110106);rng=np.random.default_rng(110106)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    model=FaintNet().cuda();init=ROOT/'fit/model.pt';model.load_state_dict(torch.load(init,map_location='cpu',weights_only=False)['model']);optimizer=torch.optim.AdamW(model.parameters(),lr=2.5e-4,weight_decay=1e-4);steps=1200;best=float('inf');best_state=None;best_step=None;logs=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats();config=dict(updates=steps,init=str(init),init_sha256=digest(init),batch=8,synthetic_per_batch=6,real_partial_per_batch=2,synthetic_training_patches=len(train),real_training_patches=len(real),real_training_parents=sorted({r['parent'] for r in real}),calibration_real_parents=sorted({r['parent'] for r in cal if 'parent' in r}),seed=110106,paired_manifest_sha256=digest(ROOT/'paired_manifest_r2.json'),source_code_sha256=digest(__file__),device=torch.cuda.get_device_name(),heads=['center','binary_visible_ink','cos2theta','sin2theta','color_only_boundary'],real_precise_center_gold_used=False,old_test_visuals_seen_during_development=True,heldout_test_labels_used=False,robot_connected=False)
    jsave(folder/'training_config.json',config)
    for step in range(1,steps+1):
        selected=[train[int(rng.integers(len(train)))] for _ in range(6)]+[real[int(rng.integers(len(real)))] for _ in range(2)];batch=[crop(r,rng) for r in selected];x,t,k=[torch.from_numpy(np.stack([a[j] for a in batch])).cuda() for j in range(3)];model.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=2.5e-4*(.25+.75*.5*(1+math.cos(math.pi*step/steps)))
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=loss_fn(model(x),t,k)
        assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if step%100==0:
            row=dict(step=step,training_loss=float(loss.detach()),seconds=time.perf_counter()-start)
            if step%200==0:
                score=validation(model,cal);row['calibration_objective']=score
                if score<best:best=score;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            logs.append(row);jsave(folder/'progress.json',dict(config=config,logs=logs));print('FAINT_JOINT_TRAIN',row,flush=True)
    torch.cuda.synchronize();torch.save(dict(model=best_state,config=config,retained_step=best_step,completed_updates=steps),checkpoint);jsave(folder/'training_summary.json',dict(completed_updates=steps,retained_step=best_step,calibration_objective=best,device=torch.cuda.get_device_name(),actual_tensor_device=str(next(model.parameters()).device),seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=digest(checkpoint),logs=logs));print('FAINT_JOINT_TRAINED',checkpoint,flush=True)
if __name__=='__main__':main()
