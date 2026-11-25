# -*- coding: utf-8 -*-
"""V20 partial clDice, continuous reference support, hard separation and ridge objectives."""
from review_guided_train_v19 import *
V19=ROOT
ROOT=LAB/'structure_generalization_v20_20261025'
from scipy.ndimage import gaussian_filter1d


def prepare():
    folder=ROOT/'training_data';folder.mkdir(parents=True,exist_ok=True);manifest=read(V19/'manifest_v2.json');out=[];display=[]
    for r in manifest['records']:
        z=np.load(r['data']);rgb=z['rgb'];h,w=rgb.shape[:2];doc=read(r['labels']);target=z['target'].copy();mask=z['mask'].copy();uncertain=z['uncertain_exclusion'];bg=z['background'];valid=z['valid'];alpha=v16.photometry(rgb)[1];center=target[0]>.5;dd,nearest=distance_transform_edt(~center,return_indices=True);curves=[];curve=np.zeros((h,w),np.uint8)
        for s in doc['strokes']:
            q=dense(s['points']);xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);near=np.column_stack((nearest[1,xy[:,1],xy[:,0]],nearest[0,xy[:,1],xy[:,0]]));offset=near-q;distance=np.linalg.norm(offset,axis=1);offset*=np.minimum(1,3/np.maximum(distance,1e-6))[:,None];offset[distance>5]=0;offset=gaussian_filter1d(offset,1,axis=0);q=q+offset;xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);visible=((alpha[xy[:,1],xy[:,0]]>.008)|(dd[xy[:,1],xy[:,0]]<2.5)).astype(np.uint8)[None];visible=cv2.morphologyEx(visible,cv2.MORPH_CLOSE,np.ones((1,5),np.uint8))[0]>0;layer=np.zeros((h,w),np.uint8)
            for i in range(len(q)-1):
                if visible[i] and visible[i+1]:cv2.line(layer,tuple(xy[i]),tuple(xy[i+1]),1,1)
            layer[(valid==0)|(uncertain>0)|((bg>0)&(target[1]<.5))]=0;curve|=layer;curves.append(layer)
        if len(curves)>1:
            distances=np.stack([distance_transform_edt(v==0).astype(np.float32) for v in curves]);two=np.partition(distances,1,axis=0)[:2];separation=(bg>0)&(two[0]>2)&(two[1]<12)&(valid>0)&(uncertain==0)
        else:separation=np.zeros((h,w),bool)
        faint=(curve>0)&(alpha>=.008)&(alpha<=.18)&(uncertain==0);weight=np.ones((h,w),np.float32)*(1. if r['id'].startswith('R') else .35);t=np.concatenate((target,z['teacher_center'][None],uncertain[None],curve[None].astype(np.float32),separation[None].astype(np.float32),faint[None].astype(np.float32),weight[None]));file=folder/(r['id']+'.npz');np.savez_compressed(file,rgb=rgb,target=t,mask=mask);out.append(dict(r,data=str(file),data_sha256=digest(file),curve_pixels=int(curve.sum()),hard_separation_pixels=int(separation.sum()),faint_curve_pixels=int(faint.sum())));print('V20_TARGETS',r['id'],'curve',curve.sum(),'hard_bg',separation.sum(),'faint',faint.sum(),flush=True)
        if r['id'] in ['R01','R02','N06']:
            overlay=rgb.copy();overlay[curve>0]=[225,40,45];overlay[separation]=[40,165,75];display.append((r['id'],rgb,overlay))
    jsave(ROOT/'training_manifest.json',dict(records=out,training_ids=manifest['training_ids'],calibration_ids=['N07','N08'],init_weights=manifest['init_weights'],init_sha256=manifest['init_sha256'],parent_sha256=manifest['original_parent_sha256'],review_sha256=manifest['review_snapshot_sha256'],external_sources_excluded=True,partial_curve_hints_are_not_full_expert_strokes=True,robot_connected=False))
    board=Image.new('RGB',(1100,360*len(display)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((6,4),'v20 训练监督审计（非预测）：红＝可见结构线连续性提示；绿＝已知背景中的近邻分离区',font=FONT,fill='black')
    for i,(key,rgb,ov) in enumerate(display):
        for j,a in enumerate([rgb,ov]):d.text((550*j+5,35+360*i),key+(' 原图' if j==0 else ' 派生训练提示'),font=SMALL,fill='black');paste(board,a,550*j+5,65+360*i,540,320)
    board.save(ROOT/'TRAINING_AUDIT.jpg',quality=94)


def erode(x):return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))
def skeleton(x,iterations=3):
    opened=F.max_pool2d(erode(x),3,1,1);sk=F.relu(x-opened)
    for _ in range(iterations):x=erode(x);opened=F.max_pool2d(erode(x),3,1,1);delta=F.relu(x-opened);sk=sk+F.relu(delta-sk*delta)
    return sk


def structural_loss(o,t,m):
    p=o.float().sigmoid();pc=p[:,0:1];pi=p[:,1:2];valid=1-t[:,6:7];curve=t[:,7:8]*t[:,10:11]*valid;sep=t[:,8:9]*valid;faint=t[:,9:10]*t[:,10:11]*valid;known=(m[:,1:2]>.05).float()*valid;allowed=torch.maximum(t[:,1:2],F.max_pool2d(curve,3,1,1));sp=skeleton(pi)*known;precision=(sp*allowed).sum((1,2,3))/(sp.sum((1,2,3))+1e-5);sensitivity=(curve*pi).sum((1,2,3))/(curve.sum((1,2,3))+1e-5);cl=(1-(2*precision*sensitivity+1e-5)/(precision+sensitivity+1e-5)).mean();cover=F.max_pool2d(pc,5,1,2);continuity=(-torch.log(cover.clamp_min(.001))*curve).sum()/curve.sum().clamp_min(1);separate=(F.binary_cross_entropy_with_logits(o[:,0:1].float(),torch.zeros_like(pc),reduction='none')*sep).sum()/sep.sum().clamp_min(1);weak=(-torch.log(cover.clamp_min(.001))*faint).sum()/faint.sum().clamp_min(1)
    b,_,h,w=pc.shape;yy,xx=torch.meshgrid(torch.arange(h,device=pc.device),torch.arange(w,device=pc.device),indexing='ij');theta=.5*torch.atan2(t[:,3],t[:,2]);radius=(t[:,4]*48*.5).clamp(2,10);nx=-torch.sin(theta);ny=torch.cos(theta)
    def side(sign):
        gx=(xx[None]+sign*nx*radius)*2/(w-1)-1;gy=(yy[None]+sign*ny*radius)*2/(h-1)-1;return F.grid_sample(pc,torch.stack((gx,gy),-1),mode='bilinear',padding_mode='border',align_corners=True)
    ridge_mask=((t[:,4:5]*48>=6)&(t[:,4:5]*48<=18)&(m[:,3:4]>.25)&(t[:,0:1]>.65)).float()*valid;ridge=(F.relu(torch.maximum(side(-1),side(1))+.15-pc)*ridge_mask).sum()/ridge_mask.sum().clamp_min(1)
    return .20*cl+.18*continuity+.12*separate+.08*ridge+.10*weak


def pool_data():
    manifest=read(ROOT/'training_manifest.json');out=[]
    for r in manifest['records']:
        z=np.load(r['data']);rgb=z['rgb'];t=z['target'];m=z['mask'];variants=[]
        for scale in [.75,1.,1.25]:
            h,w=rgb.shape[:2];size=(max(32,round(w*scale)),max(32,round(h*scale)));im=cv2.resize(rgb,size,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR);tt=np.stack([cv2.resize(a,size,interpolation=cv2.INTER_NEAREST if k in [6,7,8,9,10] else cv2.INTER_LINEAR) for k,a in enumerate(t)]);mm=np.stack([cv2.resize(a,size,interpolation=cv2.INTER_NEAREST) for a in m]);tt[4]*=scale;variants.append((v16.photometry(im)[0],tt.astype(np.float32),mm.astype(np.float32)))
        out.append(dict(id=r['id'],variants=variants))
    return manifest,out


def crop_features(a,rng):
    x,t,m=[v.copy() for v in a];h,w=x.shape[1:];ph=max(0,256-h);pw=max(0,256-w)
    if ph or pw:
        x=np.pad(x,((0,0),(0,ph),(0,pw)),mode='reflect');t=np.pad(t,((0,0),(0,ph),(0,pw)));m=np.pad(m,((0,0),(0,ph),(0,pw)));t[6,h:,:]=1;t[6,:,w:]=1;h,w=x.shape[1:]
    y=int(rng.integers(h-256+1));xx=int(rng.integers(w-256+1));x=x[:,y:y+256,xx:xx+256];t=t[:,y:y+256,xx:xx+256];m=m[:,y:y+256,xx:xx+256];k=int(rng.integers(4));x=np.rot90(x,k,axes=(1,2)).copy();t=np.rot90(t,k,axes=(1,2)).copy();m=np.rot90(m,k,axes=(1,2)).copy()
    if k%2:t[2:4]*=-1
    if rng.random()<.5:x=x[:,:,::-1].copy();t=t[:,:,::-1].copy();m=m[:,:,::-1].copy();t[3]*=-1
    return x,t,m


def train(variant):
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(202510);rng=np.random.default_rng(202510);manifest,data=pool_data();folder=ROOT/variant;folder.mkdir(exist_ok=True)
    if (folder/'final.pt').exists():raise FileExistsError('Completed checkpoint immutable')
    model=v16.RibbonNet().cuda();model.load_state_dict(torch.load(manifest['init_weights'],map_location='cuda',weights_only=False)['model']);opt=torch.optim.AdamW([dict(params=list(model.a.parameters())+list(model.b.parameters())+list(model.c.parameters()),lr=.00002),dict(params=list(model.d.parameters())+list(model.e.parameters())+list(model.head.parameters()),lr=.00012)],weight_decay=.0001);steps=1200;start=time.perf_counter();config=dict(variant=variant,steps=steps,seed=202510,training_ids=manifest['training_ids'],calibration_ids=manifest['calibration_ids'],init_sha256=manifest['init_sha256'],whole_image_features_before_crop=True,scales=[.75,1.,1.25],right_angle_rotations=True,partial_cldice_no_topology_guarantee=True,robot_connected=False);jsave(folder/'config.json',config);shutil.copy2(Path(__file__),folder/'source_at_start.py')
    for step in range(1,steps+1):
        batch=[]
        for _ in range(6):
            group=[r for r in data if r['id'].startswith('R')] if rng.random()<.65 else [r for r in data if not r['id'].startswith('R')];a=group[int(rng.integers(len(group)))];batch.append(crop_features(a['variants'][int(rng.integers(3))],rng))
        x,t,m=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(3)];opt.zero_grad(set_to_none=True);factor=.3+.7*.5*(1+math.cos(math.pi*step/steps))
        for g,lr in zip(opt.param_groups,[.00002,.00012]):g['lr']=lr*factor
        with torch.autocast('cuda',dtype=torch.bfloat16):
            o=model(x);loss=train17.loss_fn(o,t[:,:5],m);unknown=(m[:,0]<.1)&((t[:,5]>.9)|(t[:,5]<.1))&(t[:,6]<.5)&(t[:,7]<.1)&(t[:,8]<.5);distill=(F.binary_cross_entropy_with_logits(o[:,0].float(),t[:,5],reduction='none')*unknown).sum()/unknown.sum().clamp_min(1);extra=structural_loss(o,t,m) if variant=='structured' else o.new_tensor(0.);loss=loss+.2*distill+extra
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();g=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(g):raise RuntimeError('Nonfinite gradient')
        opt.step()
        if step%100==0:
            row=dict(step=step,loss=float(loss.detach()),structural_loss=float(extra.detach()),elapsed_seconds=time.perf_counter()-start)
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('V20_TRAIN',variant,json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),config=config),folder/'final.pt');jsave(folder/'training_summary.json',dict(successful_updates=steps,elapsed_seconds=time.perf_counter()-start,robot_connected=False));print('V20_TRAINED',variant,flush=True)


def load_model(variant):
    m=v16.RibbonNet().cuda();m.load_state_dict(torch.load(ROOT/variant/'final.pt',map_location='cuda',weights_only=False)['model']);return m.eval()


def evaluate(variant,mode):
    previous_root=prior_eval.ROOT;previous_load=prior_eval.load;prior_eval.ROOT=ROOT;prior_eval.load=load_model
    try:prior_eval.calibrate(variant) if mode=='calibrate' else prior_eval.evaluate(variant)
    finally:prior_eval.ROOT=previous_root;prior_eval.load=previous_load

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','calibrate','evaluate']);p.add_argument('variant',nargs='?',choices=['control','structured']);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='train':train(a.variant)
    else:evaluate(a.variant,a.mode)

