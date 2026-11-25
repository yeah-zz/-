# -*- coding: utf-8 -*-
"""Learn apparent ink-band parameters; this is not robot pressure learning."""
from rock_v15_common import LAB,read,jsave,digest,FONT,SMALL
from pathlib import Path
import json,math,time,shutil
import numpy as np,cv2,torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter1d
ROOT=LAB/'inverse_brush_curves_v31_20261027'
V30=LAB/'public_vectorizers_v30_20261027'
HOLDOUT=['U03','stage_035','replay_R02']
OFFSETS=np.linspace(-12,12,49,dtype=np.float32)

def observed(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    gray=cv2.bilateralFilter(gray,5,.025,2)
    paper=cv2.GaussianBlur(cv2.dilate(gray,np.ones((41,41),np.uint8)),(0,0),9)
    return np.clip((paper-gray)/np.maximum(paper,.1),0,1).astype(np.float32)

def features(profiles):
    x=torch.as_tensor(profiles,dtype=torch.float32,device='cuda')
    if x.ndim==1:x=x[None]
    smooth=F.avg_pool1d(F.pad(x[:,None],(2,2),mode='replicate'),5,stride=1)
    axis=torch.linspace(-1,1,49,device=x.device)[None,None].expand(len(x),1,-1)
    return torch.cat([x[:,None],smooth,torch.exp(-axis.square()*18),axis],1)

class BandNet(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Conv1d(4,24,5,padding=2),nn.SiLU(),nn.Conv1d(24,32,5,padding=2),nn.SiLU(),nn.Conv1d(32,32,3,padding=1),nn.SiLU());self.head=nn.Sequential(nn.Flatten(),nn.Linear(32*49,96),nn.SiLU(),nn.Linear(96,4))
    def forward(self,x):
        raw=self.head(self.encoder(x));return torch.stack([3*raw[:,0].tanh(),.45+11.55*raw[:,1].sigmoid(),.005+.99*raw[:,2].sigmoid(),raw[:,3].sigmoid()],1)

def brush_profile(p,softness=.6):
    t=torch.linspace(-12,12,49,device=p.device)[None];return p[:,2:3]*torch.sigmoid((p[:,1:2]-(t-p[:,0:1]).abs())/softness)

def synthetic(n,g):
    # Procedural cross-sections, not additional independent artworks.
    rand=lambda *shape:torch.rand(*shape,device='cuda',generator=g)
    p=torch.stack([rand(n)*4-2,.65+rand(n)*8.5,.025+rand(n).pow(.6)*.91,(rand(n)>.18).float()],1)
    t=torch.linspace(-12,12,49,device='cuda')[None];soft=.25+rand(n,1)*1.05
    ink=p[:,2:3]*torch.sigmoid((p[:,1:2]-(t-p[:,0:1]).abs())/soft)*p[:,3:4]
    # Neighbouring ink must not always be absorbed into the queried stroke.
    neighbour_center=(4+rand(n,1)*9)*torch.where(rand(n,1)>.5,1.,-1.);nr=.5+rand(n,1)*2;na=rand(n,1)*.7*(rand(n,1)>.35)
    neighbour=na*torch.sigmoid((nr-(t-neighbour_center).abs())/(.3+rand(n,1)*.5))
    x=1-(1-ink)*(1-neighbour);x=x+.015*torch.randn(n,49,device='cuda',generator=g)+rand(n,1)*.025
    x=(x*(.75+rand(n,1)*.35)).clamp(0,1)
    identifiable=((neighbour_center-p[:,0:1]).abs()>p[:,1:2]+nr+1).squeeze(1).float()*p[:,3]
    return x,p,identifiable

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'manifest.json').exists():raise FileExistsError('Preserve v31 data and protocol')
    manifest=read(LAB/'joint_center_affinity_v24_20261026'/'training_manifest.json');arrays={s:dict(profiles=[],masks=[],present=[],ids=[]) for s in ['train','validation']};records=[]
    for r in manifest['records']:
        if r['split']!='train':continue
        split='validation' if r['id'] in HOLDOUT else 'train';z=np.load(r['data']);rgb=z['rgb'];alpha=observed(rgb);valid=z['valid'];target=z['target'];mask=z['mask'];center=(target[0]>.65)&(mask[0]>.5)&(valid>.5);ys,xs=np.where(center);accepted=0;rejected=0
        # Use only source-valid annotations; neither pseudo widths nor exact
        # user centers are elevated to expert ground truth.
        order=np.arange(0,len(xs),max(3,len(xs)//450));chosen=[]
        for idx in order:
            x,y=int(xs[idx]),int(ys[idx]);theta=.5*math.atan2(float(target[3,y,x]),float(target[2,y,x]));normal=np.array([-math.sin(theta),math.cos(theta)],np.float32);q=np.array([x,y])+OFFSETS[:,None]*normal;xy=np.rint(q).astype(int)
            if (xy[:,0]<0).any() or (xy[:,0]>=alpha.shape[1]).any() or (xy[:,1]<0).any() or (xy[:,1]>=alpha.shape[0]).any():continue
            if not (valid[xy[:,1],xy[:,0]]>.5).all():rejected+=1;continue
            v=cv2.remap(alpha,q[:,0].astype(np.float32)[None],q[:,1].astype(np.float32)[None],cv2.INTER_LINEAR)[0];v=gaussian_filter1d(v,.65);peak=int(np.argmax(v));strength=float(v[peak]);on=v>max(.012,.28*strength);indices=np.flatnonzero(on);groups=np.split(indices,np.where(np.diff(indices)>1)[0]+1) if len(indices) else [];band=next((g for g in groups if peak in g),None)
            if strength<.035 or abs(float(OFFSETS[peak]))>2.5 or band is None or band[0]<2 or band[-1]>46 or len(band)<2 or len(band)>35:rejected+=1;continue
            others=[g for g in groups if peak not in g and float(v[g].max())>.35*strength]
            if others or np.mean(np.r_[v[:4],v[-4:]])>max(.025,.2*strength):rejected+=1;continue
            if any(np.linalg.norm(np.array([x,y])-p)<5 for p in chosen):continue
            chosen.append(np.array([x,y]));focus=np.zeros(49,np.float32);focus[max(0,band[0]-3):min(49,band[-1]+4)]=1
            arrays[split]['profiles'].append(v);arrays[split]['masks'].append(focus);arrays[split]['present'].append(1.);arrays[split]['ids'].append(r['id']);accepted+=1
        # Negatives are only explicitly labelled background, never unknown.
        if 'background' in z:background=z['background']>.5
        else:background=np.zeros(alpha.shape,bool)
        bg=cv2.erode((background&(valid>.5)).astype(np.uint8),np.ones((27,27),np.uint8));yy,xx=np.where(bg);neg=0
        for idx in np.arange(0,len(xx),max(1,len(xx)//max(4,accepted//4))):
            x,y=int(xx[idx]),int(yy[idx]);q=np.c_[x+OFFSETS,np.full(49,y)];v=cv2.remap(alpha,q[:,0].astype(np.float32)[None],q[:,1].astype(np.float32)[None],cv2.INTER_LINEAR)[0]
            if v.max()>.08:continue
            arrays[split]['profiles'].append(v);arrays[split]['masks'].append(np.ones(49,np.float32));arrays[split]['present'].append(0.);arrays[split]['ids'].append(r['id']);neg+=1
        records.append(dict(id=r['id'],split=split,source=r['data'],source_sha256=digest(r['data']),positive_profiles=accepted,explicit_background_profiles=neg,rejected_ambiguous_or_invalid=rejected))
    for split,a in arrays.items():np.savez_compressed(ROOT/(split+'_profiles.npz'),profiles=np.asarray(a['profiles'],np.float32),masks=np.asarray(a['masks'],np.float32),present=np.asarray(a['present'],np.float32),ids=np.asarray(a['ids']))
    dev_ids=['R03','R04','H01','H02','X01','X03','N10','N11','kuhw_29483_00','sina_6586334590_29'];dev=[r for r in read(V30/'manifest.json')['records'] if r['id'] in dev_ids]
    import v27_separation_faint as old
    cal=[]
    for r,rgb,refs,valid,bg in old.cases('calibration'):
        dest=ROOT/'calibration_inputs'/r['id'];dest.mkdir(parents=True,exist_ok=True);from PIL import Image
        Image.fromarray(rgb).save(dest/'image.png');np.savez_compressed(dest/'evaluation_masks.npz',valid=valid,bg=bg);jsave(dest/'partial_reference.json',dict(paths=[np.asarray(p).tolist() for p in refs]));cal.append(dict(id=r['id'],raw=str(dest/'image.png'),shape=list(rgb.shape[:2]),mask_file=str(dest/'evaluation_masks.npz'),reference_file=str(dest/'partial_reference.json'),group='calibration'))
    jsave(ROOT/'manifest.json',dict(records=records,profile_holdout_ids=HOLDOUT,calibration=cal,development=dev,development_is_reused_not_blind=True,procedural_pretrain_steps=1200,real_adaptation_steps=400,batch_size=128,fit_steps=100,maximum_normal_shift_px=2.5,ink_gray_is_apparent_not_fluid_state=True,expert_width_or_darkness_gold=False,unknown_pixels_never_background=True,robot_connected=False,parent_review_sha256=digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json'),old_raster_sha256=digest(LAB/'clean_strokes_v12_20261024'/'final.pt')))
    for split in arrays:print('V31_PROFILES',split,len(arrays[split]['profiles']),flush=True)
    print('V31_CALIBRATION',[r['id'] for r in cal],flush=True)

def real_loss(pred,x,mask,present):
    reconstruction=((brush_profile(pred)-x).abs()*mask).sum(1)/mask.sum(1).clamp_min(1)
    # Real references constrain a band, not an exact unverified center.
    center=F.relu(pred[:,0].abs()-1.5).square()/9
    return ((reconstruction+.15*center)*present).sum()/present.sum().clamp_min(1)+.12*F.binary_cross_entropy(pred[:,3].clamp(.001,.999),present)+.15*((pred[:,2]*pred[:,3])*(1-present)).mean()

def synth_loss(pred,truth,identifiable):
    positive=truth[:,3];center=F.smooth_l1_loss(pred[:,0]/3,truth[:,0]/3,reduction='none');radius=F.smooth_l1_loss(pred[:,1].log(),truth[:,1].log(),reduction='none');alpha=F.smooth_l1_loss(pred[:,2]*3,truth[:,2]*3,reduction='none')
    return ((center+.4*radius+.4*alpha)*identifiable).sum()/identifiable.sum().clamp_min(1)+.18*F.binary_cross_entropy(pred[:,3].clamp(.001,.999),positive)

def train():
    torch.set_num_threads(4);torch.manual_seed(312710);g=torch.Generator(device='cuda').manual_seed(312710);model=BandNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4);manifest=read(ROOT/'manifest.json');val=np.load(ROOT/'validation_profiles.npz');train=np.load(ROOT/'train_profiles.npz');vx=features(val['profiles']);vm=torch.tensor(val['masks'],device='cuda');vy=torch.tensor(val['present'],device='cuda');tx=features(train['profiles']);tm=torch.tensor(train['masks'],device='cuda');ty=torch.tensor(train['present'],device='cuda');logs=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats();best=1e9
    if (ROOT/'profile_model.pt').exists():raise FileExistsError('Preserve trained profile model')
    for step in range(1,1601):
        model.train();x,t,known=synthetic(128,g);pred=model(features(x));loss=synth_loss(pred,t,known)
        if step>1200:
            idx=torch.randint(len(tx),(128,),device='cuda',generator=g);loss=loss*.35+real_loss(model(tx[idx]),tx[idx,0],tm[idx],ty[idx]);opt.param_groups[0]['lr']=2e-4
        opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),3)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('Nonfinite profile training')
        opt.step()
        if step%100==0:
            model.eval()
            with torch.no_grad():vl=float(real_loss(model(vx),vx[:,0],vm,vy))
            row=dict(step=step,loss=float(loss.detach()),validation_photometric_loss=vl,elapsed_seconds=time.perf_counter()-start);logs.append(row);print('V31_TRAIN',json.dumps(row),flush=True)
            if step==1200:torch.save(dict(model=model.state_dict(),step=step),ROOT/'synthetic_only.pt')
            if step>=1200 and vl<best:best=vl;torch.save(dict(model=model.state_dict(),step=step,validation_loss=vl),ROOT/'profile_model.pt')
    torch.cuda.synchronize();jsave(ROOT/'training_summary.json',dict(actual_cuda_updates=1600,procedural_updates=1200,real_adaptation_updates=400,real_profiles=len(tx),validation_profiles=len(vx),new_independent_artworks=0,logs=logs,checkpoint_sha256=digest(ROOT/'profile_model.pt'),elapsed_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),actual_device=str(next(model.parameters()).device),gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,width_and_tone_labels_not_expert_gold=True,robot_connected=False));print('V31_TRAINED',flush=True)

def load():
    model=BandNet().cuda();model.load_state_dict(torch.load(ROOT/'profile_model.pt',map_location='cuda',weights_only=True)['model']);return model.eval()
