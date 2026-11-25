# -*- coding: utf-8 -*-
"""Small GPU crown-region router; assistant partial regional labels, not strokes."""
import sys
sys.dont_write_bytecode=True
from native_fine_faint_20261101 import BASE,ROOT,read,jsave,digest
import numpy as np,cv2,torch,time,math,argparse
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont

ANNOTATIONS={
'T01':dict(positive=[[[360,100],[400,62],[465,35],[520,42],[578,86],[618,95],[650,139],[580,143],[520,119],[464,106],[400,115]],[[610,181],[650,157],[715,146],[774,163],[790,194],[736,216],[654,208]],[[690,218],[750,220],[805,234],[850,234],[904,270],[939,304],[916,325],[866,310],[809,285],[745,267],[695,250]]],negative_boxes=[[0,0,300,180],[650,400,1200,1000],[260,350,490,850],[80,900,480,1190]]),
'T06':dict(positive=[[[0,520],[45,488],[92,471],[128,487],[205,481],[251,495],[259,528],[296,536],[335,554],[331,589],[287,596],[247,572],[190,562],[138,560],[74,570],[5,603]],[[40,644],[75,616],[118,596],[165,606],[203,590],[242,614],[247,641],[286,639],[323,650],[351,666],[390,667],[415,695],[416,714],[448,720],[485,755],[486,790],[455,793],[415,771],[382,754],[349,724],[306,715],[270,703],[229,692],[193,670],[143,660],[103,667],[66,675]]],negative_boxes=[[600,100,1200,700],[30,900,400,1200],[850,900,1200,1200],[0,0,280,400]]),
'T17':dict(positive=[[[520,375],[544,334],[600,313],[646,324],[648,349],[692,352],[724,372],[732,393],[780,399],[790,434],[749,447],[700,422],[656,403],[585,385]]],negative_boxes=[[100,90,460,300],[80,450,420,690],[850,0,1200,300],[820,600,1200,1100]]),
'T19':dict(positive=[[[130,716],[149,687],[220,672],[242,649],[306,625],[379,636],[399,671],[380,699],[341,706],[290,716],[220,716],[178,745]],[[255,743],[322,720],[356,704],[401,714],[458,729],[490,752],[475,774],[432,768],[410,788],[325,786],[281,770]],[[399,804],[460,781],[506,791],[530,815],[566,823],[633,843],[661,875],[633,893],[587,872],[534,860],[471,849],[429,843]]],negative_boxes=[[140,50,600,580],[800,260,1200,600],[20,900,400,1240],[800,30,1200,220]])}

def features(rgb):
    a=rgb.astype(np.float32)/255;g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;local=np.clip((cv2.GaussianBlur(g,(0,0),3)-g)*6,-1,1);valley=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)));sat=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[:,:,1].astype(np.float32)/255
    return np.concatenate([a,g[:,:,None],local[:,:,None],valley[:,:,None]*3,sat[:,:,None]],2).transpose(2,0,1).copy()

class FanRouter(nn.Module):
    def __init__(self):
        super().__init__()
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
        self.a=block(7,12);self.b=block(12,24);self.c=block(24,48);self.d=block(72,24);self.e=block(36,12);self.head=nn.Conv2d(12,1,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1));e=self.e(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1));return self.head(e)

def prepare():
    folder=ROOT/'router';folder.mkdir(parents=True,exist_ok=True);records=[]
    for ident,a in ANNOTATIONS.items():
        source=BASE/'originals'/(ident+'.png');rgb=np.asarray(Image.open(source).convert('RGB'));positive=np.zeros(rgb.shape[:2],np.uint8);negative=positive.copy()
        for p in a['positive']:cv2.fillPoly(positive,[np.asarray(p,np.int32)],1)
        for x,y,xx,yy in a['negative_boxes']:negative[y:yy,x:xx]=1
        # Uncertain polygon rims and every unmarked region remain unsupervised.
        rim=cv2.dilate(positive,np.ones((17,17),np.uint8))!=cv2.erode(positive,np.ones((9,9),np.uint8));known=((positive>0)|(negative>0))&~rim;target=positive.astype(np.float32);known=known.astype(np.float32)
        output=folder/(ident+'_partial.npz');np.savez_compressed(output,x=features(rgb),target=target,known=known);overlay=rgb.copy();overlay[positive>0]=(overlay[positive>0]*.5+np.array([40,210,80])*.5).astype(np.uint8);overlay[negative>0]=(overlay[negative>0]*.65+np.array([180,65,150])*.35).astype(np.uint8);Image.fromarray(overlay).save(folder/(ident+'_labels.png'));records.append(dict(id=ident,image=str(source),sha256=digest(source),file=str(output),positive_known=int(((target>0)&(known>0)).sum()),negative_known=int(((target==0)&(known>0)).sum())))
    manifest=dict(records=records,annotations=ANNOTATIONS,label_kind='Assistant non-expert partial crown-region envelopes and selected not-crown negatives; not ink masks or artist stroke ground truth',unmarked_and_boundary_regions_ignored=True,regression_ids=[f'T{i:02}' for i in range(1,21) if f'T{i:02}' not in ANNOTATIONS],real_source_images=4,independent_accuracy_not_established=True)
    jsave(folder/'manifest.json',manifest);return manifest

def train(steps=600):
    folder=ROOT/'router';checkpoint=folder/'router.pt'
    if checkpoint.exists():raise FileExistsError('Preserve completed router checkpoint')
    manifest=prepare();torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(110103);rng=np.random.default_rng(110103)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU training fallback')
    model=FanRouter().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4);data=[]
    for r in manifest['records']:
        z=np.load(r['file']);x=z['x'];target=z['target'];known=z['known'];pos=np.argwhere((target>.5)&(known>.5));neg=np.argwhere((target<.5)&(known>.5));data.append((x,target,known,pos,neg))
    config=dict(updates=steps,batch_size=8,crop=192,seed=110103,manifest_sha256=digest(folder/'manifest.json'),device=torch.cuda.get_device_name(),tensor_device='cuda:0',model_parameters=sum(p.numel() for p in model.parameters()),threshold=.65,training_reference='partial regional labels only; no extracted line used as correct stroke',test_labels_used=False,initial_weights='random',robot_connected=False)
    jsave(folder/'training_config.json',config);started=time.perf_counter();logs=[];torch.cuda.reset_peak_memory_stats()
    for step in range(1,steps+1):
        batch=[]
        for _ in range(8):
            x,t,k,pos,neg=data[int(rng.integers(len(data)))];centers=pos if rng.random()<.6 else neg;cy,cx=centers[int(rng.integers(len(centers)))];xx=int(np.clip(cx-96+rng.uniform(-45,45),0,x.shape[2]-192));yy=int(np.clip(cy-96+rng.uniform(-45,45),0,x.shape[1]-192));a=x[:,yy:yy+192,xx:xx+192].copy();b=t[yy:yy+192,xx:xx+192].copy();c=k[yy:yy+192,xx:xx+192].copy()
            if rng.random()<.5:a=a[:,:,::-1].copy();b=b[:,::-1].copy();c=c[:,::-1].copy()
            batch.append((a,b,c))
        x,t,k=[torch.from_numpy(np.stack([b[j] for b in batch])).cuda() for j in range(3)];optimizer.zero_grad(set_to_none=True);model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x)[:,0].float();losses=F.binary_cross_entropy_with_logits(logits,t,reduction='none');pos=(t>.5)*k;neg=(t<.5)*k;loss=(losses*pos).sum()/pos.sum().clamp_min(1)+(losses*neg).sum()/neg.sum().clamp_min(1)
        assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%100==0:
            row=dict(step=step,partial_training_loss=float(loss.detach()),seconds=time.perf_counter()-started);logs.append(row);print('FAN_ROUTER_TRAIN',row,flush=True)
            jsave(folder/'progress.json',dict(config=config,logs=logs))
    torch.cuda.synchronize();torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config),checkpoint);jsave(folder/'training_summary.json',dict(completed_updates=steps,device=torch.cuda.get_device_name(),actual_tensor_device=str(next(model.parameters()).device),seconds=time.perf_counter()-started,checkpoint_sha256=digest(checkpoint),peak_cuda_bytes=torch.cuda.max_memory_allocated(),logs=logs,validation_accuracy=None,not_stroke_extraction_training=True));print('FAN_ROUTER_TRAINED',str(checkpoint),flush=True)

@torch.no_grad()
def predict(rgb,model=None):
    if model is None:
        model=FanRouter().cuda();artifact=torch.load(ROOT/'router/router.pt',map_location='cpu',weights_only=False);model.load_state_dict(artifact['model'])
    model.eval();h,w=rgb.shape[:2];tile=256;stride=192;pad=24;padded=cv2.copyMakeBorder(rgb,pad,pad,pad,pad,cv2.BORDER_REFLECT_101);ph,pw=padded.shape[:2];x=features(padded);starts=lambda n:sorted(set(list(range(0,n-tile+1,stride))+[n-tile]));loc=[(y,z) for y in starts(ph) for z in starts(pw)];total=np.zeros((ph,pw),np.float32);den=total.copy();window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32)
    for i in range(0,len(loc),8):
        coords=loc[i:i+8];batch=torch.from_numpy(np.stack([x[:,y:y+tile,z:z+tile] for y,z in coords])).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model(batch)[:,0].float().sigmoid().cpu().numpy()
        for (y,z),v in zip(coords,o):total[y:y+tile,z:z+tile]+=v*window;den[y:y+tile,z:z+tile]+=window
    return (total/np.maximum(den,1e-6))[pad:pad+h,pad:pad+w]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=600);a=p.parse_args();train(a.steps)
