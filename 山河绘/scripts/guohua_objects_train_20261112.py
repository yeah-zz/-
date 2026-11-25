# -*- coding: utf-8 -*-
"""A genuine coarse object head, separate from rock roles and cunfa names."""
import sys,json,hashlib,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small
from PIL import Image
from gongbi_training.cunfa_supervision import partial_role_loss
ROOT=LAB/'guohua_objects_pilot_20261112';WEIGHTS=LAB/'cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def pack(image,target=None,size=256):
 h,w=image.shape[:2];scale=size/max(h,w);nh,nw=round(h*scale),round(w*scale);im=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR);canvas=np.full((size,size,3),int(np.median(image)),'uint8');y=(size-nh)//2;x=(size-nw)//2;canvas[y:y+nh,x:x+nw]=im
 t=np.full((size,size),255,'uint8')
 if target is not None:t[y:y+nh,x:x+nw]=cv2.resize(target,(nw,nh),interpolation=cv2.INTER_NEAREST)
 a=canvas.astype('float32')/255.;a=(a-np.array([.485,.456,.406],'float32'))/np.array([.229,.224,.225],'float32');return a.transpose(2,0,1),t,(x,y,nw,nh,w,h)
class PixelOnly(nn.Sequential):
 def __init__(self):super().__init__(nn.Conv2d(3,16,1),nn.SiLU(),nn.Conv2d(16,3,1))
class ContextObjects(nn.Module):
 def __init__(self):
  super().__init__();base=mobilenet_v3_small(weights=None);base.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True));self.encoder=base.features;self.encoder.requires_grad_(False);self.lateral=nn.ModuleList([nn.Conv2d(c,32,1) for c in [16,24,48,576]]);self.head=nn.Sequential(nn.Conv2d(128,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,3,1))
 def forward(self,x):
  shape=x.shape[-2:];self.encoder.eval();layers=[]
  with torch.set_grad_enabled(torch.is_grad_enabled() and any(p.requires_grad for p in self.encoder.parameters())):
   for i,block in enumerate(self.encoder):
    x=block(x)
    if i in [1,3,8,12]:layers.append(x)
  size=layers[0].shape[-2:];f=[F.interpolate(layer(v),size=size,mode='bilinear',align_corners=False) for layer,v in zip(self.lateral,layers)];return F.interpolate(self.head(torch.cat(f,1)),size=shape,mode='bilinear',align_corners=False)
def confusion(t,p):
 known=t!=255;return np.bincount((t[known]*3+p[known]).ravel(),minlength=9).reshape(3,3)
def metrics(c):
 support=c.sum(1);pred=c.sum(0);union=support+pred-c.diagonal();iou=[float(c[k,k]/union[k]) if union[k] else None for k in range(3)];return dict(confusion=c.tolist(),reference_support=support.tolist(),iou=iou,mean_iou=float(np.mean([v for v in iou if v is not None])),recall=[float(c[k,k]/support[k]) if support[k] else None for k in range(3)],tree_as_rock_rate=float(c[2,1]/support[2]) if support[2] else None,rock_as_tree_rate=float(c[1,2]/support[1]) if support[1] else None,source_reference_not_expert_accuracy=True)
def predict(net,image,device='cuda'):
 x,_,(left,top,nw,nh,w,h)=pack(image)
 with torch.inference_mode():prob=net(torch.from_numpy(x)[None].to(device)).softmax(1)[0].cpu().numpy()
 # Restore the complete original image; no ground-truth crop or reference mask.
 prob=prob[:,top:top+nh,left:left+nw];prob=np.stack([cv2.resize(a,(w,h),interpolation=cv2.INTER_LINEAR) for a in prob]);return prob,prob.argmax(0).astype('uint8')
def main():
 assert not (ROOT/'results.json').exists(),'Do not tune after viewing object test.';ds=json.loads((ROOT/'dataset_manifest.json').read_text('utf8'));rows=ds['records'];assert all(sha(r['image'])==r['image_sha256'] and sha(r['target'])==r['target_sha256'] for r in rows);train=[r for r in rows if r['split']=='train'];data={r['id']:(np.array(Image.open(r['image']).convert('RGB')),np.array(Image.open(r['target']))) for r in rows};seed=121207;steps=600;batch=4
 protocol=dict(seed=seed,steps_per_arm=steps,batch_size=batch,arms=['pixel_color_only','context_object_head'],dataset_sha256=sha(ROOT/'dataset_manifest.json'),mapping_sha256=sha(ROOT/'mapping_evidence.json'),encoder_weights_sha256=sha(WEIGHTS),script_sha256=sha(__file__),encoder='Existing ImageNet MobileNetV3 Small, frozen, feature stages1/3/8/12',trainable='Multiscale segmentation decoder only; color-only pixel network as control',sampling='Uniform filename work-group then uniform image in group',loss='Cross entropy with inverse square-root training class-frequency; source void255 ignored',optimizer='AdamW lr0.002 weight_decay0.0001',augmentation='Aspect-preserving256 letterbox; affine +/-7deg and0.9..1.1scale; flip and mild color/grayscale changes; same schedule per arm',selection='Last fixed step. No validation/test checkpoint or threshold selection.',confidence_display_threshold=.6,confidence_threshold_predeclared=True,provided_source_split_reorganized_by_filename_work=True,true_author_independence_verified=False,coarse_objects_only=True,cunfa_or_tree_painting_skill_trained=False,robot_connected=False);write('protocol.json',protocol)
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);counts=np.zeros(3)
 for r in train:
  t=data[r['id']][1];counts+=np.bincount(t[t!=255],minlength=3)
 weights=torch.tensor(1/np.sqrt(counts),device='cuda',dtype=torch.float32);weights/=weights.mean();groups=[[r for r in train if r['work_group']==g] for g in sorted({r['work_group'] for r in train})];models={};curves={};started=time.time()
 for name in protocol['arms']:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);net=(PixelOnly() if name=='pixel_color_only' else ContextObjects()).cuda();opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.002,weight_decay=.0001);losses=[]
  for step in range(steps):
   xs=[];ts=[]
   for _ in range(batch):
    r=random.choice(random.choice(groups));image,target=data[r['id']];x,t,_=pack(image,target);image=(x.transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255;image=np.clip(image,0,255).astype('uint8');m=cv2.getRotationMatrix2D((128,128),random.uniform(-7,7),random.uniform(.9,1.1));image=cv2.warpAffine(image,m,(256,256),borderValue=(210,210,210));t=cv2.warpAffine(t,m,(256,256),flags=cv2.INTER_NEAREST,borderValue=255)
    if random.random()<.5:image=image[:,::-1].copy();t=t[:,::-1].copy()
    if random.random()<.3:image=np.repeat(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)[...,None],3,2)
    image=np.clip(image.astype('float32')*random.uniform(.9,1.1)+random.uniform(-8,8),0,255).astype('uint8');xx=image.astype('float32')/255;xx=(xx-np.array([.485,.456,.406],'float32'))/np.array([.229,.224,.225],'float32');xs.append(xx.transpose(2,0,1));ts.append(t.astype('int64'))
   logits=net(torch.from_numpy(np.stack(xs)).cuda());targets=torch.from_numpy(np.stack(ts)).cuda();loss=partial_role_loss(logits,targets,weights[targets.clamp(0,2)]*targets.ne(255));opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
   if (step+1)%150==0:print(name,step+1,round(losses[-1],4),flush=True)
  net.eval();models[name]=net;curves[name]=losses;torch.save(dict(model=net.state_dict(),protocol=protocol,arm=name),ROOT/(name+'.pt'))
 hashes={name:sha(ROOT/(name+'.pt')) for name in models};write('checkpoint_freeze.json',hashes);(ROOT/'predictions').mkdir(exist_ok=True);results=[]
 for r in rows:
  image,t=data[r['id']];score={}
  for name,net in models.items():
   prob,p=predict(net,image);np.savez_compressed(ROOT/'predictions'/(r['id']+'_'+name+'.npz'),prediction=p,probability=prob.astype('float16'));score[name]=metrics(confusion(t,p))
  results.append(dict(id=r['id'],split=r['split'],scores=score))
 aggregates={}
 for split in sorted({r['split'] for r in rows}):
  aggregates[split]={name:metrics(sum(np.array(r['scores'][name]['confusion']) for r in results if r['split']==split)) for name in models}
 # The user's known foliage failure remains regression only, never a gradient.
 h=np.array(Image.open(LAB/'cunfa_roles_v2_20261112/references/H_14_input.png').convert('RGB'));ref=np.load(LAB/'cunfa_object_correction_20261112/H_14_object_reference.npz');region=ref['object_verified'];regression={}
 for name,net in models.items():
  prob,p=predict(net,h);np.savez_compressed(ROOT/'predictions'/('H_14_'+name+'.npz'),prediction=p,probability=prob.astype('float16'));regression[name]=dict(reference_region_pixels=int(region.sum()),object_predictions_in_user_foliage={str(k):int(((p==k)&region).sum()) for k in range(3)},tree_region_recall=float((p[region]==2).mean()),mean_tree_probability=float(prob[2][region].mean()),mean_confidence=float(prob.max(0)[region].mean()),not_new_blind_test=True,not_exact_foliage_boundary_gold=True)
 result=dict(protocol=protocol,checkpoint_hashes=hashes,records=results,aggregates=aggregates,user_foliage_regression=regression,loss_curves=curves,duration_seconds=time.time()-started,adopted=False,stage1_object_acceptance=False,stage1_cunfa_acceptance=False,tree_skill_training_enabled=False,robot_connected=False);write('results.json',result);print(json.dumps(dict(aggregates=aggregates,foliage=regression),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
