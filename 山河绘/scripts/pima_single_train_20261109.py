# -*- coding: utf-8 -*-
"""Train a real-event-conditioned neural curve tracer, never fixed stroke slots.

Input: isolated before/after event imagery + observed initial tip/direction.
No guide path, target endpoint or target length is passed to rollout. Only two
real candidate strokes train it; one same-rock stroke is DEVELOPMENT, not test.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from pima_stroke_inspect_20261109 import ROOT
import json,time,copy,math,shutil
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
STEP=1.5;PATCH=32
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def unit(v):return v/max(float(np.linalg.norm(v)),1e-6)
def patch_at(images,point):
 return np.stack([cv2.getRectSubPix(a.astype(np.float32),(PATCH,PATCH),tuple(map(float,point))) for a in images])
class CurveTracer(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.Sequential(nn.Conv2d(2,12,5,2,2),nn.SiLU(),nn.Conv2d(12,24,3,2,1),nn.SiLU(),nn.Conv2d(24,32,3,2,1),nn.SiLU(),nn.Flatten());self.head=nn.Sequential(nn.Linear(514,96),nn.SiLU(),nn.Linear(96,4))
 def forward(self,x,previous):
  raw=self.head(torch.cat([self.encoder(x),previous],1));direction=F.normalize(previous+.75*raw[:,:2].tanh(),dim=1);return direction,raw[:,2],F.softplus(raw[:,3])
def records():
 manifest=json.loads((ROOT/'single_reference_manifest.json').read_text('utf8'));out=[]
 for r in manifest['records']:
  if r['status']!='admitted_candidate':continue
  z=np.load(ROOT/'single_reference'/(r['id']+'_reference.npz'));q=z['curve'];images=np.stack([z['delta'],1-z['gray']]);out.append(dict(meta=r,q=q,images=images,width=z['width'],width_valid=z['appearance_valid']))
 return out
def samples(data):
 rows=[]
 for r in data:
  q=r['q']
  for j in range(len(q)):
   prev=unit(q[max(j,1)]-q[max(j,1)-1]);nxt=unit(q[min(j+1,len(q)-1)]-q[j]) if j<len(q)-1 else prev
   rows.append(dict(patch=patch_at(r['images'],q[j]),images=r['images'],point=q[j],future=q[min(j+2,len(q)-1)],prev=prev,nxt=nxt,stop=float(j>=len(q)-2),width=float(r['width'][j]),width_valid=float(r['width_valid'][j]),event=r['meta']['id']))
  # Recovery/termination augmentations are derived from the two real events;
  # they are not additional real strokes or fabricated artist motion labels.
  for overrun in [1.5,3.,5.,8.]:
   pr=unit(q[-1]-q[-3]);point=q[-1]+pr*overrun
   rows.append(dict(patch=patch_at(r['images'],point),images=r['images'],point=point,future=point,prev=pr,nxt=pr,stop=1.,width=0.,width_valid=0.,event=r['meta']['id']))
 return rows
def batch(rows,indices,rng,augment=True):
 patches=[];prevs=[];targets=[];stops=[];widths=[];valid=[]
 for idx in indices:
  r=rows[int(idx)];patch=r['patch'].copy();pr=r['prev'].copy();target=r['nxt'].copy()
  if augment:
   offset=rng.normal(0,1.25,2).astype(np.float32);point=r['point']+offset;patch=patch_at(r['images'],point)
   if not r['stop']:target=unit(r['future']-point)
   angle=float(rng.uniform(0,360));rot=cv2.getRotationMatrix2D((15.5,15.5),angle,1);patch=np.stack([cv2.warpAffine(a,rot,(PATCH,PATCH),borderMode=cv2.BORDER_CONSTANT) for a in patch]);pr=rot[:,:2]@pr;target=rot[:,:2]@target
   perturb=float(rng.normal(0,.13));cs,sn=np.cos(perturb),np.sin(perturb);pr=np.array([[cs,-sn],[sn,cs]])@pr
   patch=np.clip(patch*rng.uniform(.8,1.15)+rng.normal(0,.012,patch.shape),0,1)
  patches.append(patch);prevs.append(pr);targets.append(target);stops.append(r['stop']);widths.append(r['width']);valid.append(r['width_valid'])
 return [torch.tensor(np.array(a),dtype=torch.float32,device='cuda') for a in [patches,prevs,targets,stops,widths,valid]]
def lossfn(net,b):
 x,pr,target,stop,width,wvalid=b;direction,slog,w=net(x,pr)
 d=(1-(direction*target).sum(1)).mean();s=F.binary_cross_entropy_with_logits(slog,stop);wl=(F.smooth_l1_loss(w,width,reduction='none')*wvalid).sum()/wvalid.sum().clamp_min(1)
 return d*3+s*.35+wl*.02,dict(direction=float(d.detach()),stop=float(s.detach()))
def rollout(net,images,start,direction,blank=False):
 q=[np.array(start,np.float32)];direction=unit(direction).astype(np.float32);logs=[];reason='iteration_limit'
 for step in range(300):
  p=patch_at(images,q[-1]);p=p*0 if blank else p
  with torch.no_grad():d,stop,w=net(torch.tensor(p[None],device='cuda'),torch.tensor(direction[None],device='cuda'));d=d[0].cpu().numpy();prob=float(stop.sigmoid()[0]);width=float(w[0])
  logs.append(dict(stop_probability=prob,width_image_pixels=width))
  if step>=3 and prob>.65:reason='predicted_lift';break
  # The network chooses the tangent; direction smoothing stabilises subpixel noise.
  direction=unit(direction*.35+d*.65).astype(np.float32);nxt=q[-1]+STEP*direction
  h,ww=images.shape[1:]
  if nxt[0]<1 or nxt[1]<1 or nxt[0]>=ww-1 or nxt[1]>=h-1:reason='event_crop_boundary';break
  q.append(nxt)
 return np.array(q),logs,reason
def distance(a,b):return np.linalg.norm(a[:,None]-b[None,:],axis=2)
def assess(pred,truth):
 dist=distance(pred,truth);pl=float(np.linalg.norm(np.diff(pred,axis=0),axis=1).sum());tl=float(np.linalg.norm(np.diff(truth,axis=0),axis=1).sum())
 return dict(symmetric_distance_px=float((dist.min(0).mean()+dist.min(1).mean())/2),endpoint_error_px=float(np.linalg.norm(pred[-1]-truth[-1])),length_ratio=pl/max(tl,1),points=len(pred))
def main():
 previous=ROOT/'teacher_forcing_only';previous.mkdir(exist_ok=True)
 for name in ['single_curve_tracer.pt','single_curve_predictions.npz','single_curve_training_report.json','single_curve_preview.jpg']:
  p=ROOT/name
  if p.exists() and not (previous/name).exists():shutil.copy2(p,previous/name)
 torch.manual_seed(1109);torch.set_num_threads(4);rng=np.random.default_rng(1109);assert torch.cuda.is_available()
 data=records();train=[r for r in data if r['meta']['split']=='train'];dev=[r for r in data if r['meta']['split']!='train'];assert len(train)==2 and len(dev)==1
 tr=samples(train);dr=samples(dev);end=np.array([i for i,r in enumerate(tr) if r['stop']]);normal=np.array([i for i,r in enumerate(tr) if not r['stop']]);net=CurveTracer().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);best=None;bestloss=1e9;hist=[];start=time.time()
 validation=batch(dr,range(len(dr)),rng,False)
 for update in range(801):
  if update:
   ids=np.r_[rng.choice(normal,26),rng.choice(end,6)];net.train();b=batch(tr,ids,rng);opt.zero_grad(set_to_none=True);loss,_=lossfn(net,b);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
  if update%100==0:
   net.eval()
   with torch.no_grad():vl,parts=lossfn(net,validation)
   rr=dev[0];free,_,why=rollout(net,rr['images'],rr['q'][0],unit(rr['q'][2]-rr['q'][0]));fm=assess(free,rr['q'])
   score=fm['symmetric_distance_px']+.2*fm['endpoint_error_px']+(3. if why!='predicted_lift' else 0.)
   hist.append(dict(update=update,same_rock_teacher_forced_loss=float(vl),free_development=fm,stop_reason=why,selection_score=score,**parts));print('TRACE_TRAIN',hist[-1],flush=True)
   if score<bestloss:bestloss=score;best=copy.deepcopy(net.state_dict());chosen=update
 net.load_state_dict(best);net.eval();results=[];arrays={}
 for r in data:
  q=r['q'];pr=unit(q[2]-q[0]);pred,logs,reason=rollout(net,r['images'],q[0],pr);blank,_,blankreason=rollout(net,r['images'],q[0],pr,True)
  line=np.linspace(q[0],q[-1],max(len(q),2));metrics=assess(pred,q);lm=assess(line,q);bm=assess(blank,q)
  row=dict(id=r['meta']['id'],split=r['meta']['split'],metrics=metrics,straight_endpoint_oracle=lm,blank_image_ablation=bm,stop_reason=reason,blank_stop_reason=blankreason,
    initial_tip_and_direction_supplied_from_video=True,initial_state_source='candidate reference first point and initial tangent; oracle seed, not autonomous detection',target_endpoint_or_length_supplied_to_model=False,guide_curve_supplied_to_model=False,predicted_curve=pred.tolist(),logs=logs)
  results.append(row);arrays[r['meta']['id']+'_prediction']=pred;arrays[r['meta']['id']+'_reference']=q;arrays[r['meta']['id']+'_blank']=blank
 torch.save(dict(model=best,updates=800,retained_update=chosen,architecture='local-image-conditioned autoregressive curve tracer',physical_parameters=False),ROOT/'single_curve_tracer.pt');np.savez_compressed(ROOT/'single_curve_predictions.npz',**arrays)
 report=dict(device=torch.cuda.get_device_name(0),updates=800,total_updates_including_teacher_forcing_pilot=1600,retained_update=chosen,seconds=time.time()-start,real_training_events=2,same_rock_development_events=1,independent_test_works=0,
  label_provenance='assistant-reviewed motion-assisted new-ink candidates, not expert gold',augmented_examples_are_not_new_real_strokes=True,history=hist,results=results,
  style_recognition_trained=False,autonomous_stroke_discovery=False,robot_execution=False,adopted=False,off_path_recovery_augmentation=True,stop_overrun_augmentation=True,
  input_scope='isolated event before/after difference, appearance, observed initial point and tangent; event crop is supplied',
  limitation='Pilot only: the development stroke comes from the same rock and work; full-image extraction and physical brush replication remain unvalidated.')
 save(ROOT/'single_curve_training_report.json',report);gallery(data,results)
 print('TRACE_RESULT',[(r['id'],r['metrics'],r['stop_reason']) for r in results],flush=True)
def gallery(data,results):
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15);board=Image.new('RGB',(1200,370*len(data)),'#f1eee6');d=ImageDraw.Draw(board)
 for row,(r,result) in enumerate(zip(data,results)):
  q=r['q'];pred=np.array(result['predicted_curve']);h,w=r['images'].shape[1:];appearance=Image.fromarray(np.uint8(np.clip(1-r['images'][1],0,1)*255)).convert('RGB');mask=Image.fromarray(np.uint8(np.clip(1-r['images'][0],0,1)*255)).convert('RGB');ref=Image.new('RGB',(w,h),'white');rr=ImageDraw.Draw(ref);rr.line([tuple(p) for p in q],fill='#0085bd',width=1);out=Image.new('RGB',(w,h),'white');dd=ImageDraw.Draw(out);dd.line([tuple(p) for p in pred],fill='black',width=1)
  for col,im in enumerate([appearance,mask,ref,out]):
   im=im.resize((max(1,round(im.width*300/im.height)),300));board.paste(im,(col*300+40,row*370+32));d.text((col*300+8,row*370+4),[r['meta']['id']+' 原始局部','逐笔前后新增墨迹（输入）','复核候选参考（非专家真值）','模型连续输出（原始）'][col],font=small,fill='#243e37')
  m=result['metrics'];d.text((8,row*370+337),f'{result["split"]} / 距离 {m["symmetric_distance_px"]:.2f}px / 终点误差 {m["endpoint_error_px"]:.1f}px / {result["stop_reason"]}',font=font,fill='#555')
 board.save(ROOT/'single_curve_preview.jpg',quality=95)
if __name__=='__main__':main()
