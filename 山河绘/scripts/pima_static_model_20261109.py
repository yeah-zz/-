# -*- coding: utf-8 -*-
"""Static-image neural tracing, learned endpoint offset, explicit stress augmentation.

Two original training events, one SAME-ROCK development event. New public works
are never read by training. Artificial gaps/crossings/bends are labelled as
stress interventions, not new real demonstrations or named-cunfa gold data.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from pima_static_protocol_20261109 import ROOT
from pima_single_train_20261109 import records as old_records,unit,patch_at,assess
import numpy as np,cv2,json,time,copy,torch
from torch import nn
from torch.nn import functional as F
STEP=1.5
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def features(gray):
 gray=gray.astype(np.float32);paper=max(float(np.quantile(gray,.92)),.1);bg=cv2.GaussianBlur(gray,(0,0),7)
 local=np.clip((bg-gray-.008)/.23,0,1);relative=np.clip((paper-gray)/max(paper,.2),0,1)
 return np.stack([local,relative]).astype(np.float32)
def smooth_reference(q):
 qq=cv2.GaussianBlur(q[:,None,:],(1,5),1).reshape(-1,2);qq[0]=q[0];qq[-1]=q[-1];off=qq-q;norm=np.linalg.norm(off,axis=1);qq=q+off*np.minimum(1,.6/np.maximum(norm,1e-6))[:,None]
 return qq.astype(np.float32)
class StaticTracer(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.Sequential(nn.Conv2d(2,12,5,2,2),nn.SiLU(),nn.Conv2d(12,24,3,2,1),nn.SiLU(),nn.Conv2d(24,32,3,2,1),nn.SiLU(),nn.Flatten());self.head=nn.Sequential(nn.Linear(514,96),nn.SiLU(),nn.Linear(96,7))
 def forward(self,x,previous):
  a=self.head(torch.cat([self.encoder(x),previous],1));direction=F.normalize(previous+.9*a[:,:2].tanh(),dim=1)
  return direction,a[:,2],F.softplus(a[:,3]),a[:,4].sigmoid(),a[:,5:7].tanh()*16
def variants(r,all_modes=True):
 q=smooth_reference(r['q'])+25;gray=1-r['images'][1];ims=features(gray);ims=np.pad(ims,((0,0),(25,25),(25,25)));out=[]
 modes=['base','dry_gap','known_crossing','bend_left','bend_right'] if all_modes else ['base']
 for mode in modes:
  a=ims.copy();curve=q.copy();h,w=a.shape[1:];mid=int(round(curve[len(curve)//2,1]))
  if mode=='dry_gap':a[:,max(0,mid-4):mid+5]=0
  if mode=='known_crossing':
   center=curve[len(curve)//2];p=np.array([center+[-20,16],center+[20,-16]],np.int32)
   for ch in range(2):cv2.polylines(a[ch],[p],False,.85 if ch==0 else .65,2,cv2.LINE_AA)
  if mode.startswith('bend'):
   amp=-20. if mode=='bend_left' else 20.;ys=np.arange(h,dtype=np.float32);t=np.clip((ys-curve[0,1])/max(curve[-1,1]-curve[0,1],1),0,1);shift=amp*np.sin(np.pi*t)
   xx,yy=np.meshgrid(np.arange(w,dtype=np.float32),ys);mapx=xx-shift[:,None]
   a=np.stack([cv2.remap(ch,mapx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for ch in a]);curve[:,0]+=np.interp(curve[:,1],ys,shift)
  out.append(dict(id=r['meta']['id']+'_'+mode,real_event=r['meta']['id'],mode=mode,q=curve,images=a,width=r['width'],width_valid=r['width_valid'],split=r['meta']['split']))
 return out
def training_rows(data):
 rows=[]
 for r in data:
  q=r['q'];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))];remaining=arc[-1]-arc
  for j in range(len(q)):
   prev=unit(q[max(j,1)]-q[max(j,1)-1]);nxt=unit(q[min(j+3,len(q)-1)]-q[j]) if j<len(q)-1 else prev
   rows.append(dict(images=r['images'],point=q[j],future=q[min(j+3,len(q)-1)],prev=prev,nxt=nxt,stop=float(j>=len(q)-2),remaining=float(min(1,remaining[j]/16)),end=q[-1],near=float(remaining[j]<16),width=float(r['width'][j]),wvalid=float(r['width_valid'][j])))
  for overrun in [1.5,3.,5.,8.]:
   pr=unit(q[-1]-q[-3]);point=q[-1]+pr*overrun;rows.append(dict(images=r['images'],point=point,future=point,prev=pr,nxt=pr,stop=1.,remaining=0.,end=q[-1],near=1.,width=0.,wvalid=0.))
 return rows
def batch(rows,indices,rng,augment=True):
 data=[[] for _ in range(9)]
 for i in indices:
  r=rows[int(i)];point=r['point'].copy();pr=r['prev'].copy();target=r['nxt'].copy()
  if augment:point+=rng.normal(0,1.1,2)
  patch=patch_at(r['images'],point);offset=r['end']-point
  if not r['stop']:target=unit(r['future']-point)
  if augment:
   angle=float(rng.uniform(0,360));rot=cv2.getRotationMatrix2D((15.5,15.5),angle,1);patch=np.stack([cv2.warpAffine(ch,rot,(32,32)) for ch in patch]);pr=rot[:,:2]@pr;target=rot[:,:2]@target;offset=rot[:,:2]@offset
   angle=float(rng.normal(0,.1));cs,sn=np.cos(angle),np.sin(angle);pr=np.array([[cs,-sn],[sn,cs]])@pr;patch=np.clip(patch*rng.uniform(.75,1.2)+rng.normal(0,.01,patch.shape),0,1)
  for arr,item in zip(data,[patch,pr,target,r['stop'],r['remaining'],offset,r['near'],r['width'],r['wvalid']]):arr.append(item)
 return [torch.tensor(np.array(a),dtype=torch.float32,device='cuda') for a in data]
def lossfn(net,b):
 x,pr,target,stop,rem,offset,near,width,wvalid=b;d,s,w,r,e=net(x,pr)
 direction=(1-(d*target).sum(1)).mean();st=F.binary_cross_entropy_with_logits(s,stop);dist=F.smooth_l1_loss(r,rem)
 endpoint=(F.smooth_l1_loss(e/16,offset.clamp(-16,16)/16,reduction='none').mean(1)*near).sum()/near.sum().clamp_min(1)
 wid=(F.smooth_l1_loss(w,width,reduction='none')*wvalid).sum()/wvalid.sum().clamp_min(1)
 return direction*3+st*.35+dist*.3+endpoint*.5+wid*.01
def trace_many(net,images,seeds,max_steps=400):
 paths=[[np.array(s['point'],np.float32)] for s in seeds];directions=np.array([unit(np.array(s['direction'],np.float32)) for s in seeds],np.float32);active=list(range(len(seeds)));reasons=['iteration_limit']*len(seeds);terminal=[None]*len(seeds)
 h,w=images.shape[1:]
 for step in range(max_steps):
  if not active:break
  patches=np.stack([patch_at(images,paths[k][-1]) for k in active]);pr=directions[active]
  with torch.no_grad():d,s,wi,re,ep=net(torch.tensor(patches,device='cuda'),torch.tensor(pr,device='cuda'));dd=d.cpu().numpy();ss=s.sigmoid().cpu().numpy();rr=re.cpu().numpy();ee=ep.cpu().numpy()
  nxtactive=[]
  for j,k in enumerate(active):
   if step>=3 and ss[j]>.55 and rr[j]<.18:
    if np.linalg.norm(ee[j])<=6:paths[k].append(paths[k][-1]+ee[j])
    reasons[k]='learned_endpoint';terminal[k]=dict(stop=float(ss[j]),remaining=float(rr[j]),offset=ee[j].tolist());continue
   direction=unit(directions[k]*.35+dd[j]*.65).astype(np.float32);point=paths[k][-1]+STEP*direction
   if min(point)<1 or point[0]>=w-1 or point[1]>=h-1:reasons[k]='image_boundary';continue
   if len(paths[k])>15 and np.linalg.norm(point-paths[k][-12])<2:reasons[k]='loop_uncertain';continue
   directions[k]=direction;paths[k].append(point);nxtactive.append(k)
  active=nxtactive
 return [dict(curve=np.array(p,np.float32),stop_reason=r,terminal=t) for p,r,t in zip(paths,reasons,terminal)]
def seed_for(q):return dict(point=q[0].tolist(),direction=unit(q[2]-q[0]).tolist())
def main():
 torch.manual_seed(1109);torch.set_num_threads(4);rng=np.random.default_rng(1109);assert torch.cuda.is_available()
 real=old_records();train=[v for r in real if r['meta']['split']=='train' for v in variants(r)];dev=[v for r in real if r['meta']['split']!='train' for v in variants(r)]
 rows=training_rows(train);ends=np.array([i for i,r in enumerate(rows) if r['stop']]);normal=np.array([i for i,r in enumerate(rows) if not r['stop']]);net=StaticTracer().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);history=[];bestscore=1e9;best=None;started=time.time()
 for step in range(1001):
  if step:
   ids=np.r_[rng.choice(normal,26),rng.choice(ends,6)];net.train();b=batch(rows,ids,rng);opt.zero_grad(set_to_none=True);loss=lossfn(net,b);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
  if step%200==0:
   net.eval();evaluations=[]
   for r in dev:
    out=trace_many(net,r['images'],[seed_for(r['q'])])[0];m=assess(out['curve'],r['q']);evaluations.append(dict(id=r['id'],mode=r['mode'],metrics=m,stop=out['stop_reason']))
   score=float(np.mean([e['metrics']['symmetric_distance_px']+.2*e['metrics']['endpoint_error_px']+(3 if e['stop']!='learned_endpoint' else 0) for e in evaluations]));history.append(dict(update=step,selection_score=score,same_rock_development=evaluations));print('STATIC_TRAIN',step,round(score,3),flush=True)
   if score<bestscore:bestscore=score;best=copy.deepcopy(net.state_dict());chosen=step
 net.load_state_dict(best);net.eval();results=[];arrays={}
 for r in train+dev:
  out=trace_many(net,r['images'],[seed_for(r['q'])])[0];m=assess(out['curve'],r['q']);key=r['id'];arrays[key+'_prediction']=out['curve'];arrays[key+'_reference']=r['q'];results.append(dict(id=key,real_event=r['real_event'],split=r['split'],mode=r['mode'],metrics=m,stop=out['stop_reason'],oracle_initial_state=True,synthetic_intervention=r['mode']!='base'))
 torch.save(dict(model=best,retained_update=chosen,architecture='static patch tangent + remaining-distance + endpoint-offset',input_contract='single static image; no before-after difference',real_train_ids=['S01','S02']),ROOT/'static_tracer.pt');np.savez_compressed(ROOT/'static_tracer_predictions.npz',**arrays)
 report=dict(updates=1000,retained_update=chosen,seconds=time.time()-started,device=torch.cuda.get_device_name(0),real_training_events=2,same_rock_real_development_events=1,
  augmented_training_variants=len(train),augmented_development_variants=len(dev),augmentation_is_not_new_real_data=True,real_crossing_dry_bend_supervision_added=False,
  history=history,results=results,reference_pre_smoothing_max_displacement_px=.6,raw_outputs_post_smoothing=False,
  external_sources_read_by_training=False,external_used_for_checkpoint_selection=False,physical_parameters=False,adopted=False)
 save(ROOT/'static_training_report.json',report);print('STATIC_RESULT',chosen,[(r['id'],round(r['metrics']['endpoint_error_px'],2),r['stop']) for r in results if r['mode']=='base'],flush=True)
if __name__=='__main__':main()
