# -*- coding: utf-8 -*-
"""Static full-frame proposals: endpoints are detected, never supplied by gold.

All raw hypotheses are retained. Crossings remain uncertain. The partial manual
visible-curve references score exploratory shape coverage, not stroke identity,
global precision, named cunfa accuracy or physical brush performance.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from pima_static_model_20261109 import ROOT,StaticTracer,features,trace_many,unit
import numpy as np,cv2,torch,json,time
from PIL import Image,ImageDraw,ImageFont
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def thin(mask):
 a=mask.astype(np.uint8).copy()
 for _ in range(100):
  removed=0
  for phase in [0,1]:
   p=np.pad(a,1);ns=[p[:-2,1:-1],p[:-2,2:],p[1:-1,2:],p[2:,2:],p[2:,1:-1],p[2:,:-2],p[1:-1,:-2],p[:-2,:-2]];count=sum(ns);trans=sum(((ns[i]==0)&(ns[(i+1)%8]==1)).astype(np.uint8) for i in range(8))
   if phase==0:other=(ns[0]*ns[2]*ns[4]==0)&(ns[2]*ns[4]*ns[6]==0)
   else:other=(ns[0]*ns[2]*ns[6]==0)&(ns[0]*ns[4]*ns[6]==0)
   erase=(a==1)&(count>=2)&(count<=6)&(trans==1)&other;removed+=int(erase.sum());a[erase]=0
  if not removed:break
 return a
def proposals(rgb,max_seeds=80):
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;ims=features(gray);mask=(ims[0]>.16).astype(np.uint8)
 # Exclude large chromatic hand regions, preserving small coloured ink details.
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);skin=((hsv[:,:,0]<28)&(hsv[:,:,1]>45)&(hsv[:,:,2]>70)).astype(np.uint8)
 n,labels,stats,_=cv2.connectedComponentsWithStats(skin,8);large=np.zeros(mask.shape,np.uint8)
 for k in range(1,n):
  if stats[k,cv2.CC_STAT_AREA]>1200:large[labels==k]=1
 large=cv2.dilate(large,np.ones((7,7),np.uint8));mask[large>0]=0
 # The occluder mask must affect tracking evidence too, not just seed finding.
 # This heuristic is specific to these teaching-video frames, not a validated
 # semantic hand detector for arbitrary coloured landscape paintings.
 ims[:,large>0]=0
 sk=thin(mask);degree=cv2.filter2D(sk,-1,np.ones((3,3),np.uint8))-sk;ys,xs=np.where((sk>0)&(degree==1));junction=((sk>0)&(degree>=3)).astype(np.uint8);jnear=cv2.dilate(junction,np.ones((7,7),np.uint8))
 candidates=[];h,w=mask.shape
 for x,y in zip(xs,ys):
  path=[(int(x),int(y))];seen=set(path)
  for _ in range(10):
   xx,yy=path[-1];opts=[]
   for dy in [-1,0,1]:
    for dx in [-1,0,1]:
     p=(xx+dx,yy+dy)
     if (dx or dy) and 0<=p[0]<w and 0<=p[1]<h and sk[p[1],p[0]] and p not in seen:opts.append(p)
   if len(opts)!=1:break
   path.append(opts[0]);seen.add(opts[0])
  if len(path)<5:continue
  direction=unit(np.array(path[-1])-path[0]);candidates.append(dict(point=[int(x),int(y)],direction=direction.tolist(),run=len(path),grid=[min(3,int(x)*4//w),min(3,int(y)*4//h)]))
 # Spatial round-robin prevents one dense tree crown consuming the entire budget.
 buckets={}
 for c in candidates:buckets.setdefault(tuple(c['grid']),[]).append(c)
 for b in buckets.values():b.sort(key=lambda x:(-x['run'],x['point'][1],x['point'][0]))
 selected=[]
 while len(selected)<max_seeds and any(buckets.values()):
  for key in sorted(buckets):
   if buckets[key] and len(selected)<max_seeds:selected.append(buckets[key].pop(0))
 return ims,sk,jnear,selected,dict(all_eligible_endpoints=len(candidates),traced_seeds=len(selected),untraced_due_to_budget=max(0,len(candidates)-len(selected)),hand_region_masked_pixels=int(large.sum()))
def resample_reference(points):
 q=np.array(points,np.float32);length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))];t=np.linspace(0,length[-1],int(length[-1])+1);return np.column_stack([np.interp(t,length,q[:,i]) for i in [0,1]])
def evaluate_reference(curves,ref):
 q=resample_reference(ref['points']);best=None
 for c in curves:
  if len(c['curve'])<8:continue
  a=np.array(c['curve'],np.float32);dist=np.linalg.norm(q[:,None]-a[None,:],axis=2);sym=float((dist.min(0).mean()+dist.min(1).mean())/2)
  result=dict(reference_id=ref['id'],candidate_id=c['id'],symmetric_distance_px=sym,visible_reference_coverage_within_3px=float((dist.min(1)<=3).mean()),
    annotation_uncertainty_px=ref['annotation_uncertainty_px'],artist_stroke_identity_known=False,global_precision=None)
  if best is None or sym<best['symmetric_distance_px']:best=result
 return best
def main():
 torch.set_num_threads(4);ck=torch.load(ROOT/'static_tracer.pt',map_location='cpu',weights_only=False);net=StaticTracer().cuda();net.load_state_dict(ck['model']);net.eval();refs=json.loads((ROOT/'external_visible_curve_references.json').read_text('utf8'))
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);allresults=[];started=time.time()
 names=['new_howto_frame_00.jpg','new_howto_frame_01.jpg','new_fuchun_frame_01.jpg']
 for name in names:
  rgb=np.array(Image.open(ROOT/'external'/name).convert('RGB'));ims,sk,jnear,seeds,stats=proposals(rgb);raw=trace_many(net,ims,seeds);curves=[];coverage=np.zeros(sk.shape,np.uint8)
  for i,r in enumerate(raw):
   q=r['curve'];xy=np.round(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,sk.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,sk.shape[0]-1)
   support=float((ims[0,xy[:,1],xy[:,0]]>.1).mean());junction=bool(jnear[xy[:,1],xy[:,0]].any());length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum());duplicate=float((coverage[xy[:,1],xy[:,0]]>0).mean())>.85
   status='uncertain' if junction or r['stop_reason']!='learned_endpoint' or support<.65 else 'candidate_only'
   if length<10:status='short_candidate'
   if duplicate:status='overlap_duplicate_candidate'
   if not duplicate and length>=10:cv2.polylines(coverage,[xy],False,1,3)
   curves.append(dict(id=f'{name}_{i:03d}',curve=q.tolist(),seed=seeds[i],stop_reason=r['stop_reason'],ink_support_fraction=support,junction_encountered=junction,length_px=length,status=status))
  matched=[evaluate_reference(curves,r) for r in refs if r['image']==name]
  result=dict(image=name,stats=stats,candidates=curves,reference_matches=matched,manual_seed_or_curve_given_at_inference=False,
   temporal_before_after_given_at_inference=False,named_style_accuracy=None,global_precision=None,approved_drawing=False)
  allresults.append(result)
  overlay=Image.fromarray(rgb);d=ImageDraw.Draw(overlay)
  palette=['#1874a6','#b4771f','#167b60','#7c5a9a']
  for i,c in enumerate(curves):
   if c['status'] in ['short_candidate','overlap_duplicate_candidate']:continue
   d.line([tuple(p) for p in c['curve']],fill=palette[i%4],width=2)
  comparison=Image.new('RGB',(rgb.shape[1]*2,rgb.shape[0]+44),'#f1eee6');comparison.paste(Image.fromarray(rgb),(0,44));comparison.paste(overlay,(rgb.shape[1],44));draw=ImageDraw.Draw(comparison);draw.text((10,7),'新作品原始画面',font=font,fill='#243e37');draw.text((rgb.shape[1]+10,7),f'自动候选覆盖（{len(seeds)}个种子，尚未通过）',font=font,fill='#243e37');comparison.save(ROOT/(Path(name).stem+'_proposals.jpg'),quality=94)
  print('DISCOVERY',name,stats,matched,flush=True)
 save(ROOT/'static_discovery_report.json',dict(seconds=time.time()-started,images=allresults,independent_external_works=2,original_artist_disjointness_verified=False,
  external_sources_used_for_training=False,external_sources_used_for_checkpoint_selection=False,expert_stroke_truth=False,adopted=False,robot_connected=False))
if __name__=='__main__':main()
