# -*- coding: utf-8 -*-
"""Whole-module supervision. Synthetic geometry is never named-cunfa truth.

Real data use sparse assistant-reviewed structural guides and four motion
candidates. Unknown ink is ignored, including all ambiguous intersections.
"""
from pathlib import Path
import json,cv2,numpy as np
from .runtime import configure,LAB
configure()
ROOT=LAB/'rock_context_joint_20261109'
PREV=LAB/'pima_one_rock_cycle_20261109'
SIZE=256
GRID=128
POINTS=24
BOX=[265,285,490,450]
# Visible structural segments, NOT verified artist stroke identities/directions.
CONTOURS=[
 ('C01',[[330,338],[340,333],[348,330],[352,317],[360,310],[369,306],[383,307],[394,314],[408,317],[420,328],[432,337],[443,343],[447,352]]),
 ('C02',[[330,338],[328,349],[316,362],[305,369],[299,379],[291,390],[285,394]]),
 ('C03',[[285,395],[288,405],[299,414],[316,419],[328,427],[342,429]]),
 ('C04',[[342,429],[354,431],[363,431],[372,433],[383,427],[390,427],[399,432],[416,431],[426,427],[436,430],[447,429],[460,426]])]

def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sample_curve(q,n=POINTS):
 q=np.array(q,np.float32);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))];keep=np.r_[True,np.diff(arc)>1e-5];arc=arc[keep];q=q[keep]
 return np.column_stack([np.interp(np.linspace(0,arc[-1],n),arc,q[:,k]) for k in (0,1)]).astype(np.float32)
def line_mask(q,shape=(SIZE,SIZE),width=2):
 m=np.zeros(shape,np.uint8);cv2.polylines(m,[np.round(q).astype(np.int32)],False,1,width,cv2.LINE_8);return m
def crop_transform(q):
 x0,y0,x1,y1=BOX;scale=min((SIZE-24)/(x1-x0),(SIZE-24)/(y1-y0));origin=np.array([(SIZE-(x1-x0)*scale)/2,(SIZE-(y1-y0)*scale)/2]);return (np.array(q)-[x0,y0])*scale+origin
def real_example():
 rgb=cv2.cvtColor(cv2.imread(str(PREV/'finished_frame.jpg')),cv2.COLOR_BGR2RGB);x0,y0,x1,y1=BOX;crop=rgb[y0:y1,x0:x1];scale=min((SIZE-24)/(x1-x0),(SIZE-24)/(y1-y0));origin=np.array([(SIZE-(x1-x0)*scale)/2,(SIZE-(y1-y0)*scale)/2]);mat=np.float32([[scale,0,origin[0]-x0*scale],[0,scale,origin[1]-y0*scale]])
 image=cv2.warpAffine(rgb,mat,(SIZE,SIZE),borderMode=cv2.BORDER_REFLECT_101);gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(float)/255;local=np.clip((cv2.GaussianBlur(gray,(0,0),5)-gray-.007)/.22,0,1)
 strokes=[]
 for key,q in CONTOURS:
  curve=sample_curve(crop_transform(q));corr=line_mask(curve,width=6);positive=(corr>0)&(local>.1)
  strokes.append(dict(id=key,curve=curve,role=0,group=-1,oriented=False,mask=positive.astype(np.float32),mask_valid=positive.astype(np.float32),reference='assistant structural segment; not artist stroke truth'))
 reg=json.loads((PREV/'final_state_registration.json').read_text('utf8'));ref=json.loads((PREV/'reference_manifest.json').read_text('utf8'));meta={r['id']:r for r in ref['records']}
 for r in reg['records']:
  if r['status']!='reviewed_candidate':continue
  curve=sample_curve(crop_transform(r['curve']));z=np.load(PREV/'references'/(r['id']+'.npz'));align=np.array(reg['matrix'],np.float32);m=cv2.warpAffine(z['ownership_candidate'],align,(rgb.shape[1],rgb.shape[0]));unk=cv2.warpAffine(z['unknown_overlap'],align,(rgb.shape[1],rgb.shape[0]),flags=cv2.INTER_NEAREST)
  m=cv2.warpAffine(m,mat,(SIZE,SIZE));unk=cv2.warpAffine(unk,mat,(SIZE,SIZE),flags=cv2.INTER_NEAREST);positive=(m>.12)&(local>.1)&(unk==0)
  strokes.append(dict(id=r['id'],curve=curve,role=1,group=0,oriented=True,mask=positive.astype(np.float32),mask_valid=positive.astype(np.float32),reference='motion candidate; final ownership partial; endpoints +/-5 source px',order=meta[r['id']]['observed_order_within_group']))
 # These narrow margin regions were visually checked as blank paper. No other
 # unmarked ink or rock surface becomes a negative example.
 paper=np.zeros((SIZE,SIZE),np.uint8)
 for box in [(8,226,248,250),(8,8,28,38),(224,8,248,25)]:
  a,b,c,d=box;paper[b:d,a:c]=1
 paper[local>.08]=0
 # Ambiguous boundary ownership where structural and texture guides approach
 # one another is explicitly removed from role/instance supervision.
 for r in strokes:
  conflict=np.zeros((SIZE,SIZE),np.uint8)
  for other in strokes:
   if other['id']!=r['id']:conflict|=line_mask(other['curve'],width=5)
  r['mask_valid'][conflict>0]=0;r['mask'][conflict>0]=0
  # A geometric boundary is not automatically a painted centreline. Only
  # supported parts of the manually guided segment supervise curve geometry.
  distance=cv2.distanceTransform((r['mask_valid']==0).astype(np.uint8),cv2.DIST_L2,3);xy=np.clip(np.round(r['curve']).astype(int),0,SIZE-1);r['curve_valid']=(distance[xy[:,1],xy[:,0]]<=3.5).astype(np.float32)
 return dict(image=image.astype(np.float32)/255,strokes=strokes,paper=paper,complete=False,source='short_pima_work_only',split='train',manual_module_box=BOX,synthetic=False)

def synthetic(seed):
 """Known renderer paths: variable stroke count, curved shapes, width and tone.

 This exercises the representation and optimization, not real pima morphology.
 """
 rng=np.random.default_rng(seed);paper=rng.uniform(.88,1.);image=np.clip(paper+rng.normal(0,.014,(SIZE,SIZE,3)),0,1).astype(np.float32);strokes=[]
 cx,cy=rng.uniform(110,146,2);rx,ry=rng.uniform(65,100),rng.uniform(55,90);n_arc=int(rng.integers(3,6));angle=rng.uniform(-.4,.4)
 def rotate(q):
  rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);return np.clip((q-[cx,cy])@rot.T+[cx,cy],8,SIZE-9).astype(np.float32)
 for j in range(n_arc):
  t=np.linspace(2*np.pi*j/n_arc,2*np.pi*(j+1)/n_arc,7);rad=1+rng.normal(0,.055,len(t));q=np.column_stack([cx+rx*np.cos(t)*rad,cy+ry*np.sin(t)*rad]);q=sample_curve(rotate(q));strokes.append(dict(curve=q,role=0,group=-1,oriented=False))
 groups=int(rng.integers(1,4));order=0
 for group in range(groups):
  base=np.array([cx+rng.uniform(-.55,.35)*rx,cy+rng.uniform(-.55,.15)*ry]);orientation=rng.uniform(-2.8,2.8);unit=np.array([np.cos(orientation),np.sin(orientation)]);normal=np.array([-unit[1],unit[0]])
  for j in range(int(rng.integers(2,6))):
   length=rng.uniform(18,65);t=np.linspace(0,1,7);bend=rng.uniform(-10,10)*np.sin(t*np.pi)+rng.normal(0,1.2,7);q=base+normal*j*rng.uniform(5,10)+unit[None,:]*t[:,None]*length+normal[None,:]*bend[:,None];q=sample_curve(rotate(q));strokes.append(dict(curve=q,role=1,group=group,oriented=True,order=order));order+=1
 for i,r in enumerate(strokes):
  alpha=np.zeros((SIZE,SIZE),np.float32);q=r['curve'];width=rng.uniform(.8,3.5);dark=rng.uniform(.25,.9)
  for j in range(len(q)-1):
   w=max(1,round(width*(.45+.65*np.sin(np.pi*(j+.5)/(len(q)-1)))));cv2.line(alpha,tuple(np.round(q[j]).astype(int)),tuple(np.round(q[j+1]).astype(int)),float(dark),w,cv2.LINE_AA)
  if rng.random()<.2:
   # Known dry gaps preserve the complete renderer path; explicitly synthetic.
   holes=(rng.random((SIZE,SIZE))<.15);alpha[holes]*=rng.uniform(0,.3)
  image*=1-alpha[:,:,None];r.update(id=f'P{i:02d}',mask=(alpha>.035).astype(np.float32),mask_valid=np.ones((SIZE,SIZE),np.float32),curve_valid=np.ones(POINTS,np.float32),reference='known procedural path, not real brush/cunfa')
 return dict(image=np.clip(image,0,1),strokes=strokes,paper=np.ones((SIZE,SIZE),np.uint8),complete=True,source=f'procedural_{seed}',split='procedural',synthetic=True)

def augment(r,seed):
 rng=np.random.default_rng(seed);theta=rng.uniform(-22,22);scale=rng.uniform(.86,1.08);m=cv2.getRotationMatrix2D((SIZE/2,SIZE/2),theta,scale);m[:,2]+=rng.uniform(-8,8,2)
 image=cv2.warpAffine(r['image'],m,(SIZE,SIZE),borderMode=cv2.BORDER_REFLECT_101);image=np.clip(image*rng.uniform(.9,1.07)+rng.uniform(-.025,.025),0,1);strokes=[]
 for s in r['strokes']:
  q=cv2.transform(s['curve'][:,None,:],m)[:,0]
  if q.min()<2 or q.max()>SIZE-3:continue
  st=dict(s,curve=q,mask=cv2.warpAffine(s['mask'],m,(SIZE,SIZE),flags=cv2.INTER_NEAREST),mask_valid=cv2.warpAffine(s['mask_valid'],m,(SIZE,SIZE),flags=cv2.INTER_NEAREST));strokes.append(st)
 return dict(r,image=image,strokes=strokes,paper=cv2.warpAffine(r['paper'],m,(SIZE,SIZE),flags=cv2.INTER_NEAREST))

def targets(r):
 heat=np.zeros((GRID,GRID),np.float32);valid=np.ones_like(heat) if r['complete'] else cv2.resize(r['paper'].astype(np.float32),(GRID,GRID),interpolation=cv2.INTER_NEAREST)
 endpoints=np.zeros((3,GRID,GRID),np.float32);endvalid=np.broadcast_to(valid,(3,GRID,GRID)).copy();roles=np.full((GRID,GRID),-1,np.int64);paper=cv2.resize(r['paper'],(GRID,GRID),interpolation=cv2.INTER_NEAREST)
 if r['complete']:roles[:]=0
 else:roles[paper>0]=0
 anchors=[];seen=set();yy,xx=np.mgrid[:GRID,:GRID]
 for s in r['strokes']:
  q=s['curve']/SIZE
  # The centroid of a bent contour can be blank paper. The arclength midpoint
  # lies on its actual trajectory; inference still predicts this anchor.
  mid=POINTS//2
  if not r['complete']:
   known=np.flatnonzero(s['curve_valid']>0)
   if len(known):mid=int(known[np.argmin(abs(known-mid))])
  center=q[mid];ix,iy=np.clip(np.floor(center*GRID).astype(int),0,GRID-1);collision=(int(ix),int(iy)) in seen
  seen.add((int(ix),int(iy)));gauss=np.exp(-((xx-ix)**2+(yy-iy)**2)/(2*1.15**2));heat=np.maximum(heat,gauss);valid[gauss>.015]=1
  for channel,points in [(0,[q[0],q[-1]])]+([(1,[q[0]]),(2,[q[-1]])] if s['oriented'] else []):
   for pt in points:
    ex,ey=pt*GRID-.5;g=np.exp(-((xx-ex)**2+(yy-ey)**2)/(2*1.35**2));endpoints[channel]=np.maximum(endpoints[channel],g);endvalid[channel,g>.01]=1
  positive=cv2.resize((s['mask']*s['mask_valid']).astype(np.uint8),(GRID,GRID),interpolation=cv2.INTER_NEAREST)>0
  existing=(roles>0)&positive&(roles!=s['role']+1);roles[positive]=s['role']+1;roles[existing]=-1
  anchor=np.array([(ix+.5)/GRID,(iy+.5)/GRID]);mv=cv2.resize(s['mask_valid'],(GRID,GRID),interpolation=cv2.INTER_NEAREST);mask=cv2.resize(s['mask'],(GRID,GRID),interpolation=cv2.INTER_NEAREST)
  if not r['complete']:mv=np.maximum(mv,paper)
  if not collision:anchors.append(dict(x=int(ix),y=int(iy),curve=q.astype(np.float32),relative=(q-anchor).astype(np.float32),role=s['role'],group=s['group'],oriented=s['oriented'],curve_valid=s['curve_valid'],mask=mask,mask_valid=mv,id=s['id']))
 return dict(heat=heat,heat_valid=valid,endpoints=endpoints,endpoint_valid=endvalid,roles=roles,anchors=anchors)

def prepare():
 from PIL import Image,ImageDraw,ImageFont
 ROOT.mkdir(exist_ok=True);real=real_example();Image.fromarray(np.uint8(real['image']*255)).save(ROOT/'real_training_module.png');im=Image.fromarray(np.uint8(real['image']*255)).resize((768,768));d=ImageDraw.Draw(im)
 for s in real['strokes']:
  q=s['curve']*3
  for j in range(len(q)-1):
   if s['curve_valid'][j] and s['curve_valid'][j+1]:d.line([tuple(q[j]),tuple(q[j+1])],fill='#c07c22' if s['role']==0 else '#177eab',width=3)
   elif j%2==0:d.line([tuple(q[j]),tuple(q[j+1])],fill='#aaa59a',width=1)
  d.text(tuple(q[0]),s['id'],fill='#a12e2e')
 im.save(ROOT/'real_partial_reference.png')
 labels=[dict(id=s['id'],curve=s['curve'].tolist(),curve_known_samples=s['curve_valid'].astype(int).tolist(),role=['structural_contour','cun_candidate'][s['role']],group=s['group'],oriented=s['oriented'],reference=s['reference'],known_positive_mask_pixels=int(s['mask_valid'].sum())) for s in real['strokes']]
 write(ROOT/'data_manifest.json',dict(real_train_works=1,real_training_module_count=1,structural_visible_segments=4,motion_curve_candidates=4,expert_verified_strokes=0,
  module_box=BOX,module_manually_selected=True,source_image=str(PREV/'finished_frame.jpg'),input_stage='completed static image, not intermediate frame or temporal delta',records=labels,
  unlabelled_pixels_are_negative=False,contour_artist_stroke_order_known=False,final_individual_ownership_complete=False,original_author_verified=False,
  synthetic_purpose='known geometry and joint-head pretraining only; no named cunfa supervision',robot_connected=False))
 print('DATA_READY',len(labels),flush=True)
if __name__=='__main__':prepare()
