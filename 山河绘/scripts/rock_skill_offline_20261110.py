# -*- coding: utf-8 -*-
"""Auditable visible-curve references and dependency planner, never artist gold.

The raw learned output is kept separate. No robot imports or control endpoints.
"""
import sys, json, hashlib, math, heapq
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure, LAB
configure()
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = LAB / 'rock_skill_offline_20261110'
SOURCE = LAB / 'pima_one_rock_cycle_20261109/finished_frame.jpg'
BOX = [625, 225, 810, 390]
# Assistant observations in the crop, not recovered contacts or unique strokes.
# Dense/overlapping left margins and invisible closures deliberately unlabelled.
GUIDES = {
 'A': {'name':'后方山石', 'rank':0, 'contour':[
  [[45,27],[57,23],[71,27],[86,27],[97,24],[111,19],[125,19],[133,24],[138,37],[145,48],[151,55],[154,64]],
  [[155,67],[164,73],[172,80],[174,85],[169,91],[163,96],[155,99]]],
  'structure':[[[61,39],[77,47],[91,54],[108,68],[119,79],[126,92],[131,101]]],
  'cun':[
   [[75,35],[88,43],[100,53],[110,66],[115,78]],
   [[92,39],[105,48],[116,61],[123,75],[127,88]],
   [[107,45],[119,55],[128,69],[134,84],[137,95]],
   [[125,51],[135,64],[141,79],[143,95]],
   [[143,64],[148,76],[152,89],[153,98]],
   [[156,77],[161,85],[160,94]]]},
 'B': {'name':'中间山石', 'rank':1, 'contour':[
  [[25,89],[34,81],[43,78],[51,73],[60,74],[71,77],[80,77],[89,82],[98,88],[108,94]],
  [[112,99],[119,105],[126,113],[126,119],[120,122]]],
  'structure':[[[50,84],[66,90],[80,99],[89,110],[91,124],[88,135]]],
  'cun':[
   [[62,84],[77,91],[88,100],[95,112],[96,125]],
   [[77,87],[89,96],[99,107],[103,118]],
   [[94,96],[105,103],[111,115]],
   [[110,106],[115,115],[115,122]],
   [[54,94],[65,102],[74,115],[78,129]],
   [[48,101],[58,109],[65,122],[68,135]]]},
 'C': {'name':'前方山石', 'rank':2, 'contour':[
  [[3,120],[11,110],[20,103],[31,98],[40,95],[47,96],[53,100],[56,107]],
  [[5,134],[7,143],[16,147],[24,144],[30,147],[37,146],[42,145]]],
  'structure':[[[32,104],[39,114],[43,127],[43,138]]],
  'cun':[
   [[16,112],[14,124],[13,138]],
   [[23,109],[23,122],[25,137],[27,144]],
   [[29,110],[32,122],[33,137],[35,144]],
   [[38,111],[43,121],[46,133],[47,141]],
   [[45,108],[52,118],[56,130],[56,138]],
   [[54,115],[60,125],[61,136]]]},
}
ROLES = {'contour':'主轮廓','structure':'结构转折','cun':'皴笔候选'}
COLORS = {'contour':'#9b6423','structure':'#426f4c','cun':'#277a9d'}

def write(name, obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')

def resample(points, step=1.):
 q=np.asarray(points,float); a=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
 t=np.linspace(0,a[-1],max(2,math.ceil(a[-1]/step)+1))
 return np.column_stack([np.interp(t,a,q[:,i]) for i in (0,1)])

def measure(q, gray):
 """Observable profile only; never invert image width into brush pressure."""
 tangent=np.gradient(q,axis=0); tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-6)
 normal=np.column_stack([-tangent[:,1],tangent[:,0]])
 offsets=np.arange(-7,8)
 xy=q[:,None,:]+normal[:,None,:]*offsets[None,:,None]
 v=cv2.remap(gray,xy[:,:,0].astype('float32'),xy[:,:,1].astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
 background=np.quantile(v,.85,axis=1); contrast=background-v[:,7]
 widths=[]
 for i,row in enumerate(v):
  threshold=max(.018,float(contrast[i])*.35); lo=hi=7
  while lo>0 and background[i]-row[lo-1]>threshold:lo-=1
  while hi<14 and background[i]-row[hi+1]>threshold:hi+=1
  widths.append(hi-lo+1 if contrast[i]>.018 else None)
 return dict(observed_width_px=widths,relative_darkness=contrast.clip(0,1).tolist(),
  low_contrast_fraction=float((contrast<.018).mean()),profile_kind='local image estimate; overlap and paper texture may contaminate',pressure_mm=None,speed_mm_s=None,ink_quantity=None)

def dependencies(records):
 """Object completion takes precedence over distance; unknowns are excluded."""
 previous=[];edges=[]
 for module in ['A','B','C']:
  prior=previous
  for role in ['contour','structure','cun']:
   nodes=[r['id'] for r in records if r['object_id']==module and r['role']==role]
   for n in nodes:
    for p in prior:edges.append([p,n])
   if nodes:prior=nodes
  previous=prior
 return edges

def schedule(records,edges):
 by={r['id']:r for r in records};pending={k:0 for k in by};children={k:[] for k in by}
 for a,b in edges:pending[b]+=1;children[a].append(b)
 ready=[k for k in by if not pending[k]];result=[];current=np.array([0.,0.])
 while ready:
  k=min(ready,key=lambda k:(by[k]['object_rank'],['contour','structure','cun'].index(by[k]['role']),min(np.linalg.norm(np.array(by[k]['curve'])[[0,-1]]-current,axis=1))))
  ready.remove(k);r=dict(by[k]);q=np.array(r['curve']);reverse=np.linalg.norm(q[-1]-current)<np.linalg.norm(q[0]-current)
  if reverse:q=q[::-1]
  r['execution_curve']=q.tolist();r['direction_basis']='travel optimisation; original artist direction unknown';r['sequence']=len(result)+1
  result.append(r);current=q[-1]
  for j in children[k]:
   pending[j]-=1
   if not pending[j]:ready.append(j)
 if len(result)!=len(records):raise ValueError('Dependency cycle')
 return result

def draw_paths(records, width_mode=False, limit=None):
 im=Image.new('RGB',(740,660),'white');d=ImageDraw.Draw(im)
 for r in records[:limit]:
  q=np.array(r.get('execution_curve',r['curve']))*4
  if width_mode:
   p=r['profile'];ws=p['observed_width_px'];dark=p['relative_darkness']
   # Width is clipped for a target illustration, never a physical prediction.
   for i in range(len(q)-1):
    j=min(i,len(ws)-1);w=ws[j]
    if w is None:continue
    val=int(255*(1-min(.90,max(.2,dark[j]*3))))
    d.line([tuple(q[i]),tuple(q[i+1])],fill=(val,val,val),width=max(1,round(min(w,3)*2)))
  else:d.line([tuple(p) for p in q],fill=COLORS[r['role']],width=3)
 return im

def main():
 OUT.mkdir(exist_ok=True)
 rgb=np.array(Image.open(SOURCE).convert('RGB'))[BOX[1]:BOX[3],BOX[0]:BOX[2]]
 Image.fromarray(rgb).resize((740,660)).save(OUT/'source_crop.png')
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype('float32')/255
 records=[]
 for mid,m in GUIDES.items():
  for role in ROLES:
   for j,g in enumerate(m[role]):
    q=resample(g)
    records.append(dict(id=f'{mid}_{role}_{j+1:02d}',object_id=mid,object_rank=m['rank'],object_name=m['name'],role=role,
     curve=q.tolist(),guide=g,profile=measure(q,gray),group_id=mid+'_'+role,
     source_kind='assistant static visible-curve reference',artist_stroke_identity_verified=False,
     original_stroke_order=None,original_stroke_direction=None,style='uploader title: short pima; material and author unverified',
     endpoint_uncertainty_source_px=5,expert_verified=False,training_admitted=False))
 edges=dependencies(records);plan=schedule(records,edges)
 write('visible_reference.json',dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
  source_url='https://www.ganjingworld.com/video/1gc2lo9n27j5ZShwAja69iQ3N12d1c',box=BOX,
  complete_artist_stroke_annotation=False,unlabelled_regions_are_negative=False,records=records,
  omitted='Ambiguous overlaps, wash, dense left ink, faint uncertain marks and hidden object boundaries',robot_connected=False))
 write('plan.json',dict(plan_kind='reference-driven explicit dependency planner, not learned sequencing',
  object_detection='assistant assigned; not automatic',dependencies=edges,strokes=plan,robot_connected=False,approved_for_execution=False))
 draw_paths(plan).save(OUT/'reference_roles.png')
 # Retain observed ink variation instead of rendering smooth, invented cunfa.
 corridors=[]
 for r in records:
  m=np.zeros(gray.shape,np.uint8);cv2.polylines(m,[np.round(r['curve']).astype('int32')],False,1,9);corridors.append(m)
 corridors=np.stack(corridors);local_bg=cv2.GaussianBlur(gray,(0,0),7)
 ink=np.clip((local_bg-gray-.006)/.20,0,1)
 ambiguous=(corridors.sum(0)>1)&(ink>.06)
 candidate=1-ink*(corridors.any(0))
 observed=np.repeat(candidate[:,:,None],3,axis=2)
 observed[ambiguous]=observed[ambiguous]*.55+np.array([1.,.76,.3])*.45
 Image.fromarray(np.uint8(observed*255)).resize((740,660)).save(OUT/'width_target.png')
 np.savez_compressed(OUT/'observed_ink_candidates.npz',corridors=corridors,relative_ink=ink,unknown_overlap=ambiguous)
 overlay=Image.fromarray(rgb).resize((740,660));od=ImageDraw.Draw(overlay)
 for r in records:od.line([tuple(p) for p in np.array(r['curve'])*4],fill=COLORS[r['role']],width=2)
 overlay.save(OUT/'reference_overlay.png')
 # New raw inference on this exact crop; no reference, stage frame or guide input.
 import torch
 from gongbi_training.rock_context_model import RockContextNet
 sys.path.insert(0,str(Path(__file__).resolve().parent))
 from train_rock_context_joint_20261109 import infer
 torch.set_num_threads(4)
 checkpoint=LAB/'rock_context_joint_20261109/whole_context_step1250.pt'
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
 net=RockContextNet(project_curve=True).cuda();net.load_state_dict(ck['model']);net.eval()
 image=cv2.resize(rgb,(256,256)).astype('float32')/255
 pred,_=infer(net,image,threshold=.35,max_predictions=32)
 raw=[];pic=Image.new('RGB',(740,660),'white');d=ImageDraw.Draw(pic)
 for i,p in enumerate(pred):
  q=np.array(p['curve'])*[185/256,165/256];r={k:v for k,v in p.items() if k!='mask'};r['curve']=q.tolist();r['id']=f'raw_{i:02d}';raw.append(r)
  d.line([tuple(v) for v in q*4],fill=COLORS['contour' if p['role']==0 else 'cun'],width=2)
 pic.save(OUT/'model_direct.png')
 write('model_direct.json',dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
  new_weights_trained=False,reference_input=False,manual_module_crop=True,curves=raw,adopted=False,
  validation='same work as prior training; spatially different region, development only'))
 # Observable coverage diagnostics, not global accuracy or artist-stroke recall.
 coverage=[]
 for r in records:
  q=np.array(r['curve']);dist=min([np.linalg.norm(q[:,None,:]-np.array(p['curve'])[None,:,:],axis=2).min(1).mean() for p in raw],default=float('inf'))
  coverage.append(dict(id=r['id'],nearest_raw_curve_mean_distance_source_px=float(dist)))
 jumps=sum(a['object_id']!=b['object_id'] for a,b in zip(plan,plan[1:]));pos={r['id']:i for i,r in enumerate(plan)}
 assert all(pos[a]<pos[b] for a,b in edges)
 assert jumps==2
 # Repeatability + cycle rejection + role order, not a visual-quality test.
 assert [r['id'] for r in schedule(records,edges)]==[r['id'] for r in plan]
 try:schedule(records,edges+[[plan[-1]['id'],plan[0]['id']]])
 except ValueError:cycle_rejected=True
 else:raise AssertionError('cycle not rejected')
 baseline=json.loads((LAB/'pima_single_stroke_20261109/verification.json').read_text('utf8'))
 fingerprints=[]
 for r in baseline['approved_plans_fingerprints']:
  digest=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest();assert digest==r['sha256'];fingerprints.append(r)
 write('verification.json',dict(reference_segments=len(records),objects=3,raw_model_candidates=len(raw),
  object_switches=jumps,dependency_violations=0,cycle_rejected=cycle_rejected,deterministic=True,
  recognised_baselines_unchanged=fingerprints,raw_reference_coverage_diagnostic=coverage,
  new_expert_verified_strokes=0,complete_true_stroke_dataset=False,model_adopted=False,robot_connected=False))
 # Numbered reference replay is a planner demonstration, not a simulated robot.
 frames=[]
 for i in range(len(plan)+1):
  im=draw_paths(plan,limit=i).resize((555,495));frames.append(im)
 frames[0].save(OUT/'layer_replay.gif',save_all=True,append_images=frames[1:],duration=[700]+[180]*len(plan),loop=0)
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
 board=Image.new('RGB',(1480,495),'#f4f1e9');bd=ImageDraw.Draw(board)
 for j,(name,title) in enumerate([('source_crop.png','原图 · 右侧山石组'),('reference_overlay.png','新增人工结构参考 · 非模型输出'),('model_direct.png','当前模型直接输出 · 未通过'),('width_target.png','可见墨迹候选 · 橙色交叠待定')]):
  x=j*370;bd.text((x+10,15),title,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17),fill='#253c35')
  im=Image.open(OUT/name);im.thumbnail((360,650));board.paste(im,(x+5,65))
 bd.text((15,415),'本轮新增：三个对象的阶段依赖与逐笔回放；未宣称新模型已学会皴法。',font=font,fill='#253c35')
 bd.text((15,455),'金色：轮廓　绿色：结构转折　蓝色：皴笔候选；参考描线不是画家的真实分笔。',font=font,fill='#555')
 board.save(OUT/'comparison.png')
 print(json.dumps(dict(records=len(records),raw=len(raw),objects=3,jumps=jumps,root=str(OUT)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
