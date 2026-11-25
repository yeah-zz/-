# -*- coding: utf-8 -*-
"""Image-grounded centreline candidates, explicit ambiguous branches.

No neural coordinate proposals or manual curve/endpoint inputs. No white-gap joining.
Parameters selected on known procedural paths; real references evaluation only.
This is an algorithmic readout baseline, not a new trained cunfa model.
"""
import sys,json,math,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_propagation,distance_transform_edt
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ink_mask_centerline_20261105 import skeleton_graph
OUT=LAB/'source_constrained_strokes_20261110'
PREV=LAB/'rock_skill_offline_20261110'
CONFIGS=[dict(name='balanced',strong=.035,weak=.012,ridge=.006),dict(name='strict',strong=.050,weak=.018,ridge=.009),dict(name='faint',strong=.025,weak=.008,ridge=.004)]

def save(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def evidence(rgb,config):
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype('float32')/255
 smooth=cv2.GaussianBlur(gray,(0,0),.6);bg=cv2.GaussianBlur(smooth,(0,0),4)
 contrast=bg-smooth
 valley=cv2.morphologyEx(smooth,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
 yy,xx=np.mgrid[:gray.shape[0],:gray.shape[1]].astype('float32');ridge=np.zeros_like(gray)
 # Both sides of a line must be lighter at some image scale/orientation.
 for radius in [1.,2.,3.5,5.]:
  for angle in np.linspace(0,np.pi,8,endpoint=False):
   dx,dy=radius*np.cos(angle),radius*np.sin(angle)
   a=cv2.remap(smooth,xx+dx,yy+dy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
   b=cv2.remap(smooth,xx-dx,yy-dy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
   ridge=np.maximum(ridge,np.minimum(a,b)-smooth)
 weak=(valley>config['weak'])&(contrast>config['weak']*.35)&(ridge>config['ridge'])
 strong=(valley>config['strong'])&(contrast>config['strong']*.45)&(ridge>config['ridge']*2)
 mask=binary_propagation(strong,mask=weak|strong)
 # Dark interiors are eligible only next to measured ridges, not flat darkness.
 dark=(smooth<.30)&(distance_transform_edt(~strong)<3)
 mask=binary_propagation(strong,mask=mask|dark)
 return mask,dict(gray=gray,ridge=ridge,contrast=contrast)

def centre_ridges(fields,mask):
 """Scale-normalised dark-ridge maxima instead of blob medial meshes.

 Never closes a bright seam. Missing links remain missing, not guessed strokes.
 """
 gray=fields['gray'];best=np.zeros_like(gray);normal_x=np.ones_like(gray);normal_y=np.zeros_like(gray)
 for sigma in [.7,1.1,1.7,2.6,3.7]:
  g=cv2.GaussianBlur(gray,(0,0),sigma)
  xx=cv2.Sobel(g,cv2.CV_32F,2,0,ksize=3)*sigma*sigma/4
  yy=cv2.Sobel(g,cv2.CV_32F,0,2,ksize=3)*sigma*sigma/4
  xy=cv2.Sobel(g,cv2.CV_32F,1,1,ksize=3)*sigma*sigma/4
  delta=np.sqrt((xx-yy)**2+4*xy**2);large=(xx+yy+delta)/2;small=(xx+yy-delta)/2
  score=np.maximum(large,0)*np.exp(-2*(small/np.maximum(large,1e-6))**2)
  score/=math.sqrt(sigma)
  theta=.5*np.arctan2(2*xy,xx-yy);update=score>best
  normal_x[update]=np.cos(theta)[update];normal_y[update]=np.sin(theta)[update];best=np.maximum(best,score)
 y,x=np.mgrid[:gray.shape[0],:gray.shape[1]].astype('float32')
 a=cv2.remap(best,x+normal_x,y+normal_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
 b=cv2.remap(best,x-normal_x,y-normal_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
 ridge=(best>=a)&(best>b)&(best>.002)&mask
 return ridge

def paths_from_mask(mask,join=True):
 xy,adj,sk=skeleton_graph(mask)
 visited=set();chains=[]
 def edge(a,b):return (min(a,b),max(a,b))
 # Every graph edge belongs to exactly one chain; no repeated tracing.
 for start in range(len(xy)):
  if len(adj[start])==2:continue
  for nxt in adj[start]:
   if edge(start,nxt) in visited:continue
   chain=[start,nxt];visited.add(edge(start,nxt))
   while len(adj[chain[-1]])==2:
    cur=chain[-1];other=adj[cur][0] if adj[cur][0]!=chain[-2] else adj[cur][1]
    if edge(cur,other) in visited:break
    chain.append(other);visited.add(edge(cur,other))
   chains.append(chain)
 # Preserve cycles separately, without turning them into confident brush strokes.
 for start in range(len(xy)):
  for nxt in adj[start]:
   if edge(start,nxt) in visited:continue
   chain=[start,nxt];visited.add(edge(start,nxt))
   while True:
    options=[a for a in adj[chain[-1]] if edge(chain[-1],a) not in visited]
    if not options:break
    a=options[0];visited.add(edge(chain[-1],a));chain.append(a)
    if a==start:break
   chains.append(chain)
 incidence={}
 for i,chain in enumerate(chains):
  for end,node in enumerate([chain[0],chain[-1]]):incidence.setdefault(node,[]).append((i,end))
 links={};ambiguous=set()
 if join:
  for node,ends in incidence.items():
   if len(ends)<3:continue
   rays=[]
   for i,end in ends:
    ids=chains[i] if end==0 else chains[i][::-1];q=xy[ids];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
    j=min(len(q)-1,int(np.searchsorted(arc,4.)))
    v=q[j]-q[0];v/=max(np.linalg.norm(v),1e-8);rays.append(v)
   possible=[]
   for a in range(len(ends)):
    for b in range(a+1,len(ends)):
     if ends[a][0]==ends[b][0]:continue
     possible.append((float(rays[a]@rays[b]),a,b))
   possible.sort();used=set()
   for dot,a,b in possible:
    if a in used or b in used:continue
    alternatives=[v for v,c,d in possible if (a in (c,d) or b in (c,d)) and (c,d)!=(a,b)]
    margin=(min(alternatives)-dot) if alternatives else 2.
    if dot<-.82 and margin>.18:
     x,y=ends[a],ends[b];links[x]=y;links[y]=x;used.update([a,b])
    else:ambiguous.add(node)
 rows=[];usedchains=set()
 for i in range(len(chains)):
  if i in usedchains:continue
  component={i};stack=[i]
  while stack:
   k=stack.pop()
   for end in [0,1]:
    other=links.get((k,end))
    if other and other[0] not in component:component.add(other[0]);stack.append(other[0])
  starts=[(k,e) for k in component for e in [0,1] if (k,e) not in links]
  current=min(starts) if starts else (i,0);ids=[]
  while current[0] not in usedchains:
   k,end=current;usedchains.add(k);chain=chains[k] if end==0 else chains[k][::-1]
   ids.extend(chain if not ids else chain[1:]);other=links.get((k,1-end))
   if not other:break
   current=other
  q=xy[ids];length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()) if len(q)>1 else 0.
  if length==0:continue
  boundary=bool(np.any((q[:,0]<1)|(q[:,1]<1)|(q[:,0]>=mask.shape[1]-1)|(q[:,1]>=mask.shape[0]-1)))
  uncertain=ids[0] in ambiguous or ids[-1] in ambiguous or any(len(adj[z])>=3 for z in (ids[0],ids[-1]))
  rows.append(dict(id=f'path_{len(rows):04d}',curve=q.tolist(),length_px=length,
   status='short_pending' if length<4 else 'junction_pending' if uncertain else 'boundary_pending' if boundary else 'visible_connected_path',
   artist_stroke_identity_verified=False,role='unknown',object_id=None,approved_for_drawing=False))
 assert sum(len(a) for a in adj)//2==len(visited)
 assert len(usedchains)==len(chains)
 return rows,sk,dict(primitive_chains=len(chains),continuous_paths=len(rows),junction_continuations=len(links)//2,
  graph_edges=len(visited),every_edge_traced_once=True,ambiguous_junction_nodes=len(ambiguous))

def sample(q):
 q=np.array(q,float);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))];keep=np.r_[True,np.diff(arc)>1e-6];q=q[keep];arc=arc[keep]
 t=np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]*2))+1));return np.column_stack([np.interp(t,arc,q[:,i]) for i in (0,1)])

def continue_inside_ink(rows,mask):
 """Reconnect NMS sampling breaks only inside an existing measured ink ribbon.

 Endpoints must mutually prefer each other, face each other, and have a fully
 supported bridge. Competing continuations stay pending. No curve is removed.
 """
 ends=[]
 for i,r in enumerate(rows):
  q=np.array(r['curve'])
  if r['length_px']<2:continue
  for end in [0,1]:
   a=q if end==0 else q[::-1];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(a,axis=0),axis=1))]
   j=min(len(a)-1,int(np.searchsorted(arc,4.)))
   v=a[0]-a[j];v/=max(np.linalg.norm(v),1e-8)
   ends.append(dict(row=i,end=end,point=a[0],outward=v))
 candidates=[];h,w=mask.shape
 for a in range(len(ends)):
  for b in range(a+1,len(ends)):
   x,y=ends[a],ends[b]
   if x['row']==y['row']:continue
   delta=y['point']-x['point'];distance=np.linalg.norm(delta)
   if not .5<distance<=4.5:continue
   u=delta/distance
   if x['outward']@u<.72 or y['outward']@(-u)<.72:continue
   bridge=np.linspace(x['point'],y['point'],max(3,int(distance*4)+1));pix=np.round(bridge).astype(int)
   if np.any((pix[:,0]<0)|(pix[:,1]<0)|(pix[:,0]>=w)|(pix[:,1]>=h)):continue
   if not mask[pix[:,1],pix[:,0]].all():continue
   cost=float(distance+3*(2-x['outward']@u-y['outward']@(-u)))
   candidates.append((cost,a,b,bridge))
 choices={}
 for item in candidates:
  cost,a,b,bridge=item
  choices.setdefault(a,[]).append(item);choices.setdefault(b,[]).append(item)
 links={};used=set();parent=list(range(len(rows)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for cost,a,b,bridge in sorted(candidates,key=lambda x:x[0]):
  if a in used or b in used:continue
  ca=sorted(choices[a],key=lambda x:x[0]);cb=sorted(choices[b],key=lambda x:x[0])
  if ca[0][1:3]!=(a,b) or cb[0][1:3]!=(a,b):continue
  if any(len(z)>1 and z[1][0]-cost<1.0 for z in [ca,cb]):continue
  x,y=ends[a],ends[b]
  if find(x['row'])==find(y['row']):continue
  parent[find(x['row'])]=find(y['row']);used.update([a,b])
  links[(x['row'],x['end'])]=(y['row'],y['end'],bridge)
  links[(y['row'],y['end'])]=(x['row'],x['end'],bridge[::-1])
 done=set();result=[]
 for i in range(len(rows)):
  if i in done:continue
  component={k for k in range(len(rows)) if find(k)==find(i)}
  starts=[(k,e) for k in component for e in [0,1] if (k,e) not in links];current=min(starts) if starts else (i,0)
  points=[];members=[]
  while current[0] not in done:
   k,e=current;done.add(k);members.append(rows[k]['id']);q=np.array(rows[k]['curve']);q=q if e==0 else q[::-1]
   points.extend(q.tolist());other=links.get((k,1-e))
   if other is None:break
   points.extend(other[2][1:-1].tolist());current=other[:2]
  q=np.array(points);length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum())
  result.append(dict(id=f'ink_path_{len(result):04d}',curve=points,length_px=length,status='short_pending' if length<4 else 'connected_candidate_pending',
   merged_members=members,role='unknown',object_id=None,artist_stroke_identity_verified=False,approved_for_drawing=False))
 assert len(done)==len(rows)
 return result,dict(image_supported_continuations=len(links)//2,original_paths=len(rows),result_paths=len(result),white_gap_joining=False)

def raster(rows,shape):
 im=np.zeros(shape,'uint8')
 for r in rows:cv2.polylines(im,[np.round(r['curve']).astype('int32')],False,1,1)
 return im>0

def coverage(rows,refs):
 if not refs:return []
 points=np.concatenate([sample(r['curve']) for r in rows]) if rows else np.empty((0,2));tree=cKDTree(points) if len(points) else None
 results=[]
 for ref in refs:
  q=sample(ref['curve']);d=tree.query(q)[0] if tree else np.full(len(q),1e6)
  # Also measure how much ONE output path covers; union coverage hides fragments.
  one=0.
  for r in rows:
   a=sample(r['curve']);dd=cKDTree(a).query(q)[0];one=max(one,float((dd<=3).mean()))
  results.append(dict(id=ref['id'],role=ref.get('role','unknown'),union_visible_coverage_3px=float((d<=3).mean()),
   best_single_path_visible_coverage_3px=one,mean_distance_px=float(d.mean()),reference_is_partial_assistant_guide=True))
 return results

def view(rgb,rows,overlay=False):
 h,w=rgb.shape[:2];scale=min(680/w,560/h);base=Image.fromarray(rgb) if overlay else Image.new('RGB',(w,h),'white')
 im=base.resize((round(w*scale),round(h*scale)));d=ImageDraw.Draw(im)
 for r in rows:
  col='#279267' if overlay else '#292e2a'
  if r['status']=='short_pending':col='#c4c5bf'
  elif r['status'] in ('junction_pending','boundary_pending'):col='#779083' if overlay else '#737a75'
  d.line([tuple(p) for p in np.array(r['curve'])*scale],fill=col,width=2 if overlay else 1)
 return im

def main():
 OUT.mkdir(exist_ok=True);cv2.setNumThreads(2)
 from gongbi_training.rock_context_data import synthetic
 dev=[]
 for cfg in CONFIGS:
  metrics=[]
  for seed in range(771101,771107):
   r=synthetic(seed);rgb=np.uint8(r['image']*255);mask,_=evidence(rgb,cfg);truth=np.stack([s['mask'] for s in r['strokes']]).any(0)
   tp=(mask&truth).sum();metrics.append(float(2*tp/max(1,mask.sum()+truth.sum())))
  dev.append(dict(config=cfg,mean_procedural_ink_f1=float(np.mean(metrics))))
 cfg=max(dev,key=lambda z:z['mean_procedural_ink_f1'])['config'];save('config_selection.json',dict(development=dev,selected=cfg,reference_labels_used=False,real_images_used_for_selection=False))
 hold=[]
 for seed in range(773301,773309):
  r=synthetic(seed);ink,fields=evidence(np.uint8(r['image']*255),cfg);mask=centre_ridges(fields,ink)
  before,sk,stats=paths_from_mask(mask);paths,extra=continue_inside_ink(before,ink);paths0,sk0,stats0=paths_from_mask(mask,False)
  assert np.array_equal(raster(before,sk.shape),raster(paths0,sk.shape))
  assert not (raster(paths,sk.shape)&~ink).any()
  refs=[dict(id=s['id'],curve=s['curve'].tolist(),role=s['role']) for s in r['strokes']]
  hold.append(dict(seed=seed,**stats,**extra,coverage=coverage(paths,refs),junction_stage_preserves_pixels=True,added_continuations_inside_ink=True))
 save('procedural_holdout.json',hold)
 meta=json.loads((PREV/'visible_reference.json').read_text('utf8'));full=np.array(Image.open(meta['source']).convert('RGB'));a,b,c,d=meta['box'];rgb=full[b:d,a:c]
 examples=[dict(id='right_rocks',rgb=rgb,refs=meta['records'],raw=json.loads((PREV/'model_direct.json').read_text('utf8'))['curves'])]
 for ident in ['real_training_module','liupengkai_other_work','new_howto_frame_01']:
  im=np.array(Image.open(LAB/'rock_context_joint_20261109'/(ident+'_input.png')).convert('RGB'))
  examples.append(dict(id=ident,rgb=cv2.resize(im,(256,256)),refs=[]))
 results=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
 for e in examples:
  rgb=e['rgb'];ink_mask,fields=evidence(rgb,cfg);blob_rows,blob_sk,blob_stats=paths_from_mask(ink_mask)
  mask=centre_ridges(fields,ink_mask);ridge_rows,sk,stats=paths_from_mask(mask);unjoined,_,ustats=paths_from_mask(mask,False)
  rows,inside_stats=continue_inside_ink(ridge_rows,ink_mask)
  # No new gap or shortcut created by graph chain concatenation.
  assert np.array_equal(raster(ridge_rows,sk.shape),raster(unjoined,sk.shape))
  assert not (raster(rows,sk.shape)&~ink_mask).any()
  key=e['id'];Image.fromarray(rgb).save(OUT/(key+'_source.png'));Image.fromarray(np.uint8(255*(~ink_mask))).save(OUT/(key+'_ink.png'))
  view(rgb,blob_rows).save(OUT/(key+'_blob_skeleton_attempt.png'))
  view(rgb,rows).save(OUT/(key+'_paths.png'));view(rgb,rows,True).save(OUT/(key+'_overlay.png'))
  save(key+'_paths.json',dict(config=cfg,records=rows,manual_curve_input=False,manual_endpoint_input=False,manual_role_input=False,
   algorithm='source valley/ridge evidence + multiscale Hessian NMS + edge-disjoint graph chains + mutually preferred continuation entirely inside measured ink',
   trained_weights_updated=False,named_cunfa_recognition=False,complete_artist_strokes_recovered=False,approved_for_execution=False))
  refs=e['refs'];report=dict(id=key,**stats,**inside_stats,pending_short=sum(r['status']=='short_pending' for r in rows),
   source_grounded_pixels=int(sk.sum()),off_mask_pixels=int((raster(rows,sk.shape)&~ink_mask).sum()),
   earlier_blob_skeleton=blob_stats,earlier_blob_coverage=coverage(blob_rows,e['refs']),
   unjoined_coverage=coverage(unjoined,refs),source_constrained_coverage=coverage(rows,refs),
   raw_model_coverage=coverage(e.get('raw',[]),refs),reference_used_for_algorithm=False,
   original_real_annotation_complete=False,global_precision=None,real_work_blind_test=False)
  results.append(report)
  if key=='right_rocks':
   panels=[('原图',Image.fromarray(rgb)),('旧坐标模型 · 已拒绝',Image.open(PREV/'model_direct.png')),
    ('图像约束候选 · 非新权重',view(rgb,rows)),('原图叠加 · 淡灰为待定',view(rgb,rows,True))]
   board=Image.new('RGB',(1480,460),'#f4f1e9');bd=ImageDraw.Draw(board)
   for j,(title,im) in enumerate(panels):
    bd.text((j*370+8,10),title,font=font,fill='#223d34');im=im.copy();s=min(356/im.width,340/im.height);im=im.resize((round(im.width*s),round(im.height*s)));board.paste(im,(j*370+7,52))
   bd.text((12,385),'曲线从原图墨迹读取；短段、交汇与干笔归属仍待定，不强接白缝。',font=font,fill='#33473f')
   bd.text((12,422),'这不是新训练模型，也未证明恢复了全部真实笔触或皴法。',font=font,fill='#6b5840')
   board.save(OUT/'comparison.png')
 save('evaluation.json',dict(config=cfg,results=results,robot_connected=False,new_model_adopted=False))
 print(json.dumps(dict(config=cfg,examples=[{k:v for k,v in r.items() if not k.endswith('coverage')} for r in results]),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
