import sys,json,copy
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import job,imgio,vision,watercolor
from original_d_tracer import LandscapeBrushGenerator
from ink_control import refresh_stats,split_long_strokes,order_for_ink,build_plan
from shuimo import ink_palette_index
from run_mountain_balanced import load_reviewed
from run_mountain_v4_robot import safe_transfers
from run_color_experiment import module_lines
from run_color_from_job import load_plan
from collections import Counter
root=Path(__file__).resolve().parents[1]/'输出/近石远山_下压对比_20260928'
source=root/'近石远山_下压对比_黑线任务.json'
strokes,mapper,policy,meta=load_reviewed(source)
work,info=vision.prepare_image(vision.load_image(meta['image']),work_size=1800)
h,w=work.shape[:2]; scale=mapper.mm_per_px
# Tight region around the large foreground pine's curved trunk, source-relative.
top,bottom,left,right=info['pad'];cw=w-left-right;ch=h-top-bottom
x0,x1=round(left+.14*cw),round(left+.26*cw);y0,y1=round(top+.465*ch),round(top+.625*ch)
roi=np.zeros((h,w),np.uint8);roi[y0:y1,x0:x1]=1
updated=[]
for s in strokes:
 xy=np.asarray(s['points']);inside=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
 if not inside.any():updated.append(copy.deepcopy(s));continue
 # Keep outside subpaths. Do not draw chords across removed trunk region.
 indices=np.where(~inside)[0]
 for part in np.split(indices,np.where(np.diff(indices)>1)[0]+1):
  if len(part)<2:continue
  q=copy.deepcopy(s)
  for key in ('points','press','speed'):q[key]=[s[key][i] for i in part]
  updated.append(refresh_stats(q,scale))
gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
black=((gray<100)&(roi>0)).astype(np.uint8)*255
skeleton=LandscapeBrushGenerator._thin_binary(black)
paths=LandscapeBrushGenerator._trace_skeleton_paths_by_component(skeleton)
for pts in paths:
 xy=np.asarray(pts,float)
 if len(xy)<3 or np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()*scale<.7:continue
 t=np.linspace(0,1,len(xy));ramp=np.minimum(np.clip(t/.1,0,1),np.clip((1-t)/.1,0,1))
 s=copy.deepcopy(strokes[0]);s.update(points=xy.tolist(),press=(.06+.24*ramp).tolist(),speed=[20.]*len(xy),design_origin='D skeleton at prepared resolution, trunk-only refinement',pressure_band='near')
 updated.append(refresh_stats(s,scale))
updated=order_for_ink(split_long_strokes(updated,scale,policy),scale,policy)
for i,s in enumerate(updated):s['id']='refined-outline-%05d'%i
steps,_=build_plan(updated,ink_palette_index(),scale,policy)
conditioned=[s['stroke'] for s in steps if s['kind']=='stroke']
wp,_=job.steps_to_waypoints(steps,mapper)
meta=copy.deepcopy(meta);meta['strokes']=len(updated);meta['ink_execution']['base_strokes']=updated
meta['quality_stats']['strokes']=len(updated);meta['quality_stats']['local_refinement']='Pine trunk re-traced at prepared resolution, elsewhere original D600 retained; actual broad ink silhouette not guaranteed.'
ink=root/'近石远山_簇叶修正版_黑线任务.json';job.save_job(ink,safe_transfers(wp),meta);load_reviewed(ink)
plan=json.loads((root/'近石远山_密集上色_计划.json').read_text(encoding='utf-8'))
plan['modules']['outline']['strokes']=conditioned
module=plan['modules']['color'];module['strokes']=[s for s in module['strokes'] if s['color']!='green']
b,g,r=cv2.split(work.astype(float));mask=((g-r>4)&(g-b>2)&(gray>55)&(gray<165)).astype(np.uint8)
# Width envelope fits inside vegetation; retain thin branches rather than crossing them.
mask=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
greens=[]
for k in range(1,n):
 if stats[k,cv2.CC_STAT_AREA]*scale*scale<1.:continue
 x,y,ww,hh=stats[k,:4];pitch=max(3,round(.9/scale));width=.9
 for yy in range(y+pitch//2,y+hh,pitch):
  xs=np.where(labels[yy]==k)[0]
  for seg in np.split(xs,np.where(np.diff(xs)>1)[0]+1):
   if len(seg)*scale<.7:continue
   for a in range(0,len(seg),max(3,round(3./scale))):
    chunk=seg[a:a+max(3,round(3./scale))]
    if len(chunk)*scale<.7:continue
    pts=[[float(chunk[0]),float(yy)],[float((chunk[0]+chunk[-1])/2),float(yy)],[float(chunk[-1]),float(yy)]]
    greens.append(dict(id='green-cluster-%04d'%len(greens),color='green',region_id='foliage-%d'%k,recipe='cluster',layer=0,points=pts,target_opacity=[.65]*3,target_width_mm=[width]*3,brush_width_mm_estimate=width,length_mm=float((chunk[-1]-chunk[0])*scale),press=[.22]*3,path_style='cluster_short_line'))
module['strokes']+=greens
allowed=set()
for color in ('blue','yellow','green'):
 lines,_=module_lines(module,color,module['motion_candidate']);allowed.update(s['id'] for s in lines)
module['strokes']=[s for s in module['strokes'] if s['id'] in allowed];module['stroke_count']=len(module['strokes'])
pp=root/'近石远山_簇叶修正版_计划.json';pp.write_text(json.dumps(plan,ensure_ascii=False,indent=1),encoding='utf-8');load_plan(pp,ink)
canvas,_=watercolor.render_image(work,module,include_ink=False)
for s in conditioned:
 xy=np.rint(s['points']).astype(int)
 for i in range(len(xy)-1):
  press=(s['press'][i]+s['press'][i+1])/2
  cv2.line(canvas,tuple(xy[i]),tuple(xy[i+1]),(32,32,32),max(1,round(1+5*(press-.06)/.24)),cv2.LINE_AA)
imgio.imwrite(str(root/'近石远山_簇叶修正版_预览.png'),canvas)
print('black',len(updated),'color',dict(Counter(s['color'] for s in module['strokes'])))
