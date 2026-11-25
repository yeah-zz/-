from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy,json,shutil
from collections import Counter
from unittest.mock import patch
import cv2,numpy as np
import vision,imgio,job,shuimo,ink_control,watercolor
from config import ArmConfig,CanvasConfig,PALETTE,ShuimoConfig
from geometry import CanvasMap,check_waypoints
from original_d_tracer import LandscapeBrushGenerator as Trace
from run_mountain_v4_robot import safe_transfers
from run_color_from_job import load_plan
from scripts.preview_sparse_creek import spread
root=Path(__file__).resolve().parents[1]
src=root/'输出/山亭松瀑_簇叶试版_20260928'
out=root/'输出/山亭松瀑_松树专项定稿_20260928';out.mkdir(exist_ok=False)
p=json.loads((src/'山亭松瀑_簇叶_计划.json').read_text(encoding='utf8'))
_,meta=job.load_job(src/'山亭松瀑_簇叶_黑线任务.json')
work,info=vision.prepare_image(vision.load_image(str(src/'参考原图.png')),work_size=1800)
h,w=work.shape[:2];scale=p['mm_per_px'];offset=np.array([info['pad'][2],info['pad'][0]])
# Region follows the original pine, excludes nearby cliff and mountain.
polygon=np.array([[77,488],[78,443],[180,395],[190,362],[332,365],[412,450],[510,490],[636,575],[633,616],[476,604],[329,558],[290,535],[244,591],[198,652],[188,706],[93,724],[88,680],[153,624],[179,571],[229,535],[226,496],[170,507]])+offset
roi=np.zeros((h,w),np.uint8);cv2.fillPoly(roi,[polygon.astype(np.int32)],1)
gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY);b,g,r=cv2.split(work.astype(float))
centers=[(116,465),(154,448),(194,436),(235,400),(291,408),(331,433),(362,439),(367,489),(426,501),(467,514),(409,526),(452,548),(498,560),(541,560),(581,587)]
foliage=np.zeros((h,w),np.uint8)
for cx,cy in centers:
 cv2.ellipse(foliage,tuple((np.array([cx,cy-12])+offset).astype(int)),(30,17),0,0,360,1,-1)
dark=((gray<95)&(roi>0)&(foliage==0)).astype(np.uint8)
skel=Trace._thin_binary(dark*255)
paths=Trace._trace_skeleton_paths(skel)
distance=cv2.distanceTransform(dark,cv2.DIST_L2,5)
new=[]
for raw in paths:
 pts=np.asarray(raw,float)
 if len(pts)<2:continue
 arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]*scale
 if arc[-1]<1.2:continue
 ij=np.rint(pts).astype(int)
 strength=np.clip((distance[ij[:,1],ij[:,0]]-1)/5,0,1)
 press=.045+.19*strength
 ramp=np.minimum(np.clip(arc/.7,0,1),np.clip((arc[-1]-arc)/.9,0,1))
 press*=.35+.65*ramp
 s=dict(points=pts.tolist(),press=press.tolist(),speed=[12.]*len(pts),layer=1,pal_index=shuimo.ink_palette_index(),tail_mm=0.,lift_mm=4.,role='pine_source_contour',design_origin='D-skeleton-full-resolution-local',name='松树原图双侧轮廓')
 new.append(ink_control.refresh_stats(s,scale))
# Keep portions of old paths outside the pine, without joining across removed portions.
retained=[]
for s in meta['ink_execution']['base_strokes']:
 pts=np.asarray(s['points']);ij=np.rint(pts).astype(int);keep=roi[ij[:,1],ij[:,0]]==0
 starts=np.flatnonzero(np.diff(np.r_[False,keep,False].astype(int))==1)
 ends=np.flatnonzero(np.diff(np.r_[False,keep,False].astype(int))==-1)
 for a,z in zip(starts,ends):
  if z-a<2:continue
  part=copy.deepcopy(s)
  for key in ('points','press','speed','source_width_mm'):
   if key in part:part[key]=part[key][a:z]
  ink_control.refresh_stats(part,scale)
  if part['length_mm']>.3:retained.append(part)
policy=ink_control.InkPolicy(**meta['ink_execution']['policy'])
strokes=ink_control.split_long_strokes(retained+new,scale,policy)
strokes=ink_control.order_for_ink(strokes,scale,policy)
for i,s in enumerate(strokes):s['id']='outline-%05d'%i
steps,stats=ink_control.build_plan(strokes,shuimo.ink_palette_index(),scale)
conditioned=[s['stroke'] for s in steps if s['kind']=='stroke']
p['modules']['outline']['strokes']=conditioned;p['modules']['outline']['stroke_count']=len(conditioned)
# Pine needles: hand-identified fan centers on the source image, constrained to actual green pixels.
centers=[(116,465),(154,448),(194,436),(235,400),(291,408),(331,433),(362,439),(367,489),(426,501),(467,514),(409,526),(452,548),(498,560),(541,560),(581,587)]
mask=((((g-r>1)&(g-b>1)&(gray>38)&(gray<215))|((foliage>0)&(gray<95)))&(roi>0)).astype(np.uint8)
mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
green=[]
def color_stroke(points,color,width,ident,style):
 n=len(points)
 return dict(id=ident,color=color,region_id=style,points=np.asarray(points).tolist(),recipe='accent',layer=0,target_opacity=[.65]*n,target_width_mm=[width]*n,brush_width_mm_estimate=width,length_mm=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()*scale),path_style=style,pressure_mm=None,pigment_recipe_calibrated=False)
for ci,center in enumerate(centers):
 c=np.array(center)+offset
 for angle in np.linspace(-np.pi+.20,-.20,7):
  direction=np.array([np.cos(angle),np.sin(angle)])
  pts=c+np.linspace(3,24,18)[:,None]*direction
  ij=np.rint(pts).astype(int);valid=mask[ij[:,1],ij[:,0]]>0
  starts=np.flatnonzero(np.diff(np.r_[False,valid,False].astype(int))==1);ends=np.flatnonzero(np.diff(np.r_[False,valid,False].astype(int))==-1)
  if not len(starts):continue
  best=int(np.argmax(ends-starts));a,z=starts[best],ends[best]
  if z-a<5:continue
  green.append(color_stroke(pts[a:z],'green',.7,'pine-needle-%03d'%len(green),'pine_fan'))
# Keep only a few background accents so most green belongs to pine foliage.
oldcolor=p['modules']['color']['strokes']
outside=[]
for s in oldcolor:
 ij=np.rint(s['points']).astype(int)
 if not np.any(roi[ij[:,1],ij[:,0]]):outside.append(s)
colors=[s for s in outside if s['color']!='green']+spread([s for s in outside if s['color']=='green'],12)+green
# Ochre center strokes stay inside the original tan trunk/branches.
tan=((r-g>7)&(g-b>10)&(gray>120)&(roi>0)).astype(np.uint8)
tan=cv2.erode(tan,np.ones((3,3),np.uint8))
for raw in Trace._trace_skeleton_paths(Trace._thin_binary(tan*255)):
 pts=np.asarray(raw,float)
 if len(pts)<3:continue
 length=np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()*scale
 if length<1.8:continue
 # Bound each color pass length rather than making one excessive loading interval.
 for a in range(0,len(pts)-2,120):
  section=pts[a:min(a+121,len(pts))]
  if len(section)<3:continue
  colors.append(color_stroke(section,'yellow',.7,'pine-trunk-ochre-%03d'%len(colors),'pine_trunk_ochre'))
m=p['modules']['color'];m['strokes']=colors;m['stroke_count']=len(colors)
m['regions']=[]
p['pine_revision']=dict(note='Only pine black geometry retraced at source resolution; green fans prioritize pine canopy. Optical width is an estimate.',source_plan=str(src/'山亭松瀑_簇叶_计划.json'),pine_black_paths=len(new),pine_green_paths=len(green))
mapper=CanvasMap(meta['image_size']);waypoints,_=job.steps_to_waypoints(steps,mapper);waypoints=safe_transfers(waypoints)
assert check_waypoints(waypoints)[0]
meta['strokes']=len(strokes);meta['ink_execution']['base_strokes']=strokes
meta['ink_execution']['calibration']=dict(rpy=list(ArmConfig.RPY),user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,speed_factor=ArmConfig.SPEED_FACTOR,palette=PALETTE)
meta['pine_revision']=p['pine_revision'];meta['quality_stats']={'detail':'original-d-outside-pine; full-resolution-pine','strokes':len(strokes)}
prefix=out/'山亭松瀑_松树专项'
job.save_job(str(prefix)+'_黑线任务.json',waypoints,meta)
Path(str(prefix)+'_计划.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf8')
with patch('robot.E6Robot.__init__',side_effect=RuntimeError('Offline only')):load_plan(str(prefix)+'_计划.json')
# Preview the actual per-point pressure model; never paste original ink as the result.
canvas,_=watercolor.render_image(work,m,include_ink=False)
for s in conditioned:
 points=np.array(s['points']);press=s['press']
 for i,(a,z) in enumerate(zip(points,points[1:])):
  width=(shuimo.press_to_width_mm(press[i])+shuimo.press_to_width_mm(press[i+1]))/2/scale
  cv2.line(canvas,tuple(np.rint(a).astype(int)),tuple(np.rint(z).astype(int)),(35,35,35),max(1,round(width)),cv2.LINE_AA)
imgio.imwrite(str(prefix)+'_预览.png',canvas)
# Detail is a direct crop of the generated trajectory preview, not source art.
imgio.imwrite(str(out/'松树局部_路径预览.png'),canvas[350:735,130:710])
shutil.copy2(src/'参考原图.png',out/'参考原图.png')
print('black',len(strokes),'pine black',len(new),'pine green',len(green),'colors',dict(Counter(s['color'] for s in colors)))
print('Offline validation passed; no motion; pressure max',max(max(s['press']) for s in strokes))
