from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,copy,shutil
from collections import Counter
from unittest.mock import patch
import cv2,numpy as np
import vision
from painting_workflow import save_outputs
from run_color_from_job import load_plan
root=Path(__file__).resolve().parents[1]
src=root/'输出/山亭松瀑_疏淡上色_20260928'
out=root/'输出/山亭松瀑_簇叶试版_20260928'
out.mkdir(exist_ok=False)
p=json.loads((src/'山亭松瀑_疏淡_计划.json').read_text(encoding='utf-8-sig'))
original=copy.deepcopy(p)
work,_=vision.prepare_image(vision.load_image(str(src/'参考原图.png')),work_size=1800)
m=p['modules']['color'];scale=m['mm_per_px'];h,w=work.shape[:2]
b,g,r=cv2.split(work.astype(float));gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
green=((g-r>4)&(g-b>2)&(gray>55)&(gray<210)).astype(np.uint8)
ink=np.zeros((h,w),np.uint8)
for s in p['modules']['outline']['strokes']:
 cv2.polylines(ink,[np.rint(s['points']).astype(np.int32)],False,1,max(1,round((s['width_mm']+.75)/scale)))
valid=cv2.erode(green,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(3,2*int(.4/scale)+1),)*2))
valid[ink>0]=0
edge=int(np.ceil(20/scale));valid[:edge]=0;valid[-edge:]=0;valid[:,:edge]=0;valid[:,-edge:]=0
# Group vegetation into crowns/bushes, then allocate each group a few leaf clusters.
k=max(3,2*round(1.5/scale)+1)
groups=cv2.morphologyEx(green,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
n,labels,stats,_=cv2.connectedComponentsWithStats(groups,8)
components=[i for i in range(1,n) if np.sum(valid[labels==i]) *scale*scale>1.5]
areas=np.array([np.sum(valid[labels==i])*scale*scale for i in components])
weights=np.sqrt(areas);quota=np.maximum(1,np.floor(72*weights/weights.sum()).astype(int))
while quota.sum()<72: quota[np.argmax(72*weights/weights.sum()-quota)]+=1
while quota.sum()>72: quota[np.argmax(quota) ]-=1
leaves=[];used=np.zeros((h,w),np.uint8)
for component,budget in zip(components,quota):
 region=(valid>0)&(labels==component)
 distance=cv2.distanceTransform(region.astype(np.uint8),cv2.DIST_L2,5)
 local=0;attempts=0
 while local<budget and attempts<budget*8:
  score=distance.copy();score[used>0]=0
  yy,xx=np.unravel_index(np.argmax(score),score.shape)
  if score[yy,xx]<1:break
  attempts+=1
  # Three short curved marks form a cluster; all actual paths stay inside green and clear of black.
  for leaf in range(3):
   if local>=budget:break
   angle=(-.65 if leaf%2==0 else .65)
   offset=np.array([(leaf-1)*.7,-abs(leaf-1)*.55])/scale
   success=False
   for length in (2.8,2.1,1.4,.9):
    t=np.linspace(-.5,.5,9)
    curve=np.column_stack((t*length, .28*length*(1-4*t*t))) /scale
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    points=curve@rotation.T+np.array([xx,yy])+offset
    ij=np.rint(points).astype(int)
    if (ij[:,0].min()<0 or ij[:,0].max()>=w or ij[:,1].min()<0 or ij[:,1].max()>=h):continue
    if not np.all(region[ij[:,1],ij[:,0]]) or np.any(used[ij[:,1],ij[:,0]]):continue
    # Check rasterized segment interiors, not only control points.
    raster=np.zeros((h,w),np.uint8);cv2.polylines(raster,[ij],False,1,1)
    if np.any((raster>0)&~region):continue
    stroke=dict(id='green-cluster-%03d'%len(leaves),color='green',region_id='leaf-group-%03d'%component,
     points=points.tolist(),recipe='accent',layer=0,target_opacity=[.60,.65,.70,.70,.70,.70,.70,.65,.60],
     target_width_mm=[.7]*9,brush_width_mm_estimate=.7,length_mm=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()*scale),
     path_style='clustered_leaf_arc',pressure_mm=None,pigment_recipe_calibrated=False)
    leaves.append(stroke);local+=1
    cv2.polylines(used,[ij],False,1,max(1,round(.85/scale)))
    success=True;break
  # Keep groups distinct and leave small white gaps between them.
  cv2.circle(used,(int(xx),int(yy)),max(1,round(1.3/scale)),1,-1)
m['strokes']=[s for s in m['strokes'] if s['color']!='green']+leaves
m['stroke_count']=len(m['strokes'])
p['leaf_variant']=dict(source_plan=str(src/'山亭松瀑_疏淡_计划.json'),green_strokes=len(leaves),note='Curved leaf clusters within source vegetation; unchanged pressure and color speed.')
assert [s for s in m['strokes'] if s['color']!='green']==[s for s in original['modules']['color']['strokes'] if s['color']!='green']
assert p['modules']['outline']==original['modules']['outline']
prefix=out/'山亭松瀑_簇叶'
shutil.copy2(src/'山亭松瀑_疏淡_黑线任务.json',str(prefix)+'_黑线任务.json')
shutil.copy2(src/'参考原图.png',out/'参考原图.png')
save_outputs(p,work,str(prefix)+'_计划.json',str(prefix)+'_预览.png')
with patch('robot.E6Robot.__init__',side_effect=RuntimeError('Offline only')):load_plan(str(prefix)+'_计划.json')
print(dict(Counter(s['color'] for s in m['strokes'])))
print('Black/blue/yellow unchanged; full paths checked inside vegetation; offline validation passed.')
