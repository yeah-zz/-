import sys,json,copy
from pathlib import Path
from collections import Counter,defaultdict
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import vision,job,imgio,landscape_color,watercolor
from green_accents import plan_accents
from run_mountain_balanced import load_reviewed
from ink_control import build_plan
from shuimo import ink_palette_index
from color_profile import ColorProfile
from run_color_experiment import module_lines
from run_color_from_job import load_plan
r=Path(__file__).resolve().parents[1]/'输出/近石远山_下压对比_20260928'
ink=r/'近石远山_下压对比_黑线任务.json'
strokes,mapper,policy,meta=load_reviewed(ink)
work,info=vision.prepare_image(vision.load_image(meta['image']),work_size=1800)
steps,_=build_plan(strokes,ink_palette_index(),mapper.mm_per_px,policy)
conditioned=[s['stroke'] for s in steps if s['kind']=='stroke']
mask=landscape_color.analyze(work,mapper.mm_per_px)[0].astype(np.uint8)
for s in conditioned:
 cv2.polylines(mask,[np.rint(s['points']).astype(np.int32)],False,1,max(1,int(np.ceil(s['width_mm']/mapper.mm_per_px))))
paths,regions=landscape_color.plan_paths(work,mapper.mm_per_px,width=1.,gap=.35,ink_mask=mask.astype(bool))
groups=defaultdict(list)
for s in paths:
 if s['color'] in ('blue','yellow') and min(p[1] for p in s['points'])>work.shape[0]*.25:
  groups[s['region_id']].append(s)
selected=[s for group in groups.values() for s in group]
selected+=plan_accents(work,mapper.mm_per_px,conditioned,max_strokes=24)
profile=ColorProfile(press_mm=.22)
geo=dict(image_size=meta['image_size'],mm_per_px=mapper.mm_per_px,image_corners=mapper.image_corners)
for s in selected:
 s['target_opacity']=[.65]*len(s['points'])
module=dict(format='sparse-color-design-v1',strokes=selected,regions=regions,motion_candidate=profile.to_dict(),palette_bgr={'red':[130,130,190],'blue':[167,144,119],'yellow':[130,192,224],'green':[117,139,117]},**geo)
allowed=set()
for c in ('blue','yellow','green'):
 lines,_=module_lines(module,c,profile.to_dict());allowed.update(s['id'] for s in lines)
module['strokes']=[s for s in selected if s['id'] in allowed];module['stroke_count']=len(module['strokes'])
plan=dict(format='painting-workflow-design-v1',ready_for_execution=False,subject='山水',reference_image=meta['image'],stages=['outline','color'],modules=dict(outline=dict(format='outline-design-v1',strokes=conditioned,**geo),color=module),note='Three premixed cups; dense 1mm assumed-width wash paths, 0.22mm pressure; distant mountains remain white. Actual width uncalibrated.',**geo)
pp=r/'近石远山_密集上色_计划.json';pp.write_text(json.dumps(plan,ensure_ascii=False,indent=1),encoding='utf-8')
load_plan(pp,ink)
canvas,_=watercolor.render_image(work,module,include_ink=False)
for s in conditioned:
 xy=np.rint(s['points']).astype(int)
 for i in range(len(xy)-1):
  p=(s['press'][i]+s['press'][i+1])/2
  cv2.line(canvas,tuple(xy[i]),tuple(xy[i+1]),(32,32,32),max(1,round(1+5*(p-.06)/.24)),cv2.LINE_AA)
imgio.imwrite(str(r/'近石远山_密集上色_预览.png'),canvas)
print(dict(Counter(s['color'] for s in module['strokes'])))
print('PASS paired job and color paths, offline only')
