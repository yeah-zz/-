"""Offline D tracing with explicit depth bands for a pressure contrast study."""
import sys, json, shutil
from pathlib import Path
from dataclasses import asdict
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import vision, imgio, job
from original_d_adapter import build_strokes
from ink_control import InkPolicy, refresh_stats, split_long_strokes, order_for_ink, build_plan
from geometry import CanvasMap, check_waypoints
from config import ArmConfig, CanvasConfig, PALETTE
from shuimo import ink_palette_index
from run_mountain_v4_robot import safe_transfers
from run_mountain_balanced import load_reviewed

root=Path(__file__).resolve().parents[1]
out=root/'输出/近石远山_下压对比_20260928'
out.mkdir(parents=True,exist_ok=True)
src=Path(sys.argv[1]); reference=out/'近石远山_原图.png'
if not reference.exists(): shutil.copy2(src,reference)
work,info=vision.prepare_image(vision.load_image(str(reference)),work_size=1800)
mapper=CanvasMap(info['prepared_size'],info['image_corners'])
strokes,stats=build_strokes(work,mapper.mm_per_px,pad=info.get('pad'),source_image=vision.load_image(str(reference)))
top,bottom,left,right=info['pad']; content_h=work.shape[0]-top-bottom
for s in strokes:
    xy=np.asarray(s['points']); y=(np.median(xy[:,1])-top)/content_h
    low,high,band=(.06,.10,'far') if y<.245 else ((.14,.20,'middle') if y<.55 else (.24,.30,'near'))
    old=np.asarray(s['press']); t=np.linspace(0,1,len(old))
    ramp=np.minimum(np.clip(t/.10,0,1),np.clip((1-t)/.10,0,1))
    relative=np.clip((old-.06)/.24,0,1)
    target=low+(high-low)*(.5+.5*relative)
    s['press']=(.06+(target-.06)*ramp).tolist()
    s['pressure_band']=band
    s['design_origin']='D-original-600px + explicit spatial pressure bands'
    s.update(refresh_stats(s,mapper.mm_per_px))
# Comparison artifact: do not attenuate the first 12 mm after each reload.
# Fixed pressure intent remains inside the existing 0.06..0.30 mm ceiling.
policy=InkPolicy(fresh_press_factor=1.,rest_every=0)
strokes=order_for_ink(split_long_strokes(strokes,mapper.mm_per_px,policy),mapper.mm_per_px,policy)
for i,s in enumerate(strokes): s['id']='pressure-outline-%05d'%i
steps,inkstats=build_plan(strokes,ink_palette_index(),mapper.mm_per_px,policy)
points,_=job.steps_to_waypoints(steps,mapper);points=safe_transfers(points)
assert check_waypoints(points)[0]
allp=np.concatenate([s['press'] for s in strokes])
assert min(allp)>=.06-1e-9 and max(allp)<=.30+1e-9
stats.update(strokes=len(strokes),pressure_mapping='Spatial bands: far 0.06-0.10, middle 0.14-0.20, near 0.24-0.30 mm; tapered ends 0.06 mm',brush_width_calibrated_for_this_job=False)
meta=dict(mode='shuimo-framework',profile='mountain',ink_policy='balanced',image=str(reference),image_size=list(info['prepared_size']),canvas_corners=[list(x) for x in CanvasConfig.ROBOT_CORNERS],paper_z=ArmConfig.PAPER_Z,strokes=len(strokes),quality_stats=stats,ink_execution=dict(base_strokes=strokes,policy=asdict(policy),mm_per_px=mapper.mm_per_px,calibration=dict(rpy=list(ArmConfig.RPY),user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,speed_factor=ArmConfig.SPEED_FACTOR,palette=PALETTE)))
jobpath=out/'近石远山_下压对比_黑线任务.json'
job.save_job(jobpath,points,meta)
verified,_,_,_=load_reviewed(jobpath); assert verified==strokes
h,w=work.shape[:2]
pressure=np.full_like(work,255); line=np.full_like(work,255)
colors=[(205,115,30),(45,160,25),(40,75,220)]
lengths=np.zeros(3)
for s in strokes:
    xy=np.rint(s['points']).astype(int); ps=np.asarray(s['press'])
    for i in range(len(xy)-1):
        p=float((ps[i]+ps[i+1])/2); band=0 if p<.12 else (1 if p<.22 else 2)
        cv2.line(pressure,tuple(xy[i]),tuple(xy[i+1]),colors[band],3,cv2.LINE_AA)
        cv2.line(line,tuple(xy[i]),tuple(xy[i+1]),(30,30,30),max(1,round(1+5*(p-.06)/.24)),cv2.LINE_AA)
        lengths[band]+=np.linalg.norm(xy[i+1]-xy[i])*mapper.mm_per_px
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
for name,arr,title in [('下压分色预览',pressure,'蓝：轻压 <0.12 mm    绿：中压 0.12–0.22 mm    橙：重压 ≥0.22 mm'),('黑线示意预览',line,'黑线粗细为示意，未标定实际笔宽；执行下压范围 0.06–0.30 mm')]:
    canvas=Image.new('RGB',(w,h+85),'white');canvas.paste(Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)),(0,65));d=ImageDraw.Draw(canvas);d.text((25,15),title,font=font,fill=(30,30,30));canvas.save(out/('近石远山_'+name+'.png'))
summary={'strokes':len(strokes),'pressure_min_max':[float(min(allp)),float(max(allp))],'length_share_light_middle_heavy':(lengths/lengths.sum()).round(3).tolist(),'robot_connected':False,'fresh_press_factor':1.,'note':'Only black outlines. No automatic color continuation. Width schematic is not calibrated. XYZ range validation is not joint IK validation.'}
(out/'验证结果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True))
print(out)
