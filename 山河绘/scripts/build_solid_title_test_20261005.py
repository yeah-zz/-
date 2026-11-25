"""Independent black-only narrowed-letter test. Offline; existing jobs untouched."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import json,sys,socket,hashlib,csv
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def forbidden(*args,**kwargs):raise RuntimeError('Offline build: network prohibited')
socket.socket.connect=forbidden;socket.create_connection=forbidden
import imgio,ink_control,shuimo,job
from config import ArmConfig,CanvasConfig,PALETTE
from geometry import CanvasMap,check_waypoints
from original_d_tracer import LandscapeBrushGenerator as Trace
from scripts import build_filled_landscape as fill
from run_mountain_v4_robot import safe_transfers
from run_json_painting import load_artwork,manifest_for
from run_mountain_balanced import load_reviewed,run_feedback

SOURCE=ROOT/'输出/横题山河绘_青绿梯田云山_20261005'
OUT=ROOT/'输出/山河绘_收窄实心字_独立测试_20261005'
# This script owns only this independent test folder, never an active landscape.
OUT.mkdir(exist_ok=True)
original_job_hash=hashlib.sha256((SOURCE/'painting.json').read_bytes()).hexdigest()
reference=imgio.imread(str(SOURCE/'参考原图.png'))
shrink_mm=.30;guard_mm=.10;pitch_mm=.60
defaults=json.loads((SOURCE/'task_defaults.json').read_text(encoding='utf8'))
current=json.loads((SOURCE/'painting.json').read_text(encoding='utf8'))
w,h=current['ink_job']['meta']['image_size'];mapper=CanvasMap((w,h));scale=mapper.mm_per_px
crop=reference[145:385,775:1410].copy()
transform=json.loads((SOURCE/'离线检查.json').read_text(encoding='utf8'))['source_transform']
ratio=transform['scale'];crop=cv2.resize(crop,(round(crop.shape[1]*ratio),round(crop.shape[0]*ratio)),interpolation=cv2.INTER_AREA)
ch,cw=crop.shape[:2];ox=(w-cw)//2;oy=(h-ch)//2
work=np.full((h,w,3),255,np.uint8);work[oy:oy+ch,ox:ox+cw]=crop
original=(cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)<145).astype(np.uint8)
n,labels,stats,_=cv2.connectedComponentsWithStats(original,8);keep=np.zeros(n,np.uint8);keep[1:]=stats[1:,cv2.CC_STAT_AREA]>=5;original=keep[labels]
distance=cv2.distanceTransform(original,cv2.DIST_L2,5)*scale
target=(distance>=shrink_mm).astype(np.uint8)
# Protect slender connections: broad parts shrink; slender parts retain enough
# room for the narrow modeled brush where the original silhouette allows it.
source_skeleton=(Trace._thin_binary(original*255)>0).astype(np.uint8)
support=np.zeros_like(original)
for cy,cx in np.argwhere(source_skeleton):
 radius=min(distance[cy,cx],max(.62,distance[cy,cx]-shrink_mm))
 cv2.circle(support,(int(cx),int(cy)),max(1,int(np.ceil(radius/scale))),1,-1)
target|=support*original
targetdist=cv2.distanceTransform(target,cv2.DIST_L2,5)*scale
imgio.imwrite(str(OUT/'题字参考.png'),work)
imgio.imwrite(str(OUT/'收窄目标字形.png'),255-target*255)
fill.MAX_STROKE_MM=35.;fill.BRIDGE_MAX_MM=1.15;fill.BRIDGE_MIN_INSIDE=1.
raw=[];thin_width=shuimo.press_to_width_mm(.10);fill_width=shuimo.press_to_width_mm(.15)
def add(points,press,character,kind):
 p=np.asarray(points,float)
 if len(p)<2:return
 arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]*scale
 if arc[-1]<.5:return
 t=np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]/.25))+1))
 p=np.column_stack([np.interp(t,arc,p[:,k]) for k in (0,1)])
 q=np.rint(p).astype(int)
 # All center points retain half the modeled footprint plus explicit guard.
 valid=targetdist[q[:,1],q[:,0]]>=shuimo.press_to_width_mm(press)/2+guard_mm
 changes=np.diff(np.r_[False,valid,False].astype(int))
 for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
  part=p[a:b]
  if len(part)<2 or np.linalg.norm(np.diff(part,axis=0),axis=1).sum()*scale<.5:continue
  n=len(part);s=dict(points=part.tolist(),press=[press]*n,speed=[30.]*n,layer=0,
   pal_index=shuimo.ink_palette_index(),tail_mm=0.,lift_mm=2.5,
   horizontal_velocity_percent=80,lift_velocity_percent=60,touchdown_velocity_percent=30,
   role=kind,name=character+'_'+kind,character=character)
  raw.append(ink_control.refresh_stats(s,scale))

boundaries=[ox,ox+round((970-775)*ratio),ox+round((1140-775)*ratio),ox+cw]
for ci,character in enumerate('山河绘'):
 zone=np.zeros((h,w),np.uint8);zone[:,boundaries[ci]:boundaries[ci+1]]=1
 # First lay narrow center strokes, with no traced perimeter.
 skeleton=Trace._thin_binary((target*zone)*255)
 safe=(targetdist>=thin_width/2+guard_mm).astype(np.uint8)
 skeleton=(skeleton>0).astype(np.uint8)*safe
 for path in Trace._trace_skeleton_paths(skeleton*255):
  add(path,.10,character,'中心细笔')
 # Wider parts receive overlapping interior rows; holes remain excluded.
 core=((targetdist>=fill_width/2+guard_mm)&(zone>0)).astype(np.uint8)
 paths,_=fill.fill_paths(core,scale,'black',max(1,round(pitch_mm/scale)),fill_width,1.)
 for st in paths:add(st['points'],.15,character,'内部填充')

ink=deepcopy(defaults['ink_job']);meta=ink['meta']
policy=ink_control.InkPolicy(**meta['ink_execution']['policy'])
pieces=ink_control.split_long_strokes(raw,scale,policy)
for st in pieces:
 if st['segments']>1:st['press']=[raw[st['parent_stroke_id']]['press'][0]]*len(st['points']);ink_control.refresh_stats(st,scale)
for i,s in enumerate(pieces):s['id']=f'solid-title-test-{i:04}'
steps,quality=ink_control.build_plan(pieces,shuimo.ink_palette_index(),scale,policy)
waypoints,_=job.steps_to_waypoints(steps,mapper);waypoints=safe_transfers(waypoints)
assert check_waypoints(waypoints)[0]
meta.update(image=str(OUT/'题字参考.png'),image_size=[w,h],paper_z=ArmConfig.PAPER_Z,
 canvas_corners=[list(q) for q in CanvasConfig.ROBOT_CORNERS],strokes=len(pieces),quality_stats=quality)
meta['ink_execution'].update(base_strokes=pieces,mm_per_px=scale,calibration=dict(rpy=list(ArmConfig.RPY),
 user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,speed_factor=70,palette=deepcopy(PALETTE)))
meta['test_design']=dict(per_side_shrink_mm=shrink_mm,nominal_stroke_width_reduction_mm=2*shrink_mm,slender_connections_protected=True,
 fill_pitch_mm=pitch_mm,edge_guard_mm=guard_mm,thin_press_mm=.10,fill_press_mm=.15,
 independent_black_only=True,centered_on_new_paper=True,perimeter_tracing=False,
 thin_details_below_brush_width_omitted=True,estimated_brush_width_not_measured=True)
ink['waypoints']=waypoints
write=lambda name,value:(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
write('painting.json',ink)
(OUT/'speed_profile.py').write_bytes((SOURCE/'speed_profile.py').read_bytes())
(OUT/'paint_this.py').write_text("from pathlib import Path\nimport sys\nHERE=Path(__file__).resolve().parent\nsys.path.insert(0,str(HERE.parents[1]))\nimport speed_profile\nfrom run_json_painting import main\nif __name__=='__main__':\n raise SystemExit(main(['--job',str(HERE/'painting.json'),'--no-progress',*sys.argv[1:]]))\n",encoding='utf8')
# Check with the same speed profile used by the runtime, only in this process.
import importlib.util
spec=importlib.util.spec_from_file_location('test_speed_profile',OUT/'speed_profile.py');profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
loaded=load_artwork(OUT/'painting.json');manifest_for(loaded[2],loaded[1],loaded[3])
strokes,mapper,policy,meta=load_reviewed(ink)
clock=[0.];executed=[];contact_points=[0]
def draw(s):
 wp=job.stroke_waypoints(mapper,s);assert check_waypoints(wp)[0]
 contact=[p for p in wp if p['tag'].startswith(('down:','draw:'))]
 assert np.allclose([p['z']-ArmConfig.PAPER_Z for p in contact],s['press'])
 assert s['lift_mm']==2.5 and set(s['speed'])=={30.}
 executed.append(s['id']);contact_points[0]+=len(contact);clock[0]+=s['length_mm']/3.5+2
run_feedback(strokes,scale,policy,lambda _:None,draw,lambda _:None,clock=lambda:clock[0])
assert executed==[s['id'] for s in strokes]
assert [s['character'] for s in strokes]==sorted([s['character'] for s in strokes],key=lambda c:'山河绘'.index(c))
polygon=np.asarray(CanvasConfig.ROBOT_CORNERS,np.float32)
margin=min(cv2.pointPolygonTest(polygon,tuple(map(float,p)),True) for s in strokes for p in mapper.to_robot(s['points']))
assert margin>=20
preview=np.full_like(work,255);pathview=preview.copy()
for s in strokes:
 p=np.rint(s['points']).astype(np.int32)
 cv2.polylines(preview,[p],False,(20,20,20),max(1,round(shuimo.press_to_width_mm(s['press'][0])/scale)),cv2.LINE_AA)
 cv2.polylines(pathview,[p],False,(120,70,30) if s['role']=='中心细笔' else (20,20,20),1,cv2.LINE_AA)
imgio.imwrite(str(OUT/'测试路径.png'),pathview);imgio.imwrite(str(OUT/'实心字预览_笔宽为估计.png'),preview)
from PIL import Image,ImageDraw,ImageFont
ims=[Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).crop((ox-20,oy-20,ox+cw+20,oy+ch+20)) for im in [work,255-target*255,preview]]
width=max(im.width for im in ims);height=ims[0].height
comparison=Image.new('RGB',(width,height*3+165),'white');d=ImageDraw.Draw(comparison);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
for i,(im,title) in enumerate(zip(ims,['原图题字（原尺寸）','宽部两侧各内收0.30 mm；窄部少收以保连笔','最终任务路径的笔宽模型预览（非实测）'])):
 d.text((10,i*(height+55)+8),title,font=font,fill='#243b49');comparison.paste(im,(0,i*(height+55)+40))
comparison.save(OUT/'原字_收窄_任务对照.png')
report=dict(offline_passed=True,network_blocked=True,robot_connected=False,strokes=len(strokes),
 per_character=dict(Counter(s['character'] for s in strokes)),contact_points=contact_points[0],
 min_paper_margin_mm=margin,draw_velocity_percent=30,global_speed_factor=70,lift_mm=2.5,
 pressure_mm=[.10,.15],source_artwork_unchanged=hashlib.sha256((SOURCE/'painting.json').read_bytes()).hexdigest()==original_job_hash,
 settings=meta['test_design'])
assert report['source_artwork_unchanged'];write('离线检查.json',report)
with (OUT/'逐笔下压清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
 wr=csv.writer(f);wr.writerow(['笔号','字','类型','下压mm','长度mm']);wr.writerows([s['id'],s['character'],s['role'],s['press'][0],s['length_mm']] for s in strokes)
(OUT/'测试说明.md').write_text(f'''# 山河绘 · 收窄实心字独立测试

仅画三个黑色实心测试字，不画山水、不运行蓝绿上色。按山 → 河 → 绘逐字完成，图案居于新纸中央，题字尺寸沿用原任务。
待当前任务完全结束后，换一张空白纸并按现有画布四角位置放好，再启动本测试。

宽笔画每侧内收0.30 mm，总宽度减少约0.60 mm；窄笔画与连笔少收，尽量保住字形。先用中心细笔0.10 mm，再用内部填充0.15 mm，行距0.60 mm，行笔30%，全局70%，笔间离纸2.5 mm。没有另画外围描边线。
超出最细模型笔宽能力的尖尾和细小飞白会省略，字内孔洞保留；模型边缘留0.10 mm余量，实际扩散需上纸验证。

共{len(strokes)}笔，{report['per_character']}。两侧内收后的目标图、模型预览和实际路径分别保留。预览不是实际上纸结果。
已禁止网络完成离线校验与逐点下压模拟。当前山水任务及共享标定未修改。
运行入口为paint_this.py；提示时输入RUN。P暂停、C继续、Q退出；退出后重开从头开始。
''',encoding='utf8')
write('归档文件校验_SHA256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()})
print(json.dumps(report,ensure_ascii=False))
