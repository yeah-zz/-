"""Import the approved centerlines as a new artwork, entirely offline."""
from pathlib import Path
import sys, json, shutil, hashlib, csv, socket
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def no_network(*args, **kwargs):
    raise RuntimeError('Offline artwork import cannot connect to hardware')
socket.socket.connect = no_network
socket.create_connection = no_network

import cv2
import numpy as np
import imgio, vision, job, shuimo
from config import ArmConfig, CanvasConfig, ShuimoConfig, PALETTE
from geometry import CanvasMap, check_waypoints
from ink_control import InkPolicy, refresh_stats, split_long_strokes, build_plan
from color_profile import ColorProfile
from run_json_painting import make_bundle, load_artwork, manifest_for
from run_mountain_v4_robot import safe_transfers
from painting_session import save
from web.server import artwork_data
from trajectory_preview import render_contact

SOURCE = Path(r'C:\Users\Administrator\.codex\generated_images\01a11c21-af59-7b30-a2c2-337dcdaa575a')
ART = ROOT/'待画作品'/'06_松瀑云山_简化单线稿'
TASK = ART/'任务资料'
PREVIEW = ART/'预览与下压标注'
if ART.exists():
    raise FileExistsError('New artwork folder already exists; refusing replacement')
TASK.mkdir(parents=True)
PREVIEW.mkdir()
shutil.copy2(SOURCE/'exec-c0dd9dfb-31fc-46e5-92b1-df9b3f6cd0b7.png', ART/'参考原图.png')
colored = Path(r'C:\Users\ADMINI~1\AppData\Local\Temp\codex-clipboard-88e4c294-1d77-41cd-b1c2-fe107512d554.png')
if colored.exists():
    shutil.copy2(colored, ART/'原始彩色参考.png')
for name in ('landscape-centerline.svg','landscape-paths.json','conversion-stats.json'):
    shutil.copy2(SOURCE/'robot-linework'/name, TASK/name)

src = json.loads((TASK/'landscape-paths.json').read_text())
reference = str(ART/'参考原图.png')
work, info = vision.prepare_image(vision.load_image(reference), work_size=1800)
if info.get('used_corners'):
    raise ValueError('Custom image corners are not supported for this import')
mapper = CanvasMap(info['prepared_size'])
scale = mapper.mm_per_px
pad_top, _, pad_left, _ = info['pad']
geometry = dict(image_size=list(info['prepared_size']), mm_per_px=scale,
                image_corners=mapper.image_corners)
policy = InkPolicy()
pressure = min(.10, ShuimoConfig.PRESS_MAX_MM)
strokes = []
for i, original in enumerate(src['paths']):
    points = (np.asarray(original,dtype=float)*info['scale'] + [pad_left,pad_top]).tolist()
    n = len(points)
    s = dict(id='outline-centerline-%05d'%i,name='简化山水墨线',layer=0,
             pal_index=shuimo.ink_palette_index(),points=points,press=[pressure]*n,
             speed=[ShuimoConfig.BRUSHES['tip']['speed']]*n,tail_mm=0.,lift_mm=4.,
             source_path_index=i,role='outline')
    strokes.append(refresh_stats(s,scale))
strokes = split_long_strokes(strokes,scale,policy)
for i,s in enumerate(strokes):
    s['id'] = 'outline-simplified-%05d'%i
steps,stats = build_plan(strokes,shuimo.ink_palette_index(),scale,policy)
conditioned = [step['stroke'] for step in steps if step['kind']=='stroke']
waypoints,_ = job.steps_to_waypoints(steps,mapper)
waypoints = safe_transfers(waypoints)
ok,problems = check_waypoints(waypoints)
if not ok:
    raise ValueError(problems)
profile = ColorProfile(press_mm=min(.22,ShuimoConfig.PRESS_MAX_MM)).to_dict()
plan = dict(format='painting-workflow-design-v1',ready_for_execution=True,
            subject='simplified-landscape',reference_image=reference,
            stages=['outline'],sequence=[dict(kind='run_module',module='outline')],
            modules=dict(outline=dict(format='outline-design-v1',strokes=conditioned,**geometry),
                         color=dict(format='sparse-color-design-v1',strokes=[],regions=[],
                                    stroke_count=0,motion_candidate=profile,**geometry)),
            note='Approved simplified centerline artwork; no color or title. Long lines split only for existing ink budget. Preview widths are estimates.',
            **geometry)
meta = dict(mode='shuimo-framework',profile='mountain',ink_policy='balanced',
            image=reference,image_size=geometry['image_size'],
            canvas_corners=[list(p) for p in CanvasConfig.ROBOT_CORNERS],
            paper_z=ArmConfig.PAPER_Z,strokes=len(strokes),
            quality_stats=dict(algorithm='approved-single-centerline',source_paths=len(src['paths']),
                               execution_strokes=len(strokes)),
            ink_execution=dict(base_strokes=strokes,policy=asdict(policy),mm_per_px=scale,
                calibration=dict(rpy=list(ArmConfig.RPY),user=ArmConfig.USER_INDEX,
                                 tool=ArmConfig.TOOL_INDEX,speed_factor=ArmConfig.SPEED_FACTOR,
                                 palette=PALETTE)))
ink = dict(format='arm-painting job',version=1,meta=meta,waypoints=waypoints)
save(TASK/'painting.json',make_bundle(plan,ink))
loaded = load_artwork(TASK/'painting.json')
manifest = manifest_for(loaded[2],loaded[1],loaded[3])
geom = artwork_data(TASK/'painting.json')
assert geom['counts']['black']==len(strokes)
assert len(strokes)==len({s['id'] for s in strokes})
save(TASK/'黑线任务.json',ink)
save(TASK/'绘画计划.json',plan)
save(TASK/'绘画顺序索引.json',dict(strokes=[dict(order=i+1,id=s['id'],source_path=s['source_path_index'],length_mm=s['length_mm']) for i,s in enumerate(strokes)]))
with (TASK/'逐笔顺序索引.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['笔序','笔触ID','原路径编号','长度毫米','下压毫米'])
    for i,s in enumerate(strokes):
        writer.writerow([i+1,s['id'],s['source_path_index']+1,round(s['length_mm'],3),round(s['press_mean'],3)])

# Read back the saved bundle; both previews correspond to the imported task.
saved = json.loads((TASK/'painting.json').read_text(encoding='utf-8-sig'))
saved_strokes = saved['ink_job']['meta']['ink_execution']['base_strokes']
pw,ph = geometry['image_size']
center = np.full((ph*2,pw*2,3),255,np.uint8)
for s in saved_strokes:
    cv2.polylines(center,[np.rint(np.asarray(s['points'])*2).astype(np.int32)],False,(0,0,0),3,cv2.LINE_AA)
imgio.imwrite(str(PREVIEW/'单线路径预览.png'),center)
contact = render_contact(saved['ink_job']['waypoints'],mapper,output_scale=2.)
imgio.imwrite(str(PREVIEW/'实际任务笔宽估计预览.png'),contact)
summary = dict(date='2026-10-08',source_paths=len(src['paths']),execution_strokes=len(strokes),
               ink_only=True,estimated_width_mm=shuimo.press_to_width_mm(pressure),
               press_mm=pressure,canvas_mm=mapper.canvas_size_mm(),
               length_mm=sum(s['length_mm'] for s in strokes),
               validation='load_artwork + manifest_for + check_waypoints passed offline',
               painting_sha256=hashlib.sha256((TASK/'painting.json').read_bytes()).hexdigest())
save(TASK/'入库校验.json',summary)
(ART/'阅读说明.md').write_text(f'''# 松瀑云山 · 简化单线稿

已于 2026-10-08 加入待画作品。仅黑墨勾线，不含上色和题字。

原线稿 {len(src['paths'])} 条中心路径；按现有补墨预算拆分长线后，执行 {len(strokes)} 笔。
每笔按一次落笔至抬笔计算，拆分处保留共同端点。

唯一绘画入口：任务资料/painting.json。
单线路径预览显示中心路径；实际任务笔宽估计预览由导出的接触轨迹重新渲染。
本稿下压 {pressure:.2f} mm，笔宽按当前标定表估计为 {shuimo.press_to_width_mm(pressure):.2f} mm。
预览未模拟墨水扩散或控制器转角融合。

SVG 和像素路径 JSON 保留在任务资料，便于后续调整。
本次仅离线制作及校验，未连接或启动机械臂。
''',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
