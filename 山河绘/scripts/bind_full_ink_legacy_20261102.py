# -*- coding: utf-8 -*-
"""Offline export with the existing calibration explicitly reused by the user.

No robot connection or configuration mutation. Execution uses the existing
reviewed balanced executor, including its normal operator RUN prompt.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from config import ArmConfig, CanvasConfig, ShuimoConfig, PALETTE
from geometry import CanvasMap, check_waypoints
from ink_control import InkPolicy, split_long_strokes, build_plan
import job
from pigment_station import read_measurements, PigmentStation
from run_mountain_balanced import load_reviewed
from run_mountain_v4_robot import safe_transfers
from shuimo import ink_palette_index

ROOT=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\full_ink_trial_20261102')
OUT=ROOT/'legacy_calibration'
PROJECT=Path(__file__).resolve().parents[1]


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def padded_mapping(width,height,paper_width,paper_height,margin_mm=10.):
    """Place entire source within physical margins before the usual CanvasMap."""
    if min(width,height)<2 or min(paper_width,paper_height)<=margin_mm*2:
        raise ValueError('Invalid source or paper dimensions')
    ch=height
    cw=int(round((ch-1)*paper_width/paper_height))+1
    mm_per_source_px=min((paper_width-2*margin_mm)/(width-1),
                         (paper_height-2*margin_mm)/(height-1))
    local_offset=np.array([(paper_width-mm_per_source_px*(width-1))/2,
                           (paper_height-mm_per_source_px*(height-1))/2])
    canvas_px_per_mm=np.array([(cw-1)/paper_width,(ch-1)/paper_height])
    return (cw,ch),mm_per_source_px*canvas_px_per_mm,local_offset*canvas_px_per_mm,mm_per_source_px


def export(output=OUT):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    problems=config.validate(need_stations=True)
    if problems:
        raise ValueError(problems)
    before={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()
            for name in ('config.py','pigment_measurements.json')}
    source=json.loads((ROOT/'trial_plan_unbound.json').read_text('utf-8'))
    measurements=read_measurements()
    station=PigmentStation(**measurements['black']).validate()
    old_map=CanvasMap((source['width'],source['height']))
    pw,ph=old_map.canvas_size_mm()
    image_size,mult,offset,scale=padded_mapping(source['width'],source['height'],pw,ph)
    mapper=CanvasMap(image_size)
    tip=ShuimoConfig.BRUSHES['tip']
    pressure=float(tip['press_mm'])
    speed=int(tip['speed'])
    if not 0 < pressure <= ShuimoConfig.PRESS_MAX_MM:
        raise ValueError('Existing tip pressure outside current limits')
    strokes=[]
    for row in source['paths']:
        pts=(np.array(row['points'],float)*mult+offset).tolist()
        strokes.append(dict(id='N02-%04d'%row['id'],points=pts,press=[pressure]*len(pts),
            speed=[speed]*len(pts),tail_mm=0,lift_mm=ShuimoConfig.LIFT_MM,layer=1,
            pal_index=ink_palette_index(),source_ids=row['source_ids'],
            direction_verified=False,stroke_identity_verified=False))
    # Preserve the ordered geometry. Ink state may add refills but no test marks.
    policy=InkPolicy(initial_dip_s=station.dwell_s,refresh_dip_s=station.dwell_s,
                     drip_s=station.drip_s,fresh_press_factor=1.)
    base=split_long_strokes(strokes,mapper.mm_per_px,policy)
    steps,stats=build_plan(base,ink_palette_index(),mapper.mm_per_px,policy)
    if any(s['kind'] not in ('stroke','dip','rest') for s in steps):
        raise AssertionError('Unexpected extra marks or color stage')
    waypoints,_=job.steps_to_waypoints(steps,mapper)
    waypoints=safe_transfers(waypoints)
    ok,problems=check_waypoints(waypoints)
    if not ok:
        raise ValueError(problems)
    if any(w['tag'].startswith('tail:') for w in waypoints):
        raise AssertionError('Unexpected extended tail')
    snapshot=dict(user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,rpy=list(ArmConfig.RPY),
                  speed_factor=ArmConfig.SPEED_FACTOR,palette=PALETTE)
    # Same meta/schema used by the existing executor. No protection is relaxed.
    meta=dict(mode='full-black-image-trial',profile='mountain',ink_policy='balanced',
        image=str(ROOT/'source.png'),image_size=list(image_size),
        canvas_corners=[list(p) for p in CanvasConfig.ROBOT_CORNERS],paper_z=ArmConfig.PAPER_Z,
        strokes=len(base),length_mm=stats['length_mm'],dips=stats['dips'],
        ink_plan=stats,air_transfer_velocity_percent=40,preview_width_calibrated=False,
        source_coordinate_system='native_image_pixels',source_to_padded_scale=mult.tolist(),
        source_to_padded_offset=offset.tolist(),margin_mm=10.,image_fit='contain',
        calibration_reuse=dict(user_requested=True,mode='existing_configuration',new_measurements=False,
            site_change_previously_reported=True,site_match_verified=False,source_hashes=before),
        physical_brush=dict(profile='existing tip preset',press_mm=pressure,velocity_percent=speed,
            width_table_estimate_mm=float(tip['width_mm']),current_paper_test=False),
        ink_execution=dict(base_strokes=base,mm_per_px=mapper.mm_per_px,
            policy=asdict(policy),calibration=snapshot),
        actually_drawn=False,robot_connected=False)
    target=output/'N02_full_black_legacy_job.json'
    job.save_job(str(target),waypoints,meta)
    # Real existing preflight, offline. Fails if frame/limits/station disagree.
    checked,_,_,_=load_reviewed(str(target))
    if len(checked)!=len(base):
        raise AssertionError('Preflight changed stroke count')
    after={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in before}
    if before!=after:
        raise AssertionError('Shared calibration was modified')
    calibration=dict(paper_z=ArmConfig.PAPER_Z,canvas_corners=CanvasConfig.ROBOT_CORNERS,
        user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,rpy=ArmConfig.RPY,
        black=measurements['black'],paper_size_mm=[pw,ph],
        content_square_mm=scale*(source['width']-1),margin_mm=10,
        new_measurements=False,site_match_verified=False,source_sha256=before)
    save(output/'reused_calibration_snapshot.json',calibration)
    # Paper-layout view plus an explicit estimate from the OLD width table.
    center=np.full((image_size[1],image_size[0],3),255,np.uint8)
    footprint=center.copy()
    thickness=max(1,int(round(float(tip['width_mm'])/mapper.mm_per_px)))
    for s in base:
        p=np.rint(s['points']).astype(np.int32)
        cv2.polylines(center,[p],False,(0,0,0),1,cv2.LINE_AA)
        cv2.polylines(footprint,[p],False,(50,50,50),thickness,cv2.LINE_AA)
    Image.fromarray(center).save(output/'paper_centerlines.png')
    Image.fromarray(footprint).save(output/'old_width_table_estimate.png')
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',22)
    size=(820,int(round(820*image_size[1]/image_size[0])))
    comparison=Image.new('RGB',(1680,size[1]+90),'#f5f3eb')
    draw=ImageDraw.Draw(comparison)
    draw.text((15,10),'整幅纸面布局 · 复用旧标定 · 非实画',font=font,fill='#252925')
    draw.text((15,46),'中心线计划',font=font,fill='#252925')
    draw.text((855,46),'旧笔宽表估计（当前纸笔尚未验证）',font=font,fill='#252925')
    comparison.paste(Image.fromarray(center).resize(size),(15,85))
    comparison.paste(Image.fromarray(footprint).resize(size),(855,85))
    comparison.save(output/'paper_preview.jpg',quality=94)
    envlines=["$ErrorActionPreference='Stop'",
        "$env:TEMP='E:\\Codex_Workspace_Data\\CLIPascene_20261021\\temp'",
        "$env:TMP=$env:TEMP",
        "$env:PIP_CACHE_DIR='E:\\Codex_Workspace_Data\\CLIPascene_20261021\\cache\\pip'",
        "$env:TORCH_HOME='E:\\Codex_Workspace_Data\\CLIPascene_20261021\\cache\\torch'",
        "$env:HF_HOME='E:\\Codex_Workspace_Data\\CLIPascene_20261021\\cache\\huggingface'",
        "$env:XDG_CACHE_HOME='E:\\Codex_Workspace_Data\\CLIPascene_20261021\\cache'",
        "$env:CUDA_CACHE_PATH='E:\\Codex_Workspace_Data\\CLIPascene_20261021\\cache\\cuda'",
        "$env:PYTHONDONTWRITEBYTECODE='1'","$env:PYTHONIOENCODING='utf-8'",
        "$env:PYTHONUTF8='1'",f"Set-Location -LiteralPath '{PROJECT}'"]
    for name,action in [('离线校验.ps1',' --check'),('开始整幅黑线.ps1','')]:
        command=f"& 'E:\\Anaconda\\python.exe' -B '{PROJECT/'run_mountain_balanced.py'}' --job '{target}'{action}"
        # BOM supports Windows PowerShell 5.1 with Chinese absolute paths.
        (output/name).write_text('\n'.join(envlines+[command,'exit $LASTEXITCODE'])+'\n',encoding='utf-8-sig')
    summary=dict(strokes=len(base),waypoints=len(waypoints),planned_dips=stats['dips'],
        draw_length_mm=stats['length_mm'],heuristic_estimated_minutes=stats['estimated_total_s']/60,
        timing_is_not_controller_measurement=True,offline_preflight_passed=True,configuration_unchanged=True,
        pressure_mm=pressure,draw_velocity_percent=speed,global_speed_factor=ArmConfig.SPEED_FACTOR,
        paper_size_mm=[pw,ph],composition_square_mm=scale*(source['width']-1),
        site_match_verified=False,new_measurements=False,robot_connected=False,actually_drawn=False,
        existing_ink_level_compensation=measurements['black']['level_compensation_enabled'],
        note='静态蘸墨轨迹是起始液面；执行器按现有液面补偿配置在范围内调整。耗时是启发式估计。')
    save(output/'binding_statistics.json',summary)
    (output/'使用说明.md').write_text(f'''# 完整黑线山水：复用旧标定

已按用户要求使用现有 config.py 和 pigment_measurements.json。没有改写共享配置、没有连接或移动机械臂。
纸面约 {pw:.1f} × {ph:.1f} mm，完整正方形构图约 {scale*(source['width']-1):.1f} mm，等比例居中，最小留边 10 mm。
采用旧 tip 笔档：下压 {pressure:.2f} mm、行笔 v={speed}、全局倍率 {ArmConfig.SPEED_FACTOR}%。不根据图像粗细增加下压，不添加出锋尾线，不上色。

## 文件

- N02_full_black_legacy_job.json：兼容现有 balanced 山水执行器的完整绝对坐标任务。
- reused_calibration_snapshot.json：旧标定实际数值与来源哈希，可供现场核对。
- paper_preview.jpg：纸面中心线与旧笔宽表估计。右图不是已验证的物理仿真。
- 离线校验.ps1：只做静态配置/轨迹检查，不连接机械臂。
- 开始整幅黑线.ps1：操作员启动入口；调用现有执行器，输入 RUN 后才连接并执行运动。打开此脚本即会进入实机操作流程，不是预览。

## 本次现场核对

用户前面报告过夹持/现场变化，本次复用旧值不等于已重新测量。先对照 reused_calibration_snapshot.json 核对纸面位置、触纸高度、User/Tool/RPY 和黑碗液面；若不一致，应先恢复到对应布置或更新标定。现有姿态/范围检查不能自动发现纸张或毛笔移动。

可在 PowerShell 分别运行：

```powershell
& '{output/'离线校验.ps1'}'
# 确认当前纸笔与墨碗匹配旧标定后，由你启动：
& '{output/'开始整幅黑线.ps1'}'
```

整幅共 {len(base)} 条几何行笔段；静态补墨估计 {stats['dips']} 次，实际由执行器按时长/耗墨状态补墨。规划估计约 {stats['estimated_total_s']/60:.0f} 分钟，不是实测耗时，频繁抬落笔和短松针可能使实机更慢。
沿用现有黑墨液面补偿（上限 {station.level_max_mm:.2f} mm），没有新增补偿参数。
中途停止后不要在原纸上直接从头重跑；现有执行器记录每次任务进度和故障。
''',encoding='utf-8')
    (output/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>整幅黑线 · 旧标定任务</title><style>body{{font:18px system-ui;max-width:1200px;margin:25px auto;padding:0 20px;background:#f5f3eb;color:#252925}}img{{width:100%}}a{{color:#17694c}}li{{margin:12px 0}}</style><h1>完整黑线山水 · 已绑定旧标定</h1><p>纸面 {pw:.1f} × {ph:.1f} mm；构图 {scale*(source['width']-1):.1f} mm；{len(base)} 条行笔段。下压 {pressure:.2f} mm，行笔 v={speed}，全局倍率 {ArmConfig.SPEED_FACTOR}%。</p><p>已通过现有执行器的离线预检。当前现场与旧值是否一致仍需核对；没有连接机械臂、没有实画记录。</p><img src="paper_preview.jpg"><p>右侧使用旧笔宽表估计，能看到松针在该笔宽下可能变密；当前毛笔表现尚未验证。</p><ul><li><a href="使用说明.md">使用说明与启动命令</a></li><li><a href="reused_calibration_snapshot.json">复用标定数值</a></li><li><a href="N02_full_black_legacy_job.json">完整上机任务 JSON</a></li><li><a href="binding_statistics.json">规划统计</a></li><li><a href="../index.html">图像线稿与笔顺回放</a></li></ul>''',encoding='utf-8')
    return summary


if __name__=='__main__':
    print(json.dumps(export(),ensure_ascii=False,indent=2))
