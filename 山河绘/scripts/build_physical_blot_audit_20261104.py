# -*- coding: utf-8 -*-
"""Audit a failed real drawing against saved commands, never operate hardware."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json
import hashlib
from collections import Counter
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ink_drawability_20261104 import map_saved_job,summarize_geometry,drawability_gate,sensitivity
from brush_footprint_20261103 import exposure_rgb

ROOT=LAB/'physical_blot_audit_20261104'
TRIAL=LAB/'complete_G03_01_trial_20261103'
PHOTO=Path(r'E:\Codex_Workspace_Data\CLIPascene_20261021\temp\codex-clipboard-9a9748d5-af14-4645-8676-ddce94904fb7.jpg')
JOB=TRIAL/'dwell15_speed70_continuation/G03_01_dwell15_job.json'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    cv2.setNumThreads(2);ROOT.mkdir(parents=True,exist_ok=True)
    document=json.loads(JOB.read_text('utf-8'));strokes=document['meta']['ink_execution']['base_strokes']
    paths,mapping=map_saved_job(document)
    widths=Counter(round(r['width_mm'],4) for r in strokes)
    if len(widths)!=1:raise ValueError('Audit assumes one saved display width')
    legacy_width=next(iter(widths))
    geometry=summarize_geometry(paths,legacy_width)
    pressures=Counter(round(float(v),4) for s in strokes for v in s['press'])
    facts=dict(planned_strokes=len(strokes),planned_pressure_point_count=sum(pressures.values()),
        commanded_press_mm=dict((str(k),v) for k,v in pressures.items()),
        no_added_tail_count=sum(s.get('tail_mm')==0 for s in strokes),
        all_pressure_profiles_constant=all(max(s['press'])==min(s['press']) for s in strokes),
        saved_width_estimate_mm=legacy_width,width_estimate_is_current_measurement=False,
        source_curve_count_is_not_motor_waypoint_count=True,
        pen_down_count=sum(str(w['tag']).startswith('down:') for w in document['waypoints']),
        continuous_path_present=any(w.get('cp',0)>0 for w in document['waypoints']),
        photo_observation='User reports dense overlapping ink obscuring contours; visual failure',
        actual_contact_force_not_measured=True,actual_ink_volume_not_measured=True,
        photo_scale_and_camera_pose_not_calibrated=True,
        automatic_pixel_registration_not_performed=True,
        drawing_quality='failed',motion_completion_is_not_visual_quality_pass=True)
    gate=drawability_gate(geometry)
    rows,fields=sensitivity(paths,mapping['canvas_mm'])
    save(ROOT/'sensitivity.json',dict(records=rows,all_widths_are_assumed_not_measured=True,
        no_ink_flow_or_drying_simulation=True))
    save(ROOT/'geometry.json',dict(mapping=mapping,geometry=geometry))
    save(ROOT/'drawability_gate.json',gate)
    review=dict(facts=facts,geometry=geometry,gate=gate,photo=str(PHOTO),photo_sha256=sha(PHOTO),
        job=str(JOB),job_sha256=sha(JOB),robot_connected=False,job_or_shared_config_modified=False,
        confirmed_causes=['Saved task uses constant 0.60mm commanded compression for all 1184 curves',
            'Saved width estimate 2.04mm is comparable to median curve length',
            'Full task has 1184 separate vertical touchdown/lift sequences',
            'This real output fails contour/detail readability'],
        contributing_factors_not_yet_quantified=['Current actual width and endpoint pools',
            'Paper wetting, water/ink load, local accumulation',
            'Neighboring nonidentical ink bands and unverified artist stroke relationships'],
        next_changes=['Measure current moving ink width and endpoint diameter in a dense test',
            'Use width/spacing and endpoint occupancy when forming a drawable plan',
            'Continue full-curve recovery without deleting genuine pine needles silently',
            'Develop moving contact/release within supported curve, no blind extra tail',
            'Add registered per-stroke observations before training physical outcome prediction'])
    save(ROOT/'quality_review.json',review)
    before=Image.open(PHOTO).convert('RGB');before.thumbnail((850,1100),Image.Resampling.LANCZOS)
    before.save(ROOT/'observed_failure.jpg',quality=94)
    source=Image.open(TRIAL/'source.png').convert('RGB');source.save(ROOT/'reference.png')
    chosen=['w0.6_s1','w2.04_s1','w2.04_s2.5']
    for key in chosen:
        counts,starts=fields[key]
        Image.fromarray(exposure_rgb(counts)).save(ROOT/(key+'_exposure.png'))
        Image.fromarray(np.where(counts>0,0,255).astype(np.uint8)).save(ROOT/(key+'_footprint.png'))
        if key=='w2.04_s2.5':
            np.savez_compressed(ROOT/'legacy_width_endpoint_scenario.npz',counts=counts,starts=starts)
    # Crop all scenarios to the same centered source-image square used by the
    # saved 10mm-margin planning convention. This is not registration to photo.
    preview_images=[]
    for key in chosen:
        row=next(r for r in rows if r['key']==key);a=fields[key][0];p=row['summary']['margin_pixels']
        paper=a[p:a.shape[0]-p,p:a.shape[1]-p]
        width,height=mapping['canvas_mm'];side=min(width,height)-20.
        x0,y0=np.rint((np.array([width,height])-side)/2*6).astype(int)
        x1,y1=np.rint((np.array([width,height])+side)/2*6).astype(int)
        paper=paper[y0:y1,x0:x1]
        image=Image.fromarray(np.where(paper>0,0,255).astype(np.uint8)).convert('RGB')
        preview_images.append(image)
    board=Image.new('RGB',(1920,650),'#f4f1e8');d=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    labels=['原图（仅供构图对照）','假设线宽0.6mm，无额外起笔膨大','旧估值2.04mm，无额外起笔膨大','旧估值2.04mm，起笔直径×2.5']
    imgs=[source]+preview_images
    for i,(im,label) in enumerate(zip(imgs,labels)):
        im=im.copy();im.thumbnail((472,510),Image.Resampling.LANCZOS)
        board.paste(im,(i*480+(472-im.width)//2,65+(510-im.height)//2))
        d.text((i*480+6,14),label,font=font,fill='black')
    d.text((12,590),'同一批1184条保存轨迹的几何墨带敏感性；不是墨水模拟，也不是从实画拟合出的笔宽。',font=font,fill='black')
    d.text((12,620),'旧估值不是本次标定。细中心线清楚，不代表加上毛笔墨带后还能分开。',font=font,fill='black')
    board.save(ROOT/'footprint_comparison.jpg',quality=95)
    header=f'<h1>G03_01 实画粘连复核</h1><p>保存轨迹{len(paths)}条；恒定下压0.60mm；旧笔宽估值{legacy_width:.2f}mm；实际宽度尚未标定。实画质量：失败。下面是几何敏感性，不是墨水仿真。</p>'
    table='<table><tr><th>假设线宽mm</th><th>起笔直径倍数</th><th>覆盖区内多笔覆盖比例</th><th>单像素最多笔数</th><th>短于假设笔宽的曲线</th></tr>'
    for r in rows:
        s=r['summary'];table+=f'<tr><td>{s["hypothetical_width_mm"]}</td><td>{s["assumed_start_diameter_factor"]}</td><td>{s["repeated_fraction_of_covered_area"]:.1%}</td><td>{s["max_distinct_strokes_at_pixel"]}</td><td>{s["curves_shorter_than_assumed_width"]}</td></tr>'
    table+='</table>'
    page='<meta charset="utf-8"><title>实画粘连复核</title><style>body{font:17px/1.7 system-ui;max-width:1500px;margin:25px auto;background:#f4f1e8;padding:20px}img{max-width:100%}table{border-collapse:collapse;background:white}td,th{padding:8px 15px;border:1px solid #ccc}.pair{display:flex;gap:20px}.pair img{width:40%;object-fit:contain}</style>'
    page+=header+'<img src="footprint_comparison.jpg">'+table
    page+='<p>多笔覆盖只计几何范围，未考虑干燥或真实流体；数值不能解释成实际晕墨率。照片未配准，没有按视觉相似去反推2.04mm。</p><div class="pair"><img src="observed_failure.jpg"><img src="reference.png"></div>'
    page+='<p>当前可绘制性检查：缺少本次墨带与密集小样数据，不能把旧笔宽表当作上机依据。该检查是离线工具，尚未改变冻结执行器。</p><p><a href="quality_review.json">质量复核</a> · <a href="drawability_gate.json">可绘制性检查</a> · <a href="geometry.json">轨迹几何</a> · <a href="sensitivity.json">完整敏感性数据</a></p>'
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(geometry=geometry,facts=facts,gate=gate),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
