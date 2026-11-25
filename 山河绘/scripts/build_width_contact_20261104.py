# -*- coding: utf-8 -*-
"""Three-scene, three-assumed-width offline contact and detail planning study."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from width_contact_core_20261104 import (CONFIG,validate_path,length,bounded_simplify,
    plan_for_width,coverage,coverage_metrics,max_deviation)

ROOT=LAB/'width_aware_contact_20261104'
BASE=LAB/'module_stroke_planner_20261104/outputs'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
WIDTHS=[.6,1.,2.04]
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def slug(w):return 'w'+str(w).replace('.','p')
def mask_image(counts,pad):
    a=counts[pad:-pad,pad:-pad]
    return Image.fromarray(np.where(a>0,0,255).astype(np.uint8)).convert('RGB')

def render_patches(rgb,ims,labels,out,width):
    patches=[('pavilion',[108,198,231,299]),('pine',[177,72,488,240]),('rock',[178,285,326,389])]
    for patch,box in patches:
        board=Image.new('RGB',(1480,370),'#f4f1e8');draw=ImageDraw.Draw(board)
        srcbox=[int(v/496*rgb.width) for v in box]
        converted=[int(v/496*ims[0].width) for v in box]
        patches_im=[rgb.crop(srcbox)]+[im.crop(converted) for im in ims]
        for k,(im,label) in enumerate(zip(patches_im,labels)):
            factor=min(360/im.width,300/im.height)
            im=im.resize((max(1,round(im.width*factor)),max(1,round(im.height*factor))),Image.Resampling.LANCZOS)
            board.paste(im,(k*370,45));draw.text((k*370+4,8),label,font=FONT,fill='black')
        draw.text((8,340),f'{patch} · {width:g}mm假设；同画幅局部放大，非实际落笔',font=FONT,fill='black')
        board.save(out/(patch+'_comparison.jpg'),quality=95)

def build(ident):
    start=time.monotonic();folder=ROOT/ident;folder.mkdir(parents=True,exist_ok=True)
    source=BASE/ident/'source.png';file=BASE/ident/'offline_plan.json';doc=json.loads(file.read_text('utf-8'))
    rgb=Image.open(source).convert('RGB');rgb.save(folder/'source.png')
    # A declared common comparison frame, not the current robot calibration.
    scale=186.8/496
    raw=[validate_path(np.asarray(r['points'])*scale) for r in doc['curves']]
    paths=[bounded_simplify(p,CONFIG['curve_tolerance_mm']) for p in raw]
    deviation=max(max_deviation(p,q) for p,q in zip(raw,paths))
    assert deviation<=CONFIG['curve_tolerance_mm']+1e-7
    assert all(np.allclose(p[[0,-1]],q[[0,-1]]) for p,q in zip(raw,paths))
    curves=[dict(points=p.tolist(),module=c['module']) for p,c in zip(paths,doc['curves'])]
    save(folder/'complete_paths.json',dict(paths=[dict(**c,original_points_mm=raw[i].tolist(),
         source_curve_ids=doc['curves'][i]['source_curve_ids']) for i,c in enumerate(curves)],
         no_extracted_curve_deleted=True,frame='paper comparison millimeters, no robot coordinates',
         reference_width_mm=186.8,current_robot_calibration=False,physical_pressure=None,ready_for_robot=False))
    summaries=[]
    for width in WIDTHS:
        name=slug(width);out=folder/name;out.mkdir(exist_ok=True)
        plan=plan_for_width(curves,doc['actions'],width)
        constant,pad=coverage(paths,width,False)
        tapered,_=coverage(paths,width,True)
        baseline_metrics=coverage_metrics(constant);taper_metrics=coverage_metrics(tapered)
        selected_metrics=coverage_metrics(plan['counts'])
        ims=[mask_image(constant,pad),mask_image(tapered,pad),mask_image(plan['counts'],pad)]
        for im,n in zip(ims,['all_constant.png','all_tapered.png','selected.png']):im.save(out/n)
        overlay=np.array(ims[2]);ppm=CONFIG['ppm']
        for i in plan['deferred']:
            cv2.polylines(overlay,[np.rint(paths[i]*ppm).astype(np.int32)],False,(185,65,150),1,cv2.LINE_AA)
        Image.fromarray(overlay).save(out/'deferred_overlay.png')
        kept_len=sum(length(paths[i]) for i in plan['selected']);total_len=sum(length(p) for p in paths)
        summary=dict(id=ident,assumed_width_mm=width,actual_width_measured=False,input_curves=len(paths),
            selected_curves=len(plan['selected']),deferred_curves=len(plan['deferred']),
            retained_centerline_length_fraction=kept_len/total_len,
            protected_long_or_closed_candidates=len(plan['protected_curve_ids']),
            unresolved_structural_conflicts=plan['structural_conflicts'],
            all_constant=baseline_metrics,all_tapered=taper_metrics,selected_tapered=selected_metrics,
            original_waypoint_count=sum(len(p) for p in raw),simplified_waypoint_count=sum(len(p) for p in paths),
            simplification_tolerance_mm=CONFIG['curve_tolerance_mm'],
            measured_on_geometry_max_deviation_mm=deviation,endpoints_preserved=True,
            source_curve_simplification_is_not_pen_lift_reduction=True,
            full_extraction_preserved=True,painting_details_explicitly_deferred=True,
            pixel_model_trained=False,actual_blot_reduction=None,actual_drawing_time_seconds=None,
            no_blind_generalization_test=True,ready_for_robot=False,robot_connected=False)
        save(out/'statistics.json',summary)
        save(out/'drawing_candidate.json',dict(selected_curve_ids=plan['selected'],deferred_curve_ids=plan['deferred'],
            decisions=plan['decisions'],contact_envelope=plan['profiles'],full_archive='../complete_paths.json',
            contact_envelope_is_unmeasured_design=True,relative_width_is_not_relative_pressure=True,
            actual_compression_mm=None,actual_ink_load=None,artist_direction_verified=False,
            source_plan=str(file),source_plan_sha256=sha(file),configuration=CONFIG,
            robot_coordinates=None,ready_for_robot=False))
        board=Image.new('RGB',(2016,570),'#f4f1e8');draw=ImageDraw.Draw(board)
        labels=['原图','全部曲线：假设恒定笔宽','全部曲线：假设起收渐细',f'可选绘画候选：保留{len(plan["selected"])}／待定{len(plan["deferred"])}']
        for k,(im,label) in enumerate(zip([rgb]+ims,labels)):
            im=im.copy();im.thumbnail((490,490),Image.Resampling.LANCZOS);board.paste(im,(k*504,45));draw.text((k*504+5,9),label,font=FONT,fill='black')
        draw.text((8,539),f'{ident}｜假设笔宽{width:g}mm；原线条全部归档；灰黑几何墨带不代表实际墨色，候选细节有取舍，未上机。',font=FONT,fill='black')
        board.save(out/'comparison.jpg',quality=95)
        if ident=='G03_01':
            render_patches(rgb,ims,labels,out,width)
        summaries.append(summary)
        print(ident,width,'selected',len(plan['selected']),'deferred',len(plan['deferred']),
              'length',round(kept_len/total_len,3),'main_conflicts',plan['structural_conflicts'],flush=True)
    save(folder/'summary.json',dict(records=summaries,source_sha256=sha(source),source_plan_sha256=sha(file),
        all_inputs_previously_seen=True,reference_scale_is_not_current_calibration=True,runtime_s=time.monotonic()-start))
    return summaries

PAGE='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>笔宽与起收笔离线对照</title><style>body{font:17px/1.65 system-ui;max-width:1850px;padding:22px;margin:auto;background:#f4f1e8;color:#24382e}select,button{font:inherit;padding:8px;margin-right:12px}img{width:100%;background:white}#details{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}p{max-width:1450px}.note{padding:12px;background:#e4e8db}.row{display:flex;gap:14px;flex-wrap:wrap}a{color:#215d49}</style>
<h1>笔宽、密集细节与起收笔 · 离线对照</h1><p class="note">这是绘画规划实验：完整提取归档不变，绘画候选会明确暂挂部分密集细节。所有笔宽和起收渐细均是假设设计，尚未测量当前毛笔，也没有训练真实墨水模型或连接机械臂。这里的0.6mm是纸面笔宽假设，不是0.6mm下压量。</p>
<div class="row"><label>画作 <select id="scene"><option>G03_01</option><option>G05_03</option><option>T02</option></select></label><label>假设笔宽 <select id="width"><option value="0.6">0.6 mm</option><option value="1">1.0 mm</option><option value="2.04">2.04 mm（旧估值，非本次标定）</option></select></label><button id="deferred">查看暂挂细节（紫色）</button></div><p id="status"></p><img id="board"><p id="links"></p><div id="detailBlock"><h2>同图局部放大</h2><p>亭子、松树与山石分别检查；不是实画照片。</p><img id="pavilion"><img id="pine"><img id="rock"></div><p class="note">长线及闭合结构只按几何条件保护，不等于已验证全部真实主轮廓。保护线之间仍有墨带冲突时保留冲突标记，不能把少线、少黑块当作准确率提高。物理下压量、含墨量和速度仍待本次小样标定。</p>
<script>const RESULTS=__RESULTS__;const e=id=>document.getElementById(id);let overlay=false;function draw(){const id=e('scene').value,w=+e('width').value;const key='w'+(Number.isInteger(w)?w.toFixed(1):String(w)).replace('.','p');const folder=id+'/'+key+'/';const s=RESULTS.find(x=>x.id===id&&x.assumed_width_mm===w);e('board').src=folder+(overlay?'deferred_overlay.png':'comparison.jpg');e('status').textContent=`完整候选 ${s.input_curves} 条 → 可选计划 ${s.selected_curves} 条，暂挂 ${s.deferred_curves} 条；保留中心线长度 ${(100*s.retained_centerline_length_fraction).toFixed(1)}%；长线／闭合候选冲突 ${s.unresolved_structural_conflicts} 处。暂挂不代表误检；本轮不是模型准确率测试。`;e('deferred').textContent=overlay?'返回四列对照':'查看暂挂细节（紫色）';e('links').innerHTML=`<a href="${id}/complete_paths.json">完整线条归档</a> · <a href="${folder}drawing_candidate.json">候选与起收包络</a> · <a href="${folder}statistics.json">统计与限制</a>`;e('detailBlock').style.display=id==='G03_01'?'block':'none';for(const p of ['pavilion','pine','rock']){if(id==='G03_01')e(p).src=folder+p+'_comparison.jpg';else e(p).removeAttribute('src')}}e('scene').onchange=()=>{overlay=false;draw()};e('width').onchange=draw;e('deferred').onclick=()=>{overlay=!overlay;draw()};draw();</script></html>'''

def main():
    cv2.setNumThreads(2);ROOT.mkdir(exist_ok=True)
    records=[]
    for ident in ['G03_01','G05_03','T02']:records.extend(build(ident))
    save(ROOT/'results.json',dict(records=records,model_training=False,robot_connected=False))
    (ROOT/'index.html').write_text(PAGE.replace('__RESULTS__',json.dumps(records,ensure_ascii=False)),encoding='utf-8')
    save(ROOT/'current_brush_profile_to_measure.json',dict(brush_id=None,paper_id=None,ink_condition=None,
        width_p95_mm=None,start_diameter_p95_mm=None,end_diameter_p95_mm=None,dense_patch_passed=None,
        values_must_be_current_measurements=True,legacy_width_not_accepted=True,ready_for_robot=False))
    save(ROOT/'local_test_protocol.json',dict(status='offline unbound protocol only',camera_required=False,
        target='Measure moving width, endpoint pools and separable gaps before any full drawing',
        measured_paper_touch_xyz=None,measured_ink_surface_xyz=None,robot_coordinates=None,
        relative_compression_levels_mm=[.15,.30,.45,.60],levels_are_candidates_not_validated_settings=True,
        shapes=['20mm straight line','1/2/4mm short lines','parallel 10mm lines with 0.5/1/2mm center spacing',
                'one pavilion crop','one pine crop','one rock crop'],
        controlled_variables=['same paper and brush','same ink/water condition','one speed while comparing compression',
                              'record reloading history','measure after consistent drying interval'],
        geometry_taper_is_not_pressure_calibration=True,ready_for_robot=False))
    print('REPORT',str(ROOT/'index.html'),flush=True)

if __name__=='__main__':
    if '--refresh-patches' in sys.argv:
        folder=ROOT/'G03_01';rgb=Image.open(folder/'source.png').convert('RGB')
        for w in WIDTHS:
            out=folder/slug(w);s=json.loads((out/'statistics.json').read_text('utf-8'))
            ims=[Image.open(out/n).convert('RGB') for n in ['all_constant.png','all_tapered.png','selected.png']]
            labels=['原图','全部曲线：假设恒定笔宽','全部曲线：假设起收渐细',f'可选绘画候选：保留{s["selected_curves"]}／待定{s["deferred_curves"]}']
            render_patches(rgb,ims,labels,out,w)
        print('Refreshed nine local comparison images at matched visual scale',flush=True)
    else:main()
