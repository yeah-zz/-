# -*- coding: utf-8 -*-
"""Build an offline footprint-risk report and an unmeasured capture protocol."""
import sys,json,hashlib,csv,html,itertools,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from brush_footprint_20261103 import footprint_audit,exposure_rgb

ROOT=LAB/'brush_execution_model_20261103'
PRIOR=LAB/'stroke_unit_model_20261102'
WIDTHS=[.3,.6,1.0]
DETAIL={'G03_01','G01_04','G05_03','F_N02','T02','T12'}
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)


def read(p):return json.loads(Path(p).read_text('utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def capture_plan():
    out=ROOT/'capture';out.mkdir(exist_ok=True);(out/'measured').mkdir(exist_ok=True)
    paths=dict(straight=[[0,0],[20,0]],bend=[[0,0],[4,1],[8,4],[12,8],[17,10]],
               fold=[[0,0],[7,0],[10,5],[10,12]])
    sessions=[('A','train'),('B','train'),('C','calibration'),('D','test')];samples=[]
    for sid,split in sessions:
        for k,(shape,press,speed) in enumerate(itertools.product(paths,['light','reference'],['slow','reference'])):
            for pos in range(1,4):
                samples.append(dict(sample_id=f'{sid}_{k+1:02d}_{pos}',session_id=sid,split=split,
                    load_batch_id=f'{sid}_{k+1:02d}',shape=shape,press_level=press,speed_level=speed,
                    stroke_since_load=pos,local_path_mm=paths[shape],
                    actual_press_mm=None,actual_speed_mm_s=None,actual_dip_depth_mm=None,
                    actual_dwell_s=None,photo_name=None,start_width_mm=None,middle_width_mm=None,end_width_mm=None,
                    actual_ink_density=None,measured=False,ready_for_robot=False))
    plan=dict(schema='brush_capture_plan_v1',samples=samples,
        status='unmeasured',is_robot_job=False,actual_values_must_be_measured=True,
        split_unit='independent_acquisition_session; never split crops of the same stroke',
        protocol=['4 independent sessions, 12 fresh-ink batches per session, 3 strokes per batch',
            'light/reference are placeholders: set actual contact reference and safe levels on site',
            'keep photos for failed or invisible strokes, not only successful marks',
            'record brush/paper/ink identifiers, elapsed length since loading, actual robot log if available',
            'each stroke photo needs a coplanar ruler; rectify tilted images before width measurement',
            'include no-mark, dry breaks and pooling; ink density is unknown unless lighting is normalized',
            'this is a pilot protocol, not evidence that 144 strokes suffice for a general painting model'])
    save(out/'capture_plan.json',plan)
    with (out/'capture_sheet.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['sample_id','session_id','split','load_batch_id','shape','press_level','speed_level','stroke_since_load',
            'actual_press_mm','actual_speed_mm_s','actual_dip_depth_mm','actual_dwell_s','photo_name',
            'start_width_mm','middle_width_mm','end_width_mm','actual_ink_density','measured']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(samples)
    save(out/'training_readiness.json',dict(measured_strokes=0,training_performed=False,
        model_weights_created=False,reason='No paired actual action/physical stroke records available',
        required_inputs=['measured action parameters','photo with scale','independent session grouping',
            'image registration and failed-stroke labels','width/density measurement uncertainty'],
        next_model='small forward stroke-outcome predictor; compare with measured lookup/regression baseline first',
        do_not_use_hypothetical_footprints_as_real_ground_truth=True))
    shutil.copyfile(Path(__file__).with_name('brush_capture_20261103.html'),ROOT/'capture.html')
    return plan


def paths_from_page(folder):
    # The prior report has serialized baseline/new paths; no robot runner is imported.
    s=(folder/'units.html').read_text('utf-8')
    ps=json.loads(s.split('const ps=',1)[1].split(',r=document',1)[0])
    return [[np.asarray(p,float) for p in side] for side in ps]


def heat_image(counts,title):
    im=Image.fromarray(exposure_rgb(counts));im.thumbnail((550,550),Image.Resampling.LANCZOS)
    board=Image.new('RGB',(570,600),'#f5f3eb');ImageDraw.Draw(board).text((10,8),title,font=FONT,fill='black')
    board.paste(im,(10,40));return board


def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'outputs').mkdir(exist_ok=True)
    if list((ROOT/'capture'/'measured').glob('records_*.json')):
        raise RuntimeError('Physical records now exist. Use inspect_brush_measurements_20261103.py; do not reset the unmeasured report.')
    frozen=read(PRIOR/'frozen_experiment.json')
    assert all(digest(r['path'])==r['sha256'] for r in frozen['files'])
    input_checks=[];allrows=[];convergence=[]
    config=dict(width_scenarios_mm=WIDTHS,assumed_square_mm=186.8,ppm=8,
        start_footprint_diameter_factor=1.5,scenario_only=True,
        counting='distinct path IDs per swept pixel; same-stroke self-overlap counted once',
        caveats=['not ink physics, force, true wetness, or learned brush response',
          'different strokes may legitimately overlap: flags are not deletion decisions',
          'raster quantization affects thin footprints; scenario widths are not measurements',
          'new order is long-first; comparisons of first 51 include ordering effects',
          'actual speed, dwell and temporal drying are not simulated',
          'same-stroke backtracking and start/end physical ink volume are not modeled'])
    save(ROOT/'protocol.json',config)
    for rec in frozen['inference_catalog']:
        ident=rec['id'];src=PRIOR/'outputs'/ident;dst=ROOT/'outputs'/ident;dst.mkdir(exist_ok=True)
        old,new=paths_from_page(src);image=Image.open(src/'source.png').convert('RGB');size=max(image.size)
        assert digest(rec['image'])==rec['image_sha256'] and digest(rec['paths'])==rec['paths_sha256']
        mm=[[p*(186.8/size) for p in side] for side in [old,new]]
        input_checks.append(dict(id=ident,old_new_page=digest(src/'units.html'),curves=digest(src/'curves.json')))
        local=[];boards=[]
        for width in WIDTHS if ident in DETAIL else [.6]:
            result=[];ims=[]
            for name,paths in zip(['old','new'],mm):
                summary,rows,counts,starts=footprint_audit(paths,width,ppm=8)
                summary.update(id=ident,variant=name,order='prior_report_order' if name=='old' else 'long_first',
                    source_pixel_dimensions=list(image.size),source_long_side_mm=186.8)
                save(dst/f'{name}_{width:.1f}_units.json',rows)
                result.append(summary);allrows.append(summary);local.append(summary)
                if ident in DETAIL:
                    ims.append(heat_image(counts,f'{name} · 假设笔宽 {width:.1f} mm'))
                    Image.fromarray(exposure_rgb(starts)).save(dst/f'{name}_{width:.1f}_starts.png')
                if ident=='G03_01':
                    first,_,fc,_=footprint_audit(paths[:51],width,ppm=8)
                    save(dst/f'{name}_{width:.1f}_first51.json',first)
                    Image.fromarray(exposure_rgb(fc)).save(dst/f'{name}_{width:.1f}_first51.png')
            if ims:
                board=Image.new('RGB',(1140,600),'white');board.paste(ims[0],(0,0));board.paste(ims[1],(570,0))
                board.save(dst/f'footprint_{width:.1f}.jpg',quality=92)
        if ident in DETAIL:
            # Raster resolution sensitivity is numerical, not real-paper validation.
            fine,_,_,_=footprint_audit(mm[1],.6,ppm=16)
            coarse=next(r for r in local if r['variant']=='new' and r['hypothetical_width_mm']==.6)
            convergence.append(dict(id=ident,width_mm=.6,coarse_ppm=8,fine_ppm=16,
                coarse_repeat_fraction=coarse['repeated_fraction_of_covered_area'],
                fine_repeat_fraction=fine['repeated_fraction_of_covered_area'],
                absolute_difference=abs(coarse['repeated_fraction_of_covered_area']-fine['repeated_fraction_of_covered_area'])))
        save(dst/'statistics.json',local)
        old06=next(r for r in local if r['variant']=='old' and r['hypothetical_width_mm']==.6)
        new06=next(r for r in local if r['variant']=='new' and r['hypothetical_width_mm']==.6)
        body=''.join(f'<p>{r["variant"]} / 假设{r["hypothetical_width_mm"]:.1f}mm：被覆盖面积中有{r["repeated_fraction_of_covered_area"]:.1%}被不同单元重复覆盖；80%以上已覆盖的候选{r["high_overlap_review_units"]}个。</p>' for r in local)
        images=''.join(f'<h3>假设{w:.1f}mm</h3><img src="footprint_{w:.1f}.jpg">' for w in WIDTHS) if ident in DETAIL else '<p>本图仅保存0.6mm假设情形的统计与逐笔记录。</p>'
        (dst/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><style>body{{font:18px system-ui;max-width:1200px;margin:25px auto;background:#f5f3eb;padding:20px}}img{{width:100%}}</style><a href="../../index.html">返回</a><h1>{ident} · 墨带覆盖风险</h1><p>蓝：1个单元；黄：2个；橙：3–4个；红：5个以上。这是几何覆盖次数，不是墨色或墨量。</p><p>未删除任何笔画，未生成上机任务。不同笔画可以有意相交，重复覆盖不是错误的充分证据。</p>{body}{images}''',encoding='utf-8')
        print(ident,old06['high_overlap_review_units'],new06['high_overlap_review_units'],flush=True)
    save(ROOT/'all_statistics.json',allrows);save(ROOT/'raster_sensitivity.json',convergence)
    plan=capture_plan()
    rows06=[r for r in allrows if r['hypothetical_width_mm']==.6]
    def group(name):
        rs=[r for r in rows06 if r['variant']==name]
        return dict(images=len(rs),high_overlap_review_units=sum(r['high_overlap_review_units'] for r in rs),
            mean_repeated_coverage_fraction=float(np.mean([r['repeated_fraction_of_covered_area'] for r in rs])))
    summary=dict(old=group('old'),new=group('new'),measured_brush_samples=0,
        learned_brush_outcome_model_trained=False,images_newly_generated=0,
        paths_modified_or_deleted=0,ready_for_robot=False,robot_connected=False)
    save(ROOT/'summary.json',summary)
    table=''
    for ident in sorted({r['id'] for r in rows06}):
        a=next(r for r in rows06 if r['id']==ident and r['variant']=='old');b=next(r for r in rows06 if r['id']==ident and r['variant']=='new')
        table+=f'<tr><td><a href="outputs/{ident}/index.html">{ident}</a></td><td>{a["repeated_fraction_of_covered_area"]:.1%}</td><td>{b["repeated_fraction_of_covered_area"]:.1%}</td><td>{a["high_overlap_review_units"]}</td><td>{b["high_overlap_review_units"]}</td></tr>'
    target=next(r for r in rows06 if r['id']=='G03_01' and r['variant']=='new')
    page=f'''<!doctype html><meta charset="utf-8"><title>毛笔执行：覆盖风险与实测准备</title><style>body{{font:18px system-ui;max-width:1200px;margin:30px auto;background:#f5f3eb;color:#283c36;padding:20px}}section{{background:white;padding:20px;margin:18px 0;border-radius:10px}}img{{width:100%}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:10px}}table{{border-collapse:collapse;width:100%}}a{{color:#17694c}}.warn{{color:#a23f2a}}</style>
<h1>毛笔执行：墨带覆盖风险与实测准备</h1>
<section><b>本轮完成</b><p>对上一轮53张画作的旧/新曲线进行假设笔宽的几何覆盖检查，未修改提取模型或删除路径。新模型没有在本轮训练；当前真实“动作→纸面笔迹”配对数据为0。</p><p>三档0.3、0.6、1.0mm仅作敏感性情形，起笔直径假设为1.5倍，均未标定。不能解释为机械臂会画出这个笔宽或浓淡。</p><p>0.6mm情形下，逐图平均重复覆盖面积占比从{summary['old']['mean_repeated_coverage_fraction']:.2%}到{summary['new']['mean_repeated_coverage_fraction']:.2%}，变化很小；高重叠待查单元反而从{summary['old']['high_overlap_review_units']}到{summary['new']['high_overlap_review_units']}。后者受长线优先排序和分组影响，不能当成真实错误数量。6图数值分辨率检查的占比差异为0.16–0.82个百分点，也不能忽略。不能据此宣布实画积墨已解决。</p></section>
<section><b>G03_01：相同曲线，考虑墨带之后</b><p>下图以0.6mm假设演示。新分组仍有{target['high_overlap_review_units']}个单元落在80%以上已覆盖的区域，仅标记复核，不能直接删除；松针、皴线和交汇也可能合理重叠。几何曝光次数不等于积墨量。</p><img src="outputs/G03_01/footprint_0.6.jpg"><p><a href="outputs/G03_01/index.html">三档笔宽对照</a> · <a href="first51.html">前51段覆盖与落笔区对照</a></p></section>
<section><b>下一步实测：144笔小样计划，尚未执行</b><p>3种形状 × 2档下压 × 2档速度 × 蘸墨后第1/2/3笔 × 4个独立采集批次。A/B训练、C校准、D测试；全部实际参数留空，等级不自动换算为下压或速度。采集阶段要保留断墨、未画出和积墨的失败照片。</p><p><a href="capture.html">打开实测记录与照片测宽工具</a> · <a href="capture/capture_plan.json">载入用采集计划</a> · <a href="capture/capture_sheet.csv">记录表</a> · <a href="capture/training_readiness.json">训练准备状态</a></p><p>记录页不控制硬件，不把照片自动当作专家线稿。实测完成后先检验查表/回归基线，再决定是否训练小型笔触预测网络。144笔只是起步设计，不保证覆盖所有皴法、毛笔和纸张。</p></section>
<section class="warn"><b>当前边界</b><p>所有53张都是已有图片，不是新图盲测。0.6mm情形下的几何风险统计列于下表；具体数字受假设笔宽、画幅、起笔范围和栅格精度影响。没有模拟速度、含墨消耗、渗化或干燥，也没有识别真实施力。8与16像素/mm的数值敏感性见记录。旧曲线和模型保持冻结，机械臂没有连接。</p></section>
<section><b>逐图风险检查（0.6mm假设）</b><table><tr><th>图片</th><th>旧：重复覆盖面积占比</th><th>新：重复覆盖面积占比</th><th>旧：高重叠待查单元</th><th>新：高重叠待查单元</th></tr>{table}</table><p><a href="protocol.json">假设与限制</a> · <a href="all_statistics.json">全部统计</a> · <a href="raster_sensitivity.json">栅格敏感性</a> · <a href="verification.json">输入校验</a></p></section>'''
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    firstpage='<!doctype html><meta charset="utf-8"><style>body{font:18px system-ui;max-width:1200px;margin:auto;padding:20px}img{width:48%}</style><a href="index.html">返回</a><h1>G03_01 前51段</h1><p>左为旧顺序，右为长线优先的新顺序。初期覆盖变化混合了连接、去重与排序贡献，不能单独算作模型效果。颜色是不同单元覆盖次数，不是实际墨色。</p>'
    for w in WIDTHS:
        firstpage+=f'<h2>假设笔宽{w:.1f}mm</h2><img src="outputs/G03_01/old_{w:.1f}_first51.png"><img src="outputs/G03_01/new_{w:.1f}_first51.png">'
    (ROOT/'first51.html').write_text(firstpage,encoding='utf-8')
    assert all(digest(r['path'])==r['sha256'] for r in frozen['files'])
    for r in input_checks:
        src=PRIOR/'outputs'/r['id']
        assert digest(src/'units.html')==r['old_new_page'] and digest(src/'curves.json')==r['curves']
    save(ROOT/'verification.json',dict(input_images=53,prior_frozen_files_unchanged=True,
        all_input_curves_and_pages_unchanged=True,input_checks=input_checks,
        coverage_scenarios=len(allrows),capture_plan_unmeasured_samples=len(plan['samples']),
        hardware_modules_imported=False,robot_connected=False,training_performed=False,
        source_files=[dict(path=str(p),sha256=digest(p)) for p in [Path(__file__),Path(__file__).with_name('brush_footprint_20261103.py'),Path(__file__).with_name('brush_capture_20261103.html')]]))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
