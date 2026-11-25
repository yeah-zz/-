# -*- coding: utf-8 -*-
"""Report all predeclared samples, including weak cases; no extractor tuning."""
import sys
sys.dont_write_bytecode=True
from freeze_stability_brush_20261101 import ROOT,read,jsave,digest
from run_stability_brush_20261101 import verify_lock
from stroke_geometry_brush_20261101 import CONFIG
from pathlib import Path
import json,html,re,collections,shutil
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

# Assistant visual audit of original / extraction contact sheets. No expert GT.
NOTES={
'T01':('山体与松干形态可辨，轮廓仍有短缺口，细松针未完整保留。',['轮廓断续','松针漏提']),
'T02':('前景短皴与点苔形成碎线，远山淡线和小松树细节不足，水面线偏多。',['碎线','淡线漏提','松针漏提','水面偏密']),
'T03':('双峰结构清楚，部分长皴线保留；轮廓有断续，松针不完整。',['轮廓断续','松针漏提']),
'T04':('圆石主形仍可辨，但石底点苔与短纹混杂，松冠细节损失。',['碎线','松针漏提','轮廓断续']),
'T05':('台地和岩层转折较清楚，内部仍有短碎线，小松树细节不足。',['碎线','松针漏提']),
'T06':('大弯曲山面仍有断续，前景纹理较杂，远山与松冠淡细线遗漏。',['轮廓断续','碎线','淡线漏提','松针漏提']),
'T07':('瀑布两侧结构可辨，淡瀑布线与皴线不完整，水面短线偏多。',['淡线漏提','松针漏提','水面偏密']),
'T08':('大松树整体可辨，松针缺失，远岛轮廓断续，水面线较密。',['松针漏提','轮廓断续','水面偏密']),
'T09':('峰体长线部分保留，远山淡线漏提，近景点苔与皴线碎片较多。',['淡线漏提','碎线','松针漏提']),
'T10':('三峰主形保留，左侧淡峰局部遗漏，近景碎线与水面线干扰。',['淡线漏提','碎线','松针漏提','水面偏密']),
'T11':('岩拱内外关系仍有杂乱，近景细碎纹理明显，不适合作为稳定提取的成功样本。',['碎线','松针漏提','交接复杂']),
'T12':('崖壁与松枝主形可辨，岩层局部缺口，远山和内部淡纹遗漏。',['轮廓断续','淡线漏提','松针漏提']),
'T13':('山谷布局可辨，山顶与山面长线断续，前景点苔细碎，远山淡层次遗漏。',['轮廓断续','碎线','淡线漏提','松针漏提']),
'T14':('台地转折较明确，但岩面与松冠短线多，背景水线偏密。',['碎线','松针漏提','水面偏密']),
'T15':('岛石形态可辨，水纹与岛岸碎片明显偏多，松针不完整。',['碎线','水面偏密','松针漏提']),
'T16':('山体上部轮廓与淡皴线断续明显；松干可辨，松冠细节缺失。',['轮廓断续','淡线漏提','松针漏提']),
'T17':('近平行长皴线保留较多，但仍断续；图像观察不能证明分笔、连接均正确。',['轮廓断续','松针漏提','分笔待验证']),
'T18':('前后山体布局可辨，近景短线杂乱，部分山面变空，细松针遗漏。',['碎线','淡线漏提','松针漏提']),
'T19':('淡山体长线大量断续或缺失，松树主枝可辨，针叶不完整，水面线偏多。',['淡线漏提','轮廓断续','松针漏提','水面偏密']),
'T20':('松干与近山主形可辨，粗墨处碎纹仍多，远山淡线与细松针遗漏。',['碎线','淡线漏提','松针漏提'])}

def verify():
    verify_lock();records=[read(p) for p in sorted((ROOT/'input_records').glob('T*.json'))];assert len(records)==20
    assert len({r['original_sha256'] for r in records})==20
    details=[];total=valid=0;maxerr=0.;checks=0
    for native in [False,True]:
        ids=read(ROOT/'evaluation_protocol.json')['native_resolution_substudy'] if native else [r['id'] for r in records]
        for r in records:
            if r['id'] not in ids:continue
            ident=r['id'];folder=ROOT/('native_outputs' if native else 'outputs')/ident;rr=read(folder/'run_result.json');assert rr['success'];checks+=1
            source=r['original'] if native else r['input'];assert digest(source)==rr['input_sha256'];assert rr['baseline_lock_sha256']==digest(ROOT/'baseline_lock.json')
            assert digest(r['original'])==r['original_sha256']==digest(r['generated_file']);assert digest(r['input'])==r['input_sha256'];assert r['crop']==False and r['used_for_training']==False
            with Image.open(source) as im:assert list(im.size)==(r['native_size'] if native else r['input_size'])
            raw=read(folder/'complete/complete_paths.json')['paths'];curvedoc=read(folder/'brush/curves.json');curves=curvedoc['paths'];params=read(folder/'brush/image_brush_parameters.json');stats=read(folder/'brush/statistics.json')
            assert stats['config_version']==CONFIG['version'];assert curvedoc['source_paths_sha256']==digest(folder/'complete/complete_paths.json');assert params['source_sha256']==digest(source)
            assert len(raw)==len(curves)==len(params['paths'])==stats['paths'];assert not params['ready_for_robot'] and not params['actually_drawn'];assert params['physical_z'] is None and params['physical_speed'] is None and params['water_content'] is None
            assert not curvedoc['ready_for_robot'] and not curvedoc['stroke_identity_verified']
            sv=ET.parse(folder/'brush/curves.svg');assert len(sv.findall('.//{http://www.w3.org/2000/svg}path'))==len(raw)
            for a,b,p in zip(raw,curves,params['paths']):
                assert np.allclose(a['points'][0],b['endpoint_start']) and np.allclose(a['points'][-1],b['endpoint_end'])
                if b['segments']:
                    assert np.allclose(b['segments'][0][0],a['points'][0]) and np.allclose(b['segments'][-1][-1],a['points'][-1])
                    for c,d in zip(b['segments'][:-1],b['segments'][1:]):assert np.allclose(c[-1],d[0])
                else:assert np.allclose(a['points'],b['polyline_fallback'])
                assert b['max_sampled_hausdorff_px']<=.6 and b['fixed_graph_junctions'];assert b['direction_verified']==False
                n=len(p['points']);assert all(len(p[k])==n for k in ['width_px','optical_density','valid','reason','raw_profile_width_px','raw_profile_optical_density'])
                for w,d,v in zip(p['width_px'],p['optical_density'],p['valid']):
                    assert (w is not None and d is not None and np.isfinite(w) and w>0 and np.isfinite(d) and d>=0) if v else (w is None and d is None)
                assert p['pressure_z'] is None and p['speed'] is None and p['water_content'] is None
            maxerr=max(maxerr,stats['maximum_sampled_deviation_px'])
            if not native:total+=stats['profile_samples'];valid+=stats['valid_profile_samples']
            details.append(dict(id=ident,native=native,paths=stats['paths'],profile_samples=stats['profile_samples'],valid_profile_samples=stats['valid_profile_samples'],valid_profile_fraction=stats['valid_profile_fraction'],maximum_sampled_deviation_px=stats['maximum_sampled_deviation_px']))
    behavior=read(ROOT/'brush_behavior_checks.json');assert behavior['all_passed']
    # Check snapshot includes the exact locked source / model bytes.
    for r in read(ROOT/'baseline_lock.json')['files']:
        p=Path(r['path']);dest=ROOT/'baseline_snapshot'/('code/'+p.name if p.suffix=='.py' else 'complete_model.pt' if str(p).endswith('fit_r2\\model.pt') or str(p).endswith('fit_r2/model.pt') else 'previous_model.pt' if p.suffix=='.pt' else p.name)
        assert digest(dest)==r['sha256']
    result=dict(all_passed=True,independent_originals=20,primary_runs=20,native_substudy_runs=4,baseline_files_unchanged=18,behavior_checks=behavior['count'],endpoint_and_svg_checks=True,parameter_consistency_checks=True,maximum_sampled_curve_deviation_px=maxerr,primary_profile_samples=total,primary_valid_profile_samples=valid,primary_valid_profile_fraction=valid/total,unknown_profile_fraction=1-valid/total,new_extractor_training_steps=0,robot_connected=False,actual_drawing_available=False,accuracy_validated=False,details=details)
    jsave(ROOT/'final_verification.json',result);return records,result

def main():
    records,verification=verify();audit=dict(reviewer='Assistant visual review; not artist expert ground truth',scope='Original / frozen extraction contact sheets at primary 496px input; coarse qualitative review, not exhaustive stroke correctness',all_images_retained=True,full_image_accuracy_or_pass_rate=None,records=[dict(id=i,observation=n,tags=t,full_detail_acceptance_established=False) for i,(n,t) in NOTES.items()],native_review='T01 pine at native resolution retains more trunk texture but still misses needles; direct resolution increase is not a sufficient fix. T06 curved profiles, T17 parallel wrinkles and T19 faint lines were also visually reviewed in source/496/native comparisons; gaps and scale-dependent selection remain. These are diagnostics, not certified improvements.')
    jsave(ROOT/'all_results.json',dict(primary=[read(p) for p in sorted((ROOT/'outputs').glob('T*/run_result.json'))],native=[read(p) for p in sorted((ROOT/'native_outputs').glob('T*/run_result.json'))],current_brush_statistics=verification['details'],new_training_steps=0,used_for_training=False,all_samples_retained=True))
    jsave(ROOT/'qualitative_review.json',audit);tags=collections.Counter(t for row in audit['records'] for t in row['tags']);jsave(ROOT/'summary.json',dict(verification=verification,qualitative_issue_counts=dict(tags),issue_counts_are_visual_flags_not_ground_truth_accuracy=True))
    cases=[]
    for r in records:
        i=r['id'];stats=read(ROOT/'outputs'/i/'brush/statistics.json');n,t=NOTES[i];cases.append(dict(id=i,level=r['level'],note=n,tags=t,valid=round(100*stats['valid_profile_fraction'],1),paths=stats['paths']))
    payload=json.dumps(cases,ensure_ascii=False).replace('</','<\\/')
    cards=[]
    for c in cases:
        i=c['id'];cards.append(f'<article data-tags="{html.escape(" ".join(c["tags"]))}"><h3>{i} · {c["level"]}</h3><a href="outputs/{i}/comparison.jpg"><img loading="lazy" src="outputs/{i}/comparison.jpg" alt="{i} 原图、冻结提取、曲线和图像笔宽浓淡预览"></a><p>{html.escape(c["note"])}</p><p class="small">宽度墨色可测采样点 {c["valid"]}%（不等于准确率）；几何分段 {c["paths"]} 条（不等于人类笔数）。</p><p><a href="originals/{i}.png">1254原图</a> · <a href="outputs/{i}/complete/complete.png">提取</a> · <a href="outputs/{i}/brush/curves.svg">曲线SVG</a> · <a href="outputs/{i}/brush/image_brush_parameters.json">图像参数JSON</a></p></article>')
    native=''.join(f'<p>{i} <a href="native_comparisons/{i}.jpg">打开原图 / 496 / 1254对照</a></p><img loading="lazy" src="native_comparisons/{i}.jpg" alt="{i} 分辨率对照">' for i in ['T01','T06','T17','T19'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>20张新图稳定性与图像笔触预览</title><style>
    :root{font-family:"Microsoft YaHei",sans-serif;color:#23332e;background:#f1f0e8}body{max-width:1260px;margin:auto;padding:24px}h1{font-size:30px}h2{margin-top:36px}p{line-height:1.8}a{color:#176961}section,article{background:#fff;padding:20px;border-radius:12px;margin:16px 0}img{max-width:100%;height:auto}button,select,input{font:inherit;margin:6px;padding:6px}.small{font-size:13px;color:#59685e}.notice{border-left:5px solid #bb813c;padding:12px;background:#fff6e4}.metrics{display:flex;gap:16px;flex-wrap:wrap}.metrics div{background:#e4eae0;padding:16px;border-radius:10px}.metrics strong{font-size:25px}.compare{display:grid;grid-template-columns:1fr 1fr;gap:12px}.compare img{width:100%;background:white;border:1px solid #ddd}#gallery article{border:1px solid #e1e5de}label{display:inline-block}.native img{max-height:500px}.tag{font-size:13px;padding:6px 10px}@media(max-width:700px){body{padding:12px}.compare{grid-template-columns:1fr}}
    </style><h1>20张新图：稳定性测试与粗细浓淡预览</h1><p>2026-11-01 · 用户认可的完整提取版本已冻结。本轮没有更新提取模型权重，没有训练新图，没有连接机械臂。</p>
    <div class="metrics"><div><strong>20</strong><br>独立生成的全新整图</div><div><strong>4</strong><br>原生分辨率诊断</div><div><strong>VALID%</strong><br>图像宽度墨色可测采样点</div><div><strong>18</strong><br>基线文件哈希保持一致</div></div>
    <p class="notice">结论：还没有达到每张新图都稳定复现已认可效果。山体主形多可辨，但淡山轮廓、密集松针、点苔与短皴线仍有明显缺陷。所有20张均公开，未挑选成功图；没有逐笔标准答案，因此不报告提取准确率或整图通过率。</p>
    <section><h2>逐张检查</h2><p>固定496像素整图评估，便于与既有流程一致比较；保留全部1254像素原图。左侧原图，右侧切换提取或笔触预览。可放大同一位置检查，几何曲线只平滑已有轨迹，没有解决笔画归属和错误连接。</p><label>图片 <select id="sample"></select></label><label>右侧 <select id="mode"><option value="complete/complete.png">冻结提取</option><option value="brush/smooth_centerlines.png">平滑曲线</option><option value="brush/image_width_tone_preview.png">图像笔宽浓淡（非实画）</option></select></label><label>局部放大 <input id="zoom" type="range" min="1" max="4" step=".25" value="1"></label><p id="note"></p><div class="compare"><div style="overflow:auto"><img id="original" alt="测试输入原图"></div><div style="overflow:auto"><img id="result" alt="所选处理结果"></div></div></section>
    <section><h2>单条轨迹的图像粗细浓淡</h2><p>选择T01中一条较长且可测的轨迹解释参数，不代表整图准确率。蓝线为提取中心，右侧为该轨迹图像宽度和局部表观墨色的回放。此示例参与了预览程序的宽度正则化开发，不是盲测的笔触效果。</p><img src="ONE_STROKE_PREVIEW.jpg" alt="单条轨迹笔触对照"><img src="ONE_STROKE_PARAMETERS.png" alt="沿轨迹的图像笔宽和表观墨色参数"><p class="notice">笔宽单位是图像像素，墨色是相对局部背景的表观光密度。设色、纸纹、粗墨交汇和邻笔会干扰估计；UNKNOWN%采样点仍未知，以细灰线占位显示。未知不代表真实淡笔。未推断毫米笔宽、下压量、速度、含墨量、干湿或真实笔顺。</p></section>
    <section class="native"><h2>分辨率诊断：不能仅靠放大解决</h2><p>预先固定T01、T06、T17、T19四张，以同一冻结提取器处理1254原图。没有用这些结果调提取阈值。T01松树原生输入保留更多树干纹理，但针叶仍遗漏；由于部分规则依赖像素尺度，直接增大输入也可能改变筛选结果。这是诊断，不是“全部细节已恢复”。</p><img src="NATIVE_PINE_DETAIL.jpg" alt="松针局部 原图、496和1254输入结果">NATIVE</section>
    <h2>全部结果与失败案例</h2><p>每行顺序：原图 → 冻结版本提取 → 保留分笔的平滑曲线 → 图像笔宽浓淡预览。点击图像查看大图。标签来自助手目视检查，不能替代逐笔真值评估。</p><label>筛选问题 <select id="filter"><option value="">全部20张</option><option>轮廓断续</option><option>淡线漏提</option><option>松针漏提</option><option>碎线</option><option>水面偏密</option></select></label><div id="gallery">CARDS</div>
    <section><h2>可复核的方法与边界</h2><p>提取器、权重和校准配置18个文件在新图产生前冻结；新图是20次独立生成，没有把同一图裁块计数。simple / medium / stress是生成提示的计划难度，部分输出纹理实际较密，均照常保留。提取部分没有新训练；笔触预览是本轮开发的新程序，T01用于处理圆点状宽度噪声，旧预览已归档。</p><p>曲线保留已有路径端点与图交汇锚点；拟合若增加邻笔接触则回退原折线，采样双向偏差不超过0.6输入像素。这是几何保护，不证明语义连接或人类分笔正确。图像参数保留原测量与平滑值，交汇、不隔离墨带、淡色歧义和短墨块标未知。曲线和参数文件均标记ready_for_robot=false。</p><p>下一轮应以原生分辨率的山体、树木局部建立分开的尺度一致监督，解决“主线连续性 / 松针遗漏 / 点苔短皴混杂”；本轮20张已经查看，今后是回归集，再评估泛化必须新增封存测试图。继续保留现有认可版本，笔触标定待实机到位。</p><p><a href="final_verification.json">完整技术校验</a> · <a href="qualitative_review.json">逐张目视记录</a> · <a href="baseline_lock.json">基线哈希</a> · <a href="new_input_manifest.json">新图清单</a> · <a href="generation_prompts.json">生成提示</a> · <a href="evaluation_protocol.json">原始评估协议</a> · <a href="evaluation_scope_amendment.json">预览开发范围说明</a> · <a href="brush_behavior_checks.json">程序行为验证</a></p><p class="small">所有新原图、缓存和实验输出位于E盘。图像工具的本会话逻辑目录通过junction映射至E盘，未改动HOME或CODEX_HOME。没有安装新依赖，没有实画记录。</p></section>
    <script>const cases=PAYLOAD;const select=document.querySelector('#sample'),mode=document.querySelector('#mode');for(const c of cases){const o=document.createElement('option');o.value=c.id;o.textContent=c.id+' · '+c.level;select.append(o)}function update(){const c=cases.find(c=>c.id===select.value);document.querySelector('#original').src='inputs/'+c.id+'.png';document.querySelector('#result').src='outputs/'+c.id+'/'+mode.value;document.querySelector('#note').textContent=c.note+' 图像参数可测点 '+c.valid+'%；这不是提取准确率。'}select.onchange=mode.onchange=update;document.querySelector('#zoom').oninput=e=>{for(const id of ['original','result']){const im=document.getElementById(id);im.style.maxWidth='none';im.style.width=(e.target.value*100)+'%'}};document.querySelector('#filter').onchange=e=>{for(const el of document.querySelectorAll('#gallery article'))el.hidden=!!e.target.value&&!el.dataset.tags.includes(e.target.value)};update();</script></html>'''
    page=page.replace('VALID%',f'{verification["primary_valid_profile_fraction"]*100:.1f}%').replace('UNKNOWN%',f'{verification["unknown_profile_fraction"]*100:.1f}%').replace('>NATIVE</section>','>'+native+'</section>').replace('CARDS',''.join(cards)).replace('PAYLOAD',payload)
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    links=re.findall(r'(?:src|href)="([^"#]+)"',page);missing=[s for s in links if '://' not in s and not (ROOT/s).exists()];assert not missing,missing
    snapshot=ROOT/'experiment_source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('*stability*20261101*'):
        if p.is_file():shutil.copy2(p,snapshot/p.name)
    for n in ['stroke_geometry_brush_20261101.py','check_stroke_geometry_brush_20261101.py','new_landscape_manifest_20261101.py']:
        p=Path(__file__).parent/n;shutil.copy2(p,snapshot/n)
    jsave(ROOT/'report_asset_checks.json',dict(relative_assets_checked=len(links),missing=[],all_20_cases_in_report=True,all_passed=True))
    print('REPORT_READY',str(ROOT/'index.html'),json.dumps({k:v for k,v in verification.items() if k!='details'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
