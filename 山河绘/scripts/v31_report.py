# -*- coding: utf-8 -*-
from v31_profiles import ROOT,read,jsave,digest,LAB
from pathlib import Path
import json,base64,shutil,subprocess
import numpy as np,torch
from PIL import Image
RESULTS=ROOT/'adaptive_results'

def viewer():
    rows=[]
    for row in read(ROOT/'manifest.json')['development']:
        item=dict(id=row['id'],images={},methods={})
        for label,file in [('original',Path(row['raw']))]:item['images'][label]='data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode()
        item['initial']=read(ROOT/'initial_paths'/(row['id']+'.json'))['paths']
        for method in ['classical','learned']:
            doc=read(RESULTS/method/row['id']/'trajectory.json');file=RESULTS/method/row['id']/'ribbon_reconstruction.png';item['methods'][method]=dict(paths=doc['paths'],ribbon='data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode(),source_curves=doc['meta']['source_curves'],retained_curves=len(doc['paths']),width_range=[float(min(min(x) for x in doc['apparent_width_px'])),float(max(max(x) for x in doc['apparent_width_px']))],alpha_range=[float(min(min(x) for x in doc['apparent_ink_alpha'])),float(max(max(x) for x in doc['apparent_ink_alpha']))])
        rows.append(item)
    data=json.dumps(rows,ensure_ascii=False,separators=(',',':')).replace('</','<\/')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>v31 连续曲线与笔触参数实验</title><style>
    body{background:#f3f0e8;color:#242424;font:16px system-ui,"Microsoft YaHei";margin:20px}h1{font-size:24px}p{line-height:1.6;max-width:1250px}label{display:inline-flex;align-items:center;gap:8px;margin:6px 20px 6px 0}select,input{font:inherit}.grid{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:12px}.pane{background:white;border:1px solid #c9c2b2;overflow:auto;height:68vh}canvas{display:block}.title{padding:8px;font-weight:bold;background:#eee8db}#status{font-size:14px} @media(max-width:850px){.grid{grid-template-columns:1fr}.pane{height:55vh}}</style>
    <h1>v31：线条、表观笔宽与灰度重建</h1><p>这是实际训练模块与图像拟合的离线输出。左侧为原图，中间检查中心线，右侧为数值渲染器保存的真实重建图。重建图不等于机械臂实画；笔宽尚未映射为毛笔下压，灰度尚未映射为墨液浓度。</p>
    <label>图片<select id="sample"></select></label><label>初始化<select id="method"><option value="learned">训练网络＋相同图像拟合</option><option value="classical">无网络＋相同图像拟合</option></select></label><label>放大<input id="zoom" type="range" min="0.75" max="3" step="0.25" value="1.25"></label><br>
    <label>中心线背景<input id="background" type="range" min="0" max="1" step="0.05" value="0.45"></label><label><input id="old" type="checkbox">叠加原候选蓝线</label><label><input id="color" type="checkbox">按轨迹编号着色</label><p id="status"></p>
    <div class="grid"><div><div class="title">原图</div><div class="pane"><canvas id="original"></canvas></div></div><div><div class="title">拟合后的中心线</div><div class="pane"><canvas id="lines"></canvas></div></div><div><div class="title">表观笔宽与灰度重建（非实画）</div><div class="pane"><canvas id="ribbon"></canvas></div></div></div>
    <p>本轮先保留旧提取器生成候选，再优化其连续曲线、宽度与灰度。没有新增缺失笔画，也没有自动恢复笔顺或识别皴法类别。图像拟合内部最长边为320；输出坐标还原到本次输入图。局部参考线不完整，相关数值不代表整幅画准确率。</p>
    <script id="data" type="application/json">PAYLOAD</script><script>
    const rows=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);let original=new Image(),ribbon=new Image(),token=0;
    rows.forEach((r,i)=>$('sample').add(new Option(r.id,i)));$('sample').value=rows.findIndex(r=>r.id==='R04');
    function load(){const n=++token,row=rows[+$('sample').value],m=row.methods[$('method').value];const a=new Image(),b=new Image();let loaded=0;function ready(){if(++loaded===2&&n===token){original=a;ribbon=b;draw();}}a.onload=ready;b.onload=ready;a.src=row.images.original;b.src=m.ribbon;}
    function draw(){if(!original.width)return;const row=rows[+$('sample').value],m=row.methods[$('method').value],z=+$('zoom').value;for(const id of ['original','lines','ribbon']){const c=$(id);c.width=Math.round(original.width*z);c.height=Math.round(original.height*z);const ctx=c.getContext('2d');ctx.scale(z,z);ctx.fillStyle='white';ctx.fillRect(0,0,original.width,original.height);if(id==='original')ctx.drawImage(original,0,0);else if(id==='ribbon')ctx.drawImage(ribbon,0,0);else{ctx.globalAlpha=+$('background').value;ctx.drawImage(original,0,0);ctx.globalAlpha=1;ctx.lineWidth=1;ctx.lineJoin='round';ctx.lineCap='round';function curves(paths,old){paths.forEach((p,i)=>{ctx.strokeStyle=old?'#2373c4':($('color').checked?`hsl(${i*137.508%360} 70% 38%)`:'#db2437');ctx.beginPath();p.forEach((q,k)=>k?ctx.lineTo(...q):ctx.moveTo(...q));ctx.stroke();});}if($('old').checked)curves(row.initial,true);curves(m.paths,false);}}
    $('status').textContent=`${row.id} · 原候选 ${row.initial.length} 条 → 输出 ${m.retained_curves} 条；墨带宽度参数 ${m.width_range[0].toFixed(1)}–${m.width_range[1].toFixed(1)} 像素，表观灰度强度 ${m.alpha_range[0].toFixed(2)}–${m.alpha_range[1].toFixed(2)}。颜色只表示轨迹编号。`;}
    $('sample').onchange=load;$('method').onchange=load;['zoom','background','old','color'].forEach(id=>$(id).oninput=draw);load();
    </script></html>'''.replace('PAYLOAD',data)
    (ROOT/'INTERACTIVE_COMPARISON.html').write_text(html,encoding='utf-8');script=html.rsplit('<script>',1)[1].split('</script>',1)[0];file=ROOT/'viewer_check.js';file.write_text(script,encoding='utf-8');node=shutil.which('node')
    if node:
        cp=subprocess.run([node,'--check',str(file)],capture_output=True,text=True);jsave(ROOT/'viewer_syntax_check.json',dict(returncode=cp.returncode,stderr=cp.stderr));assert cp.returncode==0

def finalize():
    manifest=read(ROOT/'manifest.json');frozen=read(ROOT/'frozen_inputs.json');train=read(ROOT/'training_summary.json');ev=read(ROOT/'development_adaptive_evaluation.json');cal=read(ROOT/'calibration_adaptive_gate.json');checkpoint=torch.load(ROOT/'profile_model.pt',map_location='cpu',weights_only=True);checks={}
    checks['1600_actual_cuda_network_updates']=train['actual_cuda_updates']==1600 and train['actual_device'].startswith('cuda')
    checks['trained_checkpoint_unchanged']=digest(ROOT/'profile_model.pt')==train['checkpoint_sha256']
    checks['checkpoint_parameters_finite']=all(bool(torch.isfinite(p).all()) for p in checkpoint['model'].values())
    checks['train_validation_inputs_unchanged']=digest(ROOT/'train_profiles.npz')==frozen['training_profiles_sha256'] and digest(ROOT/'validation_profiles.npz')==frozen['validation_profiles_sha256']
    checks['all_source_data_unchanged']=all(digest(r['source'])==r['source_sha256'] for r in manifest['records'])
    checks['original_review_unchanged']=digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json')==manifest['parent_review_sha256']
    checks['old_raster_checkpoint_unchanged']=digest(LAB/'clean_strokes_v12_20261024'/'final.pt')==manifest['old_raster_sha256']
    checks['machine_initial_paths_unchanged']=all(digest(ROOT/'initial_paths'/name)==sha for name,sha in frozen['initial_path_hashes'].items())
    tx=np.load(ROOT/'train_profiles.npz');vx=np.load(ROOT/'validation_profiles.npz');checks['profile_train_validation_images_disjoint']=not(set(tx['ids'])&set(vx['ids']));checks['development_images_not_in_profile_training']=not(set(r['id'] for r in manifest['development'])&set(tx['ids']))
    outputs=[];geometry=[];gains=[];pruned=0;checks['32_matched_cuda_curve_fits_completed']=True;checks['finite_matching_curve_width_tone_arrays']=True;checks['offline_flags_and_no_reference_inputs']=True
    for row in manifest['calibration']+manifest['development']:
        h,w=row['shape']
        for variant in ['classical','learned']:
            file=RESULTS/variant/row['id']/'trajectory.json';doc=read(file);meta=doc['meta'];checks['32_matched_cuda_curve_fits_completed'] &= meta['actual_device']=='cuda:0' and meta['steps']==100 and meta['renderer_revision']==3
            checks['offline_flags_and_no_reference_inputs'] &= doc['physical_z'] is None and doc['ready_for_robot'] is False and doc['actually_drawn'] is False and not doc['reference_labels_used'] and not meta['reference_labels_used']
            for p,width,ink in zip(doc['paths'],doc['apparent_width_px'],doc['apparent_ink_alpha']):
                q=np.asarray(p);checks['finite_matching_curve_width_tone_arrays'] &= bool(len(p)==len(width)==len(ink) and np.isfinite(q).all() and np.isfinite(width).all() and np.isfinite(ink).all() and (q>=0).all() and (q[:,0]<=w-1).all() and (q[:,1]<=h-1).all())
            if variant=='learned':geometry+=meta['geometry_checks'];pruned+=len(meta['pruning']);gains.append(dict(id=row['id'],initial_photometric=doc['log'][0]['photometric'],final_photometric=meta['final_photometric_error']))
            outputs.append(dict(id=row['id'],variant=variant,sha256=digest(file),file=str(file)))
    checks['subpixel_source_curve_approximation_goal']=all(r['tolerance_met'] for r in geometry)
    checks['packed_renderer_forward_backward_consistent']=read(ROOT/'packed_renderer_check.json')['passed']
    interface=read(ROOT/'reference_free_interface_check'/'interface_check.json');checks['image_only_prediction_annotation_guard_passed']=interface['inference_reference_guard_active'] and interface['blocked_annotation_opens']==0 and not interface['reference_labels_used']
    unmet=[r for r in geometry if not r['tolerance_met']];engineering_checks={k:bool(v) for k,v in checks.items() if k!='subpixel_source_curve_approximation_goal'}
    checks={k:bool(v) for k,v in checks.items()};jsave(ROOT/'verification.json',dict(checks=checks,all_passed=all(checks.values()),engineering_checks_passed=all(engineering_checks.values()),outputs=outputs,geometry_max_source_error_px=max(r['source_parameter_aligned_max_error_px'] for r in geometry),geometry_goal_px=.75,geometry_goal_unmet_curves=len(unmet),geometry_goal_met=not unmet,source_curves_checked=len(geometry),apparent_parameters_not_physical_brush_calibration=True))
    jsave(ROOT/'deployment_gate.json',dict(default_replaced=False,calibration_passed=cal['passed'],independent_real_artwork_blind_test=False,physical_brush_calibrated=False,robot_connected=False))
    jsave(ROOT/'reconstruction_diagnostic.json',dict(records=gains,total_curves_pruned=pruned,note='Reconstruction error is the optimized objective, not independent extraction accuracy.'))
    table=['| 10张开发图，同一初始候选 | 参考点覆盖 / 3px | 单条路径覆盖 | 平均路径数 | 重叠代理 |','|---|---:|---:|---:|---:|']
    for name,title in [('initial','原候选'),('classical','无网络初始化＋相同图像拟合'),('learned','训练网络初始化＋相同图像拟合')]:
        r=ev['summary'][name];table.append(f"| {title} | {r['point_coverage_3px']:.2%} | {r['mean_best_single_path_coverage']:.2%} | {r['path_count']:.1f} | {r['duplicate_point_fraction']:.2%} |")
    initial=ev['summary']['initial'];learned=ev['summary']['learned'];width_gain=read(ROOT/'procedural_diagnostic.json')['records'];bold=[]
    for name in ['initial','classical','learned']:
        rr=[r['metrics'][name]['cross_section_proxies'] for r in ev['records']];n=sum(r['bold_samples'] for r in rr);rate=sum(r['bold_samples']*(r['bold_multiple_center_fraction'] or 0) for r in rr)/max(1,n);bold.append(dict(variant=name,automatic_broad_samples=n,multiple_center_fraction=rate,not_expert_geometry_gold=True))
    jsave(ROOT/'broad_ink_diagnostic.json',dict(records=bold))
    report=f'''# v31 连续曲线、表观笔宽与灰度重建实验

本轮完成了新的局部墨带参数网络训练，以及真实图像约束的连续曲线拟合。网络训练使用RTX 5060 Laptop GPU / PyTorch 2.8.0+cu128，实际1600步：1200步程序剖面监督、400步真实剖面适配。**这不是整个线条提取网络重新训练，也没有完成皴法类别识别。**

## 实际训练与数据

- 真实训练剖面{len(tx['profiles'])}条，来自{len(np.unique(tx['ids']))}个图像文件；验证剖面{len(vx['profiles'])}条，来自{len(np.unique(vx['ids']))}个留出文件。没有新增独立画作。
- 文件划分见manifest.json；U03、stage_035、replay_R02未进入本轮优化器。来源或画作家族可能相关，不能视为独立画作验证。真实剖面只取原有效标注范围内可辨识的墨带，明确背景才可作负例，不把未知区域当背景。
- 真实宽度、灰度没有专家真值；真实适配使用剖面重建和1.5px位置容忍带，未把自动测量结果升级成专家标注。
- 保留checkpoint为第{checkpoint['step']}步；留出真实剖面的重建损失由第1200步的{train['logs'][11]['validation_photometric_loss']:.4f}变为{checkpoint['validation_loss']:.4f}。这是剖面重建指标，不是整图线条准确率。

## 方法与对照

流程：原图 → 冻结v28生成候选 → 新网络估计局部中心偏移/墨带半径/灰度 → 自适应连续三次曲线 → 可微墨带渲染 → 按图像误差优化参数。

无网络对照用原图剖面的直接测量初始化，并采用相同曲线、相同步数和损失。两组都以每图100步优化图像参数；这32次图像拟合与1600步神经网络训练分开计数。

{chr(10).join(table)}

全部为此前反复使用的开发图，不是新的盲测集；参考仅覆盖部分线条，表中数值不是整幅画提取准确率。重叠代理统一将所有方法曲线重采样到约1px后计算，避免不同曲线采样密度干扰比较。误连代理与粗墨截面代理也不是完整专家真值。

六张校准图沿用预先门槛，最终通过：**{cal['passed']}**。本轮不自动替换正式绘画流程。单条路径覆盖的开发变化为{(learned['mean_best_single_path_coverage']-initial['mean_best_single_path_coverage'])*100:+.2f}个百分点；参考点覆盖变化为{(learned['point_coverage_3px']-initial['point_coverage_3px'])*100:+.2f}个百分点。

## 先修正了曲线表示误差

初版每14px布控制点，部分尖角转换误差达到约5px，已保留在results/、development_evaluation.json及representation_diagnostic.json。没有把这类实现损失直接归因于训练方法。

最终按机器候选本身的逼近误差自适应加控制点，不读取参考答案；本轮检查{len(geometry)}条源曲线（两组初始化几何相同，按学习组计），最大参数对齐误差{max(r['source_parameter_aligned_max_error_px'] for r in geometry):.3f}px。目标0.75px，仍有{len(unmet)}条复杂候选达到96个控制点上限后未满足；没有将这一目标标为通过。最终结果位于adaptive_results/，粗曲线版保留用于诊断。

批量渲染与逐笔渲染检查比较了前向输出、正则项和反向梯度；浮点输出差异低于0.1个8位灰度级，没有放宽训练或提取验收门槛。

## 当前能力与限制

- 已能为每条候选输出连续坐标、沿线变化的墨带宽度参数与表观灰度，提供数值渲染重建，而不只显示统一粗细的彩色中心线。
- 这些是图像域参数。宽度还没有映射为实机下压；表观灰度不等于墨液浓度，也没有识别含水量或真实飞白。
- 当前候选仍来自冻结旧模型，没有新增漏掉的笔画，也未自动恢复断点连接或画家的历史笔顺。图像像素相似也不能唯一确定交汇处分笔。
- 拟合内部最长边320，最后还原坐标；源图细节更细时会受分辨率、软边渲染和当前笔触模型限制。没有用这轮结果证明整幅青绿古画已可提取。
- 笔画数量只在高度冗余且去除后重建误差基本不变时减少，本轮两种初始化所有校准/开发图中学习组共移除{pruned}条。
- 程序诊断中，学习组总体半径误差由{width_gain['classical']['identifiable']['radius_mae_px']:.3f}px降到{width_gain['learned']['identifiable']['radius_mae_px']:.3f}px，但淡墨42个可辨程序剖面的中心误差反而由{width_gain['classical']['faint']['center_mae_px']:.3f}px变为{width_gain['learned']['faint']['center_mae_px']:.3f}px，不能声称淡线问题已解决。
- 重建误差下降是优化目标，不能单独作为线条更准确的证据。应同时检查参考线覆盖、重复描画与原图叠加。

## 可查看与复现

- RESULT_COMPARISON_ADAPTIVE.jpg：四张固定代表图，原图/原候选/拟合后中心线/表观笔宽与灰度重建。
- INTERACTIVE_COMPARISON.html：自包含离线页面，切换十张图、有/无学习初始化、放大与原图叠加。
- adaptive_comparisons/：全部校准和开发图的逐项对照。
- profile_model.pt：本轮训练权重；synthetic_only.pt：真实适配前对照权重。
- calibration_adaptive_gate.json、development_adaptive_evaluation.json：校准与开发评估。
- reference_free_interface_check/：真实图像输入的端到端接口检查；审计钩子禁止打开参考描线/训练数据，不能称为新盲测。
- source_snapshot/：脚本快照。运行入口为rock_v31.ps1；predict模式接受原图和E/F盘输出目录。

本轮数据、权重、缓存和输出在E盘。未安装新依赖，未连接、使能或移动机械臂。原标注与原基线权重哈希检查保持一致。

## 下一步应依据这些结果选择

如果中心线指标未稳定超越原候选，应保留本轮笔宽/灰度模块作为独立研究成果，先处理完整笔画候选与拓扑，而不能只继续优化重建损失。薄淡笔画与宽墨带应分开约束，增强可辨识的连续笔画监督；复杂墨迹分笔需要明确的整体笔画假设，避免让局部宽度膨胀代替正确笔画。只有在不降低中心线覆盖的条件下再扩展数据和训练规模。

工程完整性校验：{sum(engineering_checks.values())}/{len(engineering_checks)}项通过。0.75px几何表示目标单独记录为{not unmet}；校准提升门槛为{cal['passed']}，不能把工程检查通过理解为提取精度已达标。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8');snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['v31_profiles.py','v31_render.py','v31_experiment.py','v31_report.py','rock_v31.ps1']:shutil.copy2(Path(__file__).with_name(name),snap/name)
    for file in [ROOT/'RESULT_COMPARISON_ADAPTIVE.jpg']+list((ROOT/'adaptive_comparisons').glob('*.jpg')):
        with Image.open(file) as im:im.verify()
    jsave(ROOT/'artifact_manifest.json',dict(report_sha256=digest(ROOT/'REPORT.md'),viewer_sha256=digest(ROOT/'INTERACTIVE_COMPARISON.html'),comparison_sha256=digest(ROOT/'RESULT_COMPARISON_ADAPTIVE.jpg'),source_sha256={p.name:digest(p) for p in snap.iterdir()}))
    assert all(engineering_checks.values()),engineering_checks
    print('V31_FINISHED',checks,'DEVELOPMENT',ev['summary'],flush=True)

if __name__=='__main__':viewer();finalize()
