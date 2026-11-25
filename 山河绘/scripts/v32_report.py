# -*- coding: utf-8 -*-
from v32_stroke_model import *
import base64,io,hashlib,html

def image_url(path):
    im=Image.open(path).convert('RGB');b=io.BytesIO();im.save(b,format='JPEG',quality=93);return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()

def finish():
    protocol=read(ROOT/'protocol.json');train=read(ROOT/'training_summary.json');data=read(ROOT/'data_summary.json');cal=read(ROOT/'calibration.json');dev=read(ROOT/'development_evaluation.json');diag=read(ROOT/'seed_conditioned_diagnostic.json');mf=read(V31/'manifest.json');sources=[]
    checks=dict(cuda_forward_backward=read(ROOT/'cuda_smoke.json')['finite_backward'],actual_cuda_updates=train['actual_updates']==2400,real_review_unchanged=digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json')==protocol['protected_review_sha256'],original_raster_unchanged=digest(LAB/'clean_strokes_v12_20261024'/'final.pt')==protocol['protected_raster_sha256'],training_sources_unchanged=all(digest(r['data'])==r['data_sha256'] and digest(r['labels'])==r['labels_sha256'] for r in protocol['records']),calibration_before_development=not cal['development_used'],all_sixteen_predictions_present=all((ROOT/'proposals'/r['id']/'adapted.json').exists() for part in ['calibration','development'] for r in mf[part]),all_reference_free_flags=all(read(ROOT/'proposals'/r['id']/'adapted.json')['reference_labels_used']==False for part in ['calibration','development'] for r in mf[part]),real_unknown_not_stop_negatives=all(np.max(np.load(ROOT/('real_'+s+'.npz'))['stop_mask'])==0 for s in ['train','validation']),real_training_validation_file_disjoint=not (set(r['id'] for r in protocol['records'] if r['split']=='train')&set(r['id'] for r in protocol['records'] if r['split']=='validation')))
    verify=[]
    for p in list((ROOT/'comparisons').glob('*.jpg'))+[ROOT/'RESULT_COMPARISON.jpg']:
        with Image.open(p) as im:im.verify()
        verify.append(str(p))
    checks['preview_images_verified']=len(verify)==17
    guard=read(ROOT/'reference_free_interface_check'/'interface_check.json');checks['image_only_interface_annotation_guard']=guard['inference_annotation_read_guard_active'] and guard['blocked_annotation_opens']==0
    frozen=read(V31/'frozen_inputs.json');checks['old_candidate_files_unchanged']=all(digest(V31/'initial_paths'/name)==sha for name,sha in frozen['initial_path_hashes'].items())
    jsave(ROOT/'verification.json',dict(checks=checks,all_engineering_checks_passed=all(checks.values()),preview_count=len(verify),extraction_quality_gate_passed=cal['selected']['passed']))
    if not all(checks.values()):raise RuntimeError('Integrity checks failed')
    jsave(ROOT/'deployment_gate.json',dict(calibration_passed=cal['selected']['passed'],default_replace_allowed=False,robot_connected=False,physical_brush_calibration_done=False,actually_drawn=False))
    items=[]
    for r in mf['development']:
        dest=ROOT/'trajectories'/'adapted'/r['id'];old=V31/'adaptive_results'/'initial'/r['id']/'centerlines.png'
        row=next(a for a in dev['records'] if a['id']==r['id']);items.append(dict(id=r['id'],images=dict(original=image_url(r['raw']),old=image_url(old),new=image_url(dest/'new.png'),combined=image_url(dest/'combined.png'),overlay=image_url(dest/'overlay.png')),metrics=row['metrics'],additions=row['additions']))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>v32 有序笔画对照</title><style>body{font:16px system-ui;background:#f3f0e8;margin:24px;color:#252525}select,input,button{font:inherit;padding:8px}canvas{max-width:100%;background:white;border:1px solid #ccc}section{display:grid;grid-template-columns:1fr 1fr;gap:16px}p{max-width:1100px;line-height:1.6}pre{white-space:pre-wrap}label{margin:10px}</style><h2>v32 图像起点与有序曲线：实际离线输出</h2><p>左侧原图；右侧切换旧候选、网络新增线、合并线与红色叠加。未连接机械臂。数据是反复使用的开发图，参考线不完整；指标不代表整幅画准确率。当前为研究候选，没有替换正式流程。</p><label>图片 <select id="case"></select></label><label>显示 <select id="view"><option value="combined">旧候选＋网络新增线</option><option value="old">旧候选</option><option value="new">只看网络新增线</option><option value="overlay">新增红线叠加原图</option></select></label><label>缩放 <input id="zoom" type="range" min="1" max="4" step=".5" value="1"></label><section><div><p>原图</p><canvas id="left"></canvas></div><div><p id="title"></p><canvas id="right"></canvas></div></section><pre id="numbers"></pre><script>const rows=DATA;const cases=document.getElementById('case'), view=document.getElementById('view'),zoom=document.getElementById('zoom');rows.forEach((r,i)=>cases.add(new Option(r.id,i)));async function draw(id,url){let img=new Image();img.src=url;await img.decode();let c=document.getElementById(id);c.width=img.width*zoom.value;c.height=img.height*zoom.value;c.style.maxWidth='none';c.getContext('2d').drawImage(img,0,0,c.width,c.height)}function update(){let r=rows[cases.value||0];draw('left',r.images.original);draw('right',r.images[view.value]);document.getElementById('title').textContent=r.id+' '+view.selectedOptions[0].text;let names={initial:'旧候选',classical:'几何追踪补线',procedural:'程序监督网络补线',adapted:'真实适配网络补线'};document.getElementById('numbers').textContent=Object.entries(r.metrics).map(([k,m])=>`${names[k]}：参考点覆盖 ${(100*m.point_coverage_3px).toFixed(2)}%，单路径覆盖 ${(100*m.mean_best_single_path_coverage).toFixed(2)}%，路径 ${m.path_count}，重叠代理 ${(100*m.duplicate_point_fraction).toFixed(2)}%`).join('\\n')+'\\n网络新增路径：'+r.additions.adapted+'；真实笔顺、笔压和落笔效果尚未验证。'}cases.onchange=view.onchange=zoom.oninput=update;update();</script></html>'''
    (ROOT/'INTERACTIVE_COMPARISON.html').write_text(page.replace('DATA',json.dumps(items,ensure_ascii=False)),encoding='utf-8')
    file=ROOT/'INTERACTIVE_COMPARISON.html';content=file.read_text(encoding='utf-8').replace('grid-template-columns:1fr 1fr','grid-template-columns:minmax(0,1fr) minmax(0,1fr)').replace('p{max-width','section>div{overflow:auto}p{max-width');file.write_text(content,encoding='utf-8')
    import subprocess,ast
    script=content.split('<script>',1)[1].split('</script>',1)[0];(ROOT/'viewer_script.js').write_text(script,encoding='utf-8');result=subprocess.run(['node','--check',str(ROOT/'viewer_script.js')],capture_output=True,text=True,encoding='utf-8');jsave(ROOT/'viewer_syntax_check.json',dict(passed=result.returncode==0,stderr=result.stderr))
    if result.returncode:raise RuntimeError(result.stderr)
    for name in ['v32_stroke_model.py','v32_experiment.py','v32_report.py']:ast.parse((Path(__file__).parent/name).read_text(encoding='utf-8'))
    table='| 同一批10张开发图 | 参考点覆盖 / 3px | 单路径覆盖 | 平均路径数 | 重叠代理 | 误连代理 |\n|---|---:|---:|---:|---:|---:|\n'
    for key,label in [('initial','旧候选'),('classical','旧候选＋几何追踪补线'),('procedural','旧候选＋程序监督网络补线'),('adapted','旧候选＋真实适配网络补线')]:
        s=dev['summary'][key];table+=f"| {label} | {s['point_coverage_3px']:.2%} | {s['mean_best_single_path_coverage']:.2%} | {s['path_count']:.1f} | {s['duplicate_point_fraction']:.2%} | {s['mixed_reference_paths_flagged']:.2f} |\n"
    ds=diag['results'];s=dev['summary']['adapted'];b=dev['summary']['initial'];chosen=torch.load(ROOT/'curve_model.pt',map_location='cpu',weights_only=True)['step'];numadded=sum(a['additions']['adapted'] for a in dev['records']);passed=cal['selected']['passed']
    text=f'''# v32 图像起点、有序笔画生成与真实描线适配实验

本轮实际完成2400步GPU训练（RTX 5060 Laptop / {train['torch_version']}）：1800步程序监督＋600步真实/程序混合训练。保留第{chosen}步权重。**六张校准图的既定提升门槛：{passed}；未替换正式绘画流程。**

## 这次方法改变了什么

上一轮只能对旧候选调整位置、笔宽和灰度。本轮新增独立于旧候选的RGB图像起点：多尺度墨线脊线 → 起点/方向 → CNN直接预测20个有顺序的后续坐标及继续概率 → 每次推进6点并重新观察图像。每段前视60px，固定弧长步3px，最多双向各12次推进。它能提出旧模型没有的路径；不再只是给旧线换分组或颜色。

近平行与交叉笔画在程序样例中有独立曲线身份，用有序坐标监督避免只学墨带两侧边缘。真实参考仍有局部不精确、分笔不确定和来源相关的问题。本轮没有声称恢复了画家的历史笔顺，也没有完成皴法类别分类。

当前对照采取保留旧候选、增加新路径的方式，防止尚未可靠的新生成器损坏已有轮廓。因此它**不能修正旧路径内部的错误连接、删除旧环路或自动把旧碎段合成正确一笔**；新路径自身可以连续跨过局部墨带，但只把这件事作为候选能力检验，不能称为整图拓扑已解决。

## 训练与留出诊断

- 真实训练窗口{data['real_train_windows']}个，来自23个旧图像文件；真实留出{data['real_validation_windows']}窗口，来自U03、stage_035、replay_R02三个文件。窗口不是独立画作数量，没有新下载的独立画作。
- 程序训练10000窗口、程序留出1000窗口，包含近邻平行、交叉、变宽、淡墨、小缺口和杂点。它们是数学合成曲线，不是数千幅中国画。
- 真实图上的既有人工与助手参考只监督已知的有序线坐标，不能统称为专家真值；未标区域没有当背景，全部真实窗口的停笔监督掩膜为0，避免把不完整描线末端教成真实停笔。
- 给定参考起点和方向的真实留出诊断，程序监督模型有序点平均误差{ds['procedural']['real_seed_conditioned']['mean_ordered_point_error_px']:.2f}px，真实适配后{ds['adapted']['real_seed_conditioned']['mean_ordered_point_error_px']:.2f}px；3px内比例{ds['procedural']['real_seed_conditioned']['fraction_within_3px']:.2%}→{ds['adapted']['real_seed_conditioned']['fraction_within_3px']:.2%}。这是已知起点的局部诊断，不是自动整图提取准确率。真实图停笔准确率没有真值。
- 程序留出误差{ds['procedural']['procedural']['mean_ordered_point_error_px']:.2f}px→{ds['adapted']['procedural']['mean_ordered_point_error_px']:.2f}px，也不能拿它证明真实皴线准确。

## 实际整图推理对照

先在6张校准图从预先冻结的3组墨迹支撑/新增覆盖阈值选择同一配置，再在10张既有开发图对照；没有按开发图逐张调参数。推理函数只打开RGB原图和权重，参考与有效/背景掩膜只用于之后评估。

{table}

真实适配组相对旧候选，参考点覆盖变化{100*(s['point_coverage_3px']-b['point_coverage_3px']):+.2f}个百分点，单路径覆盖变化{100*(s['mean_best_single_path_coverage']-b['mean_best_single_path_coverage']):+.2f}个百分点；10张共加入{numadded}条候选。参考仅覆盖部分可见线条，覆盖率不等于完整准确率；重叠与误连是几何代理，不是完整专家判定。增加线可能提高覆盖，也可能提高重复/错误描画，必须同时看表和叠加图。

## 限制与结论使用范围

模型仍以局部前视滚动生成，长距离会累积方向误差；宽墨交汇、急转角、淡墨消失处的笔画身份并不唯一。起点来自图像Hessian脊线，纸纹仍可能成为候选；墨迹支撑筛选也不能保证是正确皴线。已知起点诊断改善不保证自动起点、自动停笔和全图分笔一起改善。不能用本轮训练损失下降宣称取得整幅复杂山水突破。

本轮不重新训练笔宽/墨色模块。v31表观宽度/灰度依然是图像参数，不能直接当成机械臂下压、墨液浓度或干湿；没有实际落笔对照。所有轨迹physical_z=null、ready_for_robot=false、actually_drawn=false。未连接、使能或移动机械臂。

## 文件

- RESULT_COMPARISON.jpg：四张固定代表图，原图 / 旧候选 / 网络新增线 / 合并线。
- INTERACTIVE_COMPARISON.html：十张图离线切换、放大、红色新增线叠加及各方法数值。
- comparisons/：全部16张逐项对照，包含几何方法与程序监督的消融。
- trajectories/adapted/：实际图像坐标轨迹和新增线；proposals/包含筛选前候选。
- curve_model.pt、procedural_only.pt：真实适配权重与程序监督对照。
- protocol.json、calibration.json、development_evaluation.json、seed_conditioned_diagnostic.json：冻结方案、选择与评估。
- verification.json：数据/原模型/人工标注完整性检查。工程检查通过不能等同提取质量通过。

所有新数据、缓存、权重、图像和页面在E盘；没有新安装依赖。首次启动的通道排列错误在正式训练前修正，CUDA前向与反向检查已保存cuda_smoke.json；正式训练统计不包含失败启动。
'''
    (ROOT/'REPORT.md').write_text(text,encoding='utf-8')
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['v32_stroke_model.py','v32_experiment.py','v32_report.py','rock_v32.ps1']:shutil.copy2(Path(__file__).parent/name,snap/name)
    jsave(ROOT/'artifact_manifest.json',dict(files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)) for p in [ROOT/'curve_model.pt',ROOT/'procedural_only.pt',ROOT/'REPORT.md',ROOT/'RESULT_COMPARISON.jpg',ROOT/'INTERACTIVE_COMPARISON.html']]))
    print('V32_FINISHED',json.dumps(dict(summary=dev['summary'],calibration_passed=passed,default_replaced=False)),flush=True)
