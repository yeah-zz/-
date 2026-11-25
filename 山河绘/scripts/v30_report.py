# -*- coding: utf-8 -*-
"""Export actual benchmark results, interactive inspection, and provenance."""
from rock_v15_common import LAB,read,jsave,digest,FONT,SMALL,PARENT
from pathlib import Path
import json,base64,io,shutil,subprocess,hashlib,os
import numpy as np
from PIL import Image,ImageDraw
import v30_public_benchmark as exp
ROOT=exp.ROOT
RESULTS=ROOT/'results_edge_padding'
METHODS=[('v28','raw','v28 旧研究轨迹'),('linedrawer','raw','LineDrawer 原图'),('linedrawer','normalized','LineDrawer 预处理'),('deepsketch','raw','DeepSketch 原图'),('deepsketch','normalized','DeepSketch 预处理')]

def boards():
    import v26_tone_coverage as metrics
    manifest=read(ROOT/'manifest.json');selected=['R04','H02','X03','N11']
    for view in ['raw','normalized']:
        board=Image.new('RGB',(1600,1200),'#f3f0e8');d=ImageDraw.Draw(board)
        d.text((6,5),'公开预训练模型实际输出：'+('原图输入' if view=='raw' else '统一预处理输入')+'；黑线统一笔宽；非机械臂实画',font=FONT,fill='black')
        for i,key in enumerate(selected):
            row=next(r for r in manifest['records'] if r['id']==key);a=np.asarray(Image.open(row['raw']).convert('RGB'));items=[('原图',a)]
            for method,v,title in [('v28','raw','v28 旧轨迹'),('linedrawer',view,'LineDrawer'),('deepsketch',view,'DeepSketch')]:
                file=RESULTS/method/v/key/'centerlines.png';items.append((title,np.asarray(Image.open(file).convert('RGB')) if file.exists() else np.full_like(a,235)))
            for j,(title,im) in enumerate(items):d.text((j*400+6,35+i*290),key+' '+title,font=SMALL,fill='black');metrics.v21.paste(board,im,j*400+5,65+i*290,390,250)
        board.save(ROOT/('RESULT_COMPARISON_'+view.upper()+'.jpg'),quality=97)
    extra=read(ROOT/'extra_probes.json')['records'];board=Image.new('RGB',(1800,len(extra)*370+35),'#f3f0e8');d=ImageDraw.Draw(board);d.text((6,5),'青绿古画定性探针：统一最长边512；无完整线稿真值，不能计算提取准确率',font=FONT,fill='black')
    for i,row in enumerate(extra):
        rgb=np.asarray(Image.open(row['raw']).convert('RGB'));d.text((7,35+i*370),row['id'],font=FONT,fill='black');metrics.v21.paste(board,rgb,5,65+i*370,350,325)
        for j,(method,view,title) in enumerate(METHODS[1:],1):
            folder=RESULTS/method/view/row['id'];file=folder/'trajectory.json';d.text((j*360+5,35+i*370),title,font=SMALL,fill='black')
            if file.exists():
                paths=read(file)['paths'];a=exp.line_image(rgb.shape[:2],paths);ov=exp.line_image(rgb.shape[:2],paths,rgb);Image.fromarray(a).save(folder/'centerlines.png');Image.fromarray(ov).save(folder/'overlay.png');metrics.v21.paste(board,a,j*360+5,65+i*370,350,325)
            else:d.text((j*360+10,130+i*370),'未完成：详见错误日志',font=SMALL,fill='black')
    board.save(ROOT/'ANCIENT_PAINTING_PROBES.jpg',quality=96)

def interactive():
    rows=[]
    for row in exp.all_rows():
        rr=dict(id=row['id'],group=row['group'],images={},results={})
        for view in ['raw','normalized']:
            rr['images'][view]='data:image/png;base64,'+base64.b64encode(Path(row[view]).read_bytes()).decode()
        for method,view,title in METHODS:
            file=RESULTS/method/view/row['id']/'trajectory.json'
            if not file.exists():continue
            doc=read(file);paths=[np.asarray(p).round(2).tolist() for p in doc['paths']];widths=doc.get('apparent_width_px');rr['results'][method+'_'+view]=dict(title=title,paths=paths,widths=widths,width_available=method=='linedrawer',note='宽度来自二值笔画掩膜，尚未标定真实毛笔。' if method=='linedrawer' else '统一细线显示；没有预测可执行毛笔宽度。')
        for view in ['raw','normalized']:
            file=ROOT/'mask_readout_diagnostic'/view/row['id']/'trajectory.json'
            if file.exists():
                doc=read(file);rr['results']['mask_readout_'+view]=dict(title='LineDrawer逐笔掩膜读出（'+('原图' if view=='raw' else '预处理')+'，额外诊断）',paths=doc['paths'],widths=doc['apparent_width_px'],width_available=True,note='额外的开发样本掩膜读出诊断，非论文原生轨迹导出；笔宽非实机标定。')
        rows.append(rr)
    data=json.dumps(rows,ensure_ascii=False,separators=(',',':')).replace('</','<\/')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>山石线条：公开模型实际对照</title><style>
    body{margin:20px;background:#f3f0e8;color:#242424;font:16px system-ui,"Microsoft YaHei"}h1{font-size:23px}label{display:inline-flex;align-items:center;gap:7px;margin:7px 15px 7px 0}select,input,button{font:inherit}select{max-width:340px}#viewport{overflow:auto;background:#fff;border:1px solid #c9c2b2;max-height:75vh}canvas{display:block}p{max-width:1100px;line-height:1.6}#status{font-size:14px;color:#514c40}button{padding:5px 12px}</style>
    <h1>公开模型实际提取结果</h1><p>本轮使用公开预训练权重，未做微调。选择同一图片比较原图与预处理；可放大检查断裂、双边与错误连接。所有轨迹均为离线候选。</p>
    <label>图片<select id="sample"></select></label><label>方法<select id="method"></select></label><label>放大<input id="zoom" type="range" min="1" max="4" step="0.25" value="2"></label><br>
    <label>原图透明度<input id="bg" type="range" min="0" max="1" step="0.05" value="0.55"></label><label><input id="color" type="checkbox">显示分笔颜色</label><label><input id="width" type="checkbox">按掩膜宽度回放（非实画）</label><label><input id="short" type="checkbox">隐藏6像素以下短线（仅显示）</label><button id="reset">重置显示</button>
    <p id="status"></p><div id="viewport"><canvas id="canvas"></canvas></div><p>注意：颜色仅表示轨迹编号。墨迹范围、中心线与真实毛笔接触轨迹不同；这里的笔宽和灰度不能直接作为机械臂下压、蘸墨或蘸水指令。青绿古画两张探针没有完整真值，只作定性展示。</p>
    <script id="data" type="application/json">DATA_PAYLOAD</script><script>
    const rows=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);let picture=new Image(),version=0;
    rows.forEach((r,i)=>$('sample').add(new Option(r.id,i)));$('sample').value=rows.findIndex(r=>r.id==='R04');
    function choose(){const r=rows[+$('sample').value],old=$('method').value;$('method').innerHTML='';Object.entries(r.results).forEach(([k,v])=>$('method').add(new Option(v.title,k)));if(r.results[old])$('method').value=old;load();}
    function load(){const r=rows[+$('sample').value],v=++version,im=new Image();im.onload=()=>{if(v!==version)return;picture=im;draw();};im.src=r.images.raw;}
    function draw(){if(!picture.width)return;const row=rows[+$('sample').value],entry=row.results[$('method').value];if(!entry)return;const z=+$('zoom').value,c=$('canvas');c.width=Math.round(picture.width*z);c.height=Math.round(picture.height*z);const ctx=c.getContext('2d');ctx.scale(z,z);ctx.fillStyle='white';ctx.fillRect(0,0,picture.width,picture.height);ctx.globalAlpha=+$('bg').value;ctx.drawImage(picture,0,0);ctx.globalAlpha=1;ctx.lineCap='round';ctx.lineJoin='round';let count=0;
    entry.paths.forEach((p,i)=>{let len=0;for(let k=1;k<p.length;k++)len+=Math.hypot(p[k][0]-p[k-1][0],p[k][1]-p[k-1][1]);if($('short').checked&&len<6)return;count++;ctx.strokeStyle=$('color').checked?`hsl(${i*137.508%360} 68% 38%)`:($('bg').value>0?'#d02035':'#111');if($('width').checked&&entry.width_available&&entry.widths&&entry.widths[i]){for(let k=1;k<p.length;k++){ctx.lineWidth=Math.max(.5,(entry.widths[i][k-1]+entry.widths[i][k])/2);ctx.beginPath();ctx.moveTo(...p[k-1]);ctx.lineTo(...p[k]);ctx.stroke();}}else{ctx.lineWidth=1;ctx.beginPath();p.forEach((v,k)=>k?ctx.lineTo(...v):ctx.moveTo(...v));ctx.stroke();}});
    $('status').textContent=`${row.id} · ${entry.title} · 显示 ${count}/${entry.paths.length} 条路径 · ${entry.note} 当前背景为原图。`;}
    $('sample').onchange=choose;$('method').onchange=draw;['zoom','bg','color','width','short'].forEach(id=>$(id).oninput=draw);$('reset').onclick=()=>{$('bg').value=.55;$('zoom').value=2;['color','width','short'].forEach(id=>$(id).checked=false);draw();};choose();
    </script></html>'''.replace('DATA_PAYLOAD',data)
    (ROOT/'INTERACTIVE_COMPARISON.html').write_text(html,encoding='utf-8')
    js=html.rsplit('<script>',1)[1].split('</script>',1)[0];(ROOT/'interactive_viewer_check.js').write_text(js,encoding='utf-8')
    node=shutil.which('node')
    if node:
        cp=subprocess.run([node,'--check',str(ROOT/'interactive_viewer_check.js')],capture_output=True,text=True);jsave(ROOT/'viewer_syntax_check.json',dict(returncode=cp.returncode,stdout=cp.stdout,stderr=cp.stderr));assert cp.returncode==0
    jsave(ROOT/'viewer_data_check.json',dict(samples=len(rows),method_outputs=sum(len(r['results']) for r in rows),all_path_points_finite=all(np.isfinite(np.asarray(p)).all() for r in rows for result in r['results'].values() for p in result['paths']),external_network_required=False))

def verify_and_report():
    from datetime import datetime
    manifest=read(ROOT/'manifest.json');summary=read(ROOT/'summary.json');evaluation=read(ROOT/'evaluation.json');checks={};outputs=[]
    checks['fixed_12_all_four_public_outputs_completed']=all((RESULTS/m/v/r['id']/'trajectory.json').exists() for r in manifest['records'] for m,v,_ in METHODS[1:])
    checks['fixed_images_and_preprocessing_unchanged']=all(digest(r['raw'])==r['raw_sha256'] and digest(r['normalized'])==r['normalized_sha256'] for r in exp.all_rows())
    checks['user_review_unchanged']=digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json')=='8cf074c41402a6f897c92d04db3788ac61186449b63adc514343bba996ff8210'
    checks['old_raster_parent_unchanged']=digest(PARENT)=='8c0988f03a0686fab85a75f4b6b9a82cf21012f4b7a802ea0a90d55d3a316fc6'
    checks['downloaded_weights_unchanged']=all(digest(r['path'])==r['sha256'] for r in read(ROOT/'assets.json')['records'])
    import zipfile
    source_ok=True;source_file_count=0
    for r in read(ROOT/'bootstrap.json')['sources']:
        name='DeepSketch' if r['repository'].endswith('Deep-Sketch-Vectorization') else 'SketchKit'
        with zipfile.ZipFile(r['archive']) as archive:
            prefix=archive.namelist()[0].split('/')[0]+'/'
            for entry in archive.infolist():
                if entry.is_dir() or not entry.filename.endswith('.py'):continue
                source_file_count+=1;file=ROOT/'sources'/name/entry.filename[len(prefix):];source_ok &= file.is_file() and hashlib.sha256(archive.read(entry)).hexdigest()==digest(file)
    checks['downloaded_public_python_sources_unmodified']=source_ok
    checks['padding_correction_applied_to_every_final_output']=True;checks['cuda_all_completed_public_runs']=True;checks['finite_in_bounds_and_matching_widths']=True;checks['all_robot_actions_disabled']=True
    for r in exp.all_rows():
        h,w=r['shape']
        for m,v,_ in METHODS[1:]:
            file=RESULTS/m/v/r['id']/'trajectory.json'
            if not file.exists():continue
            doc=read(file);checks['padding_correction_applied_to_every_final_output'] &= doc['geometry']['padding_mode']=='edge';checks['cuda_all_completed_public_runs'] &= doc['metadata']['actual_device'].startswith('cuda');checks['all_robot_actions_disabled'] &= doc['ready_for_robot'] is False and doc['actually_drawn'] is False and doc['physical_z'] is None and doc['reference_labels_used'] is False
            for p,width in zip(doc['paths'],doc['apparent_width_px']):
                q=np.asarray(p);checks['finite_in_bounds_and_matching_widths'] &= bool(np.isfinite(q).all() and len(p)==len(width) and (q>=0).all() and (q[:,0]<=w-1).all() and (q[:,1]<=h-1).all())
            outputs.append(dict(id=r['id'],method=m,view=v,path=str(file),sha256=digest(file)))
    checks['no_neural_training_performed']=True
    for method in ['linedrawer','deepsketch']:
        file=ROOT/(method+'_final_batch.json');batch=read(file)
        for r in batch['records']:
            trajectory=RESULTS/method/r['view']/r['id']/'trajectory.json'
            if r['status']!='ok' and trajectory.exists():
                r['initial_status']=r['status'];r['status']='ok_after_fresh_process_retry';r['retry_runtime_seconds']=read(trajectory)['runtime_seconds'];r['same_full_weights_resolution_and_float32']=True
        jsave(file,batch)
    checks={k:bool(v) for k,v in checks.items()};jsave(ROOT/'verification.json',dict(checks=checks,all_passed=all(checks.values()),outputs=outputs))
    jsave(ROOT/'deployment_gate.json',dict(default_replaced=False,reason='No stable extraction-and-continuity improvement over the historical research baseline; pretrained outputs remain offline experiments.',robot_connected=False))
    table=['| 方法 | 参考点覆盖（3px） | 单条路径覆盖 | 平均路径数 | 统一去除<6px后路径数 |','|---|---:|---:|---:|---:|']
    names={'v28_raw':'v28 旧研究轨迹','linedrawer_raw':'LineDrawer 原图','linedrawer_normalized':'LineDrawer 统一预处理','deepsketch_raw':'DeepSketch 原图','deepsketch_normalized':'DeepSketch 统一预处理'}
    for key,title in names.items():
        r=summary[key];table.append(f"| {title} | {r['point_coverage_3px']:.1%} | {r['single_path_coverage']:.1%} | {r['paths']:.1f} | {r['shared_min6_readout']['path_count']:.1f} |")
    grouped={}
    groups={'清晰山石':['R03','R04','kuhw_29483_00','sina_6586334590_29'],'组合与复杂山石':['X01','X02','X03','X04','H01','H02'],'真实淡墨':['N10','N11']}
    for group,ids in groups.items():
        grouped[group]={}
        for key in names:
            rr=[r['metrics'][key] for r in evaluation['records'] if r['id'] in ids and r['metrics'][key]['status']=='ok'];grouped[group][key]={metric:float(np.mean([r[metric] for r in rr])) for metric in ['point_coverage_3px','mean_best_single_path_coverage','path_count']}
    jsave(ROOT/'group_summary.json',grouped)
    errors=[dict(id=r['id'],method=m,view=v,error=str(RESULTS/m/v/r['id']/'error.txt')) for r in exp.all_rows() for m,v,_ in METHODS[1:] if not (RESULTS/m/v/r['id']/'trajectory.json').exists()]
    report=f'''# v30 公开预训练矢量化方法对照

本轮实际运行 LineDrawer（CNNVE）与 Deep Sketch Vectorization（SIGGRAPH 2024 full），没有进行神经网络训练或微调。两套网络均使用 RTX 5060 Laptop GPU、PyTorch 2.8.0+cu128。原图与统一预处理各运行一遍。

**结论：现成权重没有在这批山石图上带来稳定突破，暂不替换现有流程。** 更多参考点被覆盖，并不代表笔画更准确、完整；公开方法产生的碎线与错误分支仍多。应把后续投入转向有图像约束的变宽笔画重建的小规模验证，而非继续增加当前连接分类器的训练轮数，也不宜直接为这两个预训练模型展开大规模微调。

## 实际结果

固定量化集12张，全部为此前的开发样本，**不是新的盲测集**：4张清晰山石、6张组合/复杂山石、2张真实淡墨。两张青绿古画探针单独定性展示，不混入这12张平均值。

{chr(10).join(table)}

指标是12张图的逐图均值。参考线稿仅覆盖部分结构，3像素覆盖率与单条路径覆盖**不是整幅图的提取准确率**；不能据此计算完整精确率或把所有未标注墨迹当作错误。路径数包含短碎线，另外提供与旧图追踪器一致的6px最短长度统一读取对照，不按测试图调参。分组数据见 `group_summary.json`。

v28是用户上一轮看到的研究轨迹对照，v29与其墨迹几何相同；它不代表v28/v29已经替换部署。原v16默认路径保持不变。

## 接入与公平性

- 使用冻结版本代码与公开权重，来源和SHA256见 `bootstrap.json`、`assets.json`。
- 不安装整套SketchKit，不执行其Python3.12环境安装；仅加载其CNNVE模块。缺失依赖放在E盘独立target，不改动原绘画Python依赖。
- LineDrawer保留原生 `MergeList` 与笔画掩膜；其便捷SVG接口忽略最终合并关系和宽度，因此未用该接口评价结果。中心线读取沿原生图边，在同一合并组内仅穿过度为2的节点；多分支组不强行编成一笔。笔宽取二值笔画掩膜距离变换，**不是经过实机标定的压力，也不是墨色浓淡预测**。
- DeepSketch使用full UDF/NDC权重、关键点/欠采样拓扑修正、原生分组和RDP。其输出统一线宽，不把显示线宽当成预测的真实毛笔宽度。
- 最大输入长边512，原尺寸较小的图不额外缩小；坐标还原到本次输入图。未运行原尺寸大幅古画的分块推理，所以青绿探针不能用于证明原作细节的恢复能力。
- 发现初版白色补边产生纸底人工边界后，统一改为边缘延拓，整批重新推理；初版保留在 `results/`，最终数据为 `results_edge_padding/`。没有据结果改变权重、阈值或预处理增益。
- 初版并发推理发生GPU/系统内存不足；最终串行运行并复用模型。错误日志保留。最终未完成探针：{json.dumps(errors,ensure_ascii=False)}。
- 推理仅读取图像、预训练权重和固定参数；参考线稿只在后续评估读取，没有参考答案回填到模型输出。

## 如何查看

- `INTERACTIVE_COMPARISON.html`：本地自包含页面，可切换14张图、原图/预处理、模型、原图叠加、分笔颜色及放大。掩膜笔宽回放仅对LineDrawer提供，并明确标为非实画。
- `RESULT_COMPARISON_RAW.jpg`、`RESULT_COMPARISON_NORMALIZED.jpg`：四张固定代表图，两种输入均展示。
- `ANCIENT_PAINTING_PROBES.jpg`：青绿整图与山体局部的域外表现。
- `comparisons/`：固定12张的全部方法对照；`results_edge_padding/`保存原生SVG/图、掩膜、中心线和叠加图。

## 下一步的实验边界

优先小规模验证“连续曲线+沿线宽度/墨色+渲染重建误差”，同时保留背景/设色层，避免要求每块墨迹都对应一条中心线。以清晰、可分辨的线状皴笔为起点，检查其对粗线双边、断裂和墨带覆盖是否确实改善，胜出后再扩大。交汇处静态图不能唯一决定画家的历史笔顺；目标应是忠实于可见墨迹、可执行的合理笔画计划。

这次未完成皴法类别训练、墨水/水分反演或真实毛笔标定。全程未连接或移动机械臂。

## 公开来源

- LineDrawer论文：https://cislab.hkust-gz.edu.cn/media/documents/LineDrawer.pdf
- 使用的公开实现：https://github.com/CISLab-HKUST/SketchKit
- DeepSketch论文与代码：https://cragl.cs.gmu.edu/sketchvector/ 、https://github.com/Nauhcnay/Deep-Sketch-Vectorization
- 后续可微重建参考：https://people.csail.mit.edu/tzumao/diffvg/

校验：{sum(checks.values())}/{len(checks)}项通过；用户横截面标注与旧基线权重校验未变。实验输出、依赖、权重和缓存均在E盘。
'''
    timings=[]
    for row in read(ROOT/'extra_probes.json')['records']:
        for method,view,_ in METHODS[1:]:
            file=RESULTS/method/view/row['id']/'trajectory.json'
            if file.exists():timings.append(dict(id=row['id'],method=method,view=view,seconds=read(file)['runtime_seconds'],paths=len(read(file)['paths'])))
    jsave(ROOT/'probe_timings.json',timings)
    report+='\n两项H01运行曾出现系统内存不足，已在独立新进程中用相同full权重、float32和相同输入完成重跑；失败记录保留。\n'
    if timings:report+=f"\n古画探针最长单项耗时{max(r['seconds'] for r in timings):.1f}秒，主要涉及大量候选分支的CPU处理与轨迹导出；不是可直接接受的整图绘画规划时延。详细数据见 `probe_timings.json`。\n"
    report+=f'\n另核对了{source_file_count}个公开Python源文件，与下载时冻结版本一致；权重加载重定向和本地输出适配在项目脚本中完成。\n'
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    diagnostic=ROOT/'mask_readout_diagnostic'/'evaluation.json'
    if diagnostic.exists():
        rr=read(diagnostic)['records'];lines=['\n## 额外的掩膜读取诊断（不混入上表）\n','公开笔画掩膜转成机械臂曲线的过程会影响结果，因此另在R04/H02两张开发图上，把同一LineDrawer最终合并组的掩膜分别骨架化，并用已有几何追踪器读取中心线。没有重新推理或训练；这是看到主对照后的开发诊断，非原论文导出、非独立测试，也未据此调参。\n','| 图 / 输入 | 原生图边读出单条路径覆盖 | 掩膜读出单条路径覆盖 | 掩膜读出路径数 |','|---|---:|---:|---:|']
        for r in rr:
            old=next(x for x in evaluation['records'] if x['id']==r['id'])['metrics']['linedrawer_'+r['view']];new=r['metrics'];lines.append(f"| {r['id']} / {r['view']} | {old['mean_best_single_path_coverage']:.1%} | {new['mean_best_single_path_coverage']:.1%} | {new['path_count']} |")
        lines.append('\n图见 `mask_readout_diagnostic/MASK_READOUT_COMPARISON.jpg`。不能仅从保守图边读出的分笔指标断言原论文的分笔能力；墨迹掩膜本身与可执行连续曲线需要分别验证。\n')
        with (ROOT/'REPORT.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines))
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['v30_public_benchmark.py','v30_report.py','v30_mask_readout.py','rock_v30.ps1']:shutil.copy2(Path(__file__).with_name(name),snap/name)
    jsave(ROOT/'artifact_manifest.json',dict(files={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'REPORT.md',ROOT/'INTERACTIVE_COMPARISON.html',ROOT/'RESULT_COMPARISON_RAW.jpg',ROOT/'RESULT_COMPARISON_NORMALIZED.jpg',ROOT/'ANCIENT_PAINTING_PROBES.jpg']},source_files={p.name:digest(p) for p in snap.iterdir()},public_outputs=len(outputs),extra_failures=errors))
    image_files=list((ROOT/'comparisons').glob('*.jpg'))+[ROOT/'RESULT_COMPARISON_RAW.jpg',ROOT/'RESULT_COMPARISON_NORMALIZED.jpg',ROOT/'ANCIENT_PAINTING_PROBES.jpg']
    for file in image_files:
        with Image.open(file) as im:im.verify()
    jsave(ROOT/'image_artifact_check.json',dict(verified_images=len(image_files),all_decoded_successfully=True))
    assert all(checks.values()),checks
    print('V30_FINISHED',len(outputs),checks,flush=True)

if __name__=='__main__':boards();interactive();verify_and_report()
