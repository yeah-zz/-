# -*- coding: utf-8 -*-
"""Publish the conservative post-hoc comparison without hiding initial failures."""
from v32_report import *
import subprocess,ast

def main():
    root=ROOT/'image_verified';dev=read(root/'development_evaluation.json');cal=read(root/'calibration_evaluation.json');gate=read(root/'gate.json');raw=read(ROOT/'development_evaluation.json');file=ROOT/'INTERACTIVE_COMPARISON.html';content=file.read_text(encoding='utf-8');start=content.index('const rows=')+len('const rows=');items,n=json.JSONDecoder().raw_decode(content[start:])
    for r in items:
        path=root/'adapted'/r['id'];r['images'].update(guarded=image_url(path/'combined.png'),guarded_new=image_url(path/'new.png'),guarded_overlay=image_url(path/'overlay.png'));row=next(a for a in dev['records'] if a['id']==r['id']);r['metrics']['guarded']=row['metrics']['adapted'];r['metrics']['guarded_classical']=row['metrics']['classical'];r['additions']['guarded']=row['additions']['adapted']
    content=content[:start]+json.dumps(items,ensure_ascii=False)+content[start+n:]
    if '<option value="guarded">' not in content:
        content=content.replace('<select id="view">','<select id="view"><option value="guarded">核验后：旧候选＋新增线</option><option value="guarded_new">核验后：只看新增线</option><option value="guarded_overlay">核验后：新增红线叠加</option>')
        content=content.replace("adapted:'真实适配网络补线'","adapted:'未核验网络补线',guarded:'核验后网络补线',guarded_classical:'核验后几何补线'")
        content=content.replace("r.additions.adapted+'；真实笔顺'","r.additions.guarded+'；真实笔顺'")
        content=content.replace('网络新增路径：','核验后新增路径：').replace('v32 图像起点与有序曲线：实际离线输出','v32 逐点图像核验后的补线与原始输出对照')
    content=content.replace('r.additions.adapted','r.additions.guarded')
    file.write_text(content,encoding='utf-8');script=content.split('<script>',1)[1].split('</script>',1)[0];(ROOT/'viewer_script.js').write_text(script,encoding='utf-8');p=subprocess.run(['node','--check',str(ROOT/'viewer_script.js')],capture_output=True,text=True,encoding='utf-8');jsave(ROOT/'viewer_syntax_check.json',dict(passed=p.returncode==0,stderr=p.stderr,includes_post_hoc_guarded_comparison=True))
    if p.returncode:raise RuntimeError(p.stderr)
    # Correct a display-only heading; raw numeric trajectories remain unchanged.
    path=ROOT/'RESULT_COMPARISON.jpg';im=Image.open(path).convert('RGB');d=ImageDraw.Draw(im);d.rectangle((0,0,1600,29),fill='#f3f0e8');d.text((7,5),'v32 未逐点核验的实际失败输出：原图 / 旧候选 / 网络新增线 / 合并线（离线）',font=FONT,fill='black');im.save(path,quality=97)
    table='| 同一批10张开发图 | 参考点覆盖 / 3px | 单路径覆盖 | 平均路径数 | 重叠代理 | 误连代理 | 标注背景内路径密度 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for key,label in [('initial','旧候选'),('classical','图像核验后几何补线'),('adapted','图像核验后网络补线')]:
        s=dev['summary'][key];table+=f"| {label} | {s['point_coverage_3px']:.2%} | {s['mean_best_single_path_coverage']:.2%} | {s['path_count']:.1f} | {s['duplicate_point_fraction']:.2%} | {s['mixed_reference_paths_flagged']:.2f} | {s['paper_path_density']:.3%} |\n"
    b=dev['summary']['initial'];a=dev['summary']['adapted'];v=raw['summary']['adapted']
    report=f'''# v32 补线训练与逐点图像核验：最终对照

实际完成2400步RTX 5060 Laptop GPU训练；模型直接预测有序后续曲线，再滚动生成路径。新数据、权重、缓存、预览均在E盘。没有连接机械臂，没有替换正式流程。

**本轮没有取得可靠的完整笔画提取突破。** 网络可以提出旧模型没有的线，但原始曲线有跨越空白和不正确长弧的问题；逐点图像核验去掉了这类明显悬空路径，却增加了短碎线。不能只根据覆盖率增加宣称提取更准确。同条件的简单几何追踪并未比神经网络差。

## 修正后，十张旧开发图对照

{table}

核验后网络组相对旧候选：参考点覆盖增加{100*(a['point_coverage_3px']-b['point_coverage_3px']):.2f}个百分点，单路径覆盖增加{100*(a['mean_best_single_path_coverage']-b['mean_best_single_path_coverage']):.2f}个百分点；重叠代理降低，但平均路径数量由{b['path_count']:.1f}增至{a['path_count']:.1f}，仍存在过度分笔，标注背景中的误检也增加。没有完整专家标注，表中误连与重复均为几何代理；部分参考覆盖率不是整幅画准确率。

未逐点核验的网络组覆盖{v['point_coverage_3px']:.2%}、单路径覆盖{v['mean_best_single_path_coverage']:.2%}，但重叠{v['duplicate_point_fraction']:.2%}、误连代理{v['mixed_reference_paths_flagged']:.2f}，且肉眼可见错误长弧。该失败输出全部保留，没有作为推荐结果。

## 图像核验做了什么

对每条生成曲线查找原RGB图中的多尺度墨线脊线；只有距离≤2.5px、方向一致度≥0.75且有局部墨迹支撑的点才保留。截去与已有轨迹距离≤4.2px的部分；连续保留至少13个采样点，再输出独立路径。它更保守，但会把一条候选截为多段，也可能删掉真实近平行笔画，不能称为分笔问题已经解决。纸张拍摄边界、纹理和字迹仍可能被识别为墨线。

这是在检查原始开发输出后增加的修正。参数没有按开发图逐张选择，但开发图已经被观察，结果属于事后诊断，**不是新的独立盲测**。六张校准图核验后的既定门槛：{gate['passed']}；校准单路径覆盖仅{cal['summary']['initial']['mean_best_single_path_coverage']:.2%}→{cal['summary']['adapted']['mean_best_single_path_coverage']:.2%}，仍未达到预定0.5个百分点提升。没有放宽门槛或自动部署。

## 训练与验证范围

真实训练3680个窗口来自23个旧文件，留出480窗口来自3个旧文件；程序生成10000个训练窗口、1000个留出窗口。没有新增独立画作，不把窗口数当作画作数量。真实图参考含人工与助手描线，局部精度和笔画身份不都是专家真值；未知区域没有被标成背景，真实参考末端没有当成停笔负例。

本轮实际2400步，包含1800步程序监督与600步真实/程序混合；详见training_summary.json。已知参考起点的真实留出曲线点误差12.90px→6.05px，只说明局部适配改善，不能代表自动整图准确率。

宽度/墨色没有在这轮重新训练；v31仍是图像域笔宽和表观灰度模块。没有笔压、墨液浓度、含水量或实机落笔验证，physical_z=null、ready_for_robot=false。

## 查看结果

- RESULT_COMPARISON_GUARDED.jpg：四张固定样例，原图 / 旧候选 / 核验后新增线 / 新增红线叠加。第三列仅为新增部分，不是完整线稿。
- INTERACTIVE_COMPARISON.html：十张图可放大；默认核验后合并线，可切换新增线、原始失败输出与红色叠加，并看各方法指标。
- image_verified/adapted/：核验后实际候选坐标。image_verified/*_evaluation.json为数值。
- REPORT.md：原始训练、消融与失败结果详情。curve_model.pt为所保留神经网络权重。
- verification.json、viewer_syntax_check.json：原标注/模型/旧候选完整性与页面语法检查。reference_free_interface_check/验证推理在禁止读取标注的审计钩子下可运行。

## 据此下一步应做什么

优先处理候选的整笔选择与替换：让确有连续图像支撑的新候选替代匹配的旧碎片，而不是不断叠加新短线；对近平行、交汇和淡墨歧义保留不确定判断。先控制新增背景误检、碎片数量与错误合并，再考虑扩大模型或训练步数。当前没有证据支持仅靠多训几轮就能取得突破。
'''
    (ROOT/'GUARDED_REPORT.md').write_text(report,encoding='utf-8')
    files=[ROOT/'RESULT_COMPARISON_GUARDED.jpg']+list((root/'adapted').glob('*/*.png'));verified=0
    for path in files:
        with Image.open(path) as im:im.verify()
        verified+=1
    verification=read(ROOT/'verification.json');verification.update(post_hoc_preview_images_verified=verified,post_hoc_quality_gate_passed=gate['passed'],post_hoc_development_is_not_independent=True);verification['checks']['guarded_previews_verified']=verified==49;verification['checks']['viewer_script_syntax_passed']=p.returncode==0;verification['all_engineering_checks_passed']=all(verification['checks'].values());jsave(ROOT/'verification.json',verification)
    for name in ['v32_stroke_model.py','v32_experiment.py','v32_report.py','v32_guard.py','v32_finalize_guard.py','rock_v32.ps1']:
        shutil.copy2(Path(__file__).parent/name,ROOT/'source_snapshot'/name)
        if name.endswith('.py'):ast.parse((Path(__file__).parent/name).read_text(encoding='utf-8'))
    paths=[ROOT/n for n in ['curve_model.pt','procedural_only.pt','REPORT.md','GUARDED_REPORT.md','RESULT_COMPARISON.jpg','RESULT_COMPARISON_GUARDED.jpg','INTERACTIVE_COMPARISON.html']];jsave(ROOT/'artifact_manifest.json',dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in paths]));print('V32_GUARDED_FINALIZED',dict(gate_passed=gate['passed'],engineering_passed=verification['all_engineering_checks_passed'],preview_images=verified),flush=True)

if __name__=='__main__':main()
