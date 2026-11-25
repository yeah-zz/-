# -*- coding: utf-8 -*-
"""Paper false positives, faint-line retention and nonexpert band diagnostics."""
import sys
sys.dont_write_bytecode=True
from paper_center_refinement_data import ROOT,PARENT,LOCAL,LAB
from rock_v15_common import read,jsave,digest
from evaluate_joint_ink_structure import summarize_relation
from pathlib import Path
import numpy as np,cv2,shutil,json
from PIL import Image,ImageDraw,ImageFont

def main():
    corrected=ROOT/'corrected_renderer_stage3';results_root=corrected if (corrected/'evaluation_results.json').exists() else ROOT;prefix='corrected_renderer_stage3/' if results_root==corrected else ''
    result=read(results_root/'evaluation_results.json')['records'];a=read(ROOT/'training_summary.json');b=read(ROOT/'faint_preservation_stage2/training_summary.json');records=read(results_root/'manifest.json')['records'];actual={r['id']:r for r in records if r['kind']=='actual_refined'};stats={}
    for tag in ['before','after']:
        declared=[r for r in result if r['kind']=='declared'];positive=[r[tag]['center'] for r in declared if r[tag]['center']['f1'] is not None];blank=[r[tag]['center']['blank_fp_fraction'] for r in declared if r[tag]['center']['f1'] is None];paper=[r for r in result if r['kind']=='paper_stress'];actual_paper=[r for r in result if r['kind']=='actual' and r['id'].startswith('G05')]
        faint_count=sum(r[tag].get('faint_reference_pixels',0) for r in declared);faint_hits=sum(r[tag].get('faint_matched_pixels',0) for r in declared);pine=[r[tag]['center']['recall'] for r in declared if r['case']=='needles']
        def paper_fp(rows):return sum(r[tag]['paper_fp_pixels'] for r in rows)/max(1,sum(r[tag]['paper_pixels'] for r in rows))
        stats[tag]=dict(declared_nonblank_f1=float(np.mean([r['f1'] for r in positive])),declared_blank_fp_fraction=float(np.mean(blank)),faint_center_recall=faint_hits/max(1,faint_count),pine_center_recall=float(np.mean(pine)),source_separated_paper_stress_fp_fraction=paper_fp(paper),G05_reviewed_paper_fp_fraction=paper_fp(actual_paper),relation=summarize_relation([r[tag]['relation'] for r in declared]))
    before=stats['before'];after=stats['after'];gates=dict(fresh_blank_fp_below_0002=after['declared_blank_fp_fraction']<=.0002,faint_recall_drop_below_5pp=after['faint_center_recall']>=before['faint_center_recall']-.05,pine_recall_drop_below_5pp=after['pine_center_recall']>=before['pine_center_recall']-.05,reviewed_actual_paper_fp_reduced_by_half=after['G05_reviewed_paper_fp_fraction']<=before['G05_reviewed_paper_fp_fraction']*.5)
    coarse={}
    for ident in ['R04','N11']:
        row=next(r for r in result if r['id']==ident);coarse[ident]={tag:row[tag] for tag in ['before','after']}
    summary=dict(executed_updates=a['completed_updates']+b['completed_updates'],stage1_retained_step=a['retained_step'],stage2_retained_step=b['retained_step'],selected_new_updates=600+b['retained_step'] if b['retained_step'] else a['retained_step'],device=b['gpu'],metrics=stats,gates=gates,focused_gates_all_pass=all(gates.values()),coarse_partial_nonexpert_diagnostics=coarse,actual_full_image_accuracy=None,full_image_v4_default_unchanged=True,robot_connected=False,reference_kind='partial nonexpert cross sections; no expert full-image stroke gold',new_program_images=48,paper_stress_images=32,paper_stress_parent_group='G05, six correlated crops of one historical parent sheet',actual_visual_regression_images=32,actual_junction_stroke_identity='unknown, ink-only known regions',checkpoint=str(ROOT/'faint_preservation_stage2/refined.pt'))
    if results_root==corrected:
        c=read(corrected/'fit/training_summary.json');fallback=read(corrected/'training_summary.json');summary.update(executed_updates=summary['executed_updates']+c['completed_updates'],corrected_renderer_updates=c['completed_updates'],corrected_renderer_retained_step=c['retained_step'],selected_new_updates=800+c['retained_step'] if c['retained_step'] else fallback['fallback_new_updates'],checkpoint=str(corrected/'fit/refined.pt'),historical_sparse_fixture_metrics_invalid=True,test_renderer_version=2)
    jsave(ROOT/'summary.json',summary)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);cards=[]
    atlas=Image.new('RGB',(1280,((len(actual)+3)//4)*355),'#f3f1eb');ad=ImageDraw.Draw(atlas)
    for i,r in enumerate(actual.values()):
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));z=np.load(r['data']);out=rgb.copy();pm=z['paper_mask']>0;out[pm]=(out[pm]*.4+np.array([30,190,80])*.6).astype(np.uint8);out[(z['target'][0]>.3)&(z['weight'][0]>0)]=[220,25,50]
        x=i%4*320;y=i//4*355;ad.text((x+4,y+4),r['id']+' '+r['split'],font=font,fill='black');im=Image.fromarray(out);im.thumbnail((316,316));atlas.paste(im,(x,y+30))
    atlas.save(ROOT/'SUPERVISION_REVIEW.jpg',quality=95)
    for row in [r for r in result if r['kind']=='actual']:
        ident=row['id'];folder=results_root/'evaluation'/ident
        source=LOCAL/'baseline'/ident/'input.png' if ident.startswith('G') else Path(actual[ident]['image']);v4=LOCAL/'candidate'/ident/'lines.png' if ident.startswith('G') else LAB/'split_detail_structure_20261028/v4/real_regression'/ident/'lines.png'
        old=np.asarray(Image.open(folder/'before/joint/raw_center.png'))<128;new=np.asarray(Image.open(folder/'after/joint/raw_center.png'))<128;diff=np.full((*old.shape,3),255,np.uint8);diff[old&new]=[30,30,30];diff[old&~new]=[220,50,50];diff[new&~old]=[10,120,230];Image.fromarray(diff).save(folder/'changes.png')
        panels=[(source,'原图'),(v4,'既有 v4 对照'),(folder/'before/joint/raw_center.png','上轮联合模型'),(folder/'after/joint/raw_center.png','本轮专项修正'),(folder/'changes.png','红：删除；蓝：新增')]
        board=Image.new('RGB',(1800,410),'#f3f1eb');draw=ImageDraw.Draw(board)
        for j,(p,label) in enumerate(panels):
            draw.text((j*360+3,5),ident+' '+label,font=font,fill='black');im=Image.open(p).convert('RGB');im.thumbnail((356,356));board.paste(im,(j*360,38))
        board.save(folder/'comparison.jpg',quality=95)
        scope=actual[ident]['split'] if ident in actual else 'related reused artwork'
        cards.append(f'<section><h2>{ident} · {scope}</h2><a href="{prefix}evaluation/{ident}/comparison.jpg"><img src="{prefix}evaluation/{ident}/comparison.jpg"></a><p><a href="{prefix}evaluation/{ident}/after/joint/overlay.png">本轮轨迹叠加原图</a> · <a href="{prefix}evaluation/{ident}/after/joint/paths.json">像素路径与推理记录</a> · <a href="{prefix}evaluation/{ident}/changes.png">全部增删</a></p></section>')
    # Fixed example chosen by ID, not by an image score.
    ident='G05_01';folder=results_root/'evaluation'/ident;preview=Image.new('RGB',(1488,540),'#f3f1eb');draw=ImageDraw.Draw(preview)
    for j,(p,label) in enumerate([(LOCAL/'baseline'/ident/'input.png','原图：本轮训练外的旧回归图'),(folder/'before/joint/raw_center.png','上轮联合模型'),(folder/'after/joint/raw_center.png','本轮纸面与粗墨专项修正')]):
        draw.text((j*496+4,5),label,font=font,fill='black');preview.paste(Image.open(p).convert('RGB'),(j*496,36))
    preview.save(ROOT/'ACTUAL_PREVIEW.jpg',quality=95)
    # Coarse reference is partial and is shown explicitly beside both outputs.
    for ident in ['R04','N11']:
        folder=results_root/'evaluation'/ident;canvas=Image.new('RGB',(1440,410),'#f3f1eb');draw=ImageDraw.Draw(canvas)
        for j,(p,label) in enumerate([(Path(actual[ident]['image']),'原图'),(ROOT/(ident+'_MEASURED_REFERENCE.png'),'红：部分中心参考；蓝：已知墨迹'),(folder/'before/joint/raw_center.png','上轮联合模型'),(folder/'after/joint/raw_center.png','本轮修正')]):
            draw.text((j*360+4,5),label,font=font,fill='black');im=Image.open(p).convert('RGB');im.thumbnail((356,356));canvas.paste(im,(j*360,38))
        canvas.save(ROOT/(ident+'_COARSE_COMPARISON.jpg'),quality=95)
    metric_rows=[]
    labels=[('新程序样本中心线F1','declared_nonblank_f1'),('新程序样本淡线召回（1.5像素容差）','faint_center_recall'),('新程序松针召回','pine_center_recall'),('新程序空白误线像素占比','declared_blank_fp_fraction'),('G05已核对纸面误线像素占比','G05_reviewed_paper_fp_fraction'),('G05纸面变体误线像素占比','source_separated_paper_stress_fp_fraction')]
    for title,key in labels:metric_rows.append(f'<tr><td>{title}</td><td>{before[key]:.6f}</td><td>{after[key]:.6f}</td></tr>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>纸面误检与粗墨中心专项训练</title>
<style>body{font:17px system-ui;line-height:1.75;background:#f3f1eb;margin:24px auto;max-width:1900px;padding:0 16px;color:#222}img{width:100%;background:white}section{border-top:1px solid #ccc;margin:28px 0}aside{background:#fff0d5;padding:18px}table{border-collapse:collapse;background:white}td,th{padding:10px;border:1px solid #bbb}a{color:#175f86}</style>
<h1>纸面误检、粗墨中心与淡线保留：专项训练</h1>
<aside>实际GPU训练600＋300步。第一阶段纸面损失下降，但未通过淡线保护，未采用；第二阶段加强已知淡线正例，仍按原保留标准选择权重。本轮推理采用与上一轮相同的中心线0.35、墨迹0.25、关系0.5门槛。保留全部增删与未通过阶段记录。默认v4和机械臂程序均未替换。</aside>
<h2>具体改动</h2><p>真实山石R04中4条局部粗墨走向通过单独横截面测量建立部分中心与墨带监督；N11中2条走向用于校准。只有具备清晰两侧肩部的截面参与中心监督，相邻墨带与交汇不强行配成同一笔。交汇处仅明确深墨区域参与墨迹监督，中心和笔顺身份保持未知。</p>
<p>纸面ROI经过原图核对，按母图分开：G04为主要新增训练纸面，G02校准，G05保留回归。64训练、16校准、32回归纸面变体来自这些已核对区域，不是112幅新画作。训练与推理统一先计算整图明暗特征再切块；额外惩罚最明显的64个纸面误线像素，避免稀疏错误被平均掉。推理只输入图片，纸面标注只用于训练和评价。</p>
<h2>同阈值前后对照</h2><p>48张全新程序种子910000–910047用于检查薄线、粗线、并行和交汇；程序家族与训练相同，不能据此推断真实画作准确率。G05是旧母图的6个相关场景，纸面指标仅计算已核对的明确背景。没有完整真实线稿，整幅准确率仍未知。</p>
<table><tr><th>指标</th><th>上轮联合模型</th><th>本轮专项修正</th></tr>'''+''.join(metric_rows)+'''</table>
<p><a href="summary.json">完整指标、约束与局部粗墨诊断</a> · <a href="training_summary.json">第一阶段600步记录（未采用）</a> · <a href="faint_preservation_stage2/training_summary.json">第二阶段300步与选中权重</a> · <a href="evaluation_protocol.json">冻结阈值与测试协议</a> · <a href="manifest.json">数据来源与未知掩码</a> · <a href="preflight_checks.json">训练前检查</a></p>
<h2>实际粗墨部分参考</h2><p>以下是助手测量的局部参考，未经专家逐笔确认。R04用于训练，N11用于校准；它们不是独立盲测。蓝色交汇中没有红色中心参考的区域，只监督确定墨迹。</p><img src="R04_COARSE_COMPARISON.jpg"><img src="N11_COARSE_COMPARISON.jpg">
<h2>实际生成画预览</h2><img src="ACTUAL_PREVIEW.jpg"><h2>全部32张旧实际画面对照</h2><p>黑色保持，红色删除，蓝色新增；删除不自动等于纠错，新增不自动等于正确。复杂皴线、完整笔顺、物理笔宽仍未验收。</p>'''+''.join(cards)
    gate_text='已通过纸面、淡线与松针专项门槛' if all(gates.values()) else '未通过全部专项门槛，保持研究候选'
    page=page.replace('<h2>同阈值前后对照</h2>',f'<p>{gate_text}。这只覆盖纸面与程序薄线约束，不表示整幅画已经超过v4。选中本轮累计更新：{summary["selected_new_updates"]}步。</p><h2>同阈值前后对照</h2>')
    if results_root==corrected:
        page=page.replace('实际GPU训练600＋300步。','实际GPU训练600＋300＋200步。发现程序松针渲染只画端点的缺陷后，已修复连续线段、重建训练/校准图并补训200步；此前含该缺陷的松针与聚合指标作废。')
        page=page.replace('48张全新程序种子910000–910047','48张修正渲染后的全新程序种子950000–950047')
        page=page.replace('<h2>具体改动</h2>','<p><a href="corrected_renderer_stage3/RENDERER_AUDIT.jpg">端点渲染缺陷与修正对照</a> · <a href="corrected_renderer_stage3/renderer_rgb_checks.json">80条RGB中点着墨检查</a> · <a href="corrected_renderer_stage3/fit/training_summary.json">修正数据后的训练记录</a> · <a href="corrected_renderer_stage3/evaluation_protocol.json">最终固定阈值与新测试协议</a></p><h2>具体改动</h2>')
        page=page.replace('href="evaluation_protocol.json"','href="corrected_renderer_stage3/evaluation_protocol.json"').replace('href="manifest.json"','href="corrected_renderer_stage3/manifest.json"')
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    if (ROOT/'mountain_diagnostics/audit.json').exists():
        import runpy
        runpy.run_path(str(Path(__file__).with_name('record_mountain_regression_audit.py')))
    source=ROOT/'source_snapshot';source.mkdir(exist_ok=True);names=['inspect_paper_center_refinement.py','paper_center_refinement_data.py','extend_paper_center_data.py','check_paper_center_refinement.py','train_paper_center_refinement.py','preserve_faint_paper_refinement.py','evaluate_paper_center_refinement.py','report_paper_center_refinement.py']
    names += ['corrected_stroke_fixture.py','repair_refinement_fixtures.py','train_corrected_refinement.py','audit_corrected_stroke_rgb.py','verify_paper_center_refinement.py','record_mountain_regression_audit.py']
    for name in names:shutil.copy2(Path(__file__).with_name(name),source/name)
    jsave(ROOT/'source_snapshot_manifest.json',{n:digest(source/n) for n in names})
    print('PAPER_CENTER_REPORT',json.dumps(dict(metrics=stats,gates=gates,report=str(ROOT/'index.html')),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
