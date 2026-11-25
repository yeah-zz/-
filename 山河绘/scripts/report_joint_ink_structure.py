# -*- coding: utf-8 -*-
"""Complete comparison and conservative adoption decision for the joint pilot."""
import sys
sys.dont_write_bytecode=True
from joint_ink_structure_data import ROOT,R2,LOCAL,LAB
from rock_v15_common import read,jsave,digest
from evaluate_simple_landscape_specialist_r2 import single_path_score
from evaluate_joint_ink_structure import summarize_relation,reference_center
import numpy as np,cv2,json,html,shutil
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def main():
    result=read(ROOT/'evaluation_results.json');train=read(ROOT/'training_summary.json');selection=read(ROOT/'selection.json');manifest=read(ROOT/'manifest.json');by_id={r['id']:r for r in manifest['records']};metrics={}
    stage=ROOT/'ink_gate_stage2';latest=read(stage/'summary.json') if (stage/'summary.json').exists() else None
    for kind in ['hard_declared','existing_declared']:
        rows=[r for r in result['records'] if r['kind']==kind];group={}
        for tag in ['previous_neural','joint']:
            positive=[r[tag]['center'] for r in rows if r[tag]['center']['f1'] is not None];blank=[r[tag]['center']['blank_fp_fraction'] for r in rows if r[tag]['center'].get('blank_fp_fraction') is not None]
            relation=summarize_relation([r[tag]['relation'] for r in rows]);continuity=[]
            for r in rows:
                doc=read(by_id[r['id']]['reference'])
                if not doc['strokes']:continue
                paths=read(ROOT/'evaluation'/r['id']/tag/'paths.json')['paths']
                continuity.append(single_path_score([np.asarray(p,np.float32) for p in paths],doc))
            group[tag]=dict(images=len(rows),nonblank_images=len(positive),center={k:float(np.mean([r[k] for r in positive])) for k in ['precision','recall','f1']},blank_false_positive_fraction=float(np.mean(blank)) if blank else None,relation=relation,mean_single_path_coverage=float(np.mean(continuity)))
        metrics[kind]=group
    actual=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for row in [r for r in result['records'] if r['kind']=='actual_reused']:
        ident=row['id'];folder=ROOT/'evaluation'/ident;current=stage/'evaluation'/ident if latest else folder;rgb=Image.open(LOCAL/'baseline'/ident/'input.png').convert('RGB');canvas=Image.new('RGB',(1980,390),'#f3f1eb');d=ImageDraw.Draw(canvas)
        panels=[(LOCAL/'baseline'/ident/'input.png','原图'),(LOCAL/'baseline'/ident/'lines.png','冻结 v4'),(LOCAL/'candidate'/ident/'lines.png','上轮局部补回'),(folder/'previous_neural/raw_center.png','上一轮神经模型'),(current/'joint/raw_center.png','联合模型（双头校准）' if latest else '本轮联合模型原始线'),(current/'joint/path_identity.png','关系解码轨迹（颜色分段）')]
        for j,(p,label) in enumerate(panels):
            d.text((330*j+3,5),ident+' '+label,font=font,fill='black');im=Image.open(p).convert('RGB');im.thumbnail((326,326));canvas.paste(im,(330*j,38))
        canvas.save(folder/'comparison.jpg',quality=94)
        meta=read(current/'joint/paths.json')['meta'];actual.append(dict(id=ident,source_group=int(ident[1:3]),learned_edge_cuts=meta['removed_graph_edges'],learned_endpoint_joins=meta['endpoint_bridges'],full_image_accuracy=None))
    summary=dict(completed_gpu_updates=train['completed_updates'],retained_step=train['retained_step'],device=train['gpu'],training_samples=sum(r['split']=='train' for r in manifest['records']),calibration_samples=sum(r['split']=='calibration' for r in manifest['records']),declared_test_samples=60,actual_regression_images=30,real_partial_positive_segments=13,real_background_regions=5,expert_gold_labels=0,metrics=metrics,actual_graph_diagnostics=actual,new_actual_imagegen_test=False,new_procedural_hard_test=True,actual_full_image_accuracy=None,default_v4_unchanged=True,robot_connected=False,adoption='research checkpoint only; not promoted above v4',inference_uses_reference_paths=False,model_sha256=digest(ROOT/'joint.pt'),limitations=['Actual partial references cover only 3 parent images; 2 training sources and 1 calibration source.','13 isolated segments do not supervise every thick ink band, junction, house, or pine needle.','Actual ink labels are approximate local photometric support; unknown pixels ignored.','36 new procedural hard examples and 24 reused procedural tests do not establish real artwork accuracy.','The 30 existing generated crops come from five parent sheets and are reused regression images.','Connections use learned local affinities and conservative graph tracing; not direct Bezier regression or verified painter strokes.','Width is image-pixel appearance, not calibrated brush pressure.'])
    if latest:
        protocol=read(stage/'selection.json');latest['fresh_blank_safety_pass']=bool(latest['metrics']['joint']['blank_fp_fraction']<=protocol['caps']['declared_blank_fp_fraction']);latest['adoption']='research only; not promoted'
        jsave(stage/'summary.json',latest);summary['ink_gate_stage2']=latest;summary['fresh_blank_safety_pass']=latest['fresh_blank_safety_pass']
    jsave(ROOT/'summary.json',summary)
    fixed=['G04_01','G05_01'];board=Image.new('RGB',(1980,len(fixed)*390),'white')
    for i,ident in enumerate(fixed):board.paste(Image.open(ROOT/'evaluation'/ident/'comparison.jpg'),(0,i*390))
    board.save(ROOT/'FIXED_ACTUAL_COMPARISON.jpg',quality=95)
    preview=Image.new('RGB',(1488,540),'#f3f1eb');draw=ImageDraw.Draw(preview);current=stage/'evaluation/G04_01/joint/raw_center.png' if latest else ROOT/'evaluation/G04_01/joint/raw_center.png'
    for j,(p,label) in enumerate([(LOCAL/'baseline/G04_01/input.png','原图：本轮训练外的旧回归图'),(LOCAL/'candidate/G04_01/lines.png','上轮 v4 局部补回'),(current,'本轮联合模型：尚未推广')]):
        draw.text((j*496+6,5),label,font=font,fill='black');preview.paste(Image.open(p).convert('RGB'),(j*496,36))
    preview.save(ROOT/'ACTUAL_PREVIEW.jpg',quality=95)
    # Clear crop of the same pine region; do not silently change scale or width.
    zoom=Image.new('RGB',(1560,330),'#f3f1eb');d=ImageDraw.Draw(zoom);box=(60,75,320,210)
    for j,(p,label) in enumerate([(LOCAL/'baseline/G01_01/input.png','原图（相关训练来源）'),(LOCAL/'candidate/G01_01/lines.png','上轮局部补回'),((stage if latest else ROOT)/'evaluation/G01_01/joint/raw_center.png','本轮模型输出')]):
        d.text((j*520+5,4),label,font=font,fill='black');im=Image.open(p).convert('RGB').crop(box).resize((520,270));zoom.paste(im,(j*520,38))
    zoom.save(ROOT/'PINE_COMPARISON.jpg',quality=96)
    cards=[]
    for row in actual:
        ident=row['id'];group=row['source_group'];scope='相关训练母图' if group in [1,3] else ('相关校准母图' if group==2 else '本轮训练外的旧回归母图')
        prefix='ink_gate_stage2/' if latest else ''
        cards.append(f'<section><h2>{ident} · {scope}</h2><a href="evaluation/{ident}/comparison.jpg"><img src="evaluation/{ident}/comparison.jpg"></a><p>关系切边 {row["learned_edge_cuts"]}；有证据端点续接 {row["learned_endpoint_joins"]}。数量不是准确率。</p><p><a href="{prefix}evaluation/{ident}/joint/overlay.png">模型轨迹叠加原图</a> · <a href="{prefix}evaluation/{ident}/joint/ink_probability.png">墨迹概率</a> · <a href="{prefix}evaluation/{ident}/joint/paths.json">轨迹与续接记录</a> · <a href="{prefix}evaluation/{ident}/joint/lines.svg">SVG</a> · <a href="evaluation/{ident}/joint/raw_center.png">首轮未加墨迹门槛输出</a></p></section>')
    hard=metrics['hard_declared'];old=metrics['existing_declared'];metric_rows=[]
    for label,group in [('36张新程序难例（含空白）',hard),('24张旧程序测试图',old)]:
        a=group['previous_neural'];b=group['joint'];metric_rows.append(f'<tr><td>{label}</td><td>{a["center"]["f1"]:.3f} → {b["center"]["f1"]:.3f}</td><td>{a["mean_single_path_coverage"]:.3f} → {b["mean_single_path_coverage"]:.3f}</td><td>{b["relation"]["balanced_accuracy"]:.3f}</td></tr>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>墨迹—中心线—连接关系联合训练</title><style>body{font:17px system-ui;line-height:1.75;background:#f3f1eb;color:#222;margin:24px auto;max-width:2000px;padding:0 16px}img{width:100%;background:white}section{border-top:1px solid #ccc;margin:30px 0}aside{padding:18px;background:#fff0d4}table{border-collapse:collapse;background:white}td,th{border:1px solid #bbb;padding:10px}a{color:#175f86}</style><h1>墨迹、中心线与连接关系联合训练：首轮实验</h1><aside>实际 GPU 训练900步，独立新权重。这里显示新模型本身的输出，没有用 v4 回退替换。当前仍为研究候选，未替换用户认可的 v4；真实画面仍可能漏线、碎线或生出杂线，不能称系统已训练成型。</aside><h2>监督与未知区域</h2><p>256张已有声明路径训练图、192张新增程序难例，以及11个实际画面部分训练样本。55个校准样本按来源隔离。实际参考合计13条非专家细线、5处背景，来自3张母图；2张用于训练，1张用于校准。它们是局部监督，远非数百张完整真实线稿。未标区域及14处疑难候选保持未知，未将v4整幅结果直接作为答案。</p><p>联合输出墨迹、中心线、方向、辅助像素笔宽及8个邻接关系通道。相邻像素关系用于图追踪，较远关系用于有原图墨迹支持的端点续接；拒绝空白续接、近平行误接及跨越其他线。输出是连续像素折线，还没有端到端曲线解码器或真实画家笔顺。</p><h2>有参考的程序测试</h2><p>下面仅评价程序声明路径：中心线F1采用1.5像素容差，空白样本单列误检率，不计入F1；单路径覆盖是连续性的代理。连接关系的正负类按已知笔画ID确定。它们不能代表真实工笔山水准确率。</p><table><tr><th>数据</th><th>中心线F1：旧神经→联合</th><th>单路径覆盖：旧神经→联合</th><th>联合关系平衡准确率</th></tr>'''+''.join(metric_rows)+'''</table><p>上一轮神经模型没有关系分支；其关系输出仅为随机初始化对照，不当作已有模型能力比较。<a href="summary.json">查看分类指标、空白误检与限制</a></p><h2>实际生成画固定对照</h2><p>每行：原图 / 冻结v4 / 上轮局部补回 / 上轮神经模型 / 本轮联合模型原始线 / 本轮关系轨迹。30张均来自已有5张母图，不是新生成盲测；本轮没有新增imagegen测试图。没有完整真实线稿，因此不计算整幅真实准确率。</p><p><a href="training_summary.json">GPU训练记录</a> · <a href="manifest.json">数据来源与权重掩码</a> · <a href="selection.json">独立校准选择</a> · <a href="ACTUAL_PARTIAL_SUPERVISION.jpg">实际部分标注</a> · <a href="ADDITIONAL_ROCK_REVIEW.jpg">新增山石局部核对</a> · <a href="preflight_checks.json">数据/梯度检查</a> · <a href="graph_decoder_checks.json">连接反例检查</a></p>'''+''.join(cards)
    if latest:
        a=latest['metrics']['previous_neural'];b=latest['metrics']['joint']
        extra=f'<h2>双头校准后的另一批36张新种子程序测试</h2><p>首轮发现空白误线增加后，只用校准数据联合选择中心线阈值和墨迹阈值；训练权重仍为同一个900步模型。另一批程序测试种子760000–760035没有参与训练或阈值选择，与前批仍属同类程序图形。</p><p>同批中心线F1：{a["center"]["f1"]:.3f} → {b["center"]["f1"]:.3f}；单路径覆盖：{a["mean_single_path_coverage"]:.3f} → {b["mean_single_path_coverage"]:.3f}；关系平衡准确率：{b["relation"]["balanced_accuracy"]:.3f}。</p><p>空白误线像素占比：{a["blank_fp_fraction"]:.6f} → {b["blank_fp_fraction"]:.6f}。新批空白误检仍未达到校准设定的上限，因此不通过推广验收，不替换v4。完整真实画面的优势仍未证实。</p><p><a href="ink_gate_stage2/summary.json">新批指标与逐样本记录</a> · <a href="ink_gate_stage2/selection.json">双头门槛与校准约束</a> · <a href="ink_gate_stage2/fresh_test_manifest.json">新批测试来源</a></p>'
        page=page.replace('<h2>实际生成画固定对照</h2>',extra+'<h2>实际生成画固定对照</h2>')
        page=page.replace('本轮联合模型原始线 / 本轮关系轨迹','本轮联合模型双头校准输出 / 本轮关系轨迹')
    page=page.replace('<h2>实际生成画固定对照</h2>','<h2>实际生成画固定对照</h2><img src="ACTUAL_PREVIEW.jpg">')
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True);source_files=['joint_ink_structure_data.py','train_joint_ink_structure.py','check_joint_ink_structure.py','prepare_joint_rock_review.py','curate_joint_rock_references.py','evaluate_joint_ink_structure.py','check_joint_graph_decoder.py','report_joint_ink_structure.py']
    source_files.append('calibrate_joint_ink_gate.py')
    for name in source_files:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    jsave(ROOT/'source_snapshot_manifest.json',{name:digest(snapshot/name) for name in source_files})
    print('JOINT_REPORT',json.dumps(dict(metrics=metrics,summary=str(ROOT/'summary.json'),report=str(ROOT/'index.html')),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
