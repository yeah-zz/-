# -*- coding: utf-8 -*-
from evaluate_simple_landscape_specialist_r2 import ROOT,LAB,read,jsave,digest
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
import json,html,numpy as np,shutil

def font(size):return ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',size)

def report():
    summary=read(ROOT/'evaluation_summary.json');progress=read(ROOT/'evaluation_progress.json');assert progress['complete']
    training=read(ROOT/'training_summary.json');r1=ROOT.parent;first=read(r1/'evaluation_summary.json');rows=progress['records'];generated=[r for r in rows if r['kind']=='imagegen'];paired=[r for r in rows if r['kind']=='paired']
    assert len(generated)==30 and len(paired)==40
    assert read(ROOT/'data_checks.json')['all_passed']
    assert summary['checkpoint_sha256']==digest(ROOT/'specialist.pt')
    selection=read(ROOT/'selection.json');assert digest(ROOT/'imagegen_holdout_manifest.json')==selection['imagegen_holdout_manifest_sha256']
    ledger=dict(first_round_updates=1200,second_round_updates=1200,total_new_gpu_updates=2400,paired_train_images_per_round=512,total_new_procedural_training_images=1024,first_36_imagegen_scenes_used_as_development_after_first_eval=True,final_fresh_imagegen_scenes=30,final_generation_calls=5,within_sheet_correlation=True,imagegen_images_never_used_as_training_targets=True,old_v4_not_replaced=True,physical_robot_connected=False)
    jsave(ROOT/'iteration_ledger.json',ledger)
    panels=[]
    for r in generated:
        sid=r['id'];folder=ROOT/'outputs'/sid;flag='规则通过' if r['automatic_rule_pass'] else '规则未过，回退 v4'
        imgs=''.join(f'<figure><figcaption>{caption}</figcaption><a href="outputs/{sid}/{path}"><img loading="lazy" src="outputs/{sid}/{path}"></a></figure>' for path,caption in [('selected/input.png','原图'),('v4/lines.png','用户认可的 v4 对照'),('raw_neural/lines.png','专用网络原始输出'),('retried_neural/lines.png','固定规则重提候选'),('selected/lines.png','流程选择（可能回退）'),('selected/overlay.png','选择结果叠加')])
        panels.append(f'<details><summary>{sid}：{flag}</summary><p>自动规则状态不等于人工确认。重提：{r["retry_attempted"]}；原始规则记录：{html.escape(json.dumps(r["raw_qa"],ensure_ascii=False))}</p><div class="grid">{imgs}</div><p><a href="outputs/{sid}/raw_neural/paths.json">原始模型路径</a> · <a href="outputs/{sid}/selected/paths.json">选择路径</a></p></details>')
    # One prespecified first scene per thematic group, not best-of output selection.
    chosen=['G01_01','G03_01','G05_01'];board=Image.new('RGB',(1536,1780),'#f3f1eb');d=ImageDraw.Draw(board)
    d.text((12,12),'全新留出场景：原图 / v4 / 专用网络原始提取',font=font(28),fill='black')
    d.text((12,54),'未人工补线；这里显示模型原始输出，质检失败及回退另列。',font=font(21),fill='#444444')
    for i,sid in enumerate(chosen):
        y=100+i*552;r=next(r for r in generated if r['id']==sid);d.text((12,y),sid+'  '+('规则通过' if r['automatic_rule_pass'] else '规则未过'),font=font(21),fill='black')
        for j,path in enumerate(['selected/input.png','v4/lines.png','raw_neural/lines.png']):
            im=Image.open(ROOT/'outputs'/sid/path).convert('RGB');im.thumbnail((496,496),Image.Resampling.LANCZOS);board.paste(im,(12+j*508,y+36))
    board.save(ROOT/'FRESH_GENERATED_COMPARISON.jpg',quality=97,subsampling=0)
    # Paired benchmarks show the known vector reference as a separate column.
    refs=read(ROOT/'paired_manifest.json')['records'];board=Image.new('RGB',(1600,4*450+65),'#f3f1eb');d=ImageDraw.Draw(board);d.text((8,8),'程序配对留出样本：原图 / 已知程序线条 / v4 / 新网络',font=font(25),fill='black')
    for i,sid in enumerate(['test_0000','test_0001','test_0002','test_0003']):
        row=next(r for r in refs if r['id']==sid);im=Image.open(row['image']).convert('RGB');doc=read(row['reference']);reference=Image.new('RGB',im.size,'white');draw=ImageDraw.Draw(reference)
        for p in doc['strokes']:draw.line([tuple(q) for q in p['points']],fill='black',width=1)
        views=[im,reference,Image.open(ROOT/'outputs'/sid/'v4'/'lines.png').convert('RGB'),Image.open(ROOT/'outputs'/sid/'raw_neural'/'lines.png').convert('RGB')]
        for j,view in enumerate(views):d.text((j*400+8,65+i*450),sid,font=font(19),fill='black');board.paste(view,(j*400+8,98+i*450))
    board.save(ROOT/'PAIRED_HOLDOUT_COMPARISON.jpg',quality=95)
    metrics=summary['paired_metrics'];table=''.join(f'<tr><td>{label}</td><td>{metrics[key]["precision"]:.3f}</td><td>{metrics[key]["recall"]:.3f}</td><td>{metrics[key]["f1"]:.3f}</td><td>{metrics[key]["single_path_coverage"]:.3f}</td></tr>' for key,label in [('v4','v4'),('neural','专用网络原始输出'),('selected','规则与回退选择')])
    accepted=summary['imagegen_automatic_pass_count'];fallback=summary['imagegen_fallback_count']
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>简洁山水专项监督训练</title><style>body{font:16px system-ui,"Microsoft YaHei",sans-serif;background:#f3f1eb;color:#222;margin:24px;line-height:1.7}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0}img{width:100%;background:white}p{max-width:1150px}details{padding:14px 0;border-top:1px solid #ccc}summary{font-weight:bold;cursor:pointer}a{color:#075ea2}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px 14px;text-align:left}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><h1>简洁青绿山水：配对监督训练与新图检验</h1><p>本轮在 RTX 5060 Laptop GPU 上进行了两阶段各 1200 次更新，共 2400 次新训练。每阶段使用 512 张独立种子的程序配对训练图、64 张校准图及 40 张程序留出图；训练参考来自渲染时声明的笔画路径，不是旧算法提取结果。第一阶段较规整，第二阶段加入宽墨、颗粒干笔和不规则轮廓。程序样本仍比真正绘画简化，无法代表所有山水笔法。</p><p>第一阶段 36 个 imagegen 场景未通过从程序样本校准出的规则，随后用于观察差距，已不再作为最终留出。第二阶段重新用内置 imagegen 生成 5 组共 30 个全新场景；训练与阈值选择未读取这些图。每组来自一张 3×2 图组，场景相关，不能当作 30 个独立艺术风格。全部提示词、原图、固定网格裁切和哈希保留。</p><h2>40 张程序留出图上的测量</h2><p>以下仅是已知程序线条的中心点匹配，容差 1.5 像素；单路径覆盖使用 2 像素容差。它不是 imagegen 图、真实古画或机械臂落笔准确率，也不证明真实笔顺正确。</p><table><tr><th>方法</th><th>中心点精确率</th><th>中心点召回率</th><th>F1</th><th>平均单路径覆盖</th></tr>'''+table+'''</table><h2>30 个全新 imagegen 场景</h2><p>自动规则通过 '''+str(accepted)+'''/30；''' +str(fallback)+''' 个保留失败标记并回退 v4。规则使用图像墨迹支持、源像素覆盖和小闭环等代理，不是人工“达到昨天效果”的验收。因此不能把这个通过数当作最终画作质量通过率，更不能保证每一张都达标。页面同时显示原始网络结果与回退结果。</p><p><a href="FRESH_GENERATED_COMPARISON.jpg">预先指定的三张新图对照</a> · <a href="PAIRED_HOLDOUT_COMPARISON.jpg">程序真值对照</a> · <a href="evaluation_summary.json">完整指标</a> · <a href="training_summary.json">GPU 训练记录</a> · <a href="imagegen_prompts.json">完整生成提示词</a> · <a href="iteration_ledger.json">两阶段记录</a></p>'''+''.join(panels)+'''<h2>当前边界</h2><p>此模型只学习了受控渲染标签中的轮廓、长皴线、松树线和建筑水纹辅助类别，没有学会各种传统皴法的专家分类。宽墨交汇、淡墨和干笔的真实分笔仍有歧义；没有人工逐笔真值的生成图只做定性和规则检查。所有输出为离线像素路径，未标定物理笔宽、压力、墨色或机械臂坐标，不连接机械臂。旧 v4、旧稳定入口和旧权重保留。</p></html>'''
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    (r1/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>简洁山水专项模型</title><h1>简洁山水专项模型</h1><p><a href="round2/index.html">第二阶段训练与最终新图对照</a></p><p><a href="evaluation_summary.json">第一阶段结果：36 个生成场景未通过规则，作为开发观察保留</a></p>',encoding='utf-8')
    jsave(ROOT/'report_checks.json',dict(paired_holdout_complete=len(paired)==40,unlabeled_fresh_scenes_complete=len(generated)==30,data_checks_passed=True,checkpoint_hash_verified=True,selection_holdout_manifest_verified=True,raw_and_fallback_both_disclosed=True,old_v4_not_replaced=True))
    print('SPECIALIST_REPORT_READY',str(ROOT/'index.html'),json.dumps(summary),flush=True)

if __name__=='__main__':report()
