# -*- coding: utf-8 -*-
import sys,json,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_ink_ownership_20261110 import OUT,ROOT,SIZE,write
import numpy as np
from PIL import Image,ImageDraw,ImageFont
def main():
 e=json.loads((OUT/'evaluation.json').read_text('utf8'));a=e['mean'];real=e['real'][0];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
 board=Image.new('RGB',(1440,455),'#f2efe6');d=ImageDraw.Draw(board)
 for i,(name,title) in enumerate([('real_training_module_input.png','原图·整块山石'),('real_training_module_old_masks.png','修改前·墨迹归属预测'),('real_training_module_new_raw_masks.png','修改后·墨迹归属预测'),('real_training_module_retained_curves.png','保留曲线·定位分支未改')]):
  board.paste(Image.open(OUT/name).convert('RGB').resize((350,350)),(i*360,38));d.text((i*360+5,5),title,font=font,fill='#243e37')
 d.text((8,402),f'同图同定位：旧{real["old_candidates"]}条 → 新保留{real["new_kept_candidates"]}条。颜色表示不同候选，不是正确答案或绘画效果。',font=small,fill='#333')
 d.text((8,430),'单笔归属局部改善；完整提取未修好。新分支未超过简单几何过滤基线，暂不采用。',font=small,fill='#333');board.save(OUT/'real_before_after.jpg',quality=95)
 data=np.load(OUT/'real_training_module_raw_masks.npz');orig=Image.open(OUT/'real_training_module_input.png').convert('RGB');count=len(data['new_raw']);oldthr=.5;newthr=e['mask_threshold']
 single=Image.new('RGB',(1280,475),'#f2efe6');sd=ImageDraw.Draw(single)
 panels=[orig,Image.fromarray(np.uint8(255*(1-(data['old'][0]>oldthr)))).convert('RGB'),Image.fromarray(np.uint8(255*(1-(data['new_raw'][0]>newthr)))).convert('RGB')]
 for i,(im,label) in enumerate(zip(panels,['原图·自动候选0所在模块','旧分支：同一自动候选','新分支：同一自动候选'])):
  single.paste(im.resize((410,410)),(i*425,35));sd.text((i*425+4,4),label,font=font,fill='#243e37')
 single.save(OUT/'real_single_query_before_after.jpg',quality=95)
 for i in range(count):
  for kind,m,t in [('old',data['old'][i],oldthr),('new',data['new_raw'][i],newthr)]:
   Image.fromarray(np.uint8(255*(1-(m>t)))).resize((512,512)).save(OUT/f'query_{i:02d}_{kind}.png')
 external=''
 for scene in e['real'][1:]:
  ident=scene['id'];external+=f'<section><h2>{html.escape(ident)}：开发对照</h2><p>旧{scene["old_candidates"]}条／新保留{scene["new_kept_candidates"]}条。{html.escape(scene["scope"])}</p><div class="triple"><img src="{ident}_input.png"><img src="{ident}_old_masks.png"><img src="{ident}_new_raw_masks.png"></div></section>'
 summary=f'在16个新程序场景、统一256×256真值上，给定正确笔触位置的单笔墨迹IoU：旧{a["correct_anchor_old_iou"]:.3f}，新{a["correct_anchor_new_iou"]:.3f}，几何过滤基线{a["correct_anchor_geometry_only_iou"]:.3f}。这项提供了正确位置，只用于诊断，不能称自动提取准确率。'
 auto=f'不提供正确位置的自动实例掩码匹配F1：旧{a["old_auto_mask_f1"]:.3f}，新{a["new_auto_mask_f1"]:.3f}。漏笔仍多，整体未通过。'
 page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>单笔墨迹修正·实际前后对照</title><style>*{box-sizing:border-box}body{margin:0;background:#f2efe6;color:#263f37;font:17px/1.65 system-ui}main{max-width:1480px;margin:auto;padding:26px}h1{font-size:28px}h2{font-size:23px}section{background:white;padding:20px;margin:22px 0}img{display:block;max-width:100%}.notice{padding:17px;border-left:5px solid #b17b30;background:#fff6e3}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.triple{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #ccc;text-align:left}button,input{font:inherit}a{color:#19708c}@media(max-width:800px){.pair,.triple{grid-template-columns:1fr}main{padding:12px}}</style><main>
 <h1>单笔墨迹归属：修改前后实际对照</h1>
 <p class="notice"><b>单笔分解有局部改善，整幅提取仍未修好。</b>新分支没有超过几何过滤基线；真实图多余曲线和漏笔仍存在，部分单笔预测仍缩成局部墨团或被截短。当前不能作为可用新模型，也没有执行机械臂绘画。</p>
 <img src="real_before_after.jpg" alt="真实山石输入、旧新墨迹归属与保留曲线">
 <section><h2>这轮具体改动</h2><p>以图像、候选位置和旧模型预测曲线为条件，单独预测该候选所属墨迹。输入的曲线来自模型，不是人工参考；预测掩码来自神经网络，不是将中心线模糊加粗。训练加入正例、其他已确认笔迹和纸面的分别约束、区域Dice与空查询判断；未标区域继续忽略，真实交叉不强制拆成互斥像素。</p><p>墨迹输出改为256×256，并接入原图细节。此前128×128的最近邻监督可能完全丢掉一像素细笔，已用单元检查复现并修正。旧编码器、定位、曲线和角色分支冻结，所以本轮没有解决定位与完整曲线错误。</p><p>训练来源：新640个程序场景＋60次真实稀疏增强，随后全分辨率修正使用350个新程序场景＋50次真实增强。真实原始作品仍只有1幅、1个模块。程序路径不是国画皴法真值；真实参考也不是完整专家标注。</p></section>
 <section><h2>完整真值检验</h2><p>__SUMMARY__</p><p>__AUTO__</p><p>墨迹阈值仅在独立程序开发场景上选择，没有使用真实图片选阈值。旧概率掩码与新概率掩码在同一256×256真值网格上评价；不能与此前128×128的0.15直接当作同条件数值相减。</p><img src="single_stroke_before_after.jpg" loading="lazy" alt="程序场景同一笔的真值与旧新预测"><p>上图为最终保留集中的第一个场景，不是挑选最好案例。给定正确位置仅用于诊断；同位交汇处的多笔查询尚未解决。几何基线是原图墨迹阈值与预测曲线走廊的交集，当前学习结果没有超过它。</p></section>
 <section><h2>真实图：同一个自动候选，逐条查看</h2><p>这里由旧模型自动提供候选位置，未给参考起点或参考曲线。两张掩码对应完全相同的候选。它们都是预测，不能当作正确分笔标签。</p><input id="query" type="range" min="0" max="__MAX__" value="0"><span id="caption"></span><div class="pair"><div><p>旧分支</p><img id="old" src="query_00_old.png"></div><div><p>新分支</p><img id="new" src="query_00_new.png"></div></div></section>
 <section><h2>线条为何仍有问题</h2><p>训练模块自动候选数量仍为32条。新墨迹分支能收紧部分候选所属的区域，但并未重训曲线定位。实例掩码的空查询判断和去重也未消除这些真实图候选，因此不能宣称已经减少杂线或恢复了整块轮廓。</p><p>过滤规则另行记录：非空评分≥0.5、至少3个掩码像素；掩码IoU＞0.8且曲线距离＜6像素才判重复。这是明确的输出过滤，不是模型已经学会笔数。原始全部掩码保存在本地NPZ中。</p></section>
 __EXTERNAL__
 <section><h2>验收与限制</h2><p>本轮只修墨迹归属分支，没有训练整幅分石块、自主下一笔策略、皴法类别识别或机械臂参数。真实对照仍是开发样本，没有完整逐笔真值；不能据颜色块多少计算真实准确率。</p><p><a href="evaluation.json">完整评价</a> · <a href="training_report.json">首段训练</a> · <a href="detail_training.json">全分辨率修正</a> · <a href="mask_calibration_hr.json">阈值来源</a> · <a href="loss_checks.json">监督检查</a> · <a href="verification.json">基线与权重重放</a> · <a href="decision.json">未采用说明</a></p></section>
 <script>const slider=document.querySelector('#query');function update(){let n=String(slider.value).padStart(2,'0');document.querySelector('#old').src='query_'+n+'_old.png';document.querySelector('#new').src='query_'+n+'_new.png';document.querySelector('#caption').textContent=' 自动候选 '+slider.value+'（非正确答案）'}slider.oninput=update;update();</script></main>'''
 page=page.replace('__SUMMARY__',summary).replace('__AUTO__',auto).replace('__EXTERNAL__',external).replace('__MAX__',str(count-1));(OUT/'index.html').write_text(page,encoding='utf8')
 write(OUT/'decision.json',dict(adopted=False,whole_extraction_fixed=False,single_ownership_improved_against_failed_model=a['correct_anchor_new_iou']>a['correct_anchor_old_iou'],outperforms_geometry_baseline=a['correct_anchor_new_iou']>a['correct_anchor_geometry_only_iou'],component_mean_iou=a['correct_anchor_new_iou'],auto_instance_f1=a['new_auto_mask_f1'],real_candidate_count_reduced=real['new_kept_candidates']<real['old_candidates'],same_anchor_crossings_supported=False,robot_connected=False))
 print('REPORT_READY',OUT,summary,auto,flush=True)
if __name__=='__main__':main()
