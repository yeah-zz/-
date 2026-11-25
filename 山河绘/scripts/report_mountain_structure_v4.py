# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from mountain_structure_v4_20261101 import ROOT,LAB,CFG,read,jsave,digest,graph_cycles
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np,cv2,shutil,json

def main():
    records=read(ROOT/'results_all.json')['records'];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    roi=[150,210,350,400];x0,y0,x1,y1=roi;folder=ROOT/'outputs/G05_01';before=np.asarray(Image.open(folder/'baseline.png'))<128;after=np.asarray(Image.open(folder/'lines.png'))<128
    diagnostic=dict(image='G05_01',roi=roi)
    for key,mask in [('v4_short_recovery',before),('structure_candidate',after)]:
        m=mask[y0:y1,x0:x1];diagnostic[key]=dict(pixels=int(m.sum()),components=cv2.connectedComponents(m.astype(np.uint8),8)[0]-1,cycles=graph_cycles(m))
    withdrawn=LAB/'paper_center_refinement_20261029/corrected_renderer_stage3/evaluation/G05_01/after/joint/raw_center.png'
    m=(np.asarray(Image.open(withdrawn))<128)[y0:y1,x0:x1];diagnostic['withdrawn_neural']=dict(pixels=int(m.sum()),components=cv2.connectedComponents(m.astype(np.uint8),8)[0]-1,cycles=graph_cycles(m))
    diagnostic['counts_are_not_accuracy']=True;jsave(ROOT/'mountain_roi_diagnostic.json',diagnostic)
    trees=[]
    for ident,box in [('G01_01',[65,75,320,190]),('G05_01',[170,65,420,150]),('N02',[60,340,790,720])]:
        f=ROOT/'outputs'/ident;old=np.asarray(Image.open(f/'baseline.png'))<128;new=np.asarray(Image.open(f/'lines.png'))<128;x,y,r,b=box;original=old[y:b,x:r];selected=new[y:b,x:r]
        trees.append(dict(id=ident,roi=box,baseline_pixels=int(original.sum()),retained_pixels=int((original&selected).sum()),exactly_preserved=bool(np.array_equal(original,selected)),roi_for_audit_only_not_used_in_inference=True))
    summary=dict(images=len(records),new_neural_training_steps=0,method='CPU v4 candidate selection; geometric/color routing, not trained semantics',retained_baseline_outside_interiors=all(r['stats']['outside_interior_exactly_preserved'] for r in records),retained_protected_detail=all(r['stats']['protected_detail_exactly_preserved'] for r in records),new_pixels=sum(r['stats']['added_pixels'] for r in records),images_with_fewer_cycles=sum(r['stats']['selected_cycles']<r['stats']['baseline_cycles'] for r in records),images_with_more_cycles=sum(r['stats']['selected_cycles']>r['stats']['baseline_cycles'] for r in records),tree_roi_checks=trees,mountain_roi_diagnostic=diagnostic,full_image_accuracy=None,default_entry_unchanged=True,robot_connected=False,stage='reviewable structure-only candidate; no automatic replacement',development_examples=['G01_01','G02_01','G05_01'],dataset_scope='30 existing 496px crops from five correlated sheets; N02 native accepted image. All reused, not blind test.',known_tradeoffs=['Some true fine cun strokes are deferred with noise.','Missing original contours are not hallucinated or repaired.','Color routing only addresses this blue-green painting family; it is not full semantic segmentation.','Interior budgets count candidates, not verified painter strokes.','Tree ROI checks do not prove every tree in every image was recognized.'])
    jsave(ROOT/'summary.json',summary)
    # Four-panel view includes the rejected neural mountain for context, while
    # the baseline remains the actual v4 plus previously saved short-line stage.
    preview=Image.new('RGB',(1984,545),'#f3f1eb');draw=ImageDraw.Draw(preview)
    for j,(path,title) in enumerate([(folder/'input.png','原图'),(folder/'baseline.png','v4＋既有短线补回'),(withdrawn,'已否决的神经山体结果'),(folder/'lines.png','本轮结构简稿候选')]):
        draw.text((j*496+5,5),title,font=font,fill='black');preview.paste(Image.open(path).convert('RGB'),(j*496,38))
    preview.save(ROOT/'FIXED_PREVIEW.jpg',quality=95)
    simple=Image.new('RGB',(1488,545),'#f3f1eb');draw=ImageDraw.Draw(simple)
    for j,(path,title) in enumerate([(folder/'input.png','原图'),(folder/'baseline.png','v4＋既有短线补回'),(folder/'lines.png','本轮结构简稿候选')]):
        draw.text((j*496+5,5),title,font=font,fill='black');simple.paste(Image.open(path).convert('RGB'),(j*496,38))
    simple.save(ROOT/'STRUCTURE_PREVIEW.jpg',quality=95)
    local=Image.new('RGB',(1200,430),'#f3f1eb');draw=ImageDraw.Draw(local)
    for j,(path,title) in enumerate([(folder/'input.png','原图局部'),(folder/'baseline.png','v4 局部'),(folder/'lines.png','结构筛选局部')]):
        im=Image.open(path).convert('RGB').crop(roi).resize((400,380));draw.text((j*400+5,5),title,font=font,fill='black');local.paste(im,(j*400,38))
    local.save(ROOT/'MOUNTAIN_DETAIL.jpg',quality=95)
    panels=[]
    for r in records:
        ident=r['id'];s=r['stats'];panels.append(f'<section><h2>{ident}</h2><a href="outputs/{ident}/comparison.jpg"><img src="outputs/{ident}/comparison.jpg"></a><p>山体内部基线线像素 {s["baseline_pixels_in_interior"]} → {s["selected_pixels_in_interior"]}；全图像素图闭环 {s["baseline_cycles"]} → {s["selected_cycles"]}。这些是几何数量，不是准确率。</p><p><a href="outputs/{ident}/all_evidence.png">黑色保留／灰色待定</a> · <a href="outputs/{ident}/routing.png">分区核对</a> · <a href="outputs/{ident}/changes.png">全部删除位置</a> · <a href="outputs/{ident}/paths.json">像素路径与候选</a> · <a href="outputs/{ident}/lines.svg">SVG</a></p></section>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v4 山体结构简稿与树木细节保留</title><style>body{font:17px system-ui;line-height:1.75;background:#f3f1eb;color:#222;max-width:2050px;margin:24px auto;padding:0 16px}img{width:100%;background:white}aside{padding:18px;background:#fff1d7}section{margin:30px 0;border-top:1px solid #ccc}a{color:#185f85}</style>
<h1>山体保留结构，树木细节分开保护</h1><aside>本轮是独立CPU规则实验，没有新增神经网络训练，也未采用上一轮被否决的神经山体结果。小图基线为冻结v4加此前保存的短线补回；原生N02基线为用户认可的原v4。所有输出都保持同一像素线宽。原v4、旧权重、正式入口和机械臂程序均未替换。</aside>
<h2>本轮行为</h2><p>用青绿设色和原图墨迹定位山体内部；范围外的v4像素与保护区像素逐像素保留。树冠短线束及相连边缘枝干单独保护，不把山体内的阴影全部当细节保护。分区仍是颜色与几何规则，不是训练好的树木/山石语义模型。</p><p>以v4原有完整候选优先；保留有原图支持的长结构线和细长粗墨骨干，再按长度、直顺性、原图双侧墨迹证据和每片山体追加长线预算筛选。只沿已有相接路径尝试排列，不画跨空白的桥，不新增线像素。导出按图边去重，不能当作已确认画家笔顺。</p><p>转入待定的内容仍保存在灰色证据层和候选文件中，不当作背景负例。简化会舍弃部分真实细皴线，主线若已在基线漏掉也不会凭空补出；因此黑线减少不能单独证明更准确。这是用于检查主轮廓和少量长皴线的候选版本。</p>
<p><a href="summary.json">汇总与限制</a> · <a href="frozen_protocol.json">冻结规则与开发样例</a> · <a href="behavior_checks.json">长线／闭合轮廓／平行线保护检查</a> · <a href="mountain_roi_diagnostic.json">同区域几何诊断</a></p>
<h2>固定问题图对照</h2><img src="FIXED_PREVIEW.jpg"><h2>同一山体局部</h2><img src="MOUNTAIN_DETAIL.jpg">
<h2>全部31张已有图片</h2><p>30张小图来自5张相关母图，另含原生N02；本轮都作为旧图复测，不是新图盲测。每行：原图／v4基线／蓝色内部与绿色细节保护范围／结构简稿／红色转待定位置。</p>'''+''.join(panels)
    page=page.replace('href="frozen_protocol.json"','href="frozen_protocol_'+CFG['version']+'.json"')
    page=page.replace('树冠短线束及相连边缘枝干单独保护','树冠短线束、有限距离相连的枝干及保守包络单独保护')
    page=page.replace('<h2>全部31张已有图片</h2>','<p>原生N02未找到可靠的山体内部颜色范围，因此整张保持认可的v4原样；这张用于确认保留效果，不作为本轮改进案例。</p><h2>全部31张已有图片</h2>')
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True);names=['mountain_structure_v4_20261101.py','run_mountain_structure_v4.py','check_mountain_structure_v4.py','report_mountain_structure_v4.py','extract_mountain_structure_v4.py','verify_mountain_structure_v4.py']
    for name in names:shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    jsave(ROOT/'source_snapshot_manifest.json',{n:digest(snapshot/n) for n in names})
    print('MOUNTAIN_STRUCTURE_REPORT',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
