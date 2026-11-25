# -*- coding: utf-8 -*-
from simple_landscape_pairs_r2 import ROOT
from rock_v15_common import read,jsave,digest
from PIL import Image,ImageDraw,ImageFont
import numpy as np,cv2,html

def font(size):return ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',size)
def vector_view(path,native_size,size):
    doc=read(path);canvas=np.full((size,size,3),255,np.uint8);scale=np.array([size/native_size[0],size/native_size[1]])
    for row in doc['paths']:
        p=np.asarray(row['points'] if isinstance(row,dict) else row,np.float32)*scale
        if len(p)>1:cv2.polylines(canvas,[np.rint(p).astype(np.int32)],False,(0,0,0),1,cv2.LINE_AA)
    return Image.fromarray(canvas)

if __name__=='__main__':
    board=Image.new('RGB',(1974,2240),'#f3f1eb');d=ImageDraw.Draw(board);d.text((20,15),'全新整幅图：原图 / v4 / 专用模型原始提取',font=font(31),fill='black')
    d.text((20,65),'原生分辨率推理；中心线按统一屏幕线宽显示，不表示毛笔粗细。',font=font(23),fill='#444444')
    records=[];sections=[]
    for i,sid in enumerate(['N01','N02','N03']):
        folder=ROOT/'native_outputs'/sid;audit=read(folder/'run.json');records.append(dict(id=sid,run=audit,source_sha256=digest(ROOT/'native_holdouts'/(sid+'.png'))))
        source=Image.open(folder/'selected'/'input.png').convert('RGB');native=source.size;im=source.copy();im.thumbnail((620,620),Image.Resampling.LANCZOS)
        y=118+i*700;d.text((20,y),sid+'  采用：'+('专用模型' if not audit['fallback_to_v4'] else 'v4（专用模型候选未通过自动检查）'),font=font(24),fill='black')
        board.paste(im,(20,y+43));board.paste(vector_view(folder/'v4'/'paths.json',native,620),(674,y+43));board.paste(vector_view(folder/'raw_neural'/'paths.json',native,620),(1328,y+43))
        selected_label='流程结果：采用 v4' if audit['fallback_to_v4'] else '流程结果：采用专用模型'
        figures=''.join(f'<figure><figcaption>{label}</figcaption><a href="native_outputs/{sid}/{path}"><img src="native_outputs/{sid}/{path}"></a></figure>' for path,label in [('selected/input.png','原图'),('v4/lines.png','v4 对照'),('raw_neural/lines.png','专用模型原始输出'),('raw_neural/overlay.png','专用模型叠加'),('selected/lines.png',selected_label),('retried_neural/lines.png','专用模型固定重提候选')])
        selected_version='v4' if audit['fallback_to_v4'] else '专用模型'
        sections.append(f'<section><h2>{sid}</h2><p>本图采用：<strong>{selected_version}</strong>。专用模型候选自动检查通过：{audit["automatic_rule_pass"]}；候选检查状态不是对 v4 结果的评价，也不是人工准确率。</p><div class="grid">{figures}</div></section>')
    board.save(ROOT/'NATIVE_FULL_COMPARISON.jpg',quality=97,subsampling=0)
    # Fixed subject crop for displaying pine branches; never used by extraction.
    sid='N02';folder=ROOT/'native_outputs'/sid;box=(40,300,830,930);crop=Image.new('RGB',(2400,720),'#f3f1eb');dd=ImageDraw.Draw(crop)
    for j,(path,label) in enumerate([('selected/input.png','原图松树局部'),('v4/lines.png','v4'),('raw_neural/lines.png','专用模型原始输出')]):
        dd.text((10+j*800,10),label,font=font(26),fill='black');crop.paste(Image.open(folder/path).convert('RGB').crop(box),(10+j*800,52))
    crop.save(ROOT/'NATIVE_PINE_DETAIL.jpg',quality=97,subsampling=0)
    jsave(ROOT/'native_holdout_results.json',dict(records=records,independent_full_image_generation_calls=3,human_stroke_gold=False,not_part_of_30_grid_scene_statistic=True,model_and_rules_not_tuned_on_these_images=True,accepted_v4_not_replaced=True))
    page='<!doctype html><meta charset="utf-8"><title>原生整幅图补充检验</title><style>body{font:16px system-ui;background:#f3f1eb;margin:24px;line-height:1.7}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0}img{width:100%}section{margin-top:24px}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><h1>3 张原生整幅图：冻结模型补充检验</h1><p>这些图片在模型与规则冻结后独立生成，不参与训练或调参。用于观察网格场景约 500 像素与独立整幅图的差别，未计算真实线条准确率。原始模型输出与回退结果都保留。</p><p><a href="NATIVE_FULL_COMPARISON.jpg">整图对照</a> · <a href="NATIVE_PINE_DETAIL.jpg">松针局部</a> · <a href="index.html">完整训练报告</a> · <a href="native_prompts.json">生成提示词</a></p>'+''.join(sections)
    (ROOT/'native_report.html').write_text(page,encoding='utf-8')
    main=ROOT/'index.html';text=main.read_text(encoding='utf-8');marker='<h2>40 张程序留出图上的测量</h2>'
    if '打开三张全新整幅图及松针局部对照' not in text:text=text.replace(marker,'<h2>补充：原生整幅图</h2><p><a href="native_report.html">打开三张全新整幅图及松针局部对照</a>。单独记录，不混入 30 个网格场景通过数。</p>'+marker)
    main.write_text(text,encoding='utf-8')
    # This positive-reference screen was explored but NOT enabled: rescaling can
    # change small-cycle topology and three images do not certify a quality gate.
    explored=ROOT/'accepted_reference_quality.json'
    if explored.exists():
        doc=read(explored);doc.update(enabled=False,rejected_for_automatic_selection=True,reason_not_enabled='Three positive image-level examples and scale-sensitive topology are insufficient to establish an independent pass criterion. No holdout results were reclassified using this exploratory rule.');jsave(explored,doc)
    print('NATIVE_REPORT_READY',str(ROOT/'native_report.html'),flush=True)
