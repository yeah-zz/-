# -*- coding: utf-8 -*-
"""Same-image report, full evidence and failed real-image cases included."""
from split_detail_structure import *
from rock_v15_common import read
from PIL import ImageDraw,ImageFont
import shutil,xml.etree.ElementTree as ET

ROOT=LAB/'split_detail_structure_20261028'/'v4'
OLD=LAB/'generated_simple_landscapes_20261028'
SCENES=[('01_rocks','山石组合'),('02_pine','松树山坡'),('03_pavilion','亭子瀑布')]
def font(size):return ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',size)

def make():
    board=Image.new('RGB',(1944,2260),'#f3f1eb');d=ImageDraw.Draw(board)
    d.text((20,14),'同图对照：原图 → 上一轮原图像素基线 → 本轮分层提取',font=font(30),fill='black')
    d.text((20,62),'原分辨率处理；统一细线显示；灰色待定线另存。几何分区不代表已识别树木/皴法类别。',font=font(22),fill='#444444')
    for i,label in enumerate(['原图','上一轮像素骨架','本轮分层结果']):d.text((20+i*642,104),label,font=font(24),fill='black')
    stats=[];sections=[]
    for row,(sid,label) in enumerate(SCENES):
        folder=ROOT/sid;previous=OLD/'loss_diagnosis'/'source_pixel_baseline'/sid/'skeleton.png';shutil.copy2(previous,folder/'previous_pixel.png')
        y=145+row*690;d.text((20,y),label,font=font(24),fill='black')
        for col,path in enumerate([folder/'input.png',previous,folder/'lines.png']):
            with Image.open(path) as im:
                im=im.convert('RGB');im.thumbnail((620,620),Image.Resampling.LANCZOS);board.paste(im,(20+col*642,y+42))
        q=read(folder/'paths.json');s=read(folder/'statistics.json')
        assert q['reference_labels_used'] is False and q['ready_for_robot'] is False and q['meta']['new_training_steps']==0
        assert q['config']['version']==CFG['version']
        assert digest(q['meta']['source'])==q['meta']['sha256']
        svg=ET.parse(folder/'lines.svg');assert len(svg.findall('.//{http://www.w3.org/2000/svg}polyline'))==len(q['paths'])
        stats.append(dict(id=sid,statistics=s,source_sha256=q['meta']['sha256']))
        imgs=''.join(f'<figure><figcaption>{caption}</figcaption><a href="{sid}/{file}.png"><img loading="lazy" src="{sid}/{file}.png"></a></figure>' for file,caption in [('input','原图'),('previous_pixel','上一轮像素基线'),('lines','本轮分层中心线'),('all_evidence','灰色：待定原图骨架'),('overlay','蓝：长线；紫：细节'),('dense_regions','绿色：自动几何细节区')])
        sections.append(f'<section><h2>{label}</h2><p><a href="{sid}/index.html">单图完整结果</a> · <a href="{sid}/paths.json">路径及续接记录</a> · <a href="{sid}/lines.svg">SVG</a></p><div class="grid">{imgs}</div></section>')
    board.save(ROOT/'full_comparison.jpg',quality=96,subsampling=0)
    # Fixed display crops only; these coordinates are never passed to extraction.
    crops=[('02_pine','松冠局部',(130,240,705,565)),('01_rocks','山石内部局部',(330,450,905,920))]
    cropboard=Image.new('RGB',(1765,1050),'#f3f1eb');dd=ImageDraw.Draw(cropboard)
    dd.text((15,10),'局部对照：原图 / 上一轮像素基线 / 本轮分层',font=font(28),fill='black')
    y=60
    for sid,label,box in crops:
        dd.text((15,y),label,font=font(23),fill='black');y+=38
        for col,file in enumerate(['input.png','previous_pixel.png','lines.png']):
            with Image.open(ROOT/sid/file) as im:cropboard.paste(im.convert('RGB').crop(box),(15+col*585,y))
        y+=box[3]-box[1]+25
    dd.text((15,y),'本轮为传统图像处理流程改动，未训练新权重；断线、交汇和部分网状结构仍未解决。',font=font(21),fill='#444444')
    cropboard.save(ROOT/'local_comparison.jpg',quality=97,subsampling=0)
    real=[]
    for sid in ['R04','H02','N11']:
        folder=ROOT/'real_regression'/sid;s=read(folder/'statistics.json');q=read(folder/'paths.json');real.append(dict(id=sid,statistics=s,source_sha256=q['meta']['sha256']))
        assert q['config']['version']==CFG['version'] and q['meta']['manual_roi_used'] is False
        imgs=''.join(f'<figure><figcaption>{caption}</figcaption><a href="real_regression/{sid}/{file}.png"><img loading="lazy" src="real_regression/{sid}/{file}.png"></a></figure>' for file,caption in [('input','旧真实输入'),('lines','同参数实验输出'),('all_evidence','包含待定骨架')])
        sections.append(f'<section><h2>真实旧图回归：{sid}</h2><p>这些是反复使用的开发图，不是新的盲测。仍可见断笔和局部杂线，不能据此宣布优于旧模型。</p><div class="grid">{imgs}</div></section>')
    lock=read(LAB/'stable_line_extractor_20261028'/'baseline_lock.json')
    locked=[digest(r['path'])==r['sha256'] for r in lock['files']];assert all(locked)
    checks=read(ROOT/'behavior_checks.json');assert checks['all_passed']
    protocol=dict(config=CFG,model_weights_updated=False,new_training_steps=0,method='source grayscale valleys, fine-detail skeleton retention, multiscale Hessian ink-ridge centers, intact source curves, conservative endpoint joins',device='CPU; no neural optimization',semantic_tree_rock_classification=False,hand_roi_used_at_inference=False,samples=stats,real_regression=real,behavior_checks=checks,old_baseline_hashes_unchanged=all(locked),human_accuracy_not_measured=True,crop_boxes_display_only=[dict(id=sid,label=label,box=list(box)) for sid,label,box in crops],robot_connected=False,ready_for_robot=False,decision='experimental_only; do not replace stable model by this qualitative development comparison')
    jsave(ROOT/'report.json',protocol)
    shutil.copy2(Path(__file__).with_name('split_detail_structure.py'),ROOT/'source_snapshot.py')
    shutil.copy2(Path(__file__).with_name('check_split_detail_structure.py'),ROOT/'checks_snapshot.py')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>短线细节与长线结构分层</title><style>body{font:16px system-ui;background:#f3f1eb;color:#222;margin:24px;line-height:1.7}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}figure{margin:0}img{width:100%;background:white}section{margin-top:28px;border-top:1px solid #ccc}a{color:#075ea2}p{max-width:1100px}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><h1>密集短线与长皴线分层：本轮实验</h1><p>松树细线延续上一轮原图像素方法；长线结合墨带中心与完整原图骨架，只有方向相符、原图有墨迹支持且无歧义时尝试续接。自动分区依据几何与灰度，不是已训练的树木/山石分类器，也没有识别皴法类别。本轮在 CPU 做图像处理，未更新模型权重。</p><p>本轮 3 张生成图、3 张旧真实图使用同一固定参数。松针细节保留，部分孤立碎片移入待定层；山石内部仍有断笔和密集墨块网状线。旧真实图也有轮廓断裂，尚未达到替换当前稳定入口的条件。无完整逐笔真值，不报告准确率。页面中的长线、细节和路径数量均不等于真实画家笔顺。</p><p><a href="local_comparison.jpg">局部对照原尺寸</a> · <a href="full_comparison.jpg">三张完整对照</a> · <a href="report.json">参数、行为检查与运行记录</a></p>'''+''.join(sections)+'''<section><h2>当前边界</h2><p>待定层完整保存原图骨架未被当前筛选线覆盖的像素。它既含真实短墨线，也含纸纹和碎墨，因此灰色不能直接作为噪声或背景标签。密集区域内部仍可能有闭环，原图证据续接也不等于真实单笔判断。所有结果为离线像素文件，不含机械臂压力或落笔控制。</p></section></html>'''
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    print('REPORT_READY',str(ROOT/'index.html'),'old baseline hashes unchanged',all(locked),flush=True)

if __name__=='__main__':make()
