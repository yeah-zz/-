# -*- coding: utf-8 -*-
"""Report untouched extraction outputs for new imagegen landscape samples."""
import os
from pathlib import Path

ROOT = Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\generated_simple_landscapes_20261028')
for key, sub in {'TEMP':'temp','TMP':'temp','PIP_CACHE_DIR':'cache/pip','TORCH_HOME':'cache/torch','HF_HOME':'cache/huggingface','XDG_CACHE_HOME':'cache','CUDA_CACHE_PATH':'cache/cuda'}.items():
    os.environ[key] = str(ROOT / sub)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import json
import hashlib
import shutil
import html
from PIL import Image, ImageDraw, ImageFont

SCENES = [('01_rocks','山石组合'), ('02_pine','松树山坡'), ('03_pavilion','亭子与瀑布')]
FONT = r'C:\Windows\Fonts\msyh.ttc'
def font(size):
    return ImageFont.truetype(FONT, size)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    prompt_source = Path(__file__).with_name('generated_landscape_prompts_20261028.json')
    shutil.copy2(prompt_source, ROOT / 'generation_prompts.json')
    cell, gap, margin, top, rowh = 640, 20, 24, 132, 724
    board = Image.new('RGB', (margin*2 + cell*3 + gap*2, top+rowh*3+60), '#f3f1eb')
    draw = ImageDraw.Draw(board)
    draw.text((margin,16), '新生成山水画：原图 → 模型提取中心线 → 原图叠加', font=font(32), fill='#202020')
    draw.text((margin,65), '固定模型实际推理｜统一 640 × 640｜未人工补线｜蓝线为提取路径', font=font(23), fill='#555555')
    for col,label in enumerate(['原图（模型输入）','黑白中心线（统一细线显示）','叠加检查（蓝线）']):
        draw.text((margin+col*(cell+gap),104),label,font=font(24),fill='#202020')
    manifest = {'method':'imagegen originals followed by frozen model inference', 'training_on_these_samples':False, 'manual_path_edits':False, 'robot_connected':False, 'accuracy_not_measured_without_reference':True, 'scenes':[]}
    sections=[]
    for row,(sid,name) in enumerate(SCENES):
        folder=ROOT/'results'/sid
        data=json.loads((folder/'paths.json').read_text(encoding='utf-8'))
        meta=data['meta']
        assert meta['new_training_steps']==0 and meta['blocked_annotation_opens']==0
        assert data['reference_labels_used'] is False and data['actually_drawn'] is False
        assert meta['mode']=='center_guarded' and '5060' in meta['cuda_device']
        assert meta['source_path_count']==meta['retained_path_count']==len(data['paths'])
        y=top+row*rowh
        draw.text((margin,y+8),f'{row+1:02d} {name}  |  {len(data["paths"])} 条输出路径（不代表真实笔画数）', font=font(24),fill='#202020')
        files=['input.png','lines.png','overlay.png']
        for col,file in enumerate(files):
            with Image.open(folder/file) as im:
                im=im.convert('RGB')
                assert im.size==(640,640)
                board.paste(im,(margin+col*(cell+gap),y+52))
        manifest['scenes'].append({'id':sid,'name':name,'meta':meta,'output_path_count':len(data['paths']), 'original_sha256':hashlib.sha256((ROOT/'originals'/(sid+'.png')).read_bytes()).hexdigest()})
        figures=''.join(f'<figure><figcaption>{caption}</figcaption><a href="results/{sid}/{file}"><img src="results/{sid}/{file}" alt="{html.escape(name+caption)}"></a></figure>' for file,caption in zip(files,['原图（模型输入）','模型中心线','蓝线叠加']))
        sections.append(f'<section><h2>{row+1}. {name}</h2><p>{len(data["paths"])} 条输出路径；路径数量不等于画家真实笔画数。</p><div class="panels">{figures}</div><p><a href="originals/{sid}.png">高清生成原图</a> · <a href="results/{sid}/lines.svg">SVG</a> · <a href="results/{sid}/paths.json">路径与运行记录</a></p></section>')
    draw.text((margin,top+rowh*3+10),'离线测试；线宽仅作中心线显示，不是机械臂实画粗细。无人工真值，未计算准确率。',font=font(23),fill='#555555')
    board.save(ROOT/'comparison.png')
    board.save(ROOT/'comparison.jpg',quality=94,subsampling=0)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>新生成山水画提取对照</title><style>body{font:16px system-ui,"Microsoft YaHei",sans-serif;background:#f3f1eb;color:#222;margin:24px;line-height:1.7}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}figure{margin:0}img{width:100%;background:white}figcaption{font-weight:bold}section{border-top:1px solid #ccc;margin-top:28px;padding-top:8px}a{color:#075ea2}@media(max-width:800px){.panels{grid-template-columns:1fr}}</style><h1>新生成山水画：实际模型提取</h1><p>本批 3 张原图由内置 imagegen 生成，未用于本轮训练。提取采用当前固定的 v31 时期骨干与关系模型，加保守中心修正（center_guarded），在 RTX 5060 Laptop GPU 推理；统一输入 640 × 640，结果未人工补线、删线或挑选。<a href="generation_prompts.json">完整生成提示词</a> · <a href="run_manifest.json">运行记录</a></p><p>黑白图统一显示中心线；线宽不表示机械臂落笔粗细。没有逐笔人工真值，不能据此报告准确率；这组图也不能替代真实古画测试。</p>'''+''.join(sections)+'''<section><h2>阅读结果</h2><p>观察山体主轮廓是否跟随原图、内部皴线是否被合并或遗漏，以及树冠、亭顶等密集区域是否变成碎线或闭环。此页展示离线中心线路径，不表示已识别皴法类别、还原粗细浓淡或完成机械臂实画。</p></section></html>'''
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    (ROOT/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with Image.open(ROOT/'comparison.png') as im: im.verify()
    print(json.dumps({'comparison':str(ROOT/'comparison.png'),'gallery':str(ROOT/'index.html'),'path_counts':{x['id']:x['output_path_count'] for x in manifest['scenes']},'checks_passed':True},ensure_ascii=False))

if __name__=='__main__':
    main()
