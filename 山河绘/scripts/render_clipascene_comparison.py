"""Render exported curves without any brush styling or invented results."""
from pathlib import Path
import sys
import json
import math
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
STORAGE=Path(r"E:\Codex_Workspace_Data\CLIPascene_20261021")
sys.path.insert(0,str(STORAGE/"environment"/"deps"))
sys.dont_write_bytecode=True
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from svgpathtools import svg2paths2

OUT=ROOT/"offline_previews"/"clipascene_20261021"
FONT="C:/Windows/Fonts/msyh.ttc"
def text(im,xy,value,size=24,fill=(44,52,47)):
    ImageDraw.Draw(im).text(xy,value,font=ImageFont.truetype(FONT,size),fill=fill)

def vector_preview(svg,dest,width=2):
    paths,attrs,svg_attrs=svg2paths2(str(svg))
    sw=float(svg_attrs["width"].replace("px",""))
    S=2
    im=Image.new("RGB",(1110*S,1110*S),"white")
    draw=ImageDraw.Draw(im)
    scale=1110/sw
    length=0
    for path in paths:
        length+=path.length()*scale
        for seg in path:
            n=max(12,min(3000,int(math.ceil(seg.length()*scale))))
            pts=[seg.point(float(t))*scale for t in np.linspace(0,1,n)]
            draw.line([(p.real*S,p.imag*S) for p in pts],fill=(25,25,25),width=width*S)
    im=im.crop((0,180*S,1110*S,930*S)).resize((1110,750),Image.Resampling.LANCZOS)
    im.save(dest)
    return dict(svg=str(svg),paths=len(paths),full_square_length_px=round(length,1),preview=str(dest))

def main():
    metrics=[]
    for directory in sorted(OUT.glob("layer*")):
        svg=directory/"best_iter.svg"
        if svg.exists():
            metrics.append(vector_preview(svg,directory/"line_preview.png"))
    source=Image.open(OUT/"input_landscape.png").convert("RGB")
    baseline=Image.open(OUT/"existing_extractor.png").convert("RGB")
    existing=json.loads((OUT/"existing_strokes.json").read_text(encoding="utf-8"))
    rows=[("参考画：裁切并去除题字",source,"输入采用等比例留白；未做前后景分割或背景补绘。"),
          (f"现有提线器：{len(existing)} 条路径",baseline,"保留原图墨线细节；这里统一用等宽黑线显示。")]
    for item in metrics:
        directory=Path(item["svg"]).parent
        manifest=json.loads((directory/"experiment_manifest.json").read_text(encoding="utf-8")) if (directory/"experiment_manifest.json").exists() else {}
        layer=manifest.get("layer",directory.name)
        suffix="简化结果" if manifest.get("simplification_ratio") else f"特征层 {layer}"
        rows.append((f"CLIPascene {suffix}：{item['paths']} 条曲线",Image.open(item["preview"]),
                     f"{manifest.get('iterations','?')} 次迭代；{manifest.get('initial_paths','?')} 条初始曲线；真实算法输出。"))
    pages=[]
    if len(rows)==5:
        im=Image.new("RGB",(2000,1510),(245,244,239))
        text(im,(60,28),"山河绘 / CLIPascene 三种配置实际运行对照",40)
        text(im,(62,90),"等宽黑线展示路径几何。整图模式、单张样本；各配置均为 128 条初始曲线、501 次迭代。",24,(100,108,100))
        for j,(title,image,caption) in enumerate(rows[:2]):
            x=60+j*975;y=155
            text(im,(x,y),title,29)
            im.paste(image.resize((905,611),Image.Resampling.LANCZOS),(x,y+51))
            text(im,(x,y+678),caption,20,(90,99,92))
        for j,(title,image,caption) in enumerate(rows[2:]):
            x=60+j*640;y=920
            text(im,(x,y),title,23)
            im.paste(image.resize((600,405),Image.Resampling.LANCZOS),(x,y+44))
            text(im,(x,y+465),"501 次迭代 / 同一输入与随机种子",20,(90,99,92))
        text(im,(62,1457),"现有提线器与 CLIPascene 的笔数、工作分辨率及表示方式不同，本图为探索性对照。",22,(90,99,92))
        im.save(OUT/"comparison_overview.png")
        pages.append(str(OUT/"comparison_overview.png"))
    for page_index in range(0,len(rows),4):
        selected=rows[page_index:page_index+4]
        im=Image.new("RGB",(2000,1680),(245,244,239))
        text(im,(60,28),"山河绘  /  CLIPascene 实际运行对照",40)
        text(im,(62,90),"统一线宽展示路径几何，不添加毛笔纹理；首次小样不代表算法在全部山水画上的表现。",24,(100,108,100))
        for j,(title,image,caption) in enumerate(selected):
            x=60+(j%2)*975;y=155+(j//2)*725
            text(im,(x,y),title,29)
            im.paste(image.resize((905,611),Image.Resampling.LANCZOS),(x,y+51))
            text(im,(x,y+678),caption,20,(90,99,92))
        text(im,(62,1635),"官方代码 + Windows/PyTorch 兼容修正；整图模式。SVG 与运行参数随图保存。",22,(90,99,92))
        name=f"comparison_{page_index//4+1}.png"
        im.save(OUT/name)
        pages.append(str(OUT/name))
    (OUT/"comparison_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(pages=pages,runs=metrics),ensure_ascii=True))

if __name__=="__main__":main()
