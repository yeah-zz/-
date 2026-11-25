"""Same-framing original/current comparisons, with matching foreground crops."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image,ImageDraw,ImageFont
from web.catalog import ARTWORKS
from scripts.source_wash_comparison import render

CROPS={0:(0,310,750,1086),1:(0,590,900,1024),2:(0,375,950,887),3:(0,520,860,900)}


def main(numbers=(1,2,3)):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    title_font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',38)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
    for number in numbers:
        folder,title,_,canonical=ARTWORKS[number];path=ROOT/canonical;package=path.parent.parent
        bundle=json.loads(path.read_text(encoding='utf-8'));source=Image.open(package/('松叶简化版原图.png' if number==0 else '参考原图.png')).convert('RGB')
        tr=bundle['workflow']['source_revision']['source_transform'];ratio=tr['scale'];x,y=tr['offset'];w,h=source.size
        preview=render(bundle).crop((x,y,x+ratio*w,y+ratio*h)).resize((w,h),Image.Resampling.LANCZOS)
        board=Image.new('RGB',(2000,1500),'#f7f7f0');draw=ImageDraw.Draw(board)
        draw.text((35,20),title+(' · 原图与当前画稿预览' if number==0 else ' · 原图与最新提线修复预览'),font=title_font,fill='#304c3b')
        count=len(bundle['ink_job']['meta']['ink_execution']['base_strokes'])+len(bundle['workflow']['modules']['color']['strokes'])
        for i,(image,label) in enumerate(((source,'原图'),(preview,'当前画稿 · %d笔'%count))):
            left=35+1000*i;draw.text((left,88),label,font=font,fill='#304c3b')
            pic=image.copy();pic.thumbnail((930,640),Image.Resampling.LANCZOS)
            board.paste(pic,(left+(930-pic.width)//2,138+(640-pic.height)//2))
            draw.text((left,815),label+' · 前景局部放大',font=font,fill='#304c3b')
            pic=image.crop(CROPS[number]);pic.thumbnail((930,525),Image.Resampling.LANCZOS)
            board.paste(pic,(left+(930-pic.width)//2,870+(525-pic.height)//2))
        draw.text((35,1440),'两侧使用相同构图和局部范围；右图来自当前执行 JSON，笔宽、颜色与渗色为估计。',font=small,fill='#687562')
        out=package/'预览与下压标注'/('原图与当前画稿对照.png' if number==0 else '原图与最新提线修复对照.png');board.save(out);print(out,flush=True)


if __name__=='__main__':main()
