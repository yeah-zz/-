# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from rock_v15_common import LAB
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'paper_center_refinement_20261029';ROOT.mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
for ident in ['R04','N11']:
    p=LAB/'split_detail_structure_20261028/v4/real_regression'/ident/'input.png';im=Image.open(p).convert('RGB');w,h=im.size;layer=Image.new('RGBA',im.size);draw=ImageDraw.Draw(layer)
    for x in range(0,w,25):draw.line([(x,0),(x,h)],fill=(190,20,20,45))
    for y in range(0,h,25):draw.line([(0,y),(w,y)],fill=(190,20,20,45))
    im=Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB').resize((w*2,h*2));canvas=Image.new('RGB',(w*2+40,h*2+35),'white');canvas.paste(im,(40,35));d=ImageDraw.Draw(canvas)
    for x in range(0,w,25):d.text((40+x*2,5),str(x),font=font,fill='black')
    for y in range(0,h,25):d.text((0,35+y*2),str(y),font=font,fill='black')
    canvas.save(ROOT/(ident+'_coordinate_grid.png'))
print(ROOT)
