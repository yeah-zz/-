import sys
sys.dont_write_bytecode=True
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from audit_v4_local_repairs import OUT
f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
for ident,box in [('G01_01',(65,75,230,145)),('G02_01',(85,45,250,220)),('G04_01',(0,0,496,496)),('G05_01',(0,0,496,496))]:
    im=Image.open(OUT/'baseline'/ident/'input.png').convert('RGB');d=ImageDraw.Draw(im)
    x0,y0,x1,y1=box
    for x in range((x0//10+1)*10,x1,10):
        d.line([(x,y0),(x,y1)],fill=(180,110,110),width=1);d.text((x+1,y0),str(x),font=f,fill=(130,0,0))
    for y in range((y0//10+1)*10,y1,10):
        d.line([(x0,y),(x1,y)],fill=(180,110,110),width=1);d.text((x0,y),str(y),font=f,fill=(130,0,0))
    im=im.crop(box);im.resize((im.width*3,im.height*3)).save(OUT/(ident+'_reference_grid.png'))
