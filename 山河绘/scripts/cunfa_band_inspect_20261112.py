# -*- coding: utf-8 -*-
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_band_v3_20261112';OLD=LAB/'cunfa_roles_v2_20261112'
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'inspection').mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',15)
 # Native-pixel crops. Coordinates refer to v2 static context input, never to
 # a warped or AI-upscaled image. Nearest-neighbour enlargement is review only.
 boxes={'L_149_3':[(12,150,72,290),(58,145,116,245),(115,35,190,150),(61,215,110,350)],'M_1':[(25,35,100,100),(145,45,220,105),(195,110,264,180)],'H_14':[(75,55,170,125),(190,20,285,100),(80,175,145,244),(135,165,200,225)]}
 rows=[]
 for ident,bs in boxes.items():
  z=np.load(OLD/'references'/(ident+'.npz'));im=Image.fromarray(z['image'])
  for i,box in enumerate(bs):
   p=im.crop(box).resize(((box[2]-box[0])*6,(box[3]-box[1])*6),Image.Resampling.NEAREST);board=Image.new('RGB',(p.width+70,p.height+45),'#f4f1e8');board.paste(p,(60,35));d=ImageDraw.Draw(board)
   for x in range((box[0]+9)//10*10,box[2],10):
    px=60+(x-box[0])*6;d.line((px,35,px,board.height),fill='#9aa7a0',width=1);d.text((px,5),str(x),fill='black',font=font)
   for y in range((box[1]+9)//10*10,box[3],10):
    py=35+(y-box[1])*6;d.line((60,py,board.width,py),fill='#9aa7a0',width=1);d.text((5,py),str(y),fill='black',font=font)
   name=ident+'_'+str(i)+'.png';board.save(ROOT/'inspection'/name);rows.append(dict(id=ident,box=box,inspection=name))
 (ROOT/'inspection_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf8');print('INSPECTION_READY',len(rows),flush=True)
if __name__=='__main__':main()
