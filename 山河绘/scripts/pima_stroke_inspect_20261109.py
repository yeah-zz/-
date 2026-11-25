# -*- coding: utf-8 -*-
"""High temporal resolution inspection of real brush contact candidates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,json
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'pima_single_stroke_20261109';PREV=LAB/'pima_demonstration_20261109'
def sheet(lesson,start,end,step=.5,box=(250,100,650,470)):
 out=ROOT/'inspection';out.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(PREV/'sources'/str(lesson)/'video.mp4'));times=np.arange(start,end,step);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
 w=240;h=round((box[3]-box[1])*w/(box[2]-box[0]));board=Image.new('RGB',(w*5,(h+26)*((len(times)+4)//5)),'#eeeae1');d=ImageDraw.Draw(board)
 for i,t in enumerate(times):
  cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000);ok,bgr=cap.read()
  if not ok:continue
  im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop(box);x=(i%5)*w;y=(i//5)*(h+26);board.paste(im.resize((w,h)),(x,y+26));d.text((x+5,y+2),f'{lesson}课 {t:.2f}s',font=font,fill='#222')
 cap.release();p=out/f'{lesson}_{start}_{end}.jpg';board.save(p,quality=95);print(p,flush=True)
if __name__=='__main__':
 sheet(37,144,164)
 sheet(37,200,220)
 sheet(36,466,486,box=(380,65,750,360))
