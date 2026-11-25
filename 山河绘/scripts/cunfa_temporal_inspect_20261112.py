# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_temporal_roles_20261112';VIDEO=LAB/'pima_tutorial_search_20261109/short_pima.mp4'
def main():
 (ROOT/'frames').mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(VIDEO));font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);records=[]
 times=[0,1,2,4,9.8,19.8,29.8,39.8,47,48,49,49.7,51.4,51.6,51.8,54.1,54.3,54.5,55.2,55.4,57.2,57.4,58,60,61]
 # This fragmented H264 source gives broken references when seeking. Decode
 # sequentially and select actual frame indices, preserving timing uncertainty.
 fps=cap.get(cv2.CAP_PROP_FPS);wanted={int(round(t*fps)):t for t in times};index=0
 while index<=max(wanted):
  ok,bgr=cap.read();assert ok,index
  if index in wanted:
   t=wanted[index];p=ROOT/'frames'/f'{t:06.2f}.png';Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(p);records.append(dict(t=t,actual_t=index/fps,frame_index=index,path=str(p),size=[bgr.shape[1],bgr.shape[0]]))
  index+=1
 cap.release();canvas=Image.new('RGB',(1440,((len(records)+3)//4)*250),'#f4f1e8');d=ImageDraw.Draw(canvas)
 for i,r in enumerate(records):
  image=Image.open(r['path']);image.thumbnail((350,210));x=i%4*360;y=i//4*250;canvas.paste(image,(x,y+30));d.text((x+8,y),str(r['t'])+'s',font=font,fill='black')
 canvas.save(ROOT/'inspection.jpg',quality=94);(ROOT/'frame_manifest.json').write_text(json.dumps(dict(records=records,video=str(VIDEO),video_sha256=hashlib.sha256(VIDEO.read_bytes()).hexdigest(),source_url='https://www.ganjingworld.com/video/1gc2lo9n27j5ZShwAja69iQ3N12d1c',source_title='畫幾塊石頭，示範短披麻皴',uploader='米小鹿画画',expert_verified=False),ensure_ascii=False,indent=2),encoding='utf8');print('FRAMES',len(records),flush=True)
if __name__=='__main__':main()
