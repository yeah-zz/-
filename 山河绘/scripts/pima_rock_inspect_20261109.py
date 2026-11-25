# -*- coding: utf-8 -*-
"""Sequential decoding for review; no temporal differences across camera cuts."""
import os,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure();os.environ['OPENCV_FFMPEG_LOGLEVEL']='-8'
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'pima_one_rock_cycle_20261109'
SOURCE=LAB/'pima_tutorial_search_20261109'/'short_pima.mp4'
def main():
 ROOT.mkdir(exist_ok=True);out=ROOT/'frames';out.mkdir(exist_ok=True)
 cap=cv2.VideoCapture(str(SOURCE));fps=cap.get(cv2.CAP_PROP_FPS);n=0;kept=[]
 # Four images per second retained, decoded sequentially to avoid fragmented
 # MP4 random seeking errors. These are observations, not training examples.
 while True:
  ok,bgr=cap.read()
  if not ok:break
  t=n/fps
  if n%round(fps/4)==0:
   name=f'{t:08.3f}.jpg';cv2.imwrite(str(out/name),bgr,[cv2.IMWRITE_JPEG_QUALITY,95]);kept.append(dict(t=t,file=name))
  n+=1
 cap.release();(ROOT/'frames.json').write_text(json.dumps(dict(fps=fps,decoded=n,frames=kept),indent=2),encoding='utf8')
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
 for a,b,step in [(0,65,5),(60,110,2),(110,160,2),(160,210,2),(210,310,5)]:
  times=np.arange(a,b,step);board=Image.new('RGB',(1200,245*((len(times)+3)//4)),'#eeeae1');draw=ImageDraw.Draw(board)
  for i,t in enumerate(times):
   f=min(kept,key=lambda z:abs(z['t']-t));im=Image.open(out/f['file']).resize((300,215));x=(i%4)*300;y=(i//4)*245;board.paste(im,(x,y+28));draw.text((x+4,y+3),f"{f['t']:.2f} s",font=font,fill='#222')
  board.save(ROOT/f'overview_{a}_{b}.jpg',quality=95)
 print(json.dumps(dict(frames=len(kept),fps=fps,duration=n/fps)),flush=True)
def fine():
 out=ROOT/'fine';out.mkdir(exist_ok=True);cap=cv2.VideoCapture(str(SOURCE));fps=cap.get(cv2.CAP_PROP_FPS);n=0;kept=[]
 while n/fps<145.2:
  ok,bgr=cap.read()
  if not ok:break
  t=n/fps
  if (50<=t<=60 or 131<=t<=145) and n%3==0:
   name=f'{t:08.3f}.png';cv2.imwrite(str(out/name),bgr);kept.append(dict(t=t,file=name))
  n+=1
 cap.release();(ROOT/'fine.json').write_text(json.dumps(kept,indent=2),encoding='utf8')
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
 for a,b in [(52,53.6),(53.6,55.2),(55.5,57.1),(57.1,58.7),(131.5,133.1),(133.1,134.7),(138,139.6),(139.6,141.2),(142,143.6)]:
  frames=[f for f in kept if a<=f['t']<b];board=Image.new('RGB',(1600,225*((len(frames)+3)//4)),'#eeeae1');draw=ImageDraw.Draw(board)
  for i,f in enumerate(frames):
   im=Image.open(out/f['file']).crop((35,210,485,480)).resize((400,195));x=i%4*400;y=i//4*225;board.paste(im,(x,y+28));draw.text((x+4,y+3),f"{f['t']:.3f}s",font=font,fill='#222')
  board.save(ROOT/f'fine_{a}_{b}.jpg',quality=96)
 print('FINE',len(kept),flush=True)
def dense():
 out=ROOT/'dense';out.mkdir(exist_ok=True);cap=cv2.VideoCapture(str(SOURCE));fps=cap.get(cv2.CAP_PROP_FPS);n=0;kept=[]
 while n/fps<60:
  ok,bgr=cap.read()
  if not ok:break
  t=n/fps
  if 52.05<=t<=54.05 or 55.4<=t<=57.2 or 57.9<=t<=59.9:
   name=f'{t:08.3f}.png';cv2.imwrite(str(out/name),bgr);kept.append(dict(t=t,file=name))
  n+=1
 cap.release();(ROOT/'dense.json').write_text(json.dumps(kept,indent=2),encoding='utf8')
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
 for a,b in [(52.15,52.65),(52.65,53.15),(53.15,53.65),(55.55,56.05),(56.05,56.55),(56.55,57.05),(58,58.5),(58.5,59)]:
  rows=[z for z in kept if a<=z['t']<b];board=Image.new('RGB',(1600,880),'#eeeae1');d=ImageDraw.Draw(board)
  for i,z in enumerate(rows):
   im=Image.open(out/z['file']).crop((290,245,500,430)).resize((400,190));x=i%4*400;y=i//4*220;board.paste(im,(x,y+28));d.text((x+4,y+3),f"{z['t']:.3f}s",font=font,fill='#222')
  board.save(ROOT/f'dense_{a}_{b}.jpg',quality=96)
 print('DENSE',len(kept),flush=True)
if __name__=='__main__':
 if '--dense' in sys.argv:dense()
 elif '--fine' in sys.argv:fine()
 else:main()
