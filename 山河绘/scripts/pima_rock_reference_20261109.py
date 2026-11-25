# -*- coding: utf-8 -*-
"""New-work video observations. Tip candidates are never model inputs."""
import os,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from pima_rock_inspect_20261109 import ROOT,LAB
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
FRAMES=json.loads((ROOT/'fine.json').read_text('utf8'))
def frame(t):
 z=min(FRAMES,key=lambda z:abs(z['t']-t));return np.array(Image.open(ROOT/'fine'/z['file']).convert('RGB'))
def tip(rgb,clean):
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float);bg=cv2.cvtColor(clean,cv2.COLOR_RGB2GRAY).astype(float)
 m=((bg-gray)>55).astype(np.uint8);m[:235]=0;m[:,:200]=0
 m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));n,lab,stats,_=cv2.connectedComponentsWithStats(m,8);c=[]
 for k in range(1,n):
  x,y,w,h,area=stats[k]
  if area<150 or x+w<450 or w<60:continue
  yy,xx=np.where(lab==k);idx=xx.argmin();c.append((x,np.array([xx[idx],yy[idx]],np.float32)))
 return min(c,key=lambda a:a[0])[1] if c else None
def main():
 clean=frame(54.25);tracks=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
 for a,b in [(52,53.6),(53.6,55.2),(55.5,57.1),(57.1,58.7),(131.5,133.1),(133.1,134.7),(138,139.6),(139.6,141.2),(142,143.6)]:
  rows=[z for z in FRAMES if a<=z['t']<b];board=Image.new('RGB',(1600,225*((len(rows)+3)//4)),'#eeeae1');draw=ImageDraw.Draw(board)
  for i,z in enumerate(rows):
   rgb=frame(z['t']);p=tip(rgb,clean);im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
   if p is not None:x,y=p;d.ellipse((x-4,y-4,x+4,y+4),outline='red',width=2)
   im=im.crop((220,250,570,480)).resize((400,195));x=i%4*400;y=i//4*225;board.paste(im,(x,y+28));draw.text((x+4,y+3),f"{z['t']:.3f}s "+(str(p.tolist()) if p is not None else '-'),font=font,fill='#222')
   tracks.append(dict(t=z['t'],point=p.tolist() if p is not None else None))
  board.save(ROOT/f'tip_{a}_{b}.jpg',quality=96)
 (ROOT/'tip_candidates.json').write_text(json.dumps(tracks,indent=2),encoding='utf8');print('TIP',len(tracks),flush=True)
if __name__=='__main__':main()
