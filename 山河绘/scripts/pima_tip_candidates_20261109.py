# -*- coding: utf-8 -*-
"""Observable 2-D brush-tip candidates, with retained frames for review."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from pima_stroke_inspect_20261109 import ROOT,PREV
import cv2,numpy as np,json
from PIL import Image,ImageDraw,ImageFont

EVENTS=[
 dict(id='37_long_01',lesson=37,start=146.0,end=148.6,before=[144.5,145.0,145.5],after=[148.8,149.1,149.4],box=[285,220,430,460]),
 dict(id='36_long_01',lesson=36,start=473.3,end=475.8,before=[471.,471.3,471.6],after=[476.,476.2,476.4],box=[420,90,570,280]),
 dict(id='36_long_02',lesson=36,start=477.,end=478.4,before=[476.,476.2,476.4],after=[479.,479.2,479.4],box=[420,90,570,280]),
 dict(id='37_arc_01',lesson=37,start=151.,end=151.8,before=[149.8,150.,150.2],after=[151.8,152.,152.2],box=[350,205,430,290]),
 dict(id='37_arc_02',lesson=37,start=153.6,end=155.2,before=[153.,153.2,153.4],after=[155.6,155.8,156.],box=[350,205,440,300]),
 dict(id='37_long_02',lesson=37,start=210.4,end=212.6,before=[209.6,209.8,210.],after=[212.6,212.8,213.],box=[290,300,400,465]),
 dict(id='37_long_03',lesson=37,start=213.,end=214.6,before=[212.6,212.8,213.],after=[214.4,214.6,214.8],box=[290,300,400,465]),
 dict(id='37_long_04',lesson=37,start=215.,end=216.6,before=[214.4,214.6,214.8],after=[216.5,216.7,216.9],box=[290,300,410,465]),
 dict(id='37_long_05',lesson=37,start=217.,end=219.2,before=[216.5,216.7,216.9],after=[219.2,219.4,219.6],box=[290,300,410,465]),
 dict(id='36_split_01',lesson=36,start=473.2,end=474.0,before=[472.,472.2,472.4],after=[474.,474.1,474.2],box=[420,90,570,280]),
 dict(id='36_split_02',lesson=36,start=474.2,end=475.2,before=[474.,474.1,474.2],after=[475.1,475.2,475.3],box=[420,90,570,280]),
]
def read(cap,t):
 cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000);ok,bgr=cap.read()
 if not ok:raise RuntimeError(t)
 return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
def tip(rgb,baseline,box):
 gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);bg=cv2.cvtColor(baseline,cv2.COLOR_RGB2GRAY)
 resid=((bg.astype(float)-gray)>40).astype(np.uint8);resid[:50]=0;resid[:,:60]=0
 n,labels,stats,cent=cv2.connectedComponentsWithStats(resid,8);chosen=None
 # Brush body extends to the right from the painted rock; detect its thin body,
 # not the largest dark region or artist's hand.
 for k in range(1,n):
  x,y,w,h,area=stats[k]
  if area<150 or w<60 or w/max(h,1)<.6:continue
  yy,xx=np.where(labels==k);points=np.column_stack([xx,yy]).astype(np.float32)
  center=points.mean(0);_,sv,vt=np.linalg.svd(points-center,full_matrices=False);axis=vt[0]
  if axis[0]<0:axis=-axis
  if sv[0]/max(sv[1],1)<5:continue
  projection=(points-center)@axis;point=points[np.argmin(projection)]
  x0,y0,x1,y1=box;dist=np.linalg.norm(point-np.clip(point,[x0,y0],[x1,y1]))
  score=dist+max(0,8-sv[0]/max(sv[1],1))*10
  if chosen is None or score<chosen[0]:chosen=(score,point.tolist(),float(sv[0]/max(sv[1],1)))
 return chosen
def main():
 out=ROOT/'tip_candidates';out.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15);manifest=[]
 for e in EVENTS:
  cap=cv2.VideoCapture(str(PREV/'sources'/str(e['lesson'])/'video.mp4'));before=np.max(np.stack([read(cap,t) for t in e['before']]),axis=0);after=np.max(np.stack([read(cap,t) for t in e['after']]),axis=0)
  Image.fromarray(before).save(out/(e['id']+'_before.png'));Image.fromarray(after).save(out/(e['id']+'_after.png'))
  times=np.arange(e['start'],e['end']+.01,.2);board=Image.new('RGB',(1200,230*((len(times)+3)//4)),'#eeeae1');d=ImageDraw.Draw(board);tracks=[]
  for i,t in enumerate(times):
   rgb=read(cap,t);candidate=tip(rgb,before,e['box']);im=Image.fromarray(rgb);dd=ImageDraw.Draw(im)
   if candidate:
    x,y=candidate[1];dd.ellipse((x-5,y-5,x+5,y+5),outline='red',width=2)
   im=im.crop((250,60,750,460)).resize((300,200));bx=(i%4)*300;by=(i//4)*230;board.paste(im,(bx,by+30));d.text((bx+3,by+4),f'{t:.2f}秒 '+('候选' if candidate else '未检测'),font=font,fill='#222');tracks.append(dict(t=float(t),candidate=candidate))
  cap.release();board.save(out/(e['id']+'_track.jpg'),quality=95);manifest.append(dict(**e,tracks=tracks,status='unreviewed_2d_candidates',physical_contact_verified=False))
 (ROOT/'tip_candidates.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print('TIP_CANDIDATES',len(manifest),flush=True)
if __name__=='__main__':main()
