# -*- coding: utf-8 -*-
"""Sequential raw frames for mixed-role references; no oracle crop at inference."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_roles_v2_20261112'
SOURCES=[('liu37',LAB/'pima_demonstration_20261109/sources/37/video.mp4',[0,20,40,60,80,100,110,120,130,140,144.8,149.3,160,170,180], 'https://www.ganjingworld.com/video/1h5vi07sks9KFNZ72hjMX9VrP1331c'),('howto',LAB/'pima_independent_20261109/new_howto/video.mp4',[0,1,3,6,8,9,10,12,14,16,20,24,30,36,42,48,56,64,72,84,89], 'https://www.ganjingworld.com/video/1gc5ch0t6uo7BVAQGaZzKg2Rp1cb1c')]
def main():
 ROOT.mkdir(exist_ok=True);manifest=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
 for name,path,times,url in SOURCES:
  folder=ROOT/'frames'/name;folder.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);wanted={round(t*fps):t for t in times};rows=[];index=0
  while index<=max(wanted):
   ok,bgr=cap.read();assert ok,(name,index)
   if index in wanted:
    t=wanted[index];p=folder/f'{t:07.2f}.png';Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(p);rows.append(dict(t=t,index=index,actual_seconds=index/fps,path=str(p),size=[bgr.shape[1],bgr.shape[0]]))
   index+=1
  cap.release();board=Image.new('RGB',(1280,((len(rows)+3)//4)*210),'#f1eee5');d=ImageDraw.Draw(board)
  for i,r in enumerate(rows):
   im=Image.open(r['path']);im.thumbnail((318,180));x=i%4*320;y=i//4*210;board.paste(im,(x,y+28));d.text((x+4,y+4),f'{name} {r["t"]}s',font=font,fill='#243e32')
  board.save(ROOT/(name+'_inspection.jpg'),quality=94);manifest.append(dict(id=name,video=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),url=url,frames=rows,source_work_seen_in_prior_project=True,artist_identity_verified=name!='howto',uploader_is_not_artist_identity=True))
 (ROOT/'frame_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print('RAW_FRAMES_READY',[(r['id'],len(r['frames'])) for r in manifest],flush=True)
if __name__=='__main__':main()
