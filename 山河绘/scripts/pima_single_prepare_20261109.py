# -*- coding: utf-8 -*-
"""Prepare assistant-reviewed single-stroke observations, not expert gold."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from pima_stroke_inspect_20261109 import ROOT,PREV
from pima_tip_candidates_20261109 import read
import cv2,numpy as np,json
from PIL import Image,ImageDraw,ImageFont

# Bounds are visual contact candidates with explicitly limited precision.
# Short arcs and the original multi-contact lesson-36 windows remain excluded.
EVENTS=[
 dict(id='S01',before=[144.5,144.8,145.1],after=[148.7,149.,149.3],time=[146.,148.3],anchors=[[379,217],[361,235],[350,255],[340,272],[332,289],[324,312],[316,345],[310,368],[302,405],[295,450]],split='train'),
 dict(id='S02',before=[210.,210.1,210.2],after=[212.15,212.25,212.35],time=[210.55,212.1],anchors=[[365,296],[351,312],[338,346],[328,380],[321,415],[316,451]],split='train'),
 dict(id='S03',before=[212.3,212.4,212.5],after=[214.12,214.2,214.28],time=[212.75,214.1],anchors=[[370,298],[360,305],[349,331],[343,350],[336,380],[331,408],[326,451]],split='same_rock_development',revision='0.05s frame review removed initial hovering point at y283; contact candidate begins around y298'),
 dict(id='S04',before=[214.2,214.28,214.35],after=[215.9,216.,216.1],time=[214.5,215.9],anchors=[[359,295],[348,320],[338,345],[328,377],[322,410],[318,449]],split='train'),
 dict(id='S05',before=[217.35,217.42,217.5],after=[218.4,218.5,218.6],time=[217.6,218.4],anchors=[[340,347],[333,374],[319,395],[314,425],[311,451]],split='same_rock_development'),
]
BOX=[275,195,415,470]
def ink(im):
 gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype(np.float32);bg=cv2.GaussianBlur(gray,(0,0),7)
 return np.clip((bg-gray-1)/55,0,1)
def main():
 out=ROOT/'single_reference';out.mkdir(parents=True,exist_ok=True);cap=cv2.VideoCapture(str(PREV/'sources/37/video.mp4'));font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
 board=Image.new('RGB',(1000,len(EVENTS)*330),'#eeeae1');d=ImageDraw.Draw(board);meta=[]
 for row,e in enumerate(EVENTS):
  before=np.max(np.stack([read(cap,t) for t in e['before']]),axis=0);after=np.max(np.stack([read(cap,t) for t in e['after']]),axis=0)
  delta=np.clip(ink(after)-ink(before),0,1);x0,y0,x1,y1=BOX
  for name,rgb in [('before',before),('after',after)]:Image.fromarray(rgb).save(out/(e['id']+'_'+name+'.png'))
  np.savez_compressed(out/(e['id']+'.npz'),before=before,after=after,delta=delta)
  panel=[Image.fromarray(before).crop(BOX),Image.fromarray(after).crop(BOX),Image.fromarray(np.uint8((1-delta)*255)).convert('RGB').crop(BOX)]
  overlay=Image.fromarray(after.copy());dr=ImageDraw.Draw(overlay);dr.line([tuple(p) for p in e['anchors']],fill=(220,50,50),width=2);panel.append(overlay.crop(BOX))
  for col,im in enumerate(panel):
   im=im.resize((150,294));xx=col*250+40;yy=row*330+27;board.paste(im,(xx,yy));d.text((col*250+8,row*330+3),[e['id']+' 前帧','后帧','新增墨迹差分','视频笔尖引导·待核查'][col],font=font,fill='#222')
  meta.append(dict(**e,lesson=37,source_family='liupengkai_qiushan_same_painting',source_page='https://www.ganjingworld.com/video/1h5vi07sks9KFNZ72hjMX9VrP1331c',box=BOX,assistant_reviewed=False,expert_verified=False,contact_time_uncertainty_seconds=.2,physical_contact_sensor=False))
 cap.release();board.save(ROOT/'single_observations.jpg',quality=95);(ROOT/'single_candidates.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8');print('SINGLE_OBSERVATIONS',len(meta),flush=True)
if __name__=='__main__':main()
