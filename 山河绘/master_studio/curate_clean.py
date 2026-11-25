# -*- coding: utf-8 -*-
"""第二轮人工清洗：只从已筛出的山水作品和原有标注 patch 出拼版。"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np, math

ROOT=Path(__file__).resolve().parent
CORPUS=ROOT/"corpus"
OUT=ROOT/"data"/"curation2"
MAP={"黄公望":"pima","王蒙":"pima","吴镇":"pima","沈周":"pima","文徵明":"pima","倪瓒":"zhedai","石涛":"fupi","唐寅":"fupi","仇英":"fupi","龚贤":"midian"}
BASE={"huang_gongwang":"pima","ni_zan":"zhedai","shi_tao":"fupi","gong_xian":"midian"}

def q(im):
 g=np.asarray(im.convert("L"),dtype=np.float32); dark=(g<105).mean()
 edge=(np.abs(np.diff(g,axis=0)).mean()+np.abs(np.diff(g,axis=1)).mean())/255
 if not (.01<dark<.28) or edge<.02 or edge>.12 or g.std()<12: return None
 return edge

def collect(cls):
 rows=[]
 for aid,c in BASE.items():
  if c!=cls: continue
  for p in sorted((CORPUS/aid/"patches").glob("*.png")):
   try:
    im=Image.open(p); s=q(im)
    if s is not None: rows.append(dict(path=str(p),artist=aid,score=s))
   except: pass
 for d in sorted((CORPUS/"ltfc_landscape").iterdir()):
  if not d.is_dir(): continue
  if MAP.get(d.name)!=cls: continue
  for p in sorted((d/"patches").glob("*.png")):
   try:
    im=Image.open(p); s=q(im)
    if s is not None: rows.append(dict(path=str(p),artist=d.name,score=s))
   except: pass
 return rows

def font(sz):
 for f in ("C:/Windows/Fonts/STXINGKA.TTF","C:/Windows/Fonts/STKAITI.TTF"):
  try:return ImageFont.truetype(f,sz)
  except:pass
 return ImageFont.load_default()

def sheet(cls,rows,size=200,cols=5,per=25):
 d=OUT/cls; d.mkdir(parents=True,exist_ok=True)
 for si in range(0,len(rows),per):
  ch=rows[si:si+per]; W=cols*(size+10)+10; H=((len(ch)+cols-1)//cols)*(size+28)+20
  cv=Image.new("RGB",(W,H),(238,232,218)); dr=ImageDraw.Draw(cv)
  for k,r in enumerate(ch):
   x=10+(k%cols)*(size+10); y=10+(k//cols)*(size+28)
   try: cv.paste(Image.open(r["path"]).convert("L").resize((size,size),Image.LANCZOS).convert("RGB"),(x,y))
   except: continue
   dr.rectangle([x,y,x+size,y+size],outline=(150,145,125))
   dr.text((x+4,y+2),str(si+k),font=font(20),fill=(160,40,30))
  cv.save(d/("sheet-%02d.jpg"%(si//per)),quality=92)
 (d/"manifest.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
 print(cls,len(rows),(len(rows)+per-1)//per)

if __name__=="__main__":
 for c in ("pima","zhedai","fupi","midian"):
  sheet(c,collect(c))