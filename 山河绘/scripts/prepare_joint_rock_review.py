import sys
sys.dont_write_bytecode=True
from joint_ink_structure_data import ROOT,LOCAL
from rock_v15_common import read,jsave
import numpy as np
from PIL import Image,ImageDraw,ImageFont
old=read(LOCAL/'baseline/G03_01/paths.json')['paths'];new=read(LOCAL/'candidate/G03_01/paths.json')['paths'][len(old):];chosen=[]
for i,p in sorted(enumerate(new),key=lambda t:-len(t[1]['points'])):
    q=np.asarray(p['points']);x,y=q.mean(0);length=np.linalg.norm(np.diff(q,axis=0),axis=1).sum()
    if not(20<x<320 and 300<y<395 and 6<length<24):continue
    if any(np.linalg.norm(q.mean(0)-np.asarray(v['points']).mean(0))<18 for v in chosen):continue
    chosen.append(dict(id=f'C{len(chosen)+1:02}',image_id='G03_01',points=p['points']))
    if len(chosen)==12:break
board=Image.new('RGB',(1200,720),'#f3f1eb');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);rgb=Image.open(LOCAL/'baseline/G03_01/input.png').convert('RGB')
for k,r in enumerate(chosen):
    p=np.asarray(r['points']);x,y=np.rint(p.mean(0)).astype(int);box=(x-20,y-20,x+21,y+21);patch=rgb.crop(box);over=patch.copy();d=ImageDraw.Draw(over);d.line([(a-box[0],b-box[1]) for a,b in r['points']],fill=(230,30,50),width=1)
    tx=k%4*300;ty=k//4*240;draw.text((tx+4,ty+3),r['id']+' G03 原图/候选',font=font,fill='black');board.paste(patch.resize((145,190)),(tx,ty+30));board.paste(over.resize((145,190)),(tx+150,ty+30))
jsave(ROOT/'additional_local_proposals.json',chosen);board.save(ROOT/'ADDITIONAL_ROCK_REVIEW.jpg',quality=96);print(len(chosen))
