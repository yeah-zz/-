# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from rock_v15_common import LAB,read,jsave,digest
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np,cv2
ROOT=LAB/'faint_center_direction_20261101'
BASE=LAB/'stability_brush_preview_20261101'
PRIOR=LAB/'native_fine_faint_20261101'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
WINDOWS={
'T01':[(360,460),(505,390),(285,750)],'T06':[(790,390),(440,280),(770,650)],'T17':[(220,360),(0,400),(390,600)],'T19':[(280,330),(690,380),(250,450)],
'T03':[(250,420),(500,100),(820,430)],'T12':[(815,620),(320,560),(760,900)],'T09':[(230,580),(800,340),(480,420)],'T16':[(615,300),(180,365),(580,440)]}

def patches():
    records=[];folder=ROOT/'diagnostics';folder.mkdir(parents=True,exist_ok=True)
    for ident,windows in WINDOWS.items():
        source=BASE/'originals'/(ident+'.png');rgb=Image.open(source).convert('RGB');board=Image.new('RGB',(1536,560),'#f3f1eb');d=ImageDraw.Draw(board)
        for k,(x,y) in enumerate(windows):
            key=ident+'_'+str(k+1);box=(x,y,x+256,y+256);im=rgb.crop(box);im.save(folder/(key+'.png'));shown=im.resize((512,512),Image.Resampling.NEAREST);dd=ImageDraw.Draw(shown)
            for z in range(0,256,32):dd.line([(z*2,0),(z*2,511)],fill='#ceaaaa');dd.text((z*2+2,2),str(z),font=FONT,fill='red');dd.line([(0,z*2),(511,z*2)],fill='#ceaaaa');dd.text((2,z*2+20),str(z),font=FONT,fill='red')
            d.text((k*512+4,4),key+' source='+str((x,y)),font=FONT,fill='black');board.paste(shown,(k*512,40));records.append(dict(id=key,image=ident,box=list(box),path=str(folder/(key+'.png')),source=str(source),source_sha256=digest(source),split='train' if ident in ['T01','T06','T17','T19'] else 'calibration' if ident in ['T03','T12'] else 'old_regression_test',manual_inference_roi=False))
        board.save(folder/(ident+'_review.jpg'),quality=97)
    jsave(ROOT/'diagnostic_manifest.json',dict(records=records,total_patches=24,parent_images=8,unmarked_not_negative=True,windows_for_diagnosis_and_partial_training_only=True,whole_image_split=True,new_blind_test=False))

def overview():
    ROOT.mkdir(parents=True,exist_ok=True)
    for k,ids in enumerate([['T03','T12'],['T09','T16']]):
        board=Image.new('RGB',(1500,800),'#f3f1eb');d=ImageDraw.Draw(board)
        for j,ident in enumerate(ids):
            im=Image.open(BASE/'originals'/(ident+'.png')).convert('RGB');dd=ImageDraw.Draw(im)
            for x in range(0,1254,100):dd.line([(x,0),(x,1253)],fill='#cf8f8f');dd.text((x+2,4),str(x),font=FONT,fill='red')
            for y in range(100,1254,100):dd.line([(0,y),(1253,y)],fill='#cf8f8f');dd.text((4,y+2),str(y),font=FONT,fill='red')
            im=im.resize((750,750),Image.Resampling.LANCZOS);d.text((j*750+4,8),ident,font=FONT,fill='black');board.paste(im,(j*750,40))
        board.save(ROOT/(f'overview_{k}.jpg'),quality=97)
if __name__=='__main__':overview();patches()
