# -*- coding: utf-8 -*-
"""Experimental native-pixel fine ink evidence. No robot, no baseline mutation."""
import sys
sys.dont_write_bytecode=True
from rock_v15_common import LAB,read,jsave,digest
import split_detail_v4_frozen as graph
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np,cv2,math,argparse
from scipy.ndimage import binary_propagation,distance_transform_edt
from skimage.morphology import skeletonize
BASE=LAB/'stability_brush_preview_20261101'
ROOT=LAB/'native_fine_faint_20261101'
CFG=dict(version='native_fine_faint_r1',sigmas=[.55,.9,1.4,2.3,3.6],noise_floor=.0015,noise_factor=3.2,contrast_floor=.006,minimum_length=8.,minimum_extent=5.,needle_length=4.,needle_contrast=.035,minimum_support=.70)

def field(rgb,cfg=CFG):
    rgbf=rgb.astype(np.float32)/255.;gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.;h,w=gray.shape;y,x=np.mgrid[:h,:w].astype(np.float32)
    hp=abs(gray-cv2.GaussianBlur(gray,(0,0),.55));noise=cv2.GaussianBlur(cv2.medianBlur(hp,5),(0,0),2.);threshold=np.maximum(cfg['contrast_floor'],np.maximum(cfg['noise_floor'],noise)*cfg['noise_factor']);best=np.zeros_like(gray);contrast=best.copy();nx=best.copy();ny=best.copy();width=best.copy();coherence=best.copy()
    for sigma in cfg['sigmas']:
        g=cv2.GaussianBlur(gray,(0,0),sigma);factor=sigma**1.5/4.;a=cv2.Sobel(g,cv2.CV_32F,2,0,ksize=3)*factor;b=cv2.Sobel(g,cv2.CV_32F,0,2,ksize=3)*factor;c=cv2.Sobel(g,cv2.CV_32F,1,1,ksize=3)*factor;delta=np.sqrt((a-b)**2+4*c*c);large=(a+b+delta)/2;small=(a+b-delta)/2;ratio=abs(small)/np.maximum(large,1e-6);value=np.maximum(large,0)*np.exp(-2*ratio**2);angle=.5*np.arctan2(2*c,a-b);ux=np.cos(angle);uy=np.sin(angle);radius=max(1.7,3*sigma)
        left=cv2.remap(gray,x+ux*radius,y+uy*radius,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);right=cv2.remap(gray,x-ux*radius,y-uy*radius,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);shoulder=np.minimum(left,right)-gray
        # A bilateral valley, not just a wash edge. Keep asymmetric edge cases pending.
        value[shoulder<threshold]=0;take=value>best;best[take]=value[take];contrast[take]=shoulder[take];nx[take]=ux[take];ny[take]=uy[take];width[take]=sigma;coherence[take]=1-np.minimum(1,ratio[take])
    left=cv2.remap(best,x+nx,y+ny,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);right=cv2.remap(best,x-nx,y-ny,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);nms=(best>=left)&(best>=right);ridge=skeletonize(nms&(best>.0018)&(contrast>=threshold))
    # Pixel-mask evidence preserves short needle tips that ridge decomposition loses.
    sm=cv2.GaussianBlur(gray,(0,0),.4);local=cv2.GaussianBlur(sm,(0,0),2)-sm;valley=cv2.morphologyEx(sm,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)));weak=(local>np.maximum(.007,noise*2.8))&(valley>.025);strong=(local>np.maximum(.022,noise*5))&(valley>.060);ink=binary_propagation(strong,mask=weak|strong);fine=skeletonize(ink)
    return dict(gray=gray,noise=noise,threshold=threshold,score=best,contrast=contrast,nx=nx,ny=ny,sigma=width,coherence=coherence,ridge=ridge,fine=fine)

def proposals(rgb,cfg=CFG):
    f=field(rgb,cfg);rows=[];selected=np.zeros(rgb.shape[:2],bool);pending=selected.copy()
    for origin,mask in [('fine_dark',f['fine']),('faint_ridge',f['ridge'])]:
        paths,_=graph.pixel_chains(mask)
        for p in paths:
            length=graph.plen(p);q=np.rint(p).astype(int);x,y=q[:,0],q[:,1];extent=float(np.linalg.norm(np.ptp(p,axis=0)));span=float(np.linalg.norm(p[-1]-p[0]));con=f['contrast'][y,x];valid=f['score'][y,x]>.0018;support=float(valid.mean());closed=np.linalg.norm(p[-1]-p[0])<1.1;take=length>=cfg['minimum_length'] and extent>=cfg['minimum_extent'] and not closed and support>=cfg['minimum_support']
            if origin=='fine_dark':take=length>=cfg['needle_length'] and span/max(length,1)>=.65 and not closed
            row=dict(origin=origin,points=p.tolist(),length=length,extent=extent,span_ratio=span/max(length,1),support=support,contrast=float(np.median(con)),accepted=bool(take));rows.append(row)
            if take:selected[y,x]=True
            else:pending[y,x]=True
    return selected,pending,rows,f

def scout(ids):
    ROOT.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    for ident in ids:
        source=BASE/'originals'/(ident+'.png');rgb=np.asarray(Image.open(source).convert('RGB'));selected,pending,rows,f=proposals(rgb);out=ROOT/'scout'/ident;out.mkdir(parents=True,exist_ok=True)
        for name,mask in [('candidates',selected),('pending',pending),('fine_raw',f['fine']),('ridge_raw',f['ridge'])]:Image.fromarray(np.where(mask,0,255).astype(np.uint8)).save(out/(name+'.png'))
        np.savez_compressed(out/'evidence.npz',**f);jsave(out/'proposals.json',dict(config=CFG,source=str(source),source_sha256=digest(source),rows=rows));board=Image.new('RGB',(1500,550),'#f3f1eb');d=ImageDraw.Draw(board)
        for j,(im,title) in enumerate([(Image.fromarray(rgb),'原图'),(Image.open(BASE/'outputs'/ident/'complete/complete.png'),'上轮496提取'),(Image.open(out/'candidates.png'),'原生细线与淡线候选（待质检）')]):im.thumbnail((496,496),Image.Resampling.LANCZOS);d.text((j*500+4,8),ident+' '+title,font=font,fill='black');board.paste(im,(j*500,48))
        board.save(out/'comparison.jpg',quality=96);print('SCOUT_READY',ident,int(selected.sum()),len(rows),flush=True)
        if ident=='T01':
            board=Image.new('RGB',(1881,350),'#f3f1eb');d=ImageDraw.Draw(board);box=(376,25,1003,313)
            for j,(im,title) in enumerate([(Image.fromarray(rgb),'原图松树'),(Image.open(BASE/'outputs'/ident/'complete/complete.png').resize((1254,1254),Image.Resampling.NEAREST),'旧结果放大'),(Image.open(out/'candidates.png'),'原生细线候选')]):d.text((j*627+4,8),title,font=font,fill='black');board.paste(im.convert('RGB').crop(box),(j*627,48))
            board.save(ROOT/'PINE_SCOUT.jpg',quality=96)
    jsave(ROOT/'scout_protocol.json',dict(config=CFG,development_ids=ids,new_training_steps=0,proposal_is_not_ground_truth=True,robot_connected=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',default='T01,T06,T17,T19');a=p.parse_args();scout(a.ids.split(','))
