# -*- coding: utf-8 -*-
"""Separate fine fan recovery from long faint-curve candidates. Development."""
import sys
sys.dont_write_bytecode=True
from native_fine_faint_20261101 import *
from scipy.ndimage import gaussian_filter
CFG2=dict(version='separate_fan_faint_r1',fan_window=49,fan_count=16,fan_dispersion=.23,fan_min_length=3.5,fan_max_length=25,fan_span=.70,fan_dark=.65,fan_blue_limit=.025,fan_min_area=200,faint_min_length=36.,faint_min_extent=24.,faint_span_ratio=.35,faint_min_support=.82,faint_contrast=.013,faint_paper_margin=2.4)

def fan_region(rgb,f):
    gray=f['gray'];paths,_=graph.pixel_chains(f['fine']);h,w=gray.shape;counts=np.zeros((h,w),np.float32);vx=counts.copy();vy=counts.copy()
    for p in paths:
        length=graph.plen(p);span=float(np.linalg.norm(p[-1]-p[0]));q=np.rint(p).astype(int)
        if not CFG2['fan_min_length']<=length<=CFG2['fan_max_length'] or span/max(length,1)<CFG2['fan_span'] or float(gray[q[:,1],q[:,0]].mean())>CFG2['fan_dark']:continue
        x,y=np.rint(p.mean(0)).astype(int);t=graph.unit(p[-1]-p[0]);counts[y,x]+=1;vx[y,x]+=t[0]**2-t[1]**2;vy[y,x]+=2*t[0]*t[1]
    k=CFG2['fan_window'];counts=cv2.boxFilter(counts,-1,(k,k),normalize=False);vx=cv2.boxFilter(vx,-1,(k,k),normalize=False);vy=cv2.boxFilter(vy,-1,(k,k),normalize=False);dispersion=1-np.sqrt(vx*vx+vy*vy)/np.maximum(counts,1);blur=cv2.GaussianBlur(rgb.astype(np.float32)/255.,(0,0),5);blue=blur[:,:,2]-blur[:,:,0];seed=(counts>=CFG2['fan_count'])&(dispersion>=CFG2['fan_dispersion'])&(blue<CFG2['fan_blue_limit']);region=cv2.morphologyEx(seed.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)));region=cv2.dilate(region,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))>0
    n,cc,stats,_=cv2.connectedComponentsWithStats(region.astype(np.uint8),8);mask=np.zeros_like(region)
    for i,(x,y,bw,bh,area) in enumerate(stats[1:],1):
        if area>=CFG2['fan_min_area'] and bw/max(bh,1)>=1.05:mask|=cc==i
    return mask,dict(short_segments_density=counts,dispersion=dispersion)

def select(rgb,base,rows,f):
    fan,fanfield=fan_region(rgb,f);fine=np.zeros_like(base);faint=fine.copy();pending=fine.copy();decisions=[];s=max(base.shape)/496.
    # Retain complete native fine-ink chains only in automatically detected fans.
    for origin,mask in [('needle',f['fine']),('faint',f['ridge'])]:
        paths,_=graph.pixel_chains(mask)
        for p in paths:
            q=np.rint(p).astype(int);x,y=q[:,0],q[:,1];length=graph.plen(p);extent=float(np.linalg.norm(np.ptp(p,axis=0)));span=float(np.linalg.norm(p[-1]-p[0]));closed=span<1.1;support=float((f['contrast'][y,x]>=f['threshold'][y,x]).mean());contrast=float(np.median(f['contrast'][y,x]));reason='pending'
            if origin=='needle':take=bool(fan[y,x].mean()>=.85 and length>=2.5 and not closed);reason='native_ink_inside_automatic_fan' if take else reason
            else:
                take=bool(length>=CFG2['faint_min_length'] and extent>=CFG2['faint_min_extent'] and span/max(length,1)>=CFG2['faint_span_ratio'] and not closed and support>=CFG2['faint_min_support'] and contrast>=CFG2['faint_contrast'] and float(fan[y,x].mean())<.1);reason='long_bilateral_ink_valley' if take else reason
            if take:
                if origin=='needle':fine[y,x]=True
                else:faint[y,x]=True
            else:pending[y,x]=True
            if take:decisions.append(dict(origin=origin,points=p.tolist(),length=length,contrast=contrast,support=support,reason=reason,stroke_identity_verified=False))
    # Preserve old result outside fans. In fans replace coarse aliases, not overlay.
    result=(base&~fan)|fine
    distance=distance_transform_edt(~result);new=faint&(distance>1.6);result|=new
    return result,dict(fan=fan,fine=fine,faint=faint,new_faint=new,pending=pending),decisions

def run(ids):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for ident in ids:
        rgb=np.asarray(Image.open(BASE/'originals'/(ident+'.png')).convert('RGB'));h,w=rgb.shape[:2];doc=read(BASE/'outputs'/ident/'complete/complete_paths.json');p=[np.asarray(r['points'],np.float32)*[w/496,h/496] for r in doc['paths']];base=graph.raster((h,w),p);sc=ROOT/'scout'/ident
        if (sc/'evidence.npz').exists():z=np.load(sc/'evidence.npz');f={k:z[k] for k in z.files};rows=read(sc/'proposals.json')['rows']
        else:_,_,rows,f=proposals(rgb)
        result,masks,decisions=select(rgb,base,rows,f);out=ROOT/'development'/ident;out.mkdir(parents=True,exist_ok=True)
        for name,mask in [('lines',result),('baseline',base)]+list(masks.items()):Image.fromarray(np.where(mask,0,255).astype(np.uint8)).save(out/(name+'.png'))
        changes=np.repeat(np.where(base,0,255).astype(np.uint8)[:,:,None],3,2);changes[result&~base]=[25,130,190];changes[base&~result]=[210,60,60];Image.fromarray(changes).save(out/'changes.png');jsave(out/'decisions.json',decisions);jsave(out/'statistics.json',dict(added_pixels=int((result&~base).sum()),removed_pixels=int((base&~result).sum()),fine_pixels=int(masks['fine'].sum()),new_faint_pixels=int(masks['new_faint'].sum()),automatic_fan_pixels=int(masks['fan'].sum()),config=CFG2,new_training_steps=0,robot_connected=False))
        board=Image.new('RGB',(1984,550),'#f3f1eb');d=ImageDraw.Draw(board)
        for j,(im,title) in enumerate([(Image.fromarray(rgb),'原图'),(Image.fromarray(np.where(base,0,255).astype(np.uint8)),'旧提取（缩放轨迹）'),(Image.open(out/'lines.png'),'分流修正候选'),(Image.open(out/'changes.png'),'蓝新增 / 红替换旧片段')]):im=im.convert('RGB');im.thumbnail((496,496),Image.Resampling.LANCZOS);d.text((j*496+4,8),ident+' '+title,font=font,fill='black');board.paste(im,(j*496,48))
        board.save(out/'comparison.jpg',quality=96)
        if ident=='T01':
            board=Image.new('RGB',(1881,350),'#f3f1eb');d=ImageDraw.Draw(board);box=(376,25,1003,313)
            for j,(im,title) in enumerate([(Image.fromarray(rgb),'原图'),(Image.fromarray(np.where(base,0,255).astype(np.uint8)),'上轮轨迹'),(Image.open(out/'lines.png'),'松针分流候选')]):d.text((j*627+4,8),title,font=font,fill='black');board.paste(im.convert('RGB').crop(box),(j*627,48))
            board.save(ROOT/'PINE_DEVELOPMENT.jpg',quality=96)
        print('DEVELOPMENT_READY',ident,int(masks['fine'].sum()),int(masks['new_faint'].sum()),flush=True)
    jsave(ROOT/'development_config.json',CFG2)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',default='T01,T06,T17,T19');a=p.parse_args();run(a.ids.split(','))
