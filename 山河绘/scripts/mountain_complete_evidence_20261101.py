# -*- coding: utf-8 -*-
"""Image-supported complete candidates; sparse rendering is a separate stage."""
import sys
sys.dont_write_bytecode=True
from mountain_recovery_catalog_20261101 import ROOT as PREVIOUS,catalog
from mountain_structure_v4_20261101 import ROOT as STRUCTURE,LAB,v4,whole_candidates,read,jsave,digest
from v4_local_repair_candidates import evidence,samples,graph_cycles
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import gaussian_filter1d,distance_transform_edt,binary_propagation
from skimage.morphology import skeletonize
import numpy as np,cv2,argparse,math
ROOT=LAB/'mountain_complete_lines_20261101'
FEATURES=['log_length','log_extent','support_adaptive','support012','support006','contrast_median','contrast_q20','rgb_support','gray_mean','gray_std','bend_mean','bend_q90','total_turn','net_turn','compact_loop','hairpin','field_support','field_mean','width_median','width_std']
CFG=dict(version='complete_evidence_r2',sigmas=[.7,1.2,2.,3.2],noise_floor=.0013,noise_ceiling=.006,weak_contrast_floor=.0035,weak_contrast_noise=2.8,ridge_score_floor=.0015,ridge_score_noise=.7,min_candidate_length=7.,min_new_ridge_length=9.,min_extent=5.,minimum_supported_fraction=.65,rgb_supported_fraction=.5,max_bend_q90=1.0,minimum_faint_component_length=7.,maximum_gap=12.,bridge_support_fraction=.88,bridge_max_unsupported_run=1,model_accept_floor=.8,model_uncertainty=.20,optional_sparse_count_per_mass=6,complete_candidate_count_limit=None)

def evidence_field(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.;s=max(gray.shape)/496.;h,w=gray.shape;yy,xx=np.mgrid[:h,:w].astype(np.float32)
    smooth=cv2.GaussianBlur(gray,(0,0),max(.45,.45*s));hp=abs(gray-smooth);noise=cv2.medianBlur(hp,5);noise=cv2.GaussianBlur(noise,(0,0),max(1.,2*s));noise=np.clip(noise*1.48,CFG['noise_floor'],CFG['noise_ceiling']);threshold=np.maximum(CFG['weak_contrast_floor'],CFG['weak_contrast_noise']*noise)
    score=np.zeros_like(gray);normal=np.zeros((*gray.shape,2),np.float32);shoulder=np.zeros_like(gray);rgbcontrast=np.zeros_like(gray);rgbf=rgb.astype(np.float32)/255.
    for sigma0 in CFG['sigmas']:
        sigma=max(.55,sigma0*s);g=cv2.GaussianBlur(gray,(0,0),sigma);factor=sigma**1.5/4;a=cv2.Sobel(g,cv2.CV_32F,2,0,ksize=3)*factor;b=cv2.Sobel(g,cv2.CV_32F,0,2,ksize=3)*factor;c=cv2.Sobel(g,cv2.CV_32F,1,1,ksize=3)*factor
        delta=np.sqrt((a-b)**2+4*c*c);large=(a+b+delta)*.5;small=(a+b-delta)*.5;value=np.maximum(large,0)*np.exp(-.5*(small/np.maximum(large*.5,1e-5))**2);angle=.5*np.arctan2(2*c,a-b);nx=np.cos(angle);ny=np.sin(angle);radius=max(1.5*s,3*sigma)
        left=cv2.remap(gray,xx+nx*radius,yy+ny*radius,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);right=cv2.remap(gray,xx-nx*radius,yy-ny*radius,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);contrast=np.minimum(left,right)-gray
        lc=cv2.remap(rgbf,xx+nx*radius,yy+ny*radius,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);rc=cv2.remap(rgbf,xx-nx*radius,yy-ny*radius,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);neutral=np.minimum(lc,rc).min(2)-rgbf.min(2)
        value[(contrast<threshold)|(neutral<threshold*.3)]=0;take=value>score;score[take]=value[take];normal[take,0]=nx[take];normal[take,1]=ny[take];shoulder[take]=contrast[take];rgbcontrast[take]=neutral[take]
    nx,ny=normal[:,:,0],normal[:,:,1];left=cv2.remap(score,xx+nx,yy+ny,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);right=cv2.remap(score,xx-nx,yy-ny,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);nms=(score>=left)&(score>=right);weak=(score>np.maximum(CFG['ridge_score_floor'],noise*CFG['ridge_score_noise']))&(shoulder>=threshold);ridge=skeletonize(weak&nms)
    return dict(gray=gray,threshold=threshold,noise=noise,score=score,normal=normal,shoulder=shoulder,rgbcontrast=rgbcontrast,weak=weak,ridge=ridge)

def arc_profile(rgb,p,field):
    p=np.asarray(p,np.float32);s=max(rgb.shape[:2])/496.;delta=np.linalg.norm(np.diff(p,axis=0),axis=1);arc=np.r_[0,np.cumsum(delta)];length=float(arc[-1]);step=max(.7,s);t=np.linspace(0,length,max(4,int(length/step)+1));smooth=np.column_stack([np.interp(t,arc,p[:,j]) for j in [0,1]]).astype(np.float32)
    smooth=gaussian_filter1d(smooth,max(.55,1.2*s/step),axis=0,mode='nearest');tan=np.gradient(smooth,axis=0);tan/=np.maximum(np.linalg.norm(tan,axis=1,keepdims=True),1e-6);n=np.column_stack([-tan[:,1],tan[:,0]]);gray=field['gray'];center=samples(gray,smooth);contrast=np.full(len(t),-1.,np.float32);rgbmin=np.min(rgb.astype(np.float32)/255.,axis=2);rgbcenter=samples(rgbmin,smooth);neutral=np.full(len(t),-1.,np.float32)
    for radius in [1.5,2.5,4.,6.]:
        a=samples(gray,smooth+n*radius*s);b=samples(gray,smooth-n*radius*s);contrast=np.maximum(contrast,np.minimum(a,b)-center);a=samples(rgbmin,smooth+n*radius*s);b=samples(rgbmin,smooth-n*radius*s);neutral=np.maximum(neutral,np.minimum(a,b)-rgbcenter)
    threshold=samples(field['threshold'],smooth);angle=np.unwrap(np.arctan2(tan[:,1],tan[:,0]));bend=np.abs(np.diff(angle));q=np.rint(p).astype(int);y,x=q[:,1],q[:,0];extent=float(np.linalg.norm(np.ptp(p,axis=0)))/s;span=float(np.linalg.norm(p[-1]-p[0]))/s;closed=span<2 and extent<12;hairpin=0.
    narrow_return=0.
    if len(smooth)>10:
        dist=np.linalg.norm(smooth[:,None]-smooth[None,:],axis=2);far=np.abs(t[:,None]-t[None,:])>8*s;opposed=(tan@tan.T)<-.7;hairpin=float(((dist<3*s)&far&opposed).any(1).mean())
        narrow_return=float(((dist<7*s)&(np.abs(t[:,None]-t[None,:])>12*s)&opposed).any(1).mean())
    widths=distance_transform_edt(field['weak'])*2 if 'width' not in field else field['width'];field['width']=widths
    vals=[math.log1p(length/s),math.log1p(extent),float((contrast>=threshold).mean()),float((contrast>=.012).mean()),float((contrast>=.006).mean()),float(np.median(contrast)),float(np.quantile(contrast,.2)),float((neutral>=threshold*.4).mean()),float(center.mean()),float(center.std()),float(bend.mean()) if len(bend) else 0.,float(np.quantile(bend,.9)) if len(bend) else 0.,float(bend.sum()),float(abs(angle[-1]-angle[0])),float(closed),hairpin,float(field['weak'][y,x].mean()),float(field['score'][y,x].mean()),float(np.median(widths[y,x]))/s,float(widths[y,x].std())/s]
    features=dict(zip(FEATURES,vals));features['narrow_return']=narrow_return;return features

def build(rgb,base,previous,routing,original_rows):
    s=max(base.shape)/496.;field=evidence_field(rgb);mass=routing['mass'];area=cv2.dilate(mass.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(3,int(7*s)|1),)*2))>0;area&=~routing['protect'];rows=[]
    for r in original_rows:
        p=np.asarray(r['points'],np.float32);q=np.rint(p).astype(int);novel=~previous[q[:,1],q[:,0]]
        if not novel.any() or v4.plen(p)<4*s:continue
        if not area[q[novel,1],q[novel,0]].all():continue
        rows.append(dict(key='v4_'+str(r['id']),old_candidate=r['id'],origin='existing_v4_candidate',points=p.tolist(),features=arc_profile(rgb,p,field)))
    # A source detector proposal is not a training label. Its unreviewed areas
    # remain unknown; no component is made positive just for being detected.
    ridge=field['ridge']&area;chains,_=whole_candidates(ridge,s)
    for i,r in enumerate(chains):
        p=r['points'];length=v4.plen(p)
        if length<CFG['minimum_faint_component_length']*s:continue
        q=np.rint(p).astype(int)
        if float(previous[q[:,1],q[:,0]].mean())>.9:continue
        rows.append(dict(key='ridge_'+str(i),old_candidate=None,origin='adaptive_multiscale_ridge',points=p.tolist(),features=arc_profile(rgb,p,field)))
    return rows,field,dict(area=area,protect=routing['protect'],mass=mass)

def load_image(ident):
    rgb,base,_,_,oldrows=catalog(ident);previous=np.asarray(Image.open(PREVIOUS/'outputs'/ident/'lines.png'))<128;z=np.load(STRUCTURE/'outputs'/ident/'routing_masks.npz');routing={k:z[k] for k in z.files};return rgb,base,previous,routing,oldrows

def prepare(ident,make_cards=False):
    rgb,base,previous,routing,oldrows=load_image(ident);rows,field,masks=build(rgb,base,previous,routing,oldrows);out=ROOT/'prepared'/ident;out.mkdir(parents=True,exist_ok=True);jsave(out/'candidates.json',dict(id=ident,rows=rows,source_sha256=digest(STRUCTURE/'outputs'/ident/'input.png'),previous_sha256=digest(PREVIOUS/'outputs'/ident/'paths.json'),config=CFG));np.savez_compressed(out/'field.npz',**field,**masks);Image.fromarray(np.where(field['ridge']&masks['area'],0,255).astype(np.uint8)).save(out/'ridge_proposals.png')
    if make_cards:review_cards(ident,rgb,rows)
    print('COMPLETE_PREPARED',ident,len(rows),flush=True);return rows

def review_cards(ident,rgb,rows):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);old=[r for r in rows if r['origin']=='existing_v4_candidate'];ridge=[r for r in rows if r['origin']=='adaptive_multiscale_ridge'];known={r['candidate'] for r in read(PREVIOUS/'partial_references.json')['references'] if r['image']==ident and r['label'] is not None}
    curves=sorted([r for r in old if r['old_candidate'] not in known and r['features']['total_turn']>1.0],key=lambda r:-(r['features']['total_turn']*math.expm1(r['features']['log_length'])))[:8]
    faint=sorted(ridge,key=lambda r:r['features']['contrast_median'])[:8];strong=sorted(ridge,key=lambda r:-r['features']['log_length'])[:8];chosen=[];seen=set()
    for r in curves+faint+strong:
        if r['key'] not in seen:chosen.append(r);seen.add(r['key'])
    folder=ROOT/'review_cards';folder.mkdir(exist_ok=True)
    for start in range(0,len(chosen),4):
        board=Image.new('RGB',(1400,940),'#f3f1eb');draw=ImageDraw.Draw(board)
        for j,row in enumerate(chosen[start:start+4]):
            p=np.asarray(row['points']);a=np.maximum(0,np.floor(p.min(0)-16)).astype(int);b=np.minimum(np.array(rgb.shape[1::-1]),np.ceil(p.max(0)+17)).astype(int);x,y=a;xx,yy=b;orig=Image.fromarray(rgb).crop((x,y,xx,yy));overlay=orig.copy();d=ImageDraw.Draw(overlay);d.line([tuple(q-a) for q in p],fill=(255,30,30),width=1);ox=(j%2)*700;oy=(j//2)*470;f=row['features'];draw.text((ox+4,oy+5),f'{ident} {row["key"]} L={math.expm1(f["log_length"]):.1f} turn={f["total_turn"]:.1f}',font=font,fill='black');draw.text((ox+4,oy+29),f'profile={f["support_adaptive"]:.2f} contrast={f["contrast_median"]:.3f}',font=font,fill='black')
            factor=min(340/orig.width,390/orig.height,3.5)
            for k,im in enumerate([orig,overlay]):im=im.resize((round(orig.width*factor),round(orig.height*factor)),Image.Resampling.NEAREST);board.paste(im,(ox+k*350,oy+58))
        board.save(folder/f'{ident}_{start//4:02}.jpg',quality=96)
    jsave(folder/f'{ident}_manifest.json',[r['key'] for r in chosen])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',default='G01_01,G02_01,G03_02,G04_01,G05_01');p.add_argument('--cards',action='store_true');a=p.parse_args()
    for ident in a.ids.split(','):prepare(ident,a.cards)
