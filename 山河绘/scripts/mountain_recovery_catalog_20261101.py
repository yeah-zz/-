# -*- coding: utf-8 -*-
"""Whole-candidate catalog for local, non-expert RGB reference review."""
import sys
sys.dont_write_bytecode=True
from mountain_structure_v4_20261101 import LAB,ROOT as OLD,CFG,whole_candidates,v4,read,jsave,digest
from v4_local_repair_candidates import evidence
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np,cv2,argparse,math
ROOT=LAB/'mountain_mainline_recovery_20261101'
FEATURES=['length','span_ratio','support012','support006','contrast_median','contrast_q20','gray_mean','gray_std','turn_mean','turn_q90','extent','novel_fraction','density','inside_fraction','role_long']

def catalog(ident):
    folder=OLD/'outputs'/ident;rgb=np.asarray(Image.open(folder/'input.png').convert('RGB'));h,w=rgb.shape[:2];scale=max(h,w)/496.
    base_folder=LAB/'v4_local_repairs_20261029/candidate'/ident if ident!='N02' else LAB/'simple_landscape_specialist_20261029/round2/native_outputs/N02/v4'
    doc=read(base_folder/'paths.json');old=np.asarray(Image.open(folder/'lines.png'))<128;base=np.asarray(Image.open(folder/'baseline.png'))<128
    routing=np.load(folder/'routing_masks.npz');interior=routing['interior'];protect=routing['protect']
    joined,_=whole_candidates(base,scale);candidates=[dict(points=r['points'] if isinstance(r,dict) else r,origin='original_v4_candidate',role=r.get('role','unknown') if isinstance(r,dict) else 'unknown') for r in doc['paths']]
    for row in joined:
        p=row['points'];q=np.rint(p).astype(int)
        if v4.plen(p)>=40*scale and interior[q[:,1],q[:,0]].all():candidates.append(dict(points=p.tolist(),origin='junction_supported_candidate',role='long_structure'))
    decisions=read(folder/'decisions.json');assert len(candidates)==len(decisions)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.;density=cv2.boxFilter(base.astype(np.float32),-1,(15,15));out=[]
    for i,(row,d) in enumerate(zip(candidates,decisions)):
        p=np.asarray(row['points'],np.float32);q=np.rint(p).astype(int);length=v4.plen(p);span=float(np.linalg.norm(p[-1]-p[0]));contrast,center=evidence(gray,p,scale)
        tang=p[2:]-p[:-2] if len(p)>2 else p[-1:]-p[:1];tang=tang/np.maximum(np.linalg.norm(tang,axis=1,keepdims=True),1e-6)
        turns=np.arccos(np.clip((tang[1:]*tang[:-1]).sum(1),-1,1)) if len(tang)>1 else np.zeros(1)
        y,x=q[:,1],q[:,0];novel=float((~old[y,x]).mean());extent=float(np.linalg.norm(np.ptp(p,axis=0)))/scale
        vals=[length/scale,span/max(length,1e-6),float((contrast>=.012).mean()),float((contrast>=.006).mean()),float(np.median(contrast)),float(np.quantile(contrast,.2)),float(center.mean()),float(center.std()),float(turns.mean()),float(np.quantile(turns,.9)),extent,novel,float(density[y,x].mean()),float(interior[y,x].mean()),float(row['role']=='long_structure')]
        out.append(dict(id=i,points=p.tolist(),origin=row['origin'],role=row['role'],old_decision=d,features=dict(zip(FEATURES,vals)),protected=bool(protect[y,x].any()),fully_interior=bool(interior[y,x].all()),old_keep=d['decision']=='keep'))
    return rgb,base,old,dict(interior=interior,protect=protect),out

def cards(ident,rows,rgb,diagnostic=False):
    # Balanced length strata, deterministic; selection does not use any labels.
    eligible=[r for r in rows if r['fully_interior'] and not r['protected'] and r['features']['novel_fraction']>.4]
    strata=[sorted([r for r in eligible if r['features']['length']>=20],key=lambda r:-r['features']['length'])[:8],sorted([r for r in eligible if 8<=r['features']['length']<20],key=lambda r:-r['features']['support012'])[:8],sorted([r for r in eligible if 4<=r['features']['length']<8],key=lambda r:r['features']['contrast_median'])[:4]]
    chosen=[r for g in strata for r in g]
    if diagnostic:
        prev={r['id'] for r in chosen}
        pool=[r for r in eligible if r['id'] not in prev and r['features']['length']>=4]
        weak=sorted(pool,key=lambda r:(r['features']['support012'], -r['features']['length']))[:8]
        remaining=[r for r in pool if r not in weak and r['features']['length']>=7]
        curled=sorted(remaining,key=lambda r:r['features']['span_ratio'])[:8]
        chosen=weak+curled
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for start in range(0,len(chosen),4):
        board=Image.new('RGB',(1400,960),'#f3f1eb');draw=ImageDraw.Draw(board)
        for z,row in enumerate(chosen[start:start+4]):
            col=z%2;rr=z//2;p=np.asarray(row['points']);x0,y0=np.floor(p.min(0)-18).astype(int);x1,y1=np.ceil(p.max(0)+19).astype(int);x0=max(0,x0);y0=max(0,y0);x1=min(rgb.shape[1],x1);y1=min(rgb.shape[0],y1)
            f=row['features'];ox=col*700;oy=rr*480
            draw.text((ox+5,oy+5),f'{ident} #{row["id"]} L={f["length"]:.1f} support={f["support012"]:.2f}',font=font,fill='black')
            draw.text((ox+5,oy+28),row['old_decision']['reason'],font=font,fill='black')
            original=Image.fromarray(rgb).crop((x0,y0,x1,y1));overlay=original.copy();dr=ImageDraw.Draw(overlay);dr.line([tuple(a) for a in p-[x0,y0]],fill=(255,30,30),width=1)
            for k,im in enumerate([original,overlay]):
                im.thumbnail((340,410));factor=min(340/original.width,410/original.height,3.5);im=(original if k==0 else overlay).resize((round(original.width*factor),round(original.height*factor)),Image.Resampling.NEAREST)
                board.paste(im,(ox+k*350,oy+60))
        prefix='diagnostic_' if diagnostic else ''
        board.save(ROOT/'review_cards'/f'{prefix}{ident}_{start//4:02}.jpg',quality=96)
    return [dict(id=r['id'],features=r['features']) for r in chosen]

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',default='G01_01,G02_01,G03_01,G04_01,G05_01');a=p.parse_args();(ROOT/'review_cards').mkdir(parents=True,exist_ok=True);records=[]
    for ident in a.ids.split(','):
        rgb,base,old,masks,rows=catalog(ident);jsave(ROOT/'catalog'/f'{ident}.json',dict(id=ident,features=FEATURES,candidates=rows,source_sha256=digest(OLD/'outputs'/ident/'input.png'),baseline_sha256=digest(OLD/'outputs'/ident/'paths.json')))
        selected=cards(ident,rows,rgb)+cards(ident,rows,rgb,True);records.append(dict(id=ident,review_ids=[r['id'] for r in selected],pending=len([r for r in rows if not r['old_keep']]),candidates=len(rows)));print(ident,records[-1],flush=True)
    jsave(ROOT/'review_manifest.json',dict(records=records,annotation_scope='Local RGB candidate review by assistant; not painter ground truth; unreviewed stays unknown',split=dict(train=['G01','G03','G05'],calibration=['G02'],regression=['G04']),all_sources_reused=True))
if __name__=='__main__':main()
