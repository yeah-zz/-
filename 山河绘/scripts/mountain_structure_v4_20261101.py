# -*- coding: utf-8 -*-
"""Subset-only structure selection inside blue-green mountain interiors.

The color/geometry routing is NOT trained semantic segmentation. Existing v4
pixels outside the mountain interior and protected detail paths remain intact.
No new bridges, synthesized hatching, model weights, or robot commands are used.
"""
import sys
sys.dont_write_bytecode=True
from rock_v15_common import LAB,read,jsave,digest
import split_detail_v4_frozen as v4
from v4_local_repair_candidates import evidence,graph_cycles
import numpy as np,cv2,math
from scipy.ndimage import binary_fill_holes,binary_propagation,distance_transform_edt
from scipy.spatial import cKDTree
from pathlib import Path
from PIL import Image
ROOT=LAB/'mountain_structure_v4_20261101'
CFG=dict(version='v4_mountain_structure_r7',blue_minus_red=7,green_minus_red=5,min_pigment_area_fraction=.0035,min_pigment_height=24,material_distance=.095,interior_margin=5,min_chain_for_pair=3,pair_alignment=.88,pair_margin=.10,min_cun_length=24,min_cun_span_ratio=.48,min_cun_support=.55,contrast_threshold=.012,max_cun_per_mass=10,duplicate_distance=2.,duplicate_fraction=.70,original_anchor_min_length=35,original_anchor_support=.25,brown_geodesic_limit=40,fan_envelope_max_mass_overlap=.20)

def kernel(value,scale):return max(3,int(round(value*scale))|1)

def route(rgb,baseline,cfg=CFG):
    h,w=baseline.shape;s=max(h,w)/496.;a=rgb.astype(np.float32)/255;blur=cv2.GaussianBlur(a,(0,0),max(.7,2*s));paper=np.quantile(a.reshape(-1,3),.96,axis=0)
    blue=((blur[:,:,2]-blur[:,:,0])*255>cfg['blue_minus_red'])&((blur[:,:,1]-blur[:,:,0])*255>cfg['green_minus_red'])
    blue=cv2.morphologyEx(blue.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel(9,s),)*2))>0
    material=np.linalg.norm(blur-paper,axis=2)>cfg['material_distance']
    n,labels,stats,_=cv2.connectedComponentsWithStats(blue.astype(np.uint8),8);mass=np.zeros((h,w),bool)
    for k,(x,y,bw,bh,area) in enumerate(stats[1:],1):
        if area<h*w*cfg['min_pigment_area_fraction'] or bh<cfg['min_pigment_height']*s or bw/max(1,bh)>4.5:continue
        pad=max(int(8*s),int(max(bw,bh)*.075));x0=max(0,x-pad);y0=max(0,y-pad);x1=min(w,x+bw+pad);y1=min(h,y+bh+pad)
        seed=labels[y0:y1,x0:x1]==k;allowed=material[y0:y1,x0:x1]|seed
        grown=binary_propagation(seed,mask=allowed)
        grown=cv2.morphologyEx(grown.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel(5,s),)*2))>0
        mass[y0:y1,x0:x1]|=grown
    interior=cv2.erode(mass.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel(cfg['interior_margin']*2+1,s),)*2))>0
    # Protect multi-direction short-line bundles, then connected brown trunk ink.
    chains,_=v4.pixel_chains(baseline);count=np.zeros((h,w),np.float32);vx=count.copy();vy=count.copy();gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    for p in chains:
        length=v4.plen(p);span=float(np.linalg.norm(p[-1]-p[0]));x,y=np.rint(p.mean(0)).astype(int)
        if 2*s<=length<=18*s and span/max(length,1e-6)>=.78 and float(gray[np.rint(p[:,1]).astype(int),np.rint(p[:,0]).astype(int)].mean())<.68:
            t=v4.unit(p[-1]-p[0]);count[y,x]+=1;vx[y,x]+=t[0]**2-t[1]**2;vy[y,x]+=2*t[0]*t[1]
    window=kernel(35,s);count=cv2.boxFilter(count,-1,(window,window),normalize=False);vx=cv2.boxFilter(vx,-1,(window,window),normalize=False);vy=cv2.boxFilter(vy,-1,(window,window),normalize=False)
    dispersion=1-np.sqrt(vx*vx+vy*vy)/np.maximum(count,1)
    depth=distance_transform_edt(mass)
    fan=(count>=10)&(dispersion>.18)&((blur[:,:,2]-blur[:,:,0])*255<4)&(depth<=5*s)
    fan=cv2.dilate(fan.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel(11,s),)*2))>0
    brown=(a[:,:,2]<a[:,:,0]-.006)&(gray<.58)&(depth<=12*s);brown=cv2.morphologyEx(brown.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
    reachable=fan&brown
    for _ in range(max(1,int(round(cfg['brown_geodesic_limit']*s)))):
        grown=(cv2.dilate(reachable.astype(np.uint8),np.ones((3,3),np.uint8))>0)&brown
        if np.array_equal(grown,reachable):break
        reachable=grown
    protect=fan|reachable
    # A conservative envelope protects bark edges and branch junctions whose
    # colors blend into the rock. Bounded geodesic growth avoids flooding an
    # entire connected brown rock shadow into this envelope.
    nn,ll,ss,_=cv2.connectedComponentsWithStats(fan.astype(np.uint8),8);envelope=protect.astype(np.uint8)
    for k,(x,y,bw,bh,area) in enumerate(ss[1:],1):
        if area<20*s*s or max(bw,bh)<10*s or bw/max(1,bh)<1.15:continue
        if float(mass[ll==k].mean())>cfg['fan_envelope_max_mass_overlap']:continue
        x0=max(0,x-int(8*s));x1=min(w,x+bw+int(8*s));y0=max(0,y-int(4*s));y1=min(h,y+bh+int(20*s))
        local=((ll[y0:y1,x0:x1]==k)|reachable[y0:y1,x0:x1]);pts=np.argwhere(local)[:,::-1].astype(np.int32);pts+=np.array([x0,y0]);cv2.fillConvexPoly(envelope,cv2.convexHull(pts),1)
    protect=cv2.dilate(envelope,np.ones((3,3),np.uint8))>0
    return mass,interior,protect

def whole_candidates(mask,scale,cfg=CFG):
    chains,_=v4.pixel_chains(mask);junctions={};pair={};events=[]
    for i,p in enumerate(chains):
        if v4.plen(p)<cfg['min_chain_for_pair']*scale or np.linalg.norm(p[0]-p[-1])<1:continue
        for side in [0,1]:
            xy,t=v4.endpoint(p,side);key=tuple(np.rint(xy).astype(int));junctions.setdefault(key,[]).append((i,side,t))
    for xy,ends in junctions.items():
        if not 2<=len(ends)<=4:continue
        rankings={}
        for a,e in enumerate(ends):
            rankings[a]=sorted([(float(-e[2]@f[2]),b) for b,f in enumerate(ends) if a!=b and e[0]!=f[0]],reverse=True)
        for a,r in rankings.items():
            if not r or r[0][0]<cfg['pair_alignment'] or (len(r)>1 and r[0][0]-r[1][0]<cfg['pair_margin']):continue
            b=r[0][1];rr=rankings[b]
            if not rr or rr[0][1]!=a or (len(rr)>1 and rr[0][0]-rr[1][0]<cfg['pair_margin']):continue
            ia,sa,_=ends[a];ib,sb,_=ends[b];ea=(ia,sa);eb=(ib,sb)
            if ea in pair or eb in pair:continue
            pair[ea]=eb;pair[eb]=ea;events.append(dict(junction=list(map(int,xy)),a=list(ea),b=list(eb),alignment=r[0][0],same_artist_stroke_verified=False))
    starts=[(i,s) for i in range(len(chains)) for s in [0,1] if (i,s) not in pair]+[(i,0) for i in range(len(chains))];seen=set();out=[]
    for i,side in starts:
        if i in seen:continue
        pieces=[];members=[]
        while i not in seen:
            seen.add(i);members.append(i);p=chains[i] if side==0 else chains[i][::-1];pieces.append(p if not pieces else p[1:]);nxt=pair.get((i,1-side))
            if nxt is None:break
            i,side=nxt
        out.append(dict(points=np.concatenate(pieces),members=members))
    return out,events

def backbone_regions(rgb,interior,scale):
    ink,_=v4.source_ink(rgb);width=distance_transform_edt(ink)*2;core=(width>=2.5*scale)&interior
    n,labels,stats,_=cv2.connectedComponentsWithStats(core.astype(np.uint8),8);mask=np.zeros_like(core);direction=np.zeros((*core.shape,2),np.float32)
    for k,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area<8*scale*scale or max(w,h)<18*scale:continue
        points=np.argwhere(labels==k)[:,::-1].astype(float)
        cov=np.cov(points.T);values,vectors=np.linalg.eigh(cov)
        if values[-1]/max(1,values[0])<3:continue
        band=cv2.dilate((labels==k).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel(3,scale),)*2))>0
        mask|=band;direction[band]=vectors[:,-1]
    return mask&interior,direction

def select(rgb,base,cfg=CFG,source_records=None):
    shape=base.shape;s=max(shape)/496.;mass,interior,protect=route(rgb,base,cfg);joined,events=whole_candidates(base,s,cfg);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    candidates=[dict(points=np.asarray(r['points'] if isinstance(r,dict) else r,np.float32),members=[i],origin='original_v4_candidate',role=r.get('role','unknown') if isinstance(r,dict) else 'unknown') for i,r in enumerate(source_records)] if source_records is not None else joined
    if source_records is not None:
        for row in joined:
            p=row['points'];q=np.rint(p).astype(int)
            if v4.plen(p)>=40*s and interior[q[:,1],q[:,0]].all():candidates.append(dict(row,origin='junction_supported_candidate',role='long_structure'))
    backbone,axis=backbone_regions(rgb,interior,s)
    _,component=cv2.connectedComponents(mass.astype(np.uint8),8);kept=[];pending=[];eligible=[];decisions=[]
    for i,row in enumerate(candidates):
        p=row['points'];q=np.rint(p).astype(int);inside=interior[q[:,1],q[:,0]];protected=protect[q[:,1],q[:,0]];length=v4.plen(p);span=float(np.linalg.norm(p[-1]-p[0]))
        if not inside.all() or protected.any():
            kept.append(p);decisions.append(dict(id=i,decision='keep',reason='outside_interiors_or_protected_detail',length=length,members=row['members']));continue
        ratio=span/max(length,1e-8);on_backbone=backbone[q[:,1],q[:,0]];alignment=float(np.mean(abs(axis[q[:,1],q[:,0]]@v4.unit(p[-1]-p[0]))))
        if on_backbone.mean()>=.70 and ratio>=.65 and alignment>=.55:
            kept.append(p);decisions.append(dict(id=i,decision='keep',reason='elongated_coarse_ink_backbone',length=length,members=row['members']));continue
        contrast,_=evidence(gray,p,s);support=float((contrast>=cfg['contrast_threshold']).mean())
        if row.get('origin')=='original_v4_candidate' and row.get('role')=='long_structure' and length>=cfg['original_anchor_min_length']*s and ratio>=.65 and support>=cfg['original_anchor_support']:
            kept.append(p);decisions.append(dict(id=i,decision='keep',reason='original_v4_long_structure_anchor',length=length,image_support=support,members=row['members']));continue
        valid=length>=cfg['min_cun_length']*s and ratio>=cfg['min_cun_span_ratio'] and support>=cfg['min_cun_support']
        ids=component[q[:,1],q[:,0]];gid=int(np.bincount(ids).argmax());decision=dict(id=i,decision='pending',reason='short_curled_or_weak',length=length,span_ratio=ratio,image_support=support,mass=gid,members=row['members'])
        decisions.append(decision)
        priority=1.25 if row.get('origin')=='original_v4_candidate' and row.get('role')=='long_structure' else 1.
        if valid:eligible.append((priority*length*(.4+.6*support)*(.5+.5*ratio),i,p,gid,decision))
        else:pending.append(p)
    counts={};selected_by_mass={}
    for score,i,p,gid,decision in sorted(eligible,key=lambda r:-r[0]):
        chosen=selected_by_mass.setdefault(gid,[]);duplicate=False
        if chosen:
            d=cKDTree(np.concatenate(chosen)).query(p)[0];duplicate=float((d<=cfg['duplicate_distance']*s).mean())>=cfg['duplicate_fraction']
        if duplicate or counts.get(gid,0)>=cfg['max_cun_per_mass']:
            decision['reason']='near_duplicate' if duplicate else 'interior_stroke_budget';pending.append(p);continue
        chosen.append(p);counts[gid]=counts.get(gid,0)+1;kept.append(p);decision.update(decision='keep',reason='long_image_supported_interior',score=score)
    result=v4.raster(shape,kept);assert not (result&~base).any()
    outside=base&~interior;protected_base=base&protect
    assert np.array_equal(result&~interior,outside) and np.array_equal(result&protect,protected_base)
    return result,kept,pending,dict(mass=mass,interior=interior,protect=protect,backbone=backbone),decisions,events,counts

def run(source,baseline_folder,output):
    source=Path(source);baseline_folder=Path(baseline_folder);output=Path(output)
    if output.drive.upper() not in ['E:','F:'] or not output.is_absolute():raise ValueError('Use absolute E/F output')
    rgb=np.asarray(Image.open(source).convert('RGB'));doc=read(baseline_folder/'paths.json');paths=[np.asarray(p['points'] if isinstance(p,dict) else p,np.float32) for p in doc['paths']];base=v4.raster(rgb.shape[:2],paths)
    result,kept,pending,masks,decisions,events,counts=select(rgb,base,source_records=doc['paths']);output.mkdir(parents=True,exist_ok=True)
    # Export a canonical graph decomposition so overlapping hypotheses do not
    # become duplicated robot strokes. Artist stroke identity remains unknown.
    canonical,_=whole_candidates(result,max(result.shape)/496.);export_paths=[r['points'] for r in canonical]
    bw=lambda m:np.where(m,0,255).astype(np.uint8)
    for name,im in [('input',rgb),('baseline',bw(base)),('lines',bw(result)),('pending',bw(base&~result))]:Image.fromarray(im).save(output/(name+'.png'))
    routing=rgb.copy();routing[masks['interior']]=(routing[masks['interior']]*.55+np.array([60,150,230])*.45).astype(np.uint8);routing[masks['protect']]=(routing[masks['protect']]*.55+np.array([60,200,75])*.45).astype(np.uint8);Image.fromarray(routing).save(output/'routing.png')
    changes=np.repeat(bw(result)[:,:,None],3,2);changes[base&~result]=[220,55,55];Image.fromarray(changes).save(output/'changes.png')
    full=np.repeat(bw(result)[:,:,None],3,2);full[base&~result]=[185,185,185];Image.fromarray(full).save(output/'all_evidence.png')
    stats=dict(baseline_pixels=int(base.sum()),selected_pixels=int(result.sum()),removed_pixels=int((base&~result).sum()),added_pixels=int((result&~base).sum()),mountain_interior_pixels=int(masks['interior'].sum()),baseline_pixels_in_interior=int((base&masks['interior']).sum()),selected_pixels_in_interior=int((result&masks['interior']).sum()),outside_interior_exactly_preserved=bool(np.array_equal(result&~masks['interior'],base&~masks['interior'])),protected_detail_exactly_preserved=bool(np.array_equal(result&masks['protect'],base&masks['protect'])),baseline_cycles=graph_cycles(base),selected_cycles=graph_cycles(result),interior_strokes_per_mass=counts,selected_whole_candidates=len(kept),pending_whole_candidates=len(pending),no_trained_semantic_segmentation=True,new_training_steps=0)
    stats['routing_status']='no_confirmed_interior_keep_baseline' if not masks['interior'].any() else 'partitioned_structure_candidate'
    jsave(output/'statistics.json',stats);jsave(output/'decisions.json',decisions);np.savez_compressed(output/'routing_masks.npz',**masks)
    jsave(output/'paths.json',dict(paths=[dict(points=p.tolist(),stroke_identity_verified=False) for p in export_paths],pending_paths=[dict(points=p.tolist(),decision='deferred_not_background') for p in pending],source=str(source),source_sha256=digest(source),baseline_paths_sha256=digest(baseline_folder/'paths.json'),config=CFG,coordinate_system='input_image_pixels',junction_pair_hypotheses=events,semantic_segmentation=False,physical_z=None,physical_speed=None,ready_for_robot=False,actually_drawn=False))
    poly=''.join('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in p)+'"/>' for p in export_paths)
    (output/'lines.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {rgb.shape[1]} {rgb.shape[0]}"><g fill="none" stroke="black" stroke-width="1">{poly}</g></svg>',encoding='utf-8')
    return stats
