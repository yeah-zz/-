# -*- coding: utf-8 -*-
"""Source-ink experimental extractor: dense details and long structure separately.

Automatic routing is geometric, NOT a learned tree/rock semantic classifier.
No historical reference paths, model weights, or robot commands are read.
"""
import os,sys
from pathlib import Path
sys.dont_write_bytecode=True
LAB=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022')
for k,v in {'TEMP':'temp','TMP':'temp','PIP_CACHE_DIR':'cache/pip','TORCH_HOME':'cache/torch','HF_HOME':'cache/huggingface','XDG_CACHE_HOME':'cache','CUDA_CACHE_PATH':'cache/cuda'}.items():os.environ[k]=str(LAB/v)
from rock_v15_common import jsave,digest
from source_ink_baseline_20261028 import extract as source_ink
import numpy as np,cv2,json,argparse,html,math
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_propagation
from skimage.morphology import skeletonize
from PIL import Image,ImageOps

CFG=dict(version='source_ink_split_v4',min_structure_length=20.,max_bridge=7.,facing=.94,axis=.93,cross_track=1.5,margin=.08,min_bridge_evidence=.8,dense_window=41,dense_skeleton_pixels=110,dense_junction_pixels=6,max_coherence=.70,dense_dark_gray=.40,dense_dark_pixels=100,short_segment_window=65,short_segments_min=15)

def plen(p):return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
def unit(p):return p/max(float(np.linalg.norm(p)),1e-8)

def pixel_chains(sk):
    # Do not dilate/average junction clusters: that destroys closely spaced needles.
    pix=set(map(tuple,np.argwhere(sk)));neighbors={}
    for y,x in pix:
        near=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                q=(y+dy,x+dx)
                if (dy==dx==0) or q not in pix:continue
                if dx and dy and ((y,x+dx) in pix or (y+dy,x) in pix):continue
                near.append(q)
        neighbors[(y,x)]=sorted(near)
    nodes={p for p in pix if len(neighbors[p])!=2};used=set();paths=[]
    def edge(a,b):return tuple(sorted((a,b)))
    def follow(a,b):
        seq=[a];used.add(edge(a,b));prev=a;cur=b
        while True:
            seq.append(cur)
            if cur in nodes or cur==a:break
            opts=[q for q in neighbors[cur] if q!=prev and edge(cur,q) not in used]
            if not opts:break
            nxt=opts[0];used.add(edge(cur,nxt));prev,cur=cur,nxt
        return np.asarray([(x,y) for y,x in seq],np.float32)
    for a in sorted(nodes):
        for b in neighbors[a]:
            if edge(a,b) not in used:paths.append(follow(a,b))
    for a in sorted(pix):
        for b in neighbors[a]:
            if edge(a,b) not in used:paths.append(follow(a,b))
    # Isolated skeleton pixels remain in the ink evidence, not invented trajectories.
    return paths,neighbors

def dense_regions(rgb,sk,neighbors,cfg):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    xx=cv2.GaussianBlur(gx*gx,(0,0),8);yy=cv2.GaussianBlur(gy*gy,(0,0),8);xy=cv2.GaussianBlur(gx*gy,(0,0),8)
    coherence=np.sqrt((xx-yy)**2+4*xy*xy)/(xx+yy+1e-6)
    junction=np.zeros(sk.shape,np.float32)
    for p,near in neighbors.items():
        if len(near)>=3:junction[p]=1
    win=cfg['dense_window'];density=cv2.boxFilter(sk.astype(np.float32),-1,(win,win),normalize=False)
    joints=cv2.boxFilter(junction,-1,(win,win),normalize=False)
    dark=cv2.boxFilter((gray<cfg['dense_dark_gray']).astype(np.float32),-1,(win,win),normalize=False)
    # Pine-like fine fans contain many short nearly straight pieces, unlike most
    # broken broad-ink boundaries. This remains a geometric heuristic.
    counts=np.zeros(sk.shape,np.float32)
    detected=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect((gray*255).astype(np.uint8))[0]
    if detected is not None:
        for line in detected[:,0]:
            a=line[:2];b=line[2:];length=float(np.linalg.norm(b-a))
            if 8<=length<=48:
                x,y=np.rint((a+b)*.5).astype(int);counts[y,x]+=1
    countwin=cfg['short_segment_window'];counts=cv2.boxFilter(counts,-1,(countwin,countwin),normalize=False)
    seed=(density>=cfg['dense_skeleton_pixels'])&(joints>=cfg['dense_junction_pixels'])&(coherence<cfg['max_coherence'])&(dark>=cfg['dense_dark_pixels'])&(counts>=cfg['short_segments_min'])
    region=cv2.dilate(seed.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))>0
    return region,coherence

def ink_ridges(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    h,w=gray.shape;yy,xx=np.mgrid[:h,:w].astype(np.float32)
    best=np.zeros(gray.shape,np.float32);normal=np.zeros((*gray.shape,2),np.float32)
    for sigma in [.7,1.2,2.,3.5]:
        g=cv2.GaussianBlur(gray,(0,0),sigma)
        a=cv2.Sobel(g,cv2.CV_32F,2,0,ksize=3)*(sigma**1.5/4)
        b=cv2.Sobel(g,cv2.CV_32F,0,2,ksize=3)*(sigma**1.5/4)
        c=cv2.Sobel(g,cv2.CV_32F,1,1,ksize=3)*(sigma**1.5/4)
        delta=np.sqrt((a-b)**2+4*c*c);large=(a+b+delta)*.5;small=(a+b-delta)*.5
        score=np.maximum(large,0)*np.exp(-.5*(small/np.maximum(large*.5,1e-5))**2)
        angle=.5*np.arctan2(2*c,a-b);nx=np.cos(angle);ny=np.sin(angle)
        # Reject one-sided wash boundaries: an ink band has brighter shoulders.
        left=cv2.remap(gray,xx+nx*max(2,3*sigma),yy+ny*max(2,3*sigma),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        right=cv2.remap(gray,xx-nx*max(2,3*sigma),yy-ny*max(2,3*sigma),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        score[(np.minimum(left,right)-gray)<.006]=0
        take=score>best;best[take]=score[take];normal[take,0]=nx[take];normal[take,1]=ny[take]
    nx,ny=normal[:,:,0],normal[:,:,1]
    l=cv2.remap(best,xx+nx,yy+ny,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    r=cv2.remap(best,xx-nx,yy-ny,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    nms=(best>=l)&(best>=r)
    allowed=nms&(best>.004)
    supported=binary_propagation(nms&(best>.012),mask=allowed)
    return skeletonize(supported),best

def near_region(p,region):
    xy=np.rint(p).astype(int);return region[xy[:,1],xy[:,0]]

def consolidate_source_lines(source,ridges,dense):
    # Prefer intact source-ink chains over fitting independent ridge fragments.
    # Never clip a fragment into several novelty pieces to fill a coverage map.
    full=[p for p in source if not near_region(p,dense).any() and plen(p)>=25 and np.linalg.norm(p[0]-p[-1])>=3]
    if not full:return ridges,0
    pts=np.concatenate(full);tree=cKDTree(pts)
    output=list(full)
    for p in ridges:
        dist,idx=tree.query(p)
        covered=dist<=1.8
        if covered.mean()>=.80:continue
        # A partially overlapping ridge remains pending evidence in the source
        # mask, instead of creating two strokes on the same visible ink band.
        if covered.mean()>.20:continue
        output.append(p)
    return output,len(full)

def endpoint(p,side):
    q=p if side==0 else p[::-1]
    d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
    k=min(len(q)-1,max(1,int(np.searchsorted(d,min(9.,d[-1])))))
    return q[0],unit(q[0]-q[k])

def stitch(paths,rgb,ink,dense,cfg):
    # Only mutually unambiguous endpoint pairs, backed by original image pixels.
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    local=cv2.GaussianBlur(gray,(0,0),3)-gray
    valley=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
    evidence=(local>.008)&(valley>.018)
    coords=[];descs=[]
    for i,p in enumerate(paths):
        if plen(p)<7 or np.linalg.norm(p[0]-p[-1])<1:continue
        for side in (0,1):
            xy,t=endpoint(p,side)
            if near_region(p[[0 if side==0 else -1]],dense)[0]:continue
            coords.append(xy);descs.append((i,side,t))
    if not coords:return paths,[],[]
    coords=np.asarray(coords);tree=cKDTree(coords);candidates=[];by_end={};h,w=gray.shape
    for a,b in sorted(tree.query_pairs(cfg['max_bridge'])):
        ia,sa,ta=descs[a];ib,sb,tb=descs[b]
        if ia==ib:continue
        delta=coords[b]-coords[a];gap=float(np.linalg.norm(delta));alignment=float(ta@(-tb))
        if alignment<cfg['facing']:continue
        axis=1.;cross=0.
        if gap>.8:
            direction=delta/gap;axis=min(float(ta@direction),float(tb@(-direction)))
            cross=max(abs(float(np.cross(ta,delta))),abs(float(np.cross(tb,delta))))
            if axis<cfg['axis'] or cross>cfg['cross_track']:continue
        bridge=np.linspace(coords[a],coords[b],max(2,int(math.ceil(gap*2))+1)).astype(np.float32)
        ij=np.rint(bridge).astype(int);supported=float(evidence[ij[:,1],ij[:,0]].mean())
        if gap>1.5 and supported<cfg['min_bridge_evidence']:continue
        # A bridge must not hit an unrelated existing branch. Check its interior.
        interior=bridge[2:-2]
        if gap>2 and len(interior):
            occupied=np.zeros((h,w),np.uint8)
            for j,q in enumerate(paths):
                if j in [ia,ib]:continue
                if np.any(q.max(0)<interior.min(0)-1) or np.any(q.min(0)>interior.max(0)+1):continue
                cv2.polylines(occupied,[np.rint(q).astype(np.int32)],False,1,1)
            qq=np.rint(interior).astype(int)
            if occupied[qq[:,1],qq[:,0]].any():continue
        score=.7*alignment+.25*axis+.05*(1-gap/cfg['max_bridge'])
        k=len(candidates);candidates.append(dict(a=a,b=b,score=score,gap=gap,support=supported,bridge=bridge))
        by_end.setdefault(a,[]).append((score,k));by_end.setdefault(b,[]).append((score,k))
    accepted=[];pair={};events=[]
    for k,c in enumerate(candidates):
        ranks=[sorted(by_end[e],reverse=True) for e in [c['a'],c['b']]]
        if any(r[0][1]!=k or (len(r)>1 and r[0][0]-r[1][0]<cfg['margin']) for r in ranks):continue
        ia,sa,_=descs[c['a']];ib,sb,_=descs[c['b']];a=(ia,sa);b=(ib,sb)
        if a in pair or b in pair:continue
        # Avoid closing a cycle through previously paired chains.
        connected={ia};stack=[ia]
        while stack:
            i=stack.pop()
            for side in (0,1):
                nxt=pair.get((i,side))
                if nxt is not None and nxt[0] not in connected:connected.add(nxt[0]);stack.append(nxt[0])
        if ib in connected:continue
        pair[a]=b;pair[b]=a;accepted.append(k)
        events.append(dict(a=list(a),b=list(b),gap=c['gap'],image_support=c['support'],score=c['score'],new_bridge_pixels=c['bridge'].tolist(),hypothesis_not_artist_gold=True))
    starts=[(i,s) for i in range(len(paths)) for s in (0,1) if (i,s) not in pair]+[(i,0) for i in range(len(paths))];seen=set();out=[];members=[]
    for i,s in starts:
        if i in seen:continue
        pieces=[];ids=[]
        while i not in seen:
            seen.add(i);ids.append(i);q=paths[i] if s==0 else paths[i][::-1];pieces.append(q)
            nxt=pair.get((i,1-s))
            if nxt is None:break
            j,t=nxt;start=paths[j][0 if t==0 else -1];gap=float(np.linalg.norm(q[-1]-start))
            if gap>1:pieces.append(np.linspace(q[-1],start,int(math.ceil(gap*2))+1)[1:-1])
            i,s=j,t
        out.append(np.concatenate(pieces));members.append(ids)
    return out,events,members

def raster(shape,paths):
    sk=np.zeros(shape,np.uint8)
    for p in paths:
        if len(p)>1:cv2.polylines(sk,[np.rint(p).astype(np.int32)],False,1,1,cv2.LINE_8)
    return sk>0

def save(folder,rgb,ink,raw,dense,details,structures,pending,events,meta):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);h,w=raw.shape
    dm=raster(raw.shape,details);sm=raster(raw.shape,structures);selected=dm|sm;left=raw&~selected
    def bw(a):return np.repeat(np.where(a,0,255).astype(np.uint8)[:,:,None],3,2)
    overlay=rgb.copy();overlay[sm]=[20,105,220];overlay[dm]=[190,30,135]
    routing=rgb.copy();routing[dense]=(routing[dense]*.6+np.array([50,190,100])*.4).astype(np.uint8)
    for name,a in [('input',rgb),('ink',bw(ink)),('raw_skeleton',bw(raw)),('dense_regions',routing),('detail_lines',bw(dm)),('structure_lines',bw(sm)),('lines',bw(selected)),('pending',bw(left)),('overlay',overlay)]:Image.fromarray(a).save(folder/(name+'.png'))
    full=bw(selected);full[left]=[190,190,190];Image.fromarray(full).save(folder/'all_evidence.png')
    def row(p,role):return {'role':role,'points':p.tolist(),'stroke_identity_verified':False}
    doc=dict(meta=meta,config=CFG,coordinates='input_image_pixels',paths=[row(p,'dense_detail') for p in details]+[row(p,'long_structure') for p in structures],pending_paths=[row(p,'unresolved_short_ridge_fragment') for p in pending],unresolved_original_skeleton_mask='pending.png',pending_paths_cover_ridge_candidates_only=True,joins=events,physical_z=None,physical_speed=None,ready_for_robot=False,actually_drawn=False,reference_labels_used=False)
    jsave(folder/'paths.json',doc)
    polygons=[]
    for p in details+structures:polygons.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in p)+'"/>')
    (folder/'lines.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><metadata>Experimental source-pixel extraction, no calibrated robot commands.</metadata><g fill="none" stroke="black" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">'+''.join(polygons)+'</g></svg>',encoding='utf-8')
    stats=dict(raw_skeleton_pixels=int(raw.sum()),dense_region_pixels=int(dense.sum()),raw_pixels_in_dense_regions=int((raw&dense).sum()),retained_raw_fraction_in_dense_regions=float((selected&raw&dense).sum()/max(1,(raw&dense).sum())),selected_skeleton_pixels=int(selected.sum()),unresolved_skeleton_pixels=int(left.sum()),detail_paths=len(details),structure_paths=len(structures),pending_paths=len(pending),accepted_joins=len(events))
    jsave(folder/'statistics.json',stats)
    panels=''.join('<figure><figcaption>'+title+'</figcaption><a href="'+name+'.png"><img src="'+name+'.png"></a></figure>' for name,title in [('input','原图'),('lines','分层筛选中心线'),('overlay','蓝：长线；紫：密集细节'),('all_evidence','完整证据：灰色为待定碎线'),('dense_regions','绿色：几何检测密集交叉区'),('ink','原图像素墨迹')])
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>细节与结构分层实验</title><style>body{font:16px system-ui;background:#f3f1eb;margin:24px}main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0}img{width:100%}@media(max-width:800px){main{grid-template-columns:1fr}}</style><h1>细节与结构分层实验</h1><p>传统图像处理实验，CPU 运行，未训练新权重。区域按几何密度与方向变化自动确定，不表示已识别树木/山石语义。灰色碎线保留待核对；续接仅是有墨迹支持的几何假设。</p><main>'+panels+'</main><p><a href="lines.svg">SVG</a> · <a href="paths.json">像素路径</a> · <a href="statistics.json">保留/排除记录</a></p>',encoding='utf-8')
    return stats

def run(source,output):
    source=Path(source).resolve();output=Path(output)
    if output.drive.upper() not in ['E:','F:'] or not output.is_absolute():raise ValueError('Use E/F absolute output directory')
    rgb=np.asarray(ImageOps.exif_transpose(Image.open(source)).convert('RGB'))
    if max(rgb.shape[:2])>2600:raise ValueError('For this prototype use an image with longest side <=2600; no silent global downscale')
    ink,sk=source_ink(rgb);raw,neighbors=pixel_chains(sk);dense,coherence=dense_regions(rgb,sk,neighbors,CFG)
    ridge,response=ink_ridges(rgb)
    # Only source skeleton in dense fan regions; ridges supply long-line candidates.
    long_raw,_=pixel_chains(ridge&~dense)
    detail=[];other=[]
    for p in raw:
        if near_region(p,dense).any():detail.append(p)
    other,source_whole_count=consolidate_source_lines(raw,long_raw,dense)
    joined,events,members=stitch(other,rgb,ink,dense,CFG)
    structures=[];pending=[]
    for p in joined:
        (structures if plen(p)>=CFG['min_structure_length'] else pending).append(p)
    meta=dict(source=str(source),sha256=digest(source),input_size=[rgb.shape[1],rgb.shape[0]],method='source-pixel dual geometric layer experiment',device='CPU: OpenCV/SciPy, no neural training',new_training_steps=0,semantic_tree_rock_classification=False,manual_roi_used=False,intact_source_structure_candidates=source_whole_count)
    stats=save(output,rgb,ink,sk,dense,detail,structures,pending,events,meta)
    print('SPLIT_EXTRACTION',source.name,json.dumps(stats,ensure_ascii=False),flush=True)
    return stats

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('image');a.add_argument('--output',required=True);args=a.parse_args();run(args.image,args.output)
