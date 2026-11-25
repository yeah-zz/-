"""Versioned training-only weak-label refinement; validation remains unchanged."""
import argparse
import hashlib
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .runtime import LAB
from .guidance import apply_guidance
from .labels import annotation_regions


def ridge_proposals(rgb):
    """Color-independent dark curvilinear proposals, not semantic ground truth."""
    gray=cv2.cvtColor(cv2.bilateralFilter(rgb,7,25,5),cv2.COLOR_RGB2GRAY).astype(np.float32)
    response=np.zeros_like(gray)
    for sigma in (1.1,1.8,2.6):
        smooth=cv2.GaussianBlur(gray,(0,0),sigma)
        xx=cv2.Sobel(smooth,cv2.CV_32F,2,0,ksize=3)*sigma*sigma/4
        yy=cv2.Sobel(smooth,cv2.CV_32F,0,2,ksize=3)*sigma*sigma/4
        xy=cv2.Sobel(smooth,cv2.CV_32F,1,1,ksize=3)*sigma*sigma/4
        root=np.sqrt((xx-yy)**2+4*xy*xy)
        a=(xx+yy-root)/2;b=(xx+yy+root)/2
        swap=np.abs(a)>np.abs(b)
        small=np.where(swap,b,a);large=np.where(swap,a,b)
        score=np.maximum(large,0)*np.exp(-np.square(small/(np.abs(large)+1e-5))/.32)
        response=np.maximum(response,score)
    local=cv2.morphologyEx(cv2.GaussianBlur(gray,(0,0),.8),cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    candidate=(response>2.5)&(local>9)
    from original_d_tracer import LandscapeBrushGenerator as Trace
    skeleton=Trace._thin_binary(candidate.astype(np.uint8)*255)
    proposals=np.zeros(gray.shape,np.uint8)
    for path in Trace._trace_skeleton_paths(skeleton):
        points=np.asarray(path,np.int32)
        length=np.linalg.norm(np.diff(points,axis=0),axis=1).sum()
        extent=np.linalg.norm(points.max(axis=0)-points.min(axis=0))
        if length<26 or extent<18:continue
        # Looped blobs and dense dot clusters are details, not main contours.
        if length>extent*4:continue
        span=points.max(axis=0)-points.min(axis=0)
        if span[0]>4*max(1,span[1]):continue  # most water ripples remain auxiliary
        cv2.polylines(proposals,[points],False,255,2,cv2.LINE_AA)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    red=(((hsv[:,:,0]<12)|(hsv[:,:,0]>170))&(hsv[:,:,1]>95)).astype(np.uint8)
    red=cv2.dilate(red,np.ones((7,7),np.uint8))>0
    proposals[red]=0
    return proposals,response


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='manifest_v02');args=parser.parse_args(argv)
    base=LAB/'datasets'/'manifest.json'
    manifest=json.loads(base.read_text(encoding='utf-8'))
    destination=LAB/'datasets'/(args.name+'.json')
    if destination.exists():raise FileExistsError('Refusing to overwrite '+str(destination))
    out=LAB/'data'/('refined_'+args.name);out.mkdir(exist_ok=True)
    rows=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    for r in manifest['real']:
        if r['split']!='train':continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'))
        with np.load(r['labels']) as old:
            targets=old['targets'].astype(np.float32)/255.;weights=old['weights'].astype(np.float32)/255.
        original=targets[0].copy()
        ridge,response=ridge_proposals(rgb)
        added=(ridge>64)&(targets[0]<.3)
        targets[0]=np.maximum(targets[0],ridge.astype(np.float32)/255.)
        weights[0,added]=.3
        # Uncertain pixels near missing ridges should not be strong negatives.
        uncertain=(response>1)&(targets[0]<.3)
        weights[0,uncertain]=np.minimum(weights[0,uncertain],.08)
        targets,weights=annotation_regions(r['id'],rgb.shape,targets,weights)
        targets,weights,_,guides=apply_guidance(r['id'],rgb,targets,weights)
        # Coarse assistant-drafted training exclusions, NOT expert line labels.
        h,w=rgb.shape[:2]
        content_rectangles={
            'cma_1979_27_1_10':(.13,.10,.89,.91),
            'cma_1979_27_1_8':(.13,.10,.89,.91),
            'cma_1917_94':(.25,0,1.,1.),
            'cma_1973_72':(.34,0,.72,1.),
        }
        if r['id'] in content_rectangles:
            a,b,c,d=content_rectangles[r['id']]
            valid=np.zeros((h,w),bool);valid[round(b*h):round(d*h),round(a*w):round(c*w)]=True
            targets[:,~valid]=0;weights[:,~valid]=.75
        if r['id']=='user_qianli_02':
            targets[:,:450,:]=0;weights[:,:450,:]=.85
            water=np.zeros((h,w),bool);water[1210:,:]=True;water[1280:1360,250:355]=False
            targets[0,water]=0;weights[0,water]=.85
        weights[:,:5,:]=0;weights[:,-5:,:]=0;weights[:,:,:5]=0;weights[:,:,-5:]=0
        folder=out/r['id'];folder.mkdir(exist_ok=True)
        label=folder/'weak_labels.npz'
        np.savez_compressed(label,targets=np.rint(targets*255).astype(np.uint8),weights=np.rint(weights*255).astype(np.uint8))
        panel=Image.new('RGB',(1500,690),(246,245,239));draw=ImageDraw.Draw(panel)
        for col,(title,array) in enumerate([('原图',rgb),('第一版弱标签',255-(original*255).astype(np.uint8)),('第二版弱标签（仍非人工真值）',255-(targets[0]*255).astype(np.uint8))]):
            draw.text((col*500+10,10),title,font=font,fill=(30,45,35));im=Image.fromarray(array);im.thumbnail((480,620));panel.paste(im,(col*500+(500-im.width)//2,60+(620-im.height)//2))
        panel.save(folder/'proposal_comparison.png')
        r['previous_labels']=r['labels'];r['labels']=str(label);r['supervision']='v2_color_independent_curvilinear_weak_proposals_NOT_gold';r['sparse_guidance']=guides
        rows.append(dict(id=r['id'],added_candidate_pixels=int(added.sum()),expert_verified=False,
                         assistant_drafted_content_roi=content_rectangles.get(r['id']),
                         assistant_drafted_water_and_sky_exclusion=r['id']=='user_qianli_02'))
        print('REFINED',r['id'],int(added.sum()),flush=True)
    manifest.update(refinement='training only; original validation images and weak targets unchanged',
                    base_manifest_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),refinement_records=rows)
    destination.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REFINEMENT_READY',str(destination),flush=True)
