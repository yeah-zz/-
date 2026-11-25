# -*- coding: utf-8 -*-
"""Partial cross-section supervision and explicitly reviewed paper negatives."""
import sys
sys.dont_write_bytecode=True
from rock_v15_common import LAB,read,jsave,digest
from joint_ink_structure_data import path_fields,write_record,OFFSETS,LOCAL
from pathlib import Path
import numpy as np,cv2
from scipy.ndimage import gaussian_filter1d
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'paper_center_refinement_20261029';PARENT=LAB/'joint_ink_structure_20261029'
SPECS={
 'R04':dict(split='train',paths=[
 [[185,115],[193,102],[208,101],[219,86],[230,75],[241,75],[250,85],[255,100],[269,106],[286,119],[298,130],[304,149],[307,181],[311,214],[314,248]],
 [[65,205],[72,180],[79,150],[88,134],[99,130],[110,136],[118,154],[127,179],[137,201],[142,227],[143,243]],
 [[194,182],[198,199],[201,222],[201,245],[202,271],[203,293]],
 [[366,151],[379,168],[389,190],[395,218],[398,245],[401,272],[404,294]]],
 unknown=[[158,106,200,145],[178,143,222,189],[276,102,312,134],[110,145,142,169],[344,137,373,162]],paper=[[0,0,95,40],[325,0,430,65]]),
 'N11':dict(split='calibration',paths=[
 [[20,124],[36,111],[58,94],[79,71],[96,57],[124,45],[149,30],[165,23],[178,20],[192,25],[211,35],[230,45]],
 [[248,34],[279,26],[320,18],[354,11],[381,7],[401,11],[418,18],[436,35],[457,54],[478,71],[487,90],[491,115],[492,146],[490,173]]],
 unknown=[[465,168,533,240]],paper=[[260,38,315,53],[55,5,130,26]])}

def sample(gray,points):
    q=np.asarray(points,np.float32);return cv2.remap(gray,q[:,0].reshape(-1,1),q[:,1].reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101).ravel()

def profile_reference(rgb,seeds,unknown):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;h,w=gray.shape
    center_weight=np.zeros((h,w),np.float32);ink_weight=center_weight.copy();ink=center_weight.copy();accepted_paths=[];diagnostics=[]
    radius=12.;offset=np.arange(-radius,radius+.1,.5)
    for seed in seeds:
        p=np.asarray(seed,np.float32);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];u=np.arange(0,arc[-1]+.1,1.);p=np.stack([np.interp(u,arc,p[:,j]) for j in [0,1]],1)
        tangent=np.gradient(p,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-6);normal=np.c_[-tangent[:,1],tangent[:,0]]
        corrected=p.copy();valid=[]
        for i,(xy,n) in enumerate(zip(p,normal)):
            coords=xy+offset[:,None]*n;values=gaussian_filter1d(sample(gray,coords),.7);paper=float(np.quantile(values,.9));near=np.where(abs(offset)<=5)[0];k=int(near[np.argmin(values[near])]);contrast=paper-float(values[k]);threshold=paper-max(.06,.35*contrast);dark=values<threshold
            left=k;right=k
            while left>0 and dark[left-1]:left-=1
            while right<len(offset)-1 and dark[right+1]:right+=1
            width=float(offset[right]-offset[left]+.5);shift=float((offset[left]+offset[right])*.5)
            # Other bands may occur farther along a normal. Require clean local
            # shoulders and do not label the neighboring band's pixels as blank.
            clean_left=left
            while clean_left>0 and not dark[clean_left-1]:clean_left-=1
            clean_right=right
            while clean_right<len(offset)-1 and not dark[clean_right+1]:clean_right+=1
            left_gap=left-clean_left;right_gap=clean_right-right
            in_unknown=any(a<=xy[0]<c and b<=xy[1]<d for a,b,c,d in unknown)
            ok=contrast>=.16 and left>=3 and right<len(offset)-3 and left_gap>=3 and right_gap>=3 and 1<=width<=18 and abs(shift)<=5 and not in_unknown
            corrected[i]=xy+n*shift
            valid.append(ok)
            diagnostics.append(dict(x=float(xy[0]),y=float(xy[1]),valid=bool(ok),width=width,contrast=contrast,shift=shift,in_unknown=bool(in_unknown)))
            if not ok:continue
            for j,(x,y) in enumerate(np.rint(coords).astype(int)):
                if not(0<=x<w and 0<=y<h):continue
                if j<left-3 or j>right+3:continue
                # The known corridor is limited to this isolated band and shoulders.
                center_weight[y,x]=1;ink_weight[y,x]=1;ink[y,x]=float(dark[j])
        # Retain all ordered geometry, but ONLY accepted cross sections carry loss.
        accepted_paths.append(corrected)
    heat,co,si,dv,rel,rm,center=path_fields(accepted_paths,(h,w))
    # Tiny sampling holes are not expanded into unknown neighboring strokes.
    center_weight=cv2.dilate(center_weight,np.ones((2,2),np.uint8));ink_weight=cv2.dilate(ink_weight,np.ones((2,2),np.uint8));ink=cv2.dilate(ink,np.ones((2,2),np.uint8))
    for x0,y0,x1,y1 in unknown:
        center_weight[y0:y1,x0:x1]=0;ink_weight[y0:y1,x0:x1]=0
    direction_valid=dv*center_weight
    for k,(dy,dx) in enumerate(OFFSETS):
        shifted=np.zeros_like(center_weight);y0=max(0,-dy);y1=min(h,h-dy);x0=max(0,-dx);x1=min(w,w-dx)
        shifted[y0:y1,x0:x1]=center_weight[y0+dy:y1+dy,x0+dx:x1+dx];rm[k]*=center_weight*shifted
    # Confident dark ink in a junction is ink-only supervision, not invented centers.
    paper=cv2.dilate(gray,np.ones((25,25),np.uint8))
    for x0,y0,x1,y1 in unknown:
        deep=(paper[y0:y1,x0:x1]-gray[y0:y1,x0:x1])>.3
        ink[y0:y1,x0:x1][deep]=1;ink_weight[y0:y1,x0:x1][deep]=1
    t=np.stack([heat,ink,co,si,np.zeros_like(heat),direction_valid]);m=np.stack([center_weight,ink_weight,direction_valid,np.zeros_like(heat)])
    return t,m,rel,rm,accepted_paths,diagnostics

def build_actual(ident,rgb,split,paths=None,unknown=None,paper_boxes=None,partial=None):
    h,w=rgb.shape[:2];role=np.full((h,w),255,np.uint8);paper=np.zeros((h,w),np.float32)
    if paths:
        t,m,rel,rm,corrected,profiles=profile_reference(rgb,paths,unknown or [])
    else:t=np.zeros((6,h,w),np.float32);m=np.zeros((4,h,w),np.float32);rel=rm=np.zeros((8,h,w),np.float32);corrected=[];profiles=[]
    # Preserve previously reviewed isolated thin references; unknown elsewhere.
    if partial:
        q=[np.asarray(p,np.float32) for p in partial];heat,co,si,dv,pr,pm,center=path_fields(q,(h,w));from scipy.ndimage import distance_transform_edt
        dist=distance_transform_edt(center==0);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;bg=cv2.GaussianBlur(cv2.dilate(gray,np.ones((9,9),np.uint8)),(0,0),2);known=(dist<=1)|((dist>=2.5)&(dist<=4)&(bg-gray<.009));visible=(bg-gray>.015)&(dist<=2.2)
        t[0][known]=heat[known];m[0][known]=1;t[1][visible]=1;m[1][visible]=1;t[2]=co;t[3]=si;t[5]=dv;m[2]=dv;rel=pr;rm=pm;corrected=q
    for x0,y0,x1,y1 in paper_boxes or []:paper[y0:y1,x0:x1]=1
    assert not ((t[0]>.3)&(m[0]>0)&(paper>0)).any(),'Paper ROI overlaps reviewed center'
    for channel in [0,1]:t[channel][paper>0]=0;m[channel][paper>0]=1
    folder=ROOT/'actual'/split;meta=dict(split=split,kind='actual_refined',source_group=ident.split('_')[0],source_id=ident,strokes=[dict(points=p.tolist()) for p in corrected],expert_verified=False,reference_kind='assistant local cross-section measurements, not artist stroke truth',unknown_junction_rectangles=unknown or [],paper_boxes=paper_boxes or [],profile_decisions=profiles)
    row=write_record(folder,ident,rgb,t,m,role,rel,rm,meta)
    z=np.load(row['data']);data={k:z[k] for k in z.files};data['paper_mask']=paper.astype(np.uint8);np.savez_compressed(row['data'],**data);row['data_sha256']=digest(row['data'])
    return row

def prepare():
    if (ROOT/'manifest.json').exists():raise FileExistsError('Preserve prepared refinement data')
    ROOT.mkdir(exist_ok=True);records=[]
    old=read(PARENT/'manifest.json');records += [r for r in old['records'] if r['kind'] in ['existing_declared','hard_declared'] and r['split'] in ['train','calibration']]
    references=read(LOCAL/'actual_local_partial_references.json');extra=read(PARENT/'additional_local_reviewed.json');partial={}
    for r in references['records']+extra:
        if r['status']=='assistant_partial_positive':partial.setdefault(r['image_id'],[]).append(r['points'])
    for ident,points in partial.items():
        rgb=np.asarray(Image.open(LOCAL/'baseline'/ident/'input.png').convert('RGB'));split='calibration' if ident.startswith('G02') else 'train';boxes=[r['box'] for r in references['background_regions'] if r['image_id']==ident]
        records.append(build_actual(ident,rgb,split,paper_boxes=boxes,partial=points))
    for group,split,height in [(4,'train',40),(2,'calibration',20),(5,'regression',25)]:
        for i in range(1,7):
            ident=f'G{group:02}_{i:02}'
            if any(r['id']==ident for r in records):continue
            rgb=np.asarray(Image.open(LOCAL/'baseline'/ident/'input.png').convert('RGB'));h,w=rgb.shape[:2];boxes=[[8,0,w-8,height],[8,h-8,w-8,h]]
            if group==5:boxes.append([350,0,w-8,80])
            records.append(build_actual(ident,rgb,split,paper_boxes=boxes))
    for ident,spec in SPECS.items():
        src=LAB/'split_detail_structure_20261028/v4/real_regression'/ident/'input.png';rgb=np.asarray(Image.open(src).convert('RGB'))
        records.append(build_actual(ident,rgb,spec['split'],paths=spec['paths'],unknown=spec['unknown'],paper_boxes=spec['paper']))
    jsave(ROOT/'manifest.json',dict(records=records,parent_manifest_sha256=digest(PARENT/'manifest.json'),actual_group_split=dict(train=['G01','G03','G04','R04'],calibration=['G02','N11'],regression=['G05']),known_junction_ink_only=True,junction_stroke_identity_unknown=True,existing_artwork_is_reused=True,new_expert_gold=False,source_sha256=digest(__file__)))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);actual=[r for r in records if r['kind']=='actual_refined'];board=Image.new('RGB',(1280,((len(actual)+3)//4)*355),'#f3f1eb');draw=ImageDraw.Draw(board)
    for i,r in enumerate(actual):
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));z=np.load(r['data']);out=rgb.copy();out[z['paper_mask']>0]=(out[z['paper_mask']>0]*.4+np.array([30,190,80])*.6).astype(np.uint8);out[(z['target'][0]>.3)&(z['weight'][0]>0)]=[220,25,50]
        x=i%4*320;y=i//4*355;draw.text((x+4,y+4),r['id']+' '+r['split'],font=font,fill='black');im=Image.fromarray(out);im.thumbnail((316,316));board.paste(im,(x,y+30))
    board.save(ROOT/'SUPERVISION_REVIEW.jpg',quality=95)
    print('PAPER_CENTER_DATA',len(records),'actual',len(actual),flush=True)
if __name__=='__main__':prepare()
