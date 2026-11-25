"""Area-driven RGB-cup landscape washes; no invented pigment mixing recipe.

Widths/opacity remain optical estimates. Existing bounded motion profile supplies
pressure; this planner never increases it to achieve a target width.
"""
import cv2
import numpy as np
import watercolor as wc


def analyze(work, scale):
    b,g,r=cv2.split(work.astype(np.float32))
    chroma=np.maximum.reduce([b,g,r])-np.minimum.reduce([b,g,r])
    gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
    ink=(gray<65)|((gray<145)&(chroma<10))
    painted=(chroma>4)&(gray<249)&~ink
    red=painted&((r-g)>35)&((r-g)>1.6*np.abs(g-b)+12)&((r-b)>30)
    blue=painted&~red&((b-r)>2)
    yellow=painted&~red&~blue&((r-b)>2)
    masks=dict(blue=blue,yellow=yellow,red=red)
    fields={}
    for name,mask in masks.items():
        basis=255-np.array(wc.PALETTE[name],np.float32)
        alpha=np.clip(((255-work.astype(np.float32))@basis)/(basis@basis),0,.8)
        alpha=cv2.GaussianBlur(alpha,(0,0),max(.6,.3/scale))
        fields[name]=(mask,alpha)
    return ink,fields


def plan_paths(work,scale,width=1.,gap=.35,ink_mask=None,max_strokes=1800):
    if not np.isfinite([scale,width,gap]).all() or scale<=0 or not .3<=width<=3 or gap<0:
        raise ValueError('Invalid landscape wash geometry')
    ink,fields=analyze(work,scale)
    if ink_mask is not None:ink=ink|np.asarray(ink_mask,bool)
    flow=wc.flow_field(work,scale)
    h,w=ink.shape
    strokes=[];regions=[]
    for name in wc.COLOR_ORDER:
        evidence,strength=fields[name]
        # Join pigment texture inside a wash, while preserving white source areas.
        rr=max(1,round(.45/scale))
        mask=cv2.morphologyEx(evidence.astype(np.uint8),cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rr+1,)*2))
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        for k in range(1,n):
            area=stats[k,cv2.CC_STAT_AREA]*scale*scale
            if area<5:continue
            region=labels==k
            rad=max(1,int(np.ceil(width/(2*scale))))
            centers=cv2.erode(region.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rad+1,)*2))>0
            ir=rad+int(np.ceil(gap/scale))
            centers &= cv2.dilate(ink.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*ir+1,)*2))==0
            # Keep the existing executor's 20mm margin at planning time.
            # Exact robot-polygon check is still performed by module_lines.
            edge=max(1,int(np.ceil(20/scale)))
            centers[:edge]=False;centers[-edge:]=False;centers[:,:edge]=False;centers[:,-edge:]=False
            x,y,rw,rh=map(int,stats[k,:4]);rid='%s-wash-%03d'%(name,k)
            field=flow
            if y>.58*h and rw>rh*2.5:
                field=np.broadcast_to(np.array([1.,0.]),(h,w,2))
            pitch=max(2,round(width*.9/scale))
            seeds=[(xx,yy) for yy in range(y+pitch//2,y+rh,pitch)
                   for xx in range(x+pitch//2,x+rw,pitch) if centers[yy,xx]]
            distance=cv2.distanceTransform(centers.astype(np.uint8),cv2.DIST_L2,5)
            seeds.sort(key=lambda p:-float(distance[p[1],p[0]]*(.3+strength[p[1],p[0]])))
            occupied=np.zeros((h,w),np.uint8);footprint=np.zeros((h,w),np.uint8)
            start=len(strokes)
            for xx,yy in seeds:
                if occupied[yy,xx]:continue
                valid=centers&(occupied==0)
                points=wc.trace((xx,yy),valid,field,max(1,.7/scale),35/scale)
                if len(points)<3:continue
                length=np.linalg.norm(np.diff(points,axis=0),axis=1).sum()*scale
                if length<2.5:continue
                alpha,widths,recipe=wc.stroke_profile(points,strength,width)
                # Broad body with short end tapers; no artificial color boost.
                widths=np.maximum(width*.85,np.array(widths)).tolist()
                stroke=dict(id='landscape-color-%04d'%len(strokes),color=name,region_id=rid,
                    recipe=recipe,layer=0,points=points.tolist(),target_opacity=alpha,
                    target_width_mm=widths,brush_width_mm_estimate=width,length_mm=float(length),
                    path_style='landscape_area_wash',pressure_mm=None,pigment_recipe_calibrated=False)
                strokes.append(stroke)
                pts=np.rint(points).astype(np.int32)
                cv2.polylines(occupied,[pts],False,1,max(1,round(width*1.35/scale)))
                cv2.polylines(footprint,[pts],False,1,max(1,round(width/scale)))
                if len(strokes)>=max_strokes:break
            regions.append(dict(id=rid,color=name,area_mm2=float(area),bounds_px=[x,y,rw,rh],
                stroke_count=len(strokes)-start,status='planned' if len(strokes)>start else 'too_narrow_skip',
                planned_coverage_fraction=float(np.sum((footprint>0)&region)/np.sum(region))))
            if len(strokes)>=max_strokes:break
        if len(strokes)>=max_strokes:break
    return strokes,regions
