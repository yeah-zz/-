# -*- coding: utf-8 -*-
"""Procedural supervised fixtures with exact declared strokes, not artist gold."""
import os,sys
from pathlib import Path
sys.dont_write_bytecode=True
from rock_v15_common import LAB,jsave,read,digest
import numpy as np,cv2,math,json,hashlib
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'simple_landscape_specialist_20261029'/'round2'
ROLES={0:'background',1:'mountain_contour',2:'long_cun',3:'pine',4:'pavilion_water'}

def bezier(c,n=90):
    c=np.asarray(c,np.float32);t=np.linspace(0,1,n)[:,None];return (1-t)**3*c[0]+3*(1-t)**2*t*c[1]+3*(1-t)*t*t*c[2]+t**3*c[3]

def smooth(points,n=20):
    p=np.asarray(points,np.float32);pp=np.r_[p[:1],p,p[-1:]];out=[]
    for i in range(1,len(pp)-2):
        a,b,c,d=pp[i-1:i+3];t=np.linspace(0,1,n,endpoint=False)[:,None]
        out.append(.5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t))
    return np.r_[np.concatenate(out),p[-1:]]

def scene(seed,size=384):
    rng=np.random.default_rng(seed);N=size;s=N/384;paper=rng.uniform(244,253);broad=seed%4!=0
    rgb=np.ones((N,N,3),np.float32)*np.array([paper,paper-rng.uniform(0,3),paper-rng.uniform(2,6)])
    texture=cv2.resize(rng.normal(0,.7,(N//8,N//8)).astype(np.float32),(N,N));rgb+=texture[:,:,None]
    paths=[];regions=[]
    def stroke(p,role,width=None,alpha=None):
        p=np.asarray(p,np.float32)*N
        if len(p)>10 and broad:
            tangent=np.gradient(p,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-6);normal=np.c_[-tangent[:,1],tangent[:,0]]
            phase=np.linspace(0,rng.uniform(8,20),len(p));p+=normal*(np.sin(phase)*rng.uniform(.2,.7)*s)[:,None]
        valid=(p[:,0]>=3)&(p[:,0]<N-3)&(p[:,1]>=3)&(p[:,1]<N-3)
        if not valid.all():return
        multiplier=float(rng.choice([1.,1.5,2.5,3.5])) if broad else 1.
        if role==3:multiplier=min(multiplier,2.)
        paths.append(dict(points=p,role=role,width=float(width if width else rng.uniform(.7,2.3))*s*multiplier,alpha=float(alpha if alpha else rng.uniform(.38,.85)),phase=float(rng.uniform(0,6.28))))
    count=int(rng.integers(1,4));centers=(np.array([.50]) if count==1 else np.linspace(.25,.75,count))+rng.uniform(-.035,.035,count)
    for j,cx in enumerate(centers):
        rx=float(rng.uniform(.13,.22));top=float(rng.uniform(.15,.40));base=float(rng.uniform(.66,.79));height=base-top
        control=[[cx-rx,base],[cx-rx*.90,top+height*.60],[cx-rx*.62,top+height*.42],[cx-rx*.47,top+height*.15],[cx-rx*.15,top+height*.08],[cx+rx*.05,top],[cx+rx*.34,top+height*.18],[cx+rx*.57,top+height*.43],[cx+rx*.87,top+height*.70],[cx+rx,base]]
        outline=smooth(control);poly=np.r_[outline,[[cx+rx,base+.025],[cx-rx,base+.025]]]*N
        mask=np.zeros((N,N),np.uint8);cv2.fillPoly(mask,[np.rint(poly).astype(np.int32)],1)
        yy=np.arange(N)[:,None]/N;fade=np.clip((base+.04-yy)/.20,0,1)
        wash=cv2.GaussianBlur(mask.astype(np.float32),(0,0),max(1,s*2))*fade*rng.uniform(.12,.35)
        palette=np.array([rng.uniform(90,145),rng.uniform(150,195),rng.uniform(155,200)])
        rgb=rgb*(1-wash[:,:,None])+palette*wash[:,:,None]
        rgb+=cv2.GaussianBlur(rng.normal(0,.6,(N,N)).astype(np.float32),(0,0),.7)[:,:,None]*mask[:,:,None]
        stroke(outline,1,width=rng.uniform(1.1,2.9),alpha=rng.uniform(.50,.91))
        for k in range(int(rng.integers(3,8))):
            x=cx+rx*rng.uniform(-.62,.62);y=top+height*rng.uniform(.25,.5);endx=x+rx*rng.uniform(-.25,.25)
            p=bezier([[x,y],[x-rx*.20,y+height*.13],[endx-rx*.10,base-height*.1],[endx,base-rng.uniform(0,.05)]])
            xy=np.rint(p*N).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<N)&(xy[:,1]>=0)&(xy[:,1]<N)
            if inside.all() and (mask[xy[:,1],xy[:,0]]>0).mean()>.92:stroke(p,2,width=rng.uniform(.55,1.5),alpha=rng.choice([.12,.20,.35,.55,.78]))
        regions.append(dict(points=poly.tolist(),role='wash_not_ink'))
        if broad:
            for k in range(int(rng.integers(3,7))):
                q=outline[int(rng.integers(12,len(outline)-12))];tip=q+[rng.uniform(-.006,.006),-rng.uniform(.005,.014)]
                stroke([q,tip],2,width=rng.uniform(1.,2.2),alpha=rng.uniform(.4,.8))
    # Pines use explicit thin needles, not a texture image without stroke labels.
    if rng.random()<.72:
        x=float(rng.choice([.15,.79]));root=.78;ht=float(rng.uniform(.20,.32));lean=float(rng.uniform(-.045,.045));topx=x+lean
        trunk=bezier([[x,root],[x+.02,root-ht*.38],[topx-.02,root-ht*.7],[topx,root-ht]])
        stroke(trunk+[-.006,0],3,width=1.5,alpha=.8);stroke(trunk+[.006,0],3,width=1.3,alpha=.7)
        fans=[]
        for k in range(int(rng.integers(3,6))):
            frac=float(rng.uniform(.45,1));side=float(rng.choice([-1,1]));start=np.array([x+lean*frac,root-ht*frac]);end=start+[side*rng.uniform(.04,.10),-rng.uniform(.01,.05)]
            stroke(bezier([start,start+[side*.025,-.015],end-[side*.02,0],end]),3,width=.95,alpha=.75);fans.append(end)
        for end in fans:
            for ang in np.linspace(-math.pi*.97,-.03*math.pi,int(rng.integers(7,12))):
                length=float(rng.uniform(.017,.038));p=np.array([end,end+length*np.array([math.cos(ang),math.sin(ang)])]);stroke(p,3,width=rng.uniform(.6,1.05),alpha=rng.uniform(.55,.85))
    if rng.random()<.42:
        x=float(rng.uniform(.57,.80));y=float(rng.uniform(.71,.79));w=.11;h=.095
        for dx in [-w*.35,w*.35]:stroke([[x+dx,y],[x+dx,y-h*.65]],4,width=1.1,alpha=.75)
        stroke([[x-w*.6,y-h*.6],[x-w*.35,y-h*.70],[x,y-h],[x+w*.35,y-h*.7],[x+w*.6,y-h*.6]],4,width=1.3,alpha=.8)
        stroke([[x-w*.6,y-h*.6],[x,y-h*.7],[x+w*.6,y-h*.6]],4,width=.8,alpha=.65)
        stroke([[x-w*.5,y],[x+w*.5,y]],4,width=1.2,alpha=.7)
    for k in range(int(rng.integers(2,6))):
        x=float(rng.uniform(.12,.8));y=float(rng.uniform(.82,.93));w=float(rng.uniform(.05,.20))
        if x+w<.96:stroke(bezier([[x,y],[x+w*.33,y-.002],[x+w*.66,y+.002],[x+w,y]],35),4,width=rng.uniform(.6,1),alpha=rng.uniform(.15,.5))
    center=np.zeros((N,N),np.uint8);ink=np.zeros((N,N),np.float32);width=np.zeros((N,N),np.float32);co=np.zeros((N,N),np.float32);si=np.zeros_like(co);votes=np.zeros_like(co);roles=np.zeros((N,N),np.uint8)
    declared=[]
    for inst,row in enumerate(paths):
        p=row['points'];lo=np.maximum(np.floor(p.min(0)-row['width']*2-4).astype(int),0);hi=np.minimum(np.ceil(p.max(0)+row['width']*2+5).astype(int),N);x0,y0=lo;x1,y1=hi
        scale=3;mask=np.zeros(((y1-y0)*scale,(x1-x0)*scale),np.float32)
        length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];u=length/max(1,length[-1]);profile=row['width']*(.35+.65*np.sin(np.pi*(.1+.8*u)))*(.9+.1*np.sin(u*5+row['phase']))
        # Dense dab positions preserve the declared path while varying width.
        for a,b,wa,wb in zip(p[:-1],p[1:],profile[:-1],profile[1:]):
            n=max(2,int(np.ceil(np.linalg.norm(b-a)*2))+1)
            for t in np.linspace(0,1,n):
                q=((a*(1-t)+b*t-lo)*scale);radius=max(1,int(round((wa*(1-t)+wb*t)*scale/2)))
                cv2.circle(mask,tuple(np.rint(q).astype(int)),radius,row['alpha'],-1)
        mask=cv2.resize(mask,(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);mask=cv2.GaussianBlur(mask,(0,0),.30)
        if broad:
            grain=cv2.resize(rng.uniform(.35,1.,(max(2,(y1-y0)//2),max(2,(x1-x0)//2))).astype(np.float32),(x1-x0,y1-y0))
            mask*=grain
        inkcolor=rng.uniform(12,45);rgb[y0:y1,x0:x1]=rgb[y0:y1,x0:x1]*(1-mask[:,:,None])+inkcolor*mask[:,:,None]
        ink[y0:y1,x0:x1]=1-(1-ink[y0:y1,x0:x1])*(1-mask)
        patch=roles[y0:y1,x0:x1];patch[mask>.025]=row['role']
        ij=np.rint(p).astype(np.int32);cv2.polylines(center,[ij],False,1,1,cv2.LINE_8)
        t=np.gradient(p,axis=0);angle=np.arctan2(t[:,1],t[:,0])
        for (x,y),theta,ww in zip(ij,angle,profile):
            cv2.circle(co,(int(x),int(y)),1,float(math.cos(2*theta)), -1);cv2.circle(si,(int(x),int(y)),1,float(math.sin(2*theta)),-1);cv2.circle(votes,(int(x),int(y)),1,1.,-1);cv2.circle(width,(int(x),int(y)),1,float(ww/8),-1)
        declared.append(dict(id=inst,role=ROLES[row['role']],points=p.round(4).tolist(),width_profile_px=profile.round(4).tolist(),alpha=row['alpha']))
    heat=cv2.GaussianBlur(center.astype(np.float32),(0,0),.7);heat/=max(float(heat.max()),1e-6)
    # Directions at high-degree intersections are ambiguous for a single vector.
    degree=cv2.filter2D(center.astype(np.float32),-1,np.ones((3,3),np.float32))-center
    valid=votes*(cv2.dilate((degree>3).astype(np.uint8),np.ones((5,5),np.uint8))==0)
    target=np.stack([heat,(ink>.025).astype(np.float32),co,si,width,valid]).astype(np.float16)
    return np.clip(rgb,0,255).astype(np.uint8),target,roles,dict(seed=int(seed),size=[N,N],strokes=declared,regions=regions,reference_kind='exact_procedural_declared_paths_not_artist_annotations')

def prepare():
    protocol=ROOT/'paired_manifest.json'
    if protocol.exists():raise FileExistsError('Preserve completed paired fixtures')
    specs=[('train',400000,512),('calibration',500000,64),('test',600000,40)];records=[]
    for split,start,count in specs:
        folder=ROOT/'paired'/split;folder.mkdir(parents=True,exist_ok=True)
        for i in range(count):
            seed=start+i;rgb,target,role,doc=scene(seed)
            sid=f'{split}_{i:04d}';image=folder/(sid+'.png');data=folder/(sid+'.npz');label=folder/(sid+'_reference.json')
            Image.fromarray(rgb).save(image);np.savez_compressed(data,target=target,role=role);jsave(label,doc)
            records.append(dict(id=sid,split=split,seed=seed,image=str(image),data=str(data),reference=str(label),sha256=digest(image)))
            if i%64==0:print('PAIR_RENDER',split,i,count,flush=True)
    jsave(protocol,dict(records=records,counts={s:c for s,_,c in specs},roles=ROLES,source='procedural brush rendering, not image_gen and not human artist gold',splits_disjoint_by_seed=True,test_labels_never_used_for_training_or_threshold_selection=True,synthetic_style_family_shared=True,renderer_sha256=digest(__file__),imagegen_holdouts_are_separate_unlabeled_domain_shift_check=True))
    # Small dataset-inspection board only; no extraction result shown here.
    board=Image.new('RGB',(1536,4*430),'#f3f1eb');d=ImageDraw.Draw(board);f=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',18)
    for j,r in enumerate([r for r in records if r['split']=='train'][:16]):
        x=j%4*384;y=j//4*430;d.text((x+4,y+4),r['id']+' 配对训练输入',font=f,fill='black');board.paste(Image.open(r['image']),(x,y+30))
    board.save(ROOT/'PAIRED_TRAIN_INPUTS.jpg',quality=94)
    print('PAIRS_READY',json.dumps({s:c for s,_,c in specs}),flush=True)

if __name__=='__main__':prepare()
