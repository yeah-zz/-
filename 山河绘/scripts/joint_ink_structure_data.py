"""Joint supervision: declared renderer paths plus explicitly partial source labels.

No v4 full-image output is promoted to a target. Unknown pixels have zero weight.
"""
import sys
sys.dont_write_bytecode=True
from rock_v15_common import LAB,read,jsave,digest
from pathlib import Path
import numpy as np,cv2
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'joint_ink_structure_20261029'
R2=LAB/'simple_landscape_specialist_20261029/round2'
LOCAL=LAB/'v4_local_repairs_20261029'
OFFSETS=[(0,1),(1,0),(1,1),(1,-1),(0,3),(3,0),(3,3),(3,-3)]

def path_fields(paths,shape):
    h,w=shape;center=np.zeros(shape,np.uint8);ids=np.zeros(shape,np.int32);votes=np.zeros(shape,np.uint16);co=np.zeros(shape,np.float32);si=co.copy()
    for k,p in enumerate(paths,1):
        p=np.asarray(p,np.float32);m=np.zeros(shape,np.uint8);cv2.polylines(m,[np.rint(p).astype(np.int32)],False,1,1)
        center|=m;votes+=cv2.dilate(m,np.ones((3,3),np.uint8));ids[m>0]=k
        t=np.gradient(p,axis=0);angle=np.arctan2(t[:,1],t[:,0]);xy=np.rint(p).astype(int)
        for a,b,theta in zip(xy[:-1],xy[1:],angle[:-1]):
            cv2.line(co,tuple(a),tuple(b),float(np.cos(2*theta)),1)
            cv2.line(si,tuple(a),tuple(b),float(np.sin(2*theta)),1)
    distance,idx=distance_transform_edt(center==0,return_indices=True)
    nearest=ids[tuple(idx)];near=(distance<=1.4)&(votes<=1)
    rel=np.zeros((len(OFFSETS),h,w),np.float32);valid=np.zeros_like(rel)
    for k,(dy,dx) in enumerate(OFFSETS):
        y0=max(0,-dy);y1=min(h,h-dy);x0=max(0,-dx);x1=min(w,w-dx)
        a=(slice(y0,y1),slice(x0,x1));b=(slice(y0+dy,y1+dy),slice(x0+dx,x1+dx))
        valid[k][a]=near[a]&near[b]&(nearest[a]>0)&(nearest[b]>0)
        rel[k][a]=(nearest[a]==nearest[b]).astype(np.float32)
    heat=np.exp(-distance**2/(2*.7**2)).astype(np.float32)
    return heat,co[tuple(idx)],si[tuple(idx)],near.astype(np.float32),rel,valid,center

def write_record(folder,ident,rgb,t,m,role,rel,rm,meta):
    folder.mkdir(parents=True,exist_ok=True);image=folder/(ident+'.png');data=folder/(ident+'.npz');reference=folder/(ident+'_reference.json')
    Image.fromarray(rgb).save(image);np.savez_compressed(data,target=t.astype(np.float16),weight=m.astype(np.float16),role=role.astype(np.uint8),relation=rel.astype(np.float16),relation_weight=rm.astype(np.float16))
    jsave(reference,meta)
    return dict(id=ident,image=str(image),data=str(data),reference=str(reference),sha256=digest(image),data_sha256=digest(data),**{k:meta[k] for k in ['split','kind','source_group']})

def hard_fixture(seed,kind):
    rng=np.random.default_rng(seed);n=256;u=np.linspace(0,1,100);paths=[]
    paper=np.zeros((n,n,3),np.float32)+np.array([247,245,237]);paper+=cv2.resize(rng.normal(0,.8,(16,16)).astype(np.float32),(n,n))[:,:,None]
    yy,xx=np.mgrid[:n,:n];wash=np.exp(-((xx-125)**2/90**2+(yy-135)**2/100**2))*.32
    paper=paper*(1-wash[:,:,None])+np.array([110,174,179])*wash[:,:,None]
    if kind=='parallel':
        x=90+12*np.sin(u*2.5);paths=[np.c_[x,25+u*205],np.c_[x+rng.uniform(3,5.5),25+u*205]]
    elif kind=='crossing':paths=[np.c_[30+u*195,40+u*175],np.c_[35+u*180,220-u*185]]
    elif kind=='junction':paths=[np.c_[30+u*195,90+12*np.sin(u*3)],np.c_[130+30*u,102+u*125]]
    elif kind=='dry_continuation':paths=[np.c_[50+28*np.sin(u*2),20+u*210]]
    elif kind=='needles':
        for angle in np.linspace(-2.9,-.2,10):paths.append(np.array([[130,155],[130+65*np.cos(angle),155+65*np.sin(angle)]],np.float32))
    elif kind=='blank':paths=[]
    ink=np.zeros((n,n),np.float32);width=np.zeros((n,n),np.float32)
    for j,p in enumerate(paths):
        a=float(rng.choice([.07,.12,.24,.55,.82]));base=float(rng.uniform(1.0,3.0) if kind in ['parallel','needles'] else rng.uniform(3,13));layer=np.zeros_like(ink)
        for i,(x,y) in enumerate(p):
            if kind=='dry_continuation' and (i//2)%13==6:continue
            radius=max(0,int(round(base*(.30+.7*i/max(1,len(p)-1))/2)))
            cv2.circle(layer,(round(float(x)),round(float(y))),radius,a,-1)
        ink=1-(1-ink)*(1-layer)
        cv2.polylines(width,[np.rint(p).astype(np.int32)],False,base/8,1)
    rgb=np.clip(paper*(1-ink[:,:,None])+25*ink[:,:,None]+rng.normal(0,.3,(n,n,1)),0,255).astype(np.uint8)
    if paths:
        heat,co,si,dv,rel,rm,center=path_fields(paths,(n,n))
    else:
        heat=co=si=dv=np.zeros((n,n),np.float32);rel=rm=np.zeros((8,n,n),np.float32);center=np.zeros((n,n),np.uint8)
    t=np.stack([heat,(ink>.025).astype(np.float32),co,si,width,dv]);m=np.stack([np.ones_like(heat),np.ones_like(heat),dv,dv])
    role=np.full((n,n),255,np.uint8)
    return rgb,t,m,role,rel,rm,dict(strokes=[dict(id=i,points=p.tolist()) for i,p in enumerate(paths)],seed=seed,case=kind,reference_kind='declared procedural paths, not artist strokes')

def partial_reference(row,split):
    ident=row['image_id'];rgb=np.asarray(Image.open(LOCAL/'baseline'/ident/'input.png').convert('RGB'));h,w=rgb.shape[:2]
    if 'points' in row:
        p=np.asarray(row['points'],np.float32);cx,cy=p.mean(0);x0=int(np.clip(cx-128,0,w-256));y0=int(np.clip(cy-128,0,h-256));p=p-[x0,y0]
        patch=rgb[y0:y0+256,x0:x0+256].copy();heat,co,si,dv,rel,rm,center=path_fields([p],(256,256))
        dist=distance_transform_edt(center==0);gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        paper=cv2.GaussianBlur(cv2.dilate(gray,np.ones((9,9),np.uint8)),(0,0),2)
        visible=(paper-gray>.015)&(dist<=2.2)
        # Nearby dark ink outside the labeled line stays UNKNOWN, not negative.
        positive=dist<=1.;clear_shoulder=(dist>=2.5)&(dist<=4.)&(paper-gray<.009)
        m=np.stack([(positive|clear_shoulder).astype(np.float32),(visible|clear_shoulder).astype(np.float32),dv,dv*0])
        t=np.stack([heat,visible.astype(np.float32),co,si,np.zeros_like(heat),dv]);role=np.full((256,256),255,np.uint8)
        meta=dict(partial_points=p.tolist(),stroke_identity_verified=False,center_tolerance_px=1.5,ink_scope='approximate local photometric support, not expert ink-band labels')
    else:
        l,top,r,b=row['box'];x0=int(np.clip((l+r)/2-128,0,w-256));y0=int(np.clip((top+b)/2-128,0,h-256));patch=rgb[y0:y0+256,x0:x0+256].copy()
        t=np.zeros((6,256,256),np.float32);m=np.zeros((4,256,256),np.float32);m[:2,top-y0:b-y0,l-x0:r-x0]=1
        role=np.full((256,256),255,np.uint8);rel=rm=np.zeros((8,256,256),np.float32);meta=dict(background_box=row['box'])
    meta.update(split=split,kind='actual_partial',source_group=ident.split('_')[0],source_id=ident,source_sha256=digest(LOCAL/'baseline'/ident/'input.png'),assistant_review=True,expert_verified=False,unknown_outside_supervision=True,neural_test=False)
    return write_record(ROOT/'partial'/split,row['id'],patch,t,m,role,rel,rm,meta)

def prepare():
    if (ROOT/'manifest.json').exists():raise FileExistsError('Preserve prepared experiment')
    ROOT.mkdir(parents=True,exist_ok=True);records=[]
    old=read(R2/'paired_manifest.json')['records']
    for split,count in [('train',256),('calibration',24),('test',24)]:
        for r in [r for r in old if r['split']==split][:count]:
            z=np.load(r['data']);t=z['target'].astype(np.float32);doc=read(r['reference']);paths=[np.asarray(p['points'],np.float32) for p in doc['strokes']]
            _,_,_,valid,rel,rm,_=path_fields(paths,t.shape[-2:]);m=np.stack([np.ones_like(t[0]),np.ones_like(t[0]),t[5],t[5]])
            records.append(dict(id='old_'+r['id'],image=r['image'],data=r['data'],reference=r['reference'],sha256=r['sha256'],split=split,kind='existing_declared',source_group='renderer_seed_'+str(r['seed'])))
            # Cache derived relations without altering original targets/files.
            p=ROOT/'derived'/('old_'+r['id']+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(p,target=t.astype(np.float16),weight=m.astype(np.float16),role=z['role'],relation=rel.astype(np.float16),relation_weight=rm.astype(np.float16));records[-1]['data']=str(p);records[-1]['data_sha256']=digest(p)
        print('DERIVED',split,count,flush=True)
    kinds=['parallel','crossing','junction','dry_continuation','needles','blank']
    for split,count,start in [('train',192,710000),('calibration',24,720000),('test',36,730000)]:
        for i in range(count):
            rgb,t,m,role,rel,rm,meta=hard_fixture(start+i,kinds[i%6]);meta.update(split=split,kind='hard_declared',source_group='hard_seed_'+str(start+i))
            records.append(write_record(ROOT/'hard'/split,f'hard_{split}_{i:03}',rgb,t,m,role,rel,rm,meta))
        print('HARD_READY',split,count,flush=True)
    refs=read(LOCAL/'actual_local_partial_references.json')
    for row in refs['records']+refs['background_regions']:
        if row['status'] not in ['assistant_partial_positive','assistant_background']:continue
        records.append(partial_reference(row,'train' if row['image_id'].startswith('G01') else 'calibration'))
    jsave(ROOT/'manifest.json',dict(records=records,offsets_dy_dx=OFFSETS,actual_train_source_groups=['G01'],actual_calibration_source_groups=['G02'],unknown_proposals_ignored=[r['id'] for r in refs['records'] if r['status']=='unknown'],existing_test_is_reused=True,new_hard_test_has_new_seeds=True,actual_test_labels_available=False,v4_outputs_used_as_full_image_targets=False,split_unit='renderer seed or whole generated parent sheet',source_sha256=digest(__file__)))
    # Visual label audit, only training examples; unknown is gray, positive red.
    cards=[r for r in records if r['kind']=='actual_partial'];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(1024,((len(cards)+3)//4)*290),'#f3f1eb');draw=ImageDraw.Draw(board)
    for k,r in enumerate(cards):
        rgb=np.asarray(Image.open(r['image']).convert('RGB')).copy();z=np.load(r['data']);m=z['weight'];t=z['target'];rgb[(m[0]>0)&(t[0]>.3)]=[230,20,60];rgb[(m[0]>0)&(t[0]<.05)]=[20,170,80]
        x=k%4*256;y=k//4*290;draw.text((x+3,y+4),r['id']+' '+r['split'],font=font,fill='black');board.paste(Image.fromarray(rgb),(x,y+28))
    board.save(ROOT/'ACTUAL_PARTIAL_SUPERVISION.jpg',quality=94)
    print('JOINT_DATA_READY',len(records),str(ROOT),flush=True)
if __name__=='__main__':prepare()
