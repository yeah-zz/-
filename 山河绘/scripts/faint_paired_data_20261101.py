# -*- coding: utf-8 -*-
from faint_joint_model_20261101 import *
from faint_partial_references_20261101 import path_data
from PIL import Image
from scipy.ndimage import distance_transform_edt
import math

def synth(seed,n=192):
    rng=np.random.default_rng(seed);y,x=np.mgrid[:n,:n].astype(np.float32);paper=np.array([rng.uniform(224,250),rng.uniform(211,240),rng.uniform(181,222)],np.float32);rgb=np.ones((n,n,3),np.float32)*paper;rgb+=cv2.resize(rng.normal(0,1,(8,8)).astype(np.float32),(n,n))[:,:,None]*rng.uniform(.3,2.)
    angle=rng.uniform(0,2*math.pi);signed=(x-n/2)*math.cos(angle)+(y-n/2)*math.sin(angle)+rng.uniform(-50,50);blur=rng.uniform(1.5,18);wash=1/(1+np.exp(np.clip(signed/blur,-30,30)));color=np.array([rng.uniform(85,160),rng.uniform(150,205),rng.uniform(145,215)]);strength=rng.uniform(.12,.65);rgb=rgb*(1-wash[:,:,None]*strength)+color*wash[:,:,None]*strength;boundary=abs(signed)<1.8
    target=np.zeros((5,n,n),np.float32);known=np.ones((4,n,n),np.float32);known[2]=0;target[4]=boundary;ink=np.zeros((n,n),np.float32);owner=np.zeros((n,n),np.int16);paths=[];centers=[];dirfields=[]
    count=0 if seed%9==0 else int(rng.integers(1,5))
    for k in range(count):
        t=np.linspace(0,1,100);theta=rng.uniform(0,2*math.pi);u=np.array([math.cos(theta),math.sin(theta)]);normal=np.array([-u[1],u[0]]);center=rng.uniform(55,n-55,2);length=rng.uniform(90,190);p=center[None]+u[None]*(t[:,None]-.5)*length+normal[None]*np.sin(t[:,None]*math.pi)*rng.uniform(-35,35);inside=(p[:,0]>5)&(p[:,0]<n-5)&(p[:,1]>5)&(p[:,1]<n-5);p=p[inside]
        if len(p)<20:continue
        width=float(rng.choice([.8,1.2,1.8,2.8,4.,6.]));alpha=float(rng.choice([.025,.045,.075,.12,.2,.35,.6]));c,d,di=path_data(p,(n,n));layer=np.exp(-.5*(d/max(.4,width/2.355))**2).astype(np.float32)*alpha
        # Some exactly declared strokes have dry gaps. Pixel supervision does not invent real artist identity.
        if seed%4==0:
            a=np.linspace(0,1,len(p));gate=np.ones(len(p),np.float32)
            for _ in range(2):z=rng.uniform(.1,.9);gate[abs(a-z)<rng.uniform(.012,.045)]=rng.uniform(0,.1)
            grid=np.zeros((n,n),np.float32);xy=np.rint(p).astype(int);grid[xy[:,1],xy[:,0]]=gate;_,indices=distance_transform_edt(~c,return_indices=True);layer*=grid[indices[0],indices[1]]
        ink=1-(1-ink)*(1-layer);visible=layer>.009;target[0]=np.maximum(target[0],np.exp(-.5*(d/.8)**2)*visible);nearest=d<2;overlap=nearest&(owner>0);target[2:4,nearest]=di[:,nearest];known[2,nearest]=visible[nearest];known[2,overlap]=0;owner[nearest]=k+1;paths.append(dict(points=p.tolist(),width_px=width,opacity=alpha));centers.append(c);dirfields.append(di)
    brown=np.array([rng.uniform(40,110),rng.uniform(35,90),rng.uniform(20,70)]);rgb=rgb*(1-ink[:,:,None])+brown*ink[:,:,None];target[1]=np.clip(ink/.12,0,1);target[4,ink>.009]=0
    # Moss dots and paper fibres are not declared center strokes.
    for _ in range(int(rng.integers(0,25))):
        dot=np.zeros((n,n),np.uint8);cv2.circle(dot,(int(rng.integers(n)),int(rng.integers(n))),int(rng.integers(1,4)),1,-1);a=dot*rng.uniform(.04,.3);rgb=rgb*(1-a[:,:,None])+brown*a[:,:,None];target[1]=np.maximum(target[1],a/.3)
    noise=cv2.GaussianBlur(rng.normal(0,rng.uniform(.5,2.),(n,n)).astype(np.float32),(0,0),rng.uniform(.25,.7));rgb=np.clip(rgb+noise[:,:,None],0,255).astype(np.uint8);return rgb,target,known,dict(seed=seed,declared_paths=paths,procedural_not_artist_paintings=True)

def prepare():
    folder=ROOT/'paired_data';folder.mkdir(exist_ok=True);records=[]
    for split,start,count in [('train',11011000,320),('calibration',11012000,48),('test',11013000,48)]:
        for i in range(count):
            seed=start+i;file=folder/(str(seed)+'.npz');rgb,t,k,meta=synth(seed);np.savez_compressed(file,rgb=rgb,target=t.astype(np.float16),known=k.astype(np.float16));records.append(dict(seed=seed,split=split,file=str(file),sha256=digest(file),kind='exact_declared_vector_fixture'))
    real=[]
    for r in read(ROOT/'audited_reference_manifest.json')['records']:
        z=np.load(r['label_file']);target=z['target'].copy();known=z['known'].copy()
        # Real center locations are uncertain: use weak ink/direction constraints,
        # not exact positive pixel-center supervision from a photometric trace.
        known[0,target[0]>.5]=0;known[2]*=.4
        file=folder/(r['id']+'_partial.npz');np.savez_compressed(file,rgb=np.asarray(Image.open(r['path']).convert('RGB')),target=target.astype(np.float16),known=known.astype(np.float16));real.append(dict(id=r['id'],split=r['split'],parent=r['image'],file=str(file),sha256=digest(file),kind='assistant_partial_ink_direction_and_boundary_soft_constraints',precise_real_center_supervision=False))
    jsave(ROOT/'paired_manifest.json',dict(synthetic=records,real=real,real_parent_count=8,real_training_parent_count=4,synthetic_patch_count=416,synthetic_patches_are_not_new_paintings=True,reference_uncertainty='Real selected centers are not precise enough to treat as positive pixel-center gold; train centers on declared vectors, use real partial ink/direction and boundary/blank negatives.',test_parent_overlap_with_train=False));print('FAINT_PAIRED_DATA',len(records),len(real),flush=True)
if __name__=='__main__':prepare()
