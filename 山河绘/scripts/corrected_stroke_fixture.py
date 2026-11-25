"""Corrected continuous segment rendering; preserves the historical buggy renderer."""
import sys
sys.dont_write_bytecode=True
from joint_ink_structure_data import path_fields
import numpy as np,cv2

def fixture(seed,kind):
    rng=np.random.default_rng(seed);n=256;u=np.linspace(0,1,100);paths=[]
    paper=np.zeros((n,n,3),np.float32)+np.array([247,245,237]);paper+=cv2.resize(rng.normal(0,.8,(16,16)).astype(np.float32),(n,n))[:,:,None]
    yy,xx=np.mgrid[:n,:n];wash=np.exp(-((xx-125)**2/90**2+(yy-135)**2/100**2))*.32;paper=paper*(1-wash[:,:,None])+np.array([110,174,179])*wash[:,:,None]
    if kind=='parallel':
        x=90+12*np.sin(u*2.5);paths=[np.c_[x,25+u*205],np.c_[x+rng.uniform(3,5.5),25+u*205]]
    elif kind=='crossing':paths=[np.c_[30+u*195,40+u*175],np.c_[35+u*180,220-u*185]]
    elif kind=='junction':paths=[np.c_[30+u*195,90+12*np.sin(u*3)],np.c_[130+30*u,102+u*125]]
    elif kind=='dry_continuation':paths=[np.c_[50+28*np.sin(u*2),20+u*210]]
    elif kind=='needles':
        for angle in np.linspace(-2.9,-.2,10):paths.append(np.array([[130,155],[130+65*np.cos(angle),155+65*np.sin(angle)]],np.float32))
    ink=np.zeros((n,n),np.float32);width=np.zeros_like(ink);descriptions=[]
    for p in paths:
        alpha=float(rng.choice([.07,.12,.24,.55,.82]));base=float(rng.uniform(1.,3.) if kind in ['parallel','needles'] else rng.uniform(3,13));layer=np.zeros_like(ink)
        # Densify every segment, including a stroke declared by only TWO points.
        for i,(a,b) in enumerate(zip(p[:-1],p[1:])):
            count=max(2,int(np.ceil(np.linalg.norm(b-a)*3))+1)
            for s in np.linspace(0,1,count):
                progress=(i+s)/max(1,len(p)-1)
                if kind=='dry_continuation' and (int(progress*99)//2)%13==6:continue
                x,y=a*(1-s)+b*s;radius=max(0,int(round(base*(.30+.7*progress)/2)))
                cv2.circle(layer,(round(float(x)),round(float(y))),radius,alpha,-1)
        ink=1-(1-ink)*(1-layer);cv2.polylines(width,[np.rint(p).astype(np.int32)],False,base/8,1)
        descriptions.append(dict(points=p.tolist(),alpha=alpha,base_width_px=base))
    rgb=np.clip(paper*(1-ink[:,:,None])+25*ink[:,:,None]+rng.normal(0,.3,(n,n,1)),0,255).astype(np.uint8)
    if paths:heat,co,si,dv,rel,rm,center=path_fields(paths,(n,n))
    else:
        heat=co=si=dv=np.zeros((n,n),np.float32);rel=rm=np.zeros((8,n,n),np.float32);center=np.zeros((n,n),np.uint8)
    t=np.stack([heat,(ink>.025).astype(np.float32),co,si,width,dv]);m=np.stack([np.ones_like(heat),np.ones_like(heat),dv,dv]);role=np.full((n,n),255,np.uint8)
    return rgb,t,m,role,rel,rm,dict(strokes=descriptions,seed=seed,case=kind,reference_kind='declared continuous segments; sparse-vertex paint bug corrected',renderer_version=2)
