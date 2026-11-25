"""Group-balanced real weak supervision mixed with known synthetic geometry."""
import json
from collections import OrderedDict
import cv2
import numpy as np
import torch
from PIL import Image

from .runtime import LAB


class TileSampler:
    def __init__(self,split='train',size=320,manifest_path=None,augmentation='v1'):
        self.size=size;self.split=split;self.augmentation=augmentation
        self.manifest=json.loads((manifest_path or LAB/'datasets'/'manifest.json').read_text(encoding='utf-8'))
        self.real=[r for r in self.manifest['real'] if r['split']==split]
        self.synthetic=[r for r in self.manifest['synthetic'] if r['split']==split]
        self.groups={}
        for r in self.real:self.groups.setdefault(r['group'],[]).append(r)
        self.group_names=sorted(self.groups)
        self.cache=OrderedDict()

    def load(self,r,kind):
        key=r['id']
        if key not in self.cache:
            rgb=np.asarray(Image.open(r['image']).convert('RGB')).copy()
            with np.load(r['labels']) as labels:
                targets=labels['targets'].astype(np.float32)/255.
                weights=labels['weights'].astype(np.float32)/255. if 'weights' in labels else np.ones_like(targets)
            if kind=='real':weights*=.5
            self.cache[key]=(rgb,targets,weights)
            if len(self.cache)>40:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def sample(self,rng,real_probability=.4,augment=True,forced_record=None):
        real=forced_record is not None or bool(rng.random()<real_probability and self.real)
        if forced_record is not None:r=forced_record
        elif real:
            # User's one painting group is a target-style reference, but does
            # not become seven independent groups just because it has 7 crops.
            if self.split=='train' and 'wang_ximeng_qianli_jiangshan' in self.groups and rng.random()<.22:
                group='wang_ximeng_qianli_jiangshan'
            else:group=self.group_names[int(rng.integers(len(self.group_names)))]
            pool=self.groups[group];r=pool[int(rng.integers(len(pool)))]
        else:r=self.synthetic[int(rng.integers(len(self.synthetic)))]
        image,target,weight=self.load(r,'real' if real else 'synthetic')
        s=self.size
        if augment and self.augmentation in ('v2','ink'):
            s=round(self.size/float(rng.choice([.65,.85,1.,1.25,1.6])))
        h,w=image.shape[:2]
        bottom=max(0,s-h);right=max(0,s-w)
        if bottom or right:
            image=cv2.copyMakeBorder(image,0,bottom,0,right,cv2.BORDER_REFLECT_101)
            target=np.pad(target,((0,0),(0,bottom),(0,right)))
            weight=np.pad(weight,((0,0),(0,bottom),(0,right)))
        h,w=image.shape[:2]
        for _ in range(4):
            x=int(rng.integers(max(1,w-s+1)));y=int(rng.integers(max(1,h-s+1)))
            if rng.random()<.2 or target[0,y:y+s,x:x+s].mean()>.002:break
        image=image[y:y+s,x:x+s].copy()
        target=target[:,y:y+s,x:x+s].copy();weight=weight[:,y:y+s,x:x+s].copy()
        if s!=self.size:
            image=cv2.resize(image,(self.size,self.size),interpolation=cv2.INTER_AREA if s>self.size else cv2.INTER_LINEAR)
            target=np.stack([cv2.resize(t,(self.size,self.size),interpolation=cv2.INTER_LINEAR) for t in target])
            weight=np.stack([cv2.resize(t,(self.size,self.size),interpolation=cv2.INTER_NEAREST) for t in weight])
        if augment:
            if rng.random()<.5:image=image[:,::-1].copy();target=target[:,:,::-1].copy();weight=weight[:,:,::-1].copy()
            if rng.random()<.7:
                alpha=rng.uniform(.75,1.23);beta=rng.uniform(-16,16)
                factors=rng.uniform(.92,1.08,(1,1,3))
                image=np.clip(image.astype(float)*alpha*factors+beta,0,255).astype(np.uint8)
            if self.augmentation=='v2':
                if rng.random()<.6:
                    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)
                    blend=rng.uniform(.35,1.)
                    tint=rng.choice([0,1,2])
                    factors=[np.array([1.,1.,1.]),np.array([1.10,1.,.76]),np.array([.78,1.,1.08])][tint]
                    image=np.clip(image*(1-blend)+gray[:,:,None]*factors*blend,0,255).astype(np.uint8)
                if not real and rng.random()<.35:
                    # Known synthetic mounting dividers: distractors, not ink paths.
                    negative=np.zeros((self.size,self.size),np.uint8)
                    for _ in range(int(rng.integers(1,4))):
                        x=int(rng.integers(4,self.size-4));width=int(rng.integers(2,6))
                        cv2.line(image,(x,0),(x,self.size-1),(25,24,19),width)
                        cv2.line(negative,(x,0),(x,self.size-1),255,width+2)
                    if rng.random()<.5:
                        cv2.rectangle(image,(0,0),(self.size-1,self.size-1),(28,25,20),5)
                        cv2.rectangle(negative,(0,0),(self.size-1,self.size-1),255,7)
                    target[:,negative>0]=0;weight[:,negative>0]=1.
            if self.augmentation=='ink':
                # Photometric changes only: all target geometry remains aligned.
                h,w=image.shape[:2]
                paper=rng.uniform(.82,1.,(1,1,3)).astype(np.float32)
                darkness=rng.uniform(.72,1.12)
                image=255-(255-image.astype(np.float32))*darkness
                image*=paper
                if rng.random()<.6:
                    small=rng.normal(0,1,(5,5)).astype(np.float32)
                    shadow=cv2.resize(small,(w,h),interpolation=cv2.INTER_CUBIC)*rng.uniform(2,10)
                    image+=shadow[:,:,None]
                image+=rng.normal(0,rng.uniform(0,2.2),image.shape)
                image=np.clip(image,0,255).astype(np.uint8)
                if rng.random()<.2:image=cv2.GaussianBlur(image,(3,3),rng.uniform(.3,.65))
        return (torch.from_numpy(image.transpose(2,0,1).copy()).float()/255.,
                torch.from_numpy(target),torch.from_numpy(weight),'real_weak' if real else 'synthetic')

    def batch(self,rng,count,real_probability=.4,augment=True):
        rows=[self.sample(rng,real_probability,augment) for _ in range(count)]
        return tuple(torch.stack([row[k] for row in rows]) for k in range(3)),[r[3] for r in rows]
