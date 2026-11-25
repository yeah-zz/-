# -*- coding: utf-8 -*-
"""Declared synthetic strokes + matched partial references + explicit background.
Unmatched predictions are unknown, not negatives. Family overlap is recorded.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib
import cv2,numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from context_curve_model_20261105 import pack,raster
from module_stroke_core_20261104 import samples,length

ROOT=LAB/'context_curve_selection_20261105'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def source_records():
    records=[]
    for r in read(LAB/'reviewed_rocks_v14_20261024/corrected/manifest.json')['records']:
        doc=read(r['labels']);rgb=cv2.cvtColor(cv2.imread(r['image']),cv2.COLOR_BGR2RGB)
        refs=[np.array(s['points'],float) for s in doc['strokes'] if s['id'].startswith('s_') and len(s['points'])>=2]
        region=np.zeros(rgb.shape[:2],np.uint8)
        for a in doc['regions']:
            if a['kind']=='background':cv2.fillPoly(region,[np.rint(a['points']).astype(np.int32)],1)
        predfile=LAB/'reviewed_rocks_v14_20261024/corrected/replay/evaluation'/r['id']/'trajectory.json'
        stored=np.array(Image.open(predfile.with_name('original.png')).convert('RGB'))
        assert np.array_equal(rgb,stored),'Prediction/reference source mismatch'
        paths=[np.array(p,float) for p in read(predfile)['paths'] if length(np.asarray(p,float))>1e-6]
        records.append(dict(id=r['id'],image=r['image'],labels=r['labels'],rgb=rgb,refs=refs,background=region,
            paths=paths,prediction_file=str(predfile),split='train' if r['id'] in ['R01','R02'] else 'calibration' if r['id']=='R03' else 'test',
            source_family=r['source_group']))
    full=LAB/'user_review_v11_20261024'
    for r in read(full/'package_manifest.json')['samples']:
        if r['id'] not in ['A02','A03','A04','A05']:continue
        label=full/'labels'/(r['id']+'.json');doc=read(label)
        if doc['review_state']!='checked_partial':continue
        image=full/r['image'];assert sha(image)==r['image_sha256'];rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
        refs=[np.array(s['points'],float) for s in doc['strokes'] if s['id'].startswith('s_') and len(s['points'])>=2]
        region=np.zeros(rgb.shape[:2],np.uint8)
        for a in doc['regions']:
            if a['kind']=='background':cv2.fillPoly(region,[np.rint(a['points']).astype(np.int32)],1)
        records.append(dict(id=r['id'],image=str(image),labels=str(label),rgb=rgb,refs=refs,background=region,paths=[],
            prediction_file=None,split='train' if r['id'] in ['A02','A03'] else 'calibration' if r['id']=='A04' else 'test',source_family=r['source_group']))
    return records

def candidate_label(path,refs,background,tolerance=2.):
    q=samples(path,.5);xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,background.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,background.shape[0]-1)
    fraction=float(background[xy[:,1],xy[:,0]].mean())
    near=float((cKDTree(np.concatenate([samples(p,.5) for p in refs])).query(q)[0]<=tolerance).mean()) if refs else 0.
    if fraction>=.95 and near<.05:return 1,dict(explicit_background_fraction=fraction,near_reference_fraction=near)
    if near>=.85 and fraction<.05:return 0,dict(explicit_background_fraction=fraction,near_reference_fraction=near)
    return -1,dict(explicit_background_fraction=fraction,near_reference_fraction=near)

def bg_paths(background,rng,count):
    coords=np.argwhere(background>0);out=[]
    if not len(coords):return out
    for _ in range(5000):
        if len(out)>=count:break
        y,x=coords[rng.integers(len(coords))];a=rng.uniform(-np.pi,np.pi);l=rng.uniform(3,25)
        p=np.array([[x,y],[x+l*np.cos(a),y+l*np.sin(a)]])
        q=np.rint(samples(p,.4)).astype(int)
        if (q.min(0)<0).any() or (q.max(0)>=[background.shape[1],background.shape[0]]).any():continue
        if background[q[:,1],q[:,0]].all():out.append(p)
    return out

def real_examples(record,count,seed):
    rng=np.random.default_rng(seed);rgb=record['rgb'];refs=record['refs'];existing=record['paths'] or refs
    neighbors=raster(rgb.shape[:2],existing);out=[];labels=[];origins=[]
    # A matched real prediction is a partial-reference label, not an inference gold image.
    for index,p in enumerate(record['paths']):
        label,detail=candidate_label(p,refs,record['background'])
        if label<0:continue
        out.append(pack(rgb,p,[q for j,q in enumerate(existing) if j!=index]));labels.append(label);origins.append('actual_prediction_'+str(label))
    for i in range(count):
        p=refs[rng.integers(len(refs))];q=samples(p,.5)
        if len(q)>6 and rng.random()<.75:
            a=int(rng.integers(0,len(q)-3));b=min(len(q),a+int(rng.integers(3,max(4,min(50,len(q)-a)+1))));p=q[a:b]
        if length(p)<.5:continue
        # Reference paths supply trusted positive geometry, not synthetic ink widths.
        out.append(pack(rgb,p,neighbors));labels.append(0);origins.append('explicit_user_curve_or_subcurve')
    for p in bg_paths(record['background'],rng,count):
        out.append(pack(rgb,p,neighbors));labels.append(1);origins.append('explicit_background_generated_probe')
    return out,labels,origins

def stroke(p0,p1,bend=0):
    t=np.linspace(0,1,35);v=np.asarray(p1)-p0;n=np.array([-v[1],v[0]])/max(np.linalg.norm(v),1e-8)
    return p0+t[:,None]*v+np.sin(t*np.pi)[:,None]*bend*n

def synthetic(count,seed,target_center=False):
    rng=np.random.default_rng(seed);out=[];labels=[]
    yy,xx=np.mgrid[:96,:96]
    for i in range(count):
        label=i%3;paper=np.array([rng.uniform(.84,.99),rng.uniform(.82,.98),rng.uniform(.74,.95)])
        rgb=np.ones((96,96,3))*paper
        for _ in range(3):
            cx,cy=rng.uniform(5,91,2);sx,sy=rng.uniform(10,40,2);blob=np.exp(-((xx-cx)/sx)**2-((yy-cy)/sy)**2)
            tint=rng.choice([[-.18,-.05,-.02],[-.05,-.02,-.08],[.01,-.01,-.04]])
            rgb+=blob[:,:,None]*np.array(tint)*rng.uniform(.1,1.)
        rgb+=rng.normal(0,rng.uniform(.001,.012),(96,96,1));rgb=np.clip(rgb*255,0,255).astype(np.uint8)
        paths=[]
        kind=i%4
        if kind==0:
            paths=[np.array([[12,78],[24,25],[39,14],[51,39],[66,27],[83,79]],float)]
            paths += [stroke(np.array([x,40]),np.array([x-8,75]),rng.uniform(-4,5)) for x in [28,43,58,73]]
        elif kind==1:
            paths=[stroke(np.array([12,76]),np.array([76,20]),8)]
            paths += [np.array([[x,75-(x-12)*.8],[x-7,75-(x-12)*.8-10]],float) for x in range(24,80,8)]
        elif kind==2:
            paths=[np.array([[15,40],[48,18],[81,40],[15,40]],float),np.array([[22,42],[22,76],[74,76],[74,42]],float),np.array([[48,42],[48,76]],float)]
        else:
            theta=np.linspace(0,2*np.pi,45);paths=[np.column_stack([48+25*np.cos(theta),49+18*np.sin(theta)])]
        widths=[]
        for p in paths:
            width=int(rng.integers(4,10)) if target_center and label==2 else int(rng.integers(1,10 if target_center else 5));widths.append(width)
            layer=np.zeros((96,96),np.uint8);cv2.polylines(layer,[np.rint(p).astype(np.int32)],False,255,width,cv2.LINE_AA)
            alpha=(layer/255*rng.uniform(.06,.85))[:,:,None]
            rgb=(rgb*(1-alpha)+rng.uniform(10,80)*alpha).astype(np.uint8)
        index=int(rng.integers(len(paths)));target=paths[index].copy()
        if label==0:
            if rng.random()<.4 and len(target)>5:
                a=int(rng.integers(0,len(target)-3));target=target[a:min(len(target),a+int(rng.integers(3,16)))]
            others=[p for j,p in enumerate(paths) if j!=index]
        elif label==1:
            # Candidate follows background color/texture, not a known artist stroke.
            a=rng.uniform(6,90,2);b=rng.uniform(6,90,2);target=stroke(a,b,rng.uniform(-12,12))
            tree=cKDTree(np.concatenate([samples(p,.6) for p in paths]))
            if (tree.query(samples(target,.6))[0]>3).mean()<.90:
                target=np.array([[4,5],[18,5]],float)
            others=paths
        else:
            if target_center:
                tangent=np.gradient(target,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-8)
                normal=np.column_stack([-tangent[:,1],tangent[:,0]])
                offset=max(1.3,widths[index]*rng.uniform(.30,.48))
                target=target+normal*offset
                others=paths if rng.random()<.5 else [p for j,p in enumerate(paths) if j!=index]
            else:target=target+rng.normal(0,.08,target.shape);others=paths
        # Offset copies are still visible ink, not background. Duplicate removal
        # remains a separate source-gap/coverage operation, not a misleading label.
        out.append(pack(rgb,target,others));labels.append(1 if label==1 or target_center and label==2 else 0)
    return out,labels

def prepare():
    ROOT.mkdir(exist_ok=True);records=source_records();manifest=[];sets={}
    for split,number,seed in [('train',3300,110501),('calibration',600,110502),('test',600,110503)]:
        x,y=synthetic(number,seed);origin=['declared_synthetic']*len(y)
        for j,r in enumerate(records):
            if r['split']!=split:continue
            a,b,o=real_examples(r,80 if split=='train' else 36,seed+100+j);x+=a;y+=b;origin+=o
            manifest.append(dict(id=r['id'],split=split,image=r['image'],image_sha256=sha(r['image']),labels=r['labels'],
                label_sha256=sha(r['labels']),source_family=r['source_family'],prediction_file=r['prediction_file'],
                actual_prediction_sha256=sha(r['prediction_file']) if r['prediction_file'] else None,
                examples=len(b),known_keep=b.count(0),known_background=b.count(1)))
        x=np.stack(x);y=np.array(y,np.int64);sets[split]=(x,y)
        np.savez_compressed(ROOT/(split+'_data.npz'),x=x,y=y)
        print('DATA',split,len(y),np.bincount(y,minlength=2).tolist(),flush=True)
    write(ROOT/'protocol.json',dict(records=manifest,classes=['visible_ink_candidate','explicit_background_or_declared_noise'],
        synthetic_counts=dict(train=3300,calibration=600,test=600),synthetic_not_real_paintings=True,
        unmarked_predictions_not_negative=True,partial_actual_prediction_matching_tolerance_px=2.,
        source_families_overlap_across_splits=True,independent_painting_generalization_not_established=True,
        actual_prediction_audit_uses_archived_replay_extractor=True,latest_generated_images_are_unlabeled_development_only=True,
        duplicate_decisions_are_separate_from_neural_visible_ink_classification=True,
        parameter_selection_uses_calibration_only=True,test_not_used_for_checkpoint_selection=True,robot_connected=False))
    return sets

if __name__=='__main__':prepare()
