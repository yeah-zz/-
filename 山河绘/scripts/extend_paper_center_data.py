# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from paper_center_refinement_data import ROOT,PARENT,LOCAL,build_actual
from joint_ink_structure_data import write_record
from rock_v15_common import read,jsave,digest
import numpy as np,cv2
from PIL import Image

def extend():
    manifest=read(ROOT/'manifest.json')
    if manifest.get('paper_stress_added'):raise FileExistsError('Preserve completed paper stress data')
    rows=manifest['records']
    # Tighten sky margins after viewing the supervision atlas, before training.
    for i,r in enumerate(rows):
        ident=r['id']
        if r['kind']!='actual_refined' or not ident.startswith(('G04','G05')):continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));h,w=rgb.shape[:2];height=40 if ident.startswith('G04') else 25;boxes=[[8,0,w-8,height],[8,h-8,w-8,h]]
        if ident.startswith('G05'):boxes.append([350,0,w-8,80])
        rows[i]=build_actual(ident,rgb,r['split'],paper_boxes=boxes)
    for split,count,start,group in [('train',64,820000,'G04'),('calibration',16,830000,'G02'),('regression',32,840000,'G05')]:
        sources=[r for r in rows if r['kind']=='actual_refined' and r['id'].startswith(group)]
        for i in range(count):
            rng=np.random.default_rng(start+i);src=sources[i%len(sources)];doc=read(src['reference']);boxes=doc['paper_boxes'];box=boxes[int(rng.integers(len(boxes)))];rgb=np.asarray(Image.open(src['image']).convert('RGB'));x0,y0,x1,y1=box;patch=cv2.resize(rgb[y0:y1,x0:x1],(256,256),interpolation=cv2.INTER_CUBIC).astype(np.float32)
            yy,xx=np.mgrid[:256,:256];gradient=(xx/256*rng.uniform(-2,2)+yy/256*rng.uniform(-2,2));grain=rng.normal(0,rng.uniform(.2,.7),(256,256));patch+=gradient[:,:,None]+grain[:,:,None];patch=np.clip(patch*rng.uniform(.97,1.02),0,255).astype(np.uint8)
            t=np.zeros((6,256,256),np.float32);m=np.zeros((4,256,256),np.float32);m[:2]=1;role=np.full((256,256),255,np.uint8);rel=rm=np.zeros((8,256,256),np.float32)
            meta=dict(split=split,kind='paper_stress',source_group=group,source_id=src['id'],source_sha256=src['sha256'],source_box=box,seed=start+i,reference_kind='known reviewed paper transformed without adding ink; not new artwork',strokes=[])
            r=write_record(ROOT/'paper_stress'/split,f'paper_{split}_{i:03}',patch,t,m,role,rel,rm,meta);z=np.load(r['data']);data={k:z[k] for k in z.files};data['paper_mask']=np.ones((256,256),np.uint8);np.savez_compressed(r['data'],**data);r['data_sha256']=digest(r['data']);rows.append(r)
    manifest.update(records=rows,paper_stress_added=True,sky_margin_adjusted_before_fit=True,source_sha256=digest(__import__('paper_center_refinement_data').__file__))
    jsave(ROOT/'manifest.json',manifest);print('REFINEMENT_EXTENDED',len(rows),flush=True)
if __name__=='__main__':extend()
