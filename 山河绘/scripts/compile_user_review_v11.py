# -*- coding: utf-8 -*-
"""Compile only checked user marks. Outside explicit marks stays unknown."""
import sys,json,math,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image,ImageDraw

def rasterize(doc):
    w,h=doc['width'],doc['height'];center=Image.new('L',(w,h));c=ImageDraw.Draw(center)
    # 1px centerline localization tolerance is not the observed brush width.
    for s in doc['strokes']:c.line([tuple(p) for p in s['points']],fill=1,width=3)
    background=Image.new('L',(w,h));uncertain=Image.new('L',(w,h))
    for region in doc['regions']:
        im=background if region['kind']=='background' else uncertain
        ImageDraw.Draw(im).polygon([tuple(p) for p in region['points']],fill=1)
    pos=np.asarray(center,dtype=np.uint8);neg=np.asarray(background,dtype=np.uint8);unknown=np.asarray(uncertain,dtype=np.uint8)
    conflicts=(pos>0)&(neg>0)&(unknown==0)
    if conflicts.any():raise ValueError(f'{doc["sample_id"]}: {conflicts.sum()} positive/background conflict pixels; correct the annotations first')
    known=((pos>0)|(neg>0))&(unknown==0)
    return dict(centerline=pos,explicit_background=neg,uncertain_region=unknown,centerline_confidence=known.astype(np.uint8),positive_role_paths=doc['strokes'])

def compile_labels(root):
    root=Path(root).resolve();sys.path.insert(0,str(root));from review_app import validate_document
    manifest=json.loads((root/'package_manifest.json').read_text(encoding='utf-8'));dest=root/'compiled';dest.mkdir(exist_ok=True);records=[];pending=[]
    for r in manifest['samples']:
        path=root/'labels'/(r['id']+'.json')
        if not path.exists():pending.append(r['id']);continue
        doc=json.loads(path.read_text(encoding='utf-8'));validate_document(doc,r,manifest['package_id'])
        if doc['review_state']!='checked_partial':pending.append(r['id']);continue
        image=root/r['image']
        if hashlib.sha256(image.read_bytes()).hexdigest()!=r['image_sha256']:raise ValueError('image changed '+r['id'])
        data=rasterize(doc);paths=data.pop('positive_role_paths');target=dest/(r['id']+'.npz');np.savez_compressed(target,**data)
        records.append(dict(id=r['id'],image=str(image),labels=str(target),path_annotations=str(path),source_group=r['source_group'],
          split='train_calibration',checked_strokes=len(paths),checked_background_regions=sum(a['kind']=='background' for a in doc['regions']),
          provenance='user_reviewed_partial',unlabeled='unknown',dense_gold=False,independent_test=False))
    status='awaiting_user_labels' if not records else 'reviewed_labels_compiled_not_trained'
    report=dict(status=status,records=records,pending=pending,training_started=False,notes='centerline confidence supervises only explicit center traces and background; nearby untraced ink remains unknown. Not compatible with treating centerlines as full ink masks. No automatic labels or appearance values invented.')
    (dest/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=status,checked_images=len(records),pending=len(pending)),ensure_ascii=False));return report
if __name__=='__main__':compile_labels(Path(__file__).resolve().parent)
