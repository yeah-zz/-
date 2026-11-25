# -*- coding: utf-8 -*-
"""Freeze named teaching pairs and audit scope before G1/G2 model comparison."""
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gongbi_training.evidence_packet import audit_lineage, supervision_channels
from gongbi_training.runtime import LAB, configure


OUT = LAB/'recognition_main_tracks_20261122'
HKMA = 'https://hk.art.museum/Archive/resources/S_Wucius_WONG_Suggested_Class_Activities_CHIN.pdf'
FROZEN = LAB/'cunfa_fresh_validation_20261111'
SCENE = LAB/'guohua_scene4_mapping_fix_20261116'
PROSPECTIVE = LAB/'object_prospective_20261118'
PROTECTED = LAB/'guohua_faint_sources_20261116/selection_before_download.json'
# Same two printed examples, each with an unaltered photo and a tighter rock view.
BOXES = [
    ('axe','斧劈皴',[185,1084,399,1243],[281,1178,347,1239]),
    ('hemp','披麻皴',[410,1084,622,1243],[455,1128,565,1209]),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def teaching_pair():
    page = OUT/'hkma_page-4.png'
    doc = OUT/'hkma_teaching_reference.pdf'
    image = Image.open(page).convert('RGB')
    assert image.size==(1275,1650),image.size
    records = []
    for key,label,full,tight in BOXES:
        for view,box in [('example',full),('rock_crop',tight)]:
            image.crop(box).save(OUT/f'{key}_{view}.png')
        example = OUT/f'{key}_example.png'
        rock = OUT/f'{key}_rock_crop.png'
        row = dict(id='hkma_'+key,scope='named_style_example',label=label,source_id=HKMA,
                   source_group='HKMA_Wucius_class_activity_pdf',work_group='HKMA_Wucius_class_activity_pair',
                   illustrator=None,source_named=True,page=4,box_150dpi=full,rock_crop_box_150dpi=tight,
                   image=str(example),image_sha256=sha(example),rock_crop=str(rock),rock_crop_sha256=sha(rock),
                   source_pdf_sha256=sha(doc),reviewer='assistant_visual_review',seen_before_round=False,
                   usage='new_source_diagnostic',expert_review=False,
                   evidence='Publisher caption identifies the example. Illustrator and individual stroke ownership are unverified.')
        row['allowed_supervision'] = supervision_channels(row)
        records.append(row)
    audit_lineage(records)
    return records


def training_audit():
    old_style = read(LAB/'cunfa_context_reference_20261111/input_manifest.json')['rows']
    test = read(FROZEN/'test_manifest.json')
    previous = [row for row in test['records'] if row['admission']=='source_label_test']
    rows = read(SCENE/'dataset_manifest.json')['records']
    frozen = read(PROSPECTIVE/'dataset_manifest.json')['records']
    reference = read(PROSPECTIVE/'local_references.json')
    frozen_by_id = {row['id']:row for row in frozen}
    evidence = []
    for row in reference['records']:
        source = frozen_by_id[row['image_id']]
        item = dict(id=row['id'],scope='object_region',source_id=source['image_url'],
                    image_id=row['image_id'],
                    work_group=source['work_group'],source_group='CLP_dataset',seen_before_round=True,
                    reviewer=row['reviewer'],evidence=row['evidence'],usage='known_regression',
                    object_checked=row['object_type'] in ('rock','tree'),object_type=row['object_type'],
                    image=source['image'],image_sha256=sha(source['image']),box=row['box'],train=False)
        item['allowed_supervision'] = supervision_channels(item)
        evidence.append(item)
    audit_lineage(evidence)
    by_scope = {}
    for r in rows:
        by_scope.setdefault(r['split'],[]).append(r)
    held_groups = sorted({re.sub(r'_\d+$','',Path(p).stem) for p in read(PROTECTED)['reserved_paths']})
    protected = dict(path_metadata=len(read(PROTECTED)['reserved_paths']),groups=held_groups,
                     pixels_accessed_this_round=False)
    old_train = by_scope.get('train',[])
    opened = [r for k,v in by_scope.items() if k!='train' for r in v]
    overlap = {field:sorted({r[field] for r in old_train}&{r[field] for r in opened})
               for field in ('work_group','artist_group','decoded_pixel_sha256')}
    # Read the frozen predictions from the same original-coordinate reference
    # regions. Count the two confusion directions instead of averaging them.
    object_errors = []
    for mode in ('global256','detail512','fixed_mean','learned_gate'):
        counts = {'tree':0,'rock':0,'tree_to_rock':0,'rock_to_tree':0}
        for row in evidence:
            target = row['object_type']
            if target not in ('rock','tree'):
                continue
            with np.load(PROSPECTIVE/f"predictions/{row['image_id']}_{mode}.npz") as saved:
                x0,y0,x1,y1 = row['box']
                patch = saved['prediction'][y0:y1,x0:x1]
            assert patch.shape==(y1-y0,x1-x0)
            opposite = 'rock' if target=='tree' else 'tree'
            code = 1 if opposite=='rock' else 2
            counts[target] += patch.size
            counts[target+'_to_'+opposite] += int((patch==code).sum())
        object_errors.append(dict(mode=mode,**counts))
    return dict(
        g1=dict(previous_train_views=len(old_style),previous_train_work_groups=len({r['work_group'] for r in old_style}),
                previous_train_artist_groups=len({r['artist_group'] for r in old_style}),
                previous_train_classes=dict(Counter(r['label'] for r in old_style)),
                previous_exposed_checks=len(previous),previous_expert_local_test=int(test['expert_local_test_samples'])),
        g2=dict(splits={k:dict(images=len(v),work_groups=len({r['work_group'] for r in v}))
                        for k,v in by_scope.items()}, historical_overlap=overlap,
                opened_regression_images=len(frozen),opened_regression_work_groups=len({r['work_group'] for r in frozen}),
                known_object_regions=dict(Counter(r['object_type'] for r in evidence)),
                pairs=reference['pairs'],metadata_only_protected=protected,object_evidence=evidence,
                known_region_errors=object_errors))


def frozen_comparison(records):
    configure()
    import joblib
    import torch
    from torchvision.models import mobilenet_v3_small
    from train_cunfa_gabor_20261110 import CLASSES, phash

    before = read(FROZEN/'freeze_manifest.json')
    for entry in before['files']:
        assert sha(entry['path'])==entry['sha256']
    old = read(LAB/'cunfa_context_reference_20261111/input_manifest.json')['rows']
    hashes = [phash(Image.open(r['image'])) for r in old]
    spec = importlib.util.spec_from_file_location('cunfa_locked_preprocessing',FROZEN/'frozen/encoder_and_preprocessing.py')
    preprocess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess)
    head = joblib.load(FROZEN/'frozen/local_only_head.joblib')
    assert set(head['trained_ids'])==set(before['train_ids'])
    weight = next(r['path'] for r in before['files'] if r['path'].endswith('.pth'))
    torch.set_num_threads(4)
    net = mobilenet_v3_small(weights=None)
    net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
    net = net.features.eval()
    results = []
    for row in records:
        for view in ('image','rock_crop'):
            path = row[view]
            assert sha(path)==row[view+'_sha256']
            im = Image.open(path).convert('RGB')
            h = phash(im)
            nearby = min(int(np.count_nonzero(h!=prior)) for prior in hashes)
            assert nearby > 4,'Previously used near-duplicate candidate requires review'
            with torch.inference_mode():
                z = net(torch.from_numpy(preprocess.preprocess(im)[None])).mean((2,3))
                features = torch.nn.functional.normalize(z,dim=1).numpy()
            prediction = CLASSES[int(head['model'].predict(features)[0])]
            results.append(dict(id=row['id'],view=view,label=row['label'],predicted=prediction,
                                closest_old_crop_phash_distance=nearby))
    return results


def board(records,predictions):
    sheet = Image.new('RGB',(850,525),'#f5f2e8')
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    d.text((12,10),'同一教材的具名对照：示例照片与山石裁片',font=font,fill='#243a31')
    for j,row in enumerate(records):
        for i,view in enumerate(('image','rock_crop')):
            x,y = 12+i*421,46+j*216
            im = Image.open(row[view]).convert('RGB')
            im.thumbnail((390,175),Image.Resampling.LANCZOS)
            sheet.paste(im,(x,y+30))
            p=next(r['predicted'] for r in predictions if (r['id'],r['view'])==(row['id'],view))
            title='教材照片' if view=='image' else '仅山石裁片'
            d.text((x,y),f"{row['label']} · {title} · 预测 {p}",font=font,fill='#243a31')
    d.text((12,494),'两类来自同一份教材。相同示例的两个裁剪只计为一份图例。',font=font,fill='#243a31')
    sheet.save(OUT/'teaching_pair_comparison.png')


def main():
    pair=teaching_pair()
    audit=training_audit()
    protocol=dict(new_pair=pair, audit_source=str(PROTECTED), freeze_manifest_sha256=sha(FROZEN/'freeze_manifest.json'),
                  inference_only=True,planned_views=['image','rock_crop'],recognized_classes=['披麻皴','斧劈皴'],
                  scope='Two named teaching examples from a single publication; cropped variants are not independent labels.')
    lock=OUT/'protocol.json'
    if lock.exists():
        assert read(lock)==protocol
    else:
        save('protocol.json',protocol)
    predictions=frozen_comparison(pair)
    board(pair,predictions)
    save('recognition_data_register.json',audit)
    baselines=read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    checks={k:sha(baselines[k])==baselines[k+'_sha256'] for k in ('approved500','working489')}
    assert all(checks.values())
    result=dict(predictions=predictions, protocol_sha256=sha(lock),baseline_hashes=checks,
                new_weight_training=False,adopted=False,independent_acceptance=False,robot_connected=False)
    save('results.json',result)
    print(json.dumps(dict(g1=audit['g1'],g2={k:v for k,v in audit['g2'].items() if k!='object_evidence'},
                          predictions=predictions),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
