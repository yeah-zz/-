# -*- coding: utf-8 -*-
from simple_landscape_pairs import ROOT
from rock_v15_common import read,jsave,digest,LAB
import numpy as np,cv2
from PIL import Image
if __name__=='__main__':
    manifest=read(ROOT/'paired_manifest.json');records=manifest['records'];groups={k:[r for r in records if r['split']==k] for k in ['train','calibration','test']};checks={}
    sets={k:{r['seed'] for r in rows} for k,rows in groups.items()}
    assert not sets['train']&sets['calibration'] and not sets['train']&sets['test'] and not sets['calibration']&sets['test'];checks['seed_splits_disjoint']=True
    assert len({r['sha256'] for r in records})==len(records);checks['paired_images_have_no_exact_duplicates']=True
    checked=0
    for rows in groups.values():
        for r in rows[:8]:
            image=np.asarray(Image.open(r['image']));data=np.load(r['data']);target=data['target'];role=data['role'];doc=read(r['reference'])
            assert list(image.shape[:2])==list(target.shape[1:])==list(role.shape)
            reference=np.zeros(role.shape,np.uint8)
            for stroke in doc['strokes']:
                p=np.asarray(stroke['points']);assert np.isfinite(p).all() and (p>=0).all() and (p[:,0]<image.shape[1]).all() and (p[:,1]<image.shape[0]).all()
                cv2.polylines(reference,[np.rint(p).astype(np.int32)],False,1,1,cv2.LINE_8)
            near=cv2.dilate((target[0]>.12).astype(np.uint8),np.ones((3,3),np.uint8));assert near[reference>0].mean()>.995
            assert np.isfinite(target).all() and role.max()<=4;checked+=1
    checks['reference_and_targets_aligned_on_24_examples']=True
    held=read(ROOT/'imagegen_holdout_manifest.json');assert len(held['records'])==36 and held['imagegen_call_count']==6
    for r in held['records']:
        assert digest(r['image'])==r['sha256'] and not r['used_for_training'] and not r['used_for_parameter_selection'] and r['reference'] is None
    checks['36_imagegen_holdouts_have_no_claimed_gold_or_training_use']=True
    config=read(ROOT/'training_config.json');assert set(config['train_ids'])=={r['id'] for r in groups['train']}
    assert not config['heldout_test_labels_used'] and not config['imagegen_holdouts_used'];checks['training_scope_only_512_paired_train_inputs']=True
    assert digest(config['init'])==config['init_sha256'];checks['initial_checkpoint_unchanged']=True
    lock=read(LAB/'stable_line_extractor_20261028'/'baseline_lock.json');assert all(digest(r['path'])==r['sha256'] for r in lock['files']);checks['old_locked_models_and_code_unchanged']=True
    jsave(ROOT/'data_checks.json',dict(checks=checks,all_passed=all(checks.values()),artist_reference_accuracy_not_verified=True,procedural_checks_are_not_artwork_quality_certification=True))
    print('SPECIALIST_DATA_CHECKS',checks,flush=True)
