# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image
ROOT=LAB/'regional_presence_trial_20261117';FOLLOW=LAB/'regional_presence_headonly_20261117';BASE=LAB/'guohua_scene4_mapping_fix_20261116'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    first=read(ROOT/'results.json');second=read(FOLLOW/'results.json');catalog=read(ROOT/'catalog.json')
    for folder,result in [(ROOT,first),(FOLLOW,second)]:
        p=read(folder/'protocol.json')
        assert sha(p['initialization'])==p['initial_sha256']
        assert sha(LAB/'clear_rock_sources_20261117/local_references.json')==p['region_reference_sha256']
        for arm,digest in result['checkpoint_hashes'].items():assert sha(folder/(arm+'.pt'))==digest
        assert all(result['checks'].values())
    assert sha(ROOT/'schedule.json')==sha(FOLLOW/'schedule.json')
    assert read(FOLLOW/'freeze_checks.json')['only_final_classifier_changed']
    # Frozen prediction reproducibility against the original checkpoint's cache.
    for row in first['records']:
        old=np.load(BASE/f'predictions/{row["id"]}_mapping_fixed.npz')['prediction']
        frozen=np.load(ROOT/f'predictions/{row["id"]}_frozen_mapping.npz')['prediction']
        assert np.array_equal(old,frozen),row['id']
        other=np.load(FOLLOW/f'predictions/{row["id"]}_frozen_mapping.npz')['prediction']
        assert np.array_equal(frozen,other)
    assert len(catalog['records'])==43
    for r in catalog['records']:
        assert set(r['predictions'])=={'frozen','control','presence','head_control','head_presence'}
        for p in [r['source']]+list(r['predictions'].values()):
            with Image.open(ROOT/p) as im:im.verify()
    # Recompute the known14 average from confusion, not from hand-picked cases.
    for result in [first,second]:
        for arm in ['frozen_mapping','continuation_control','regional_presence']:
            c=sum(np.array(r['scores'][arm]['confusion']) for r in result['records'] if r['historical_split']=='new_test')
            u=c.sum(0)+c.sum(1)-c.diagonal();mean=np.mean([c[k,k]/u[k] for k in range(4) if u[k]])
            assert abs(mean-result['aggregates']['new_test'][arm]['mean_iou'])<1e-12
    admission=read(ROOT/'admission.json')['records']
    assert sum(r['admitted_for']=='weak_object_presence_only' for r in admission)==8
    assert sum(r['admitted_for'] is None for r in admission)==2
    assert all(not r['dense_pixel_target'] and not r['ink_target'] and r['cunfa_target'] is None for r in admission)
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    unchanged={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(unchanged.values())
    decision=read(ROOT/'decision.json');assert not decision['adopted'] and not decision['independent_validation']
    checks=dict(four_candidates_have_frozen_checkpoints=True,same_inputs_and_steps_within_pairs=True,
        same_schedule_across_update_scope_trials=True,encoder_frozen=True,head_followup_only_final_layer_changes=True,
        original_frozen_outputs_pixel_identical=True,all43_cases_and5arms_rendered=True,
        known14_confusion_and_mean_recomputed=True,eight_positive_regions_two_unknown=True,
        region_is_not_dense_ink_or_cunfa_target=True,source_split_retirement_recorded=True,
        baseline_hashes=unchanged,independent_validation=False,model_adopted=False,
        tree_skill_trained=False,cunfa_trained=False,robot_connected=False)
    (ROOT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(checks,ensure_ascii=False))
if __name__=='__main__':main()
