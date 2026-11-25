# -*- coding: utf-8 -*-
import sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image
from gongbi_training.evidence_packet import validate_evidence,supervision_channels,audit_lineage

ROOT=LAB/'whole_rock_evidence_20261117'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    packet=read(ROOT/'packet.json');rs=packet['records'];ss={s['id']:s for s in packet['sources']};checks={}
    assert len({r['id'] for r in rs})==len(rs)
    for s in ss.values():assert sha(s['path'])==s['sha256']
    for r in rs:
        assert r['source_id'] in ss
        assert (r['work_group'],r['source_group'])==(ss[r['source_id']]['work_group'],ss[r['source_id']]['source_group'])
        validate_evidence(r)
        assert packet['supervision_channels'][r['id']]==supervision_channels(r)
    checks['source_hashes_and_task_channels']=True
    legacy=read(LAB/'rock_skill_offline_20261110/visible_reference.json')
    guides=[r for r in rs if r['scope']=='visible_curve_candidate'];lookup={r['id']:r for r in legacy['records']}
    for r in guides:
        assert r['curve']==lookup[r['id']]['curve']
        assert not any(supervision_channels(r).values())
    checks['old_27_guides_not_promoted_or_redrawn']=True
    raw=np.array(Image.open(legacy['source']).convert('RGB').crop(legacy['box']))
    assert np.array_equal(raw,np.array(Image.open(ROOT/'views/rock_native.png')))
    corridors=np.load(ROOT/'data/candidate_corridors.npz')
    assert (corridors['instance_targets']==255).all()
    fragment=np.array(Image.open(ROOT/'views/source_corridor_fragments.png'))
    known=corridors['corridors'].any(axis=0)
    assert np.array_equal(raw[known],fragment[known])
    assert (fragment[~known]==255).all()
    checks['native_coordinates_and_source_fragments_exact']=True
    for r in rs:
        if r['scope']=='temporal_role_core':
            before=np.load(r['source_npz']);after=np.load(r['packet_npz'])
            assert set(before.files)==set(after.files)
            for k in before.files:assert np.array_equal(before[k],after[k])
            assert set(np.unique(after['targets'])).issubset({0,1,2,255})
    checks['temporal_unknown_pixels_preserved']=True
    h=np.load(ROOT/'data/H14_object_region.npz')
    assert (h['rock_role_targets']==255).all()
    assert set(np.unique(h['object_targets']))=={1,255}
    checks['tree_region_has_no_cun_or_paper_role_targets']=True
    rejected=[]
    def expect(label,rec):
        try:validate_evidence(rec)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label+' was accepted')
    tree=next(r for r in rs if r['id']=='H14_foliage')
    expect('tree_as_cun',dict(tree,role='cun'))
    expect('object_polygon_as_ink_mask',dict(tree,region_is_ink_mask=True))
    expect('known_as_independent',dict(tree,independent_test=True))
    expect('prediction_as_label',dict(tree,prediction_as_label=True))
    expect('image_width_as_pressure',dict(tree,pressure_mm=.6))
    expect('static_guide_as_artist_stroke',dict(guides[0],instance_supervision=True))
    checks['six_invalid_supervision_promotions_rejected']=rejected
    leaking=copy.deepcopy(tree);leaking.update(id='leak',seen_before_round=False,independent_test=True,usage='independent_test')
    try:audit_lineage([tree,leaking])
    except ValueError:checks['work_source_leakage_rejected']=True
    else:raise AssertionError('lineage leakage accepted')
    replay=read(ROOT/'reference_replay.json')['steps']
    assert [r['object_id'] for r in replay]==list('AAABBBCCC')
    assert [r['role'] for r in replay]==['contour','structure','cun']*3
    assert sorted(i for s in replay for i in s['ids'])==sorted(r['id'] for r in guides)
    checks['reference_replay_roles_and_no_repeated_ids']=True
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    checks['approved500_hash_unchanged']=sha(lock['approved500'])==lock['approved500_sha256']
    checks['working489_hash_unchanged']=sha(lock['working489'])==lock['working489_sha256']
    assert checks['approved500_hash_unchanged'] and checks['working489_hash_unchanged']
    result=dict(engineering_checks=checks,engineering_passed=True,independent_validation_passed=False,
                new_recognition_capability=False,new_model_trained=False,robot_connected=False)
    (ROOT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
