# -*- coding: utf-8 -*-
import sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image
from gongbi_training.source_identity import source_identity,check_previously_seen
ROOT=LAB/'clear_rock_sources_20261117'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    m=read(ROOT/'fuchun_manifest.json');refs=read(ROOT/'local_references.json');pre=read(ROOT/'source_preflight.json')
    assert sha(m['video_path'])==m['video_sha256']
    assert pre['previously_seen'] and pre['source_new_to_project'] is False
    ident='1gga38p7dgh3iKX2kYUhFzF7k1ns1c'
    assert source_identity('https://www.ganjingworld.com/video/'+ident)==source_identity('https://www.ganjingworld.com/video/any-title/'+ident+'?tracking=yes')
    assert source_identity('https://youtu.be/EvnuvzSU4Y0')==source_identity('https://www.youtube.com/watch?v=EvnuvzSU4Y0&feature=share')
    assert not check_previously_seen('https://www.ganjingworld.com/video/different',pre['matches'])['previously_seen']
    for r in refs['records']:
        assert sha(r['frame'])==r['frame_sha256']
        original=Image.open(r['frame']).convert('RGB');crop=Image.open(ROOT/f'views/{r["id"]}.png')
        assert original.size==tuple(r['native_size'])
        assert np.array_equal(np.array(original.crop(r['box'])),np.array(crop))
        assert r['style_training_target'] is None and r['complete_curve'] is None
        assert r['stroke_order'] is None and r['pressure_mm'] is None
        assert not r['region_is_ink_mask'] and not r['unmarked_is_background'] and not r['training_admitted']
        assert r['source_seen_before_round'] and not r['new_independent_work']
        assert not (r['object_type']=='tree' and r['role']=='cun')
    assert sum(r['object_checked'] for r in refs['records'])==8
    assert sum(r['object_type']=='unknown' for r in refs['records'])==2
    assert len({r['work_group'] for r in refs['records']})==1
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    baseline={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(baseline.values())
    checks=dict(media_hash=True,native_frame_hashes=True,ten_crops_pixel_exact=True,
        canonical_video_id_duplicates_detected=True,tracking_title_variants_deduplicated=True,
        all_regions_same_work_group=True,eight_object_refs_two_unknown=True,
        no_title_to_local_style_propagation=True,no_region_to_ink_mask_promotion=True,
        no_complete_stroke_or_pressure_targets=True,baseline_hashes=baseline,
        new_independent_works=0,new_complete_artist_strokes=0,new_model_trained=False,
        independent_validation_passed=False,production_replaced=False,robot_connected=False)
    (ROOT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(checks,ensure_ascii=False))
if __name__=='__main__':main()
