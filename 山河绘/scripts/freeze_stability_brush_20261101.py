# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from rock_v15_common import LAB,read,jsave,digest
from pathlib import Path
import shutil,json
ROOT=LAB/'stability_brush_preview_20261101'
ACCEPTED=LAB/'mountain_complete_lines_20261101'

def main():
    root=Path(__file__).resolve().parents[1];scripts=root/'scripts';names=['extract_complete_lines_20261101.py','mountain_complete_lines_20261101.py','mountain_complete_evidence_20261101.py','train_complete_candidates_20261101.py','extract_mainline_recovery_20261101.py','mountain_mainline_recovery_20261101.py','mountain_recovery_catalog_20261101.py','train_mountain_candidate_recovery_20261101.py','extract_mountain_structure_v4.py','mountain_structure_v4_20261101.py','v4_local_repair_candidates.py','split_detail_v4_frozen.py','source_ink_baseline_20261028.py','rock_v15_common.py'];files=[scripts/n for n in names]+[root/'gongbi_training/runtime.py',ACCEPTED/'fit_r2/model.pt',ACCEPTED/'bucket_calibration.json',LAB/'mountain_mainline_recovery_20261101/fit/model.pt']
    lock=dict(version='user_accepted_complete_r2_bucket',files=[dict(path=str(p),sha256=digest(p)) for p in files],accepted_report=str(ACCEPTED/'index.html'),user_approval='User said this version looks good; image-level approval, not expert approval of every stroke',frozen_before_new_inputs=True,robot_connected=False)
    path=ROOT/'baseline_lock.json'
    if path.exists():assert read(path)==lock,'Accepted baseline changed'
    else:jsave(path,lock)
    snapshot=ROOT/'baseline_snapshot';snapshot.mkdir(parents=True,exist_ok=True)
    for p in files:
        dest=snapshot/('code/'+p.name if p.suffix=='.py' else 'complete_model.pt' if p==ACCEPTED/'fit_r2/model.pt' else 'previous_model.pt' if p.suffix=='.pt' else p.name);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert digest(dest)==digest(p)
        else:shutil.copy2(p,dest)
    protocol=dict(samples=20,independent_generation_calls=20,same_style=True,groups=dict(simple=8,medium=8,stress=4),source_resolution='Original generated PNG retained unchanged',primary_input='Whole image resized to long edge 496px, matching approved development family; no crops',native_resolution_substudy=['T01','T06','T17','T19'],training_on_test_images=False,tuning_on_test_images=False,selective_regeneration=False,failed_results_retained=True,reference_truth='No pixel/stroke ground truth for new generated images; assistant visual review only',review_rubric=['main contours','mountain interior clutter','pine detail','blank-paper false lines'],curves=dict(max_hausdorff_error_px=.6,endpoints_fixed=True,unknown_junctions_not_joined=True),brush_parameters=dict(width='local image FWHM, pixels',ink='apparent local optical density; not pigment concentration',physical_z=None,velocity=None,water_content=None,ready_for_robot=False),baseline_lock_sha256=digest(path))
    if (ROOT/'evaluation_protocol.json').exists():assert read(ROOT/'evaluation_protocol.json')==protocol
    else:jsave(ROOT/'evaluation_protocol.json',protocol)
    print('STABILITY_BASELINE_FROZEN',len(files),str(ROOT),flush=True)
if __name__=='__main__':main()
