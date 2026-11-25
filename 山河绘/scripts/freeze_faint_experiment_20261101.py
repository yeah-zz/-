from faint_candidate_fusion_20261101 import *
from pathlib import Path
from run_stability_brush_20261101 import verify_lock
import shutil
def main():
    verify_lock()
    for r in read(PRIOR/'r3_lock.json')['files']:assert digest(r['path'])==r['sha256']
    summary=read(ROOT/'local_repair_summary.json')['records'];cal=[r for r in summary if r['split']=='calibration'];known=[r for r in cal if r['repaired']['partial_reference_coverage_3px'] is not None];before=float(np.mean([r['previous']['partial_reference_coverage_3px'] for r in known]));after=float(np.mean([r['repaired']['partial_reference_coverage_3px'] for r in known]));neg=sum(r['repaired']['line_pixels_in_checked_negative_boxes']-r['previous']['line_pixels_in_checked_negative_boxes'] for r in cal);edge=sum(r['repaired']['color_boundary_ref_hits_3px']-r['previous']['color_boundary_ref_hits_3px'] for r in cal);protected=all(r['protected_exact'] for r in summary);gate=dict(calibration_partial_coverage_before=before,calibration_partial_coverage_after=after,partial_references_are_photometric_and_not_expert_accuracy=True,new_pixels_in_calibration_checked_negative_boxes=neg,increase_in_color_boundary_hits=edge,protected_exact_all_24=protected,passed=bool(after-before>=.03 and neg<=3 and edge<=0 and protected),not_a_whole_image_accuracy_gate=True)
    jsave(ROOT/'local_gate.json',gate);print('FAINT_LOCAL_GATE',gate,flush=True)
    if not gate['passed']:return
    scripts=Path(__file__).parent;names=['faint_joint_model_20261101.py','faint_context_routing_20261101.py','evaluate_faint_joint_20261101.py','faint_candidate_fusion_20261101.py','extract_faint_joint_20261101.py'];files=[scripts/n for n in names]+[ROOT/p for p in ['fit_r2/model.pt','context_calibration.json','audited_reference_manifest.json','paired_manifest_r2.json','fit_r2/training_config.json']];lock=dict(files=[dict(path=str(p),sha256=digest(p)) for p in files],parent_r3_lock_sha256=digest(PRIOR/'r3_lock.json'),accepted_baseline_lock_sha256=digest(BASE/'baseline_lock.json'),fusion_config=CFG_FUSION,fresh_generation_after_freeze=True,fresh_images_used_for_training=False,fresh_image_tuning_allowed=False,fresh_sample_count=3,all_fresh_outputs_retained=True,source_ground_truth_for_fresh_images=False,robot_connected=False)
    file=ROOT/'frozen_experiment.json'
    if file.exists():assert read(file)==lock
    else:jsave(file,lock)
    folder=ROOT/'frozen_source';folder.mkdir(exist_ok=True)
    for p in files:
        destination=folder/(p.parent.name+'_'+p.name if p.name in ['model.pt','training_config.json'] else p.name)
        if destination.exists():assert digest(destination)==digest(p)
        else:shutil.copy2(p,destination)
if __name__=='__main__':main()
