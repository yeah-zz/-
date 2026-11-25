"""Preserve initial metadata; correct exercise-vs-rock semantics, not targets."""
import sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.cunfa_concept_supervision import concept_targets,validate_concept_split
ROOT=LAB/'cunfa_concept_supervision_20261116'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 src=ROOT/'reference_manifest.json';old=json.loads(src.read_text('utf8'));fixed=copy.deepcopy(old)
 changes=[]
 for r in fixed['records']:
  if r['id'] in {'hkma_axe_0','hkma_axe_1','hkma_axe_3'}:
   r['object_type']='teaching_mark';r['reference_kind']='named_brush_mark_exercise'
   changes.append(dict(id=r['id'],old_object='rock',corrected_object='teaching_mark',
    reason='Isolated named brush-mark practice on paper, not an actual rock instance'))
  else:r['reference_kind']='rock_surface_region'
  assert concept_targets(r).tolist()==r['targets'],r['id']
 validate_concept_split(fixed['records']);fixed['metadata_correction_applied_after_first_fit']=True
 fixed['original_manifest_sha256']=sha(src);fixed['morphology_targets_and_windows_unchanged']=True
 (ROOT/'reference_manifest_corrected.json').write_text(json.dumps(fixed,ensure_ascii=False,indent=2),encoding='utf8')
 audit=dict(changes=changes,source_manifest_sha256=sha(src),
  corrected_manifest_sha256=sha(ROOT/'reference_manifest_corrected.json'),
  original_module_snapshot=str(ROOT/'frozen_concept_contract_before_metadata_fix.py'),
  original_module_sha256=sha(ROOT/'frozen_concept_contract_before_metadata_fix.py'),
  corrected_module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/cunfa_concept_supervision.py'),
  object_only_metadata_correction=True,all25_concept_target_arrays_identical=True,
  crop_pixels_windows_splits_and_predictions_unchanged=True,no_refit_or_test_selection=True,
  teaching_marks_not_accepted_as_rock_instances_or_draw_actions=True,robot_connected=False)
 protocol=json.loads((ROOT/'protocol.json').read_text('utf8'))
 assert audit['original_module_sha256']==protocol['module_sha256']
 (ROOT/'reference_metadata_correction.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
