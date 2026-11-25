import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
ROOT=LAB/'cunfa_concept_supervision_20261116'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 old=LAB/'cunfa_input_scale_20261116';frozen=read(old/'results.json');checks={}
 for arm,expected in frozen['head_hashes'].items():
  checks[arm+'_head_unchanged']=sha(old/(arm+'.joblib'))==expected
 checks['legacy_train_manifest_unchanged']=sha(LAB/'cunfa_context_reference_20261111/input_manifest.json')==frozen['protocol']['train_manifest_sha256']
 checks['legacy_13_manifest_unchanged']=sha(LAB/'cunfa_fresh_validation_20261111/test_manifest.json')==frozen['protocol']['test_manifest_sha256']
 checks['encoder_matches_prior_pinned_weights']=sha(LAB/'cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth')==frozen['protocol']['encoder_sha256']
 assert all(checks.values());browser=read(ROOT/'browser_checks.json')
 assert browser['all25_regions_switch_and_load'] and browser['weights_not_served']
 checks.update(browser_passed=True,robot_connected=False)
 (ROOT/'postexperiment_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
 statuspath=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(statuspath)
 assert not any(s['capability_state'].values())
 s.update(latest_cunfa_concept_result=str(ROOT/'results.json'),latest_cunfa_concept_manifest=str(ROOT/'reference_manifest_corrected.json'),
  latest_cunfa_concept_report=read(ROOT/'server.json')['url'],latest_cunfa_concept_checks=str(ROOT/'postexperiment_checks.json'),
  concept_model_trained=True,concept_training_scope='3 partial observable-region morphology heads; illustrative concept-only style bottleneck; provided windows, unknown roles',
  latest_cunfa_concept_status='AAM4 illustration units train, HKMA2 units from1 source photo test; angular5of6, curved-group5of7, line7of7 positives only; direct-style2of2 vs bottleneck1of2; not adopted',
  concept_labels_expert_verified=False,complete_stroke_labels_added=0,fully_resolved_role_labels_added=0,
  latest_cunfa_concept_independent_validation=False,production_replaced=False,robot_connected=False,
  native_xia_assets_reconciled=str(ROOT/'native_asset_reconciliation.json'))
 s['input_scale_previous_result']=s['latest_cunfa_result']
 s['latest_cunfa_result']=str(ROOT/'results.json');s['latest_cunfa_verification']=str(ROOT/'verification.json')
 s['latest_cunfa_status']=s['latest_cunfa_concept_status']
 s['latest_cunfa_browser_checks']=str(ROOT/'browser_checks.json')
 s['new_cunfa_training_scope']=s['concept_training_scope']
 s['latest_cunfa_report']=read(ROOT/'server.json')['url']
 statuspath.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(checks),flush=True);print(s['latest_cunfa_concept_report'],flush=True)

if __name__=='__main__':main()
