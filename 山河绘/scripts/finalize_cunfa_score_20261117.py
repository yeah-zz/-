import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,joblib
from gongbi_training.cunfa_score_alignment import predict_concepts,aggregate_regions
ROOT=LAB/'cunfa_score_alignment_20261117';OLD=LAB/'cunfa_concept_supervision_20261116'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 protocol=read(ROOT/'protocol.json');checks={}
 checks['previous_artifacts_unchanged']=all(sha(OLD/n)==h for n,h in protocol['old_artifacts'].items())
 ds=read(ROOT/'reference_manifest.json');rows=ds['records'];z=np.load(ROOT/'features_and_scores.npz');model=joblib.load(ROOT/'score_alignment_models.joblib')
 scores=predict_concepts(model['expanded'],z['features'],True)
 checks['model_roundtrip_all48_scores']=bool(np.allclose(scores,z['oof_expanded_sigmoid'],atol=1e-12))
 units=[r['reference_unit'] for r in rows];names=['hkma_axe','hkma_hemp']
 prediction=model['style_heads']['oof_expanded_sigmoid'].predict(aggregate_regions(scores,units,names))
 results=read(ROOT/'results.json')
 checks['style_roundtrip']=prediction.tolist()==[r['predictions']['oof_expanded_sigmoid'] for r in results['style_probe']]
 checks['browser_all_regions_passed']=read(ROOT/'browser_checks.json')['all48_regions_switch_and_load']
 assert all(checks.values());checks['robot_connected']=False
 (ROOT/'postexperiment_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
 path=LAB/'guohua_hierarchy_20261112/current_milestones.json';state=read(path)
 assert not any(state['capability_state'].values())
 state.update(latest_cunfa_result=str(ROOT/'results.json'),latest_cunfa_report=read(ROOT/'server.json')['url'],
  latest_cunfa_verification=str(ROOT/'verification.json'),latest_cunfa_browser_checks=str(ROOT/'browser_checks.json'),
  latest_cunfa_concept_result=str(ROOT/'results.json'),latest_cunfa_concept_manifest=str(ROOT/'reference_manifest.json'),
  latest_cunfa_concept_report=read(ROOT/'server.json')['url'],latest_cunfa_concept_checks=str(ROOT/'postexperiment_checks.json'),
  latest_cunfa_date='2026-11-17',concept_model_trained=True,new_cunfa_model_trained=True,
  concept_training_scope='Illustration-group OOF base predictions, train-only downstream input scaling and sigmoid fitting; 9 AAM units still1 illustrator; given region windows',
  latest_cunfa_concept_status='Known HKMA2 named examples1to2 after input alignment even without added data; curved group5of7to6of7; added3compact-ink negatives still2 line false positives; new angular false positive; no independent acceptance',
  latest_cunfa_status='Attribute-score alignment and counterexamples show local changes but not independently validated cunfa recognition',
  concept_labels_expert_verified=False,complete_stroke_labels_added=0,fully_resolved_role_labels_added=0,
  latest_cunfa_concept_independent_validation=False,production_replaced=False,robot_connected=False)
 path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
 queue=[]
 for r in results['records']:
  if r['split']!='test' or not r['outputs']['oof_expanded_sigmoid']['errors']:continue
  queue.append(dict(id=r['id'],errors=r['outputs']['oof_expanded_sigmoid']['errors'],scope=r['evaluation_scope'],
   use='diagnosis_only; preserve as check case, seek separate-source references for training',
   automatic_prediction_as_label=False,style_target=None,stroke_order=None))
 (ROOT/'remaining_failure_queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(checks),flush=True);print(state['latest_cunfa_report'],flush=True)
if __name__=='__main__':main()
