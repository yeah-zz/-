import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,joblib
from gongbi_training.cunfa_score_alignment import predict_concepts
from gongbi_training.cunfa_context_roles import ROLE_NAMES
ROOT=LAB/'cunfa_cross_source_20261117';OLD=LAB/'cunfa_score_alignment_20261117'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=read(ROOT/'protocol.json');checks={}
 checks['previous_model_and_reference_hashes_unchanged']=all(sha(OLD/n)==v for n,v in p['old_artifacts'].items())
 data=read(ROOT/'reference_manifest.json');rows=data['records'];z=np.load(ROOT/'features_and_scores.npz');model=joblib.load(ROOT/'cross_source_models.joblib');r=read(ROOT/'results.json')
 checks['all93_attribute_scores_roundtrip']=bool(np.allclose(predict_concepts(model['concepts'],z['local'],True),z['new_source_candidate'],atol=1e-12))
 both=np.concatenate([z['local'],z['context']],1)/np.sqrt(2)
 for name,features in [('local_only',z['local']),('local_plus_context',both)]:
  predictions=model['roles'][name].predict(features)
  checks[name+'_role_scores_roundtrip']=predictions.tolist()==[ROLE_NAMES.index(case['role_predictions'][name]['predicted']) for case in r['records']]
 checks['all93_report_views_passed']=read(ROOT/'browser_checks.json')['all93_regions_switch_and_load']
 checks['new_NT_source_not_used_for_fit']=all(row['split']=='test' for row in rows if row['source_group'].startswith('ntmofa_'))
 assert all(checks.values());checks.update(role_context_not_pixel_truth=True,robot_connected=False)
 (ROOT/'postexperiment_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
 path=LAB/'guohua_hierarchy_20261112/current_milestones.json';state=read(path);assert not any(state['capability_state'].values())
 state.update(latest_cunfa_result=str(ROOT/'results.json'),latest_cunfa_report=read(ROOT/'server.json')['url'],latest_cunfa_verification=str(ROOT/'verification.json'),
  latest_cunfa_browser_checks=str(ROOT/'browser_checks.json'),latest_cunfa_concept_result=str(ROOT/'results.json'),latest_cunfa_concept_manifest=str(ROOT/'reference_manifest.json'),
  latest_cunfa_concept_report=read(ROOT/'server.json')['url'],latest_cunfa_concept_checks=str(ROOT/'postexperiment_checks.json'),
  latest_cunfa_date='2026-11-17',new_cunfa_model_trained=True,new_role_context_model_trained=True,
  concept_training_scope='AAM plus Qingpu source development; NTM3work holdout; partial attributes and given-region local/context role heads, no stroke ownership learning',
  latest_cunfa_status='Cross-source candidate not accepted: line8of10to9of10 but angular4of4to2of4 negatives; roles6of12 local,5of12 context; paper/tone recall0; Mi_dot forced to fupi',
  latest_cunfa_concept_status='New source check exposes role and unknown-method failures, despite2 known named methods corresponding in both old/new models',
  latest_cunfa_concept_independent_validation=False,concept_labels_expert_verified=False,
  role_context_expert_verified=False,complete_stroke_labels_added=0,fully_resolved_role_labels_added=0,
  NTM3work_status='All3 opened for checks; further crops from these works are no longer new unseen works',
  production_replaced=False,robot_connected=False)
 path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(checks),flush=True);print(state['latest_cunfa_report'],flush=True)
if __name__=='__main__':main()
