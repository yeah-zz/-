import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,joblib
from gongbi_training.cunfa_relational_roles import predict_role_probe
from gongbi_training.cunfa_context_roles import ROLE_NAMES
ROOT=LAB/'cunfa_relational_roles_20261117';OLD=LAB/'cunfa_photometry_20261117'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=read(ROOT/'protocol.json');r=read(ROOT/'results.json');z=np.load(ROOT/'features.npz');m=joblib.load(ROOT/'relational_models.joblib');checks={}
 checks['previous_models_features_unchanged']=all(sha(OLD/n)==h for n,h in p['old_artifacts'].items())
 for a,model in m['models'].items():
  x=z['near'] if a=='expanded_flat_near' else z['spatial'];pred=predict_role_probe(model,x,z['appearance'])
  checks[a+'_roundtrip']=pred.tolist()==[ROLE_NAMES.index(c['predictions'][a]) for c in r['records']]
 checks['all112_views_browser_passed']=read(ROOT/'browser_checks.json')['all112_regions_switch_and_load']
 checks['selection_matches_record']=r['selected_arm']==read(ROOT/'training_selection.json')['selected_arm']
 assert all(checks.values());checks['robot_connected']=False
 (ROOT/'postexperiment_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
 path=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(path);assert not any(s['capability_state'].values())
 s.update(latest_cunfa_result=str(ROOT/'results.json'),latest_cunfa_report=read(ROOT/'server.json')['url'],latest_cunfa_verification=str(ROOT/'verification.json'),
  latest_cunfa_browser_checks=str(ROOT/'browser_checks.json'),latest_role_context_result=str(ROOT/'results.json'),
  latest_relational_research_model=str(ROOT/'relational_models.joblib'),latest_role_reference_manifest=str(ROOT/'reference_manifest.json'),
  latest_cunfa_status='Role-reference expansion23to38, spatial/hierarchical probes; known12 selected6 vs previous7, structure0to1of2 but other roles regress; not adopted',
  latest_cunfa_date='2026-11-17',role_context_independent_validation=False,
  new_model_trained_this_round=True,complete_stroke_labels_added=0,fully_resolved_role_labels_added=0,
  rejector_retrained_this_round=False,production_replaced=False,robot_connected=False)
 path.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(checks),flush=True);print(s['latest_cunfa_report'],flush=True)
if __name__=='__main__':main()
