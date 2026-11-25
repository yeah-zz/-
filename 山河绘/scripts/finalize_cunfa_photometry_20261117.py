import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,joblib
from gongbi_training.cunfa_context_roles import ROLE_NAMES
ROOT=LAB/'cunfa_photometry_20261117';OLD=LAB/'cunfa_cross_source_20261117'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=read(ROOT/'protocol.json');r=read(ROOT/'results.json');model=joblib.load(ROOT/'photometry_models.joblib');z=np.load(ROOT/'features.npz');checks={}
 checks['previous_model_reference_features_unchanged']=all(sha(OLD/n)==v for n,v in p['old_artifacts'].items())
 for name,m in model['role_models'].items():checks[name+'_roundtrip']=m.predict(z[name]).tolist()==[ROLE_NAMES.index(c['predictions'][name]) for c in r['records']]
 scope=joblib.load(ROOT/'rejection_scope_model.joblib');rr=read(ROOT/'rejection_scope_results.json')
 checks['scoped_rejection_roundtrip']=np.allclose(scope['model'].predict_proba(z['rejection'])[:,1],[u['known_score'] for u in rr['records']])
 checks['ambiguous_rejection_units_not_trained']=not(set(read(ROOT/'rejection_scope_protocol.json')['ambiguous_units'])&set(scope['trained_ids']))
 checks['all93_browser_views_passed']=read(ROOT/'browser_checks.json')['all93_regions_switch_and_load']
 assert all(checks.values());checks['robot_connected']=False
 # The first strict-name rejector remains archived, never the active probe.
 joblib.dump(dict(role_models=model['role_models'],selected_arm=model['selected_arm'],
  rejector=scope['model'],rejection_scope=scope['protocol'],unreviewed_first_rejection_branch_superseded=True,
  protocol=p,production_use=False),ROOT/'photometry_scoped_models.joblib')
 (ROOT/'postexperiment_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
 path=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(path);assert not any(s['capability_state'].values())
 s.update(latest_cunfa_result=str(ROOT/'results.json'),latest_cunfa_report=read(ROOT/'server.json')['url'],latest_cunfa_verification=str(ROOT/'verification.json'),
  latest_cunfa_browser_checks=str(ROOT/'browser_checks.json'),latest_role_context_result=str(ROOT/'results.json'),
  latest_rejection_probe=str(ROOT/'rejection_scope_results.json'),latest_rejection_probe_model=str(ROOT/'rejection_scope_model.joblib'),
  latest_photometric_research_model=str(ROOT/'photometry_scoped_models.joblib'),
  latest_cunfa_status='Known-case photometry regression: selected linear+contrast5to7of12, blank0to2of3, outline1to0of2, tone0of2; scoped rejection known3of4 accepted, MiDot not rejected; not adopted',
  latest_cunfa_date='2026-11-17',role_context_independent_validation=False,rejection_independent_validation=False,
  photometry_source_masks_used=False,unknown_rejection_trained_on_limited_named_units=True,general_unknown_recognition=False,
  new_model_trained_this_round=True,production_replaced=False,robot_connected=False)
 path.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(checks),flush=True);print(s['latest_cunfa_report'],flush=True)
if __name__=='__main__':main()
