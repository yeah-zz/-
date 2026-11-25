import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,joblib
from gongbi_training.cunfa_context_roles import ROLE_NAMES
ROOT=LAB/'cunfa_spatial_pairs_20261117'
def read(p):return json.loads(Path(p).read_text('utf8'))
def main():
 p=read(ROOT/'protocol.json');r=read(ROOT/'results.json');z=np.load(ROOT/'features.npz');models=joblib.load(ROOT/'spatial_models.joblib');checks={}
 checks['old_artifacts_unchanged']=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in p['old_artifacts'].items())
 for arm,model in models['models'].items():checks[arm+'_roundtrip']=model.predict(z[arm]).tolist()==[ROLE_NAMES.index(v['predictions'][arm]) for v in r['records']]
 checks['all137_views_browser_passed']=read(ROOT/'browser_checks.json')['all137_regions_switch_and_load'];checks['training_selection_preserved']=r['selected_arm']==read(ROOT/'training_selection.json')['selected_arm']
 assert all(checks.values());checks['robot_connected']=False;(ROOT/'postexperiment_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
 status=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(status);assert not any(s['capability_state'].values())
 s.update(latest_cunfa_result=str(ROOT/'results.json'),latest_cunfa_report=read(ROOT/'server.json')['url'],latest_cunfa_verification=str(ROOT/'verification.json'),latest_cunfa_browser_checks=str(ROOT/'browser_checks.json'),latest_role_context_result=str(ROOT/'results.json'),latest_role_reference_manifest=str(ROOT/'reference_manifest.json'),latest_spatial_research_model=str(ROOT/'spatial_models.joblib'),latest_cunfa_status='Added11 development role references;49train. Selectedglobal known7/12, spatial8/12; all models fresh4/6, fresh outline0/2. No adoption',latest_cunfa_date='2026-11-17',role_context_independent_validation=False,new_model_trained_this_round=True,complete_stroke_labels_added=0,fully_resolved_role_labels_added=0,rejector_retrained_this_round=False,production_replaced=False,robot_connected=False)
 status.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(checks));print(s['latest_cunfa_report'])
if __name__=='__main__':main()
