import sys,json,hashlib,os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
from gongbi_training.cunfa_branch_isolation import IsolatedRoleHead
from gongbi_training.cunfa_context_roles import ROLE_NAMES
ROOT=LAB/'cunfa_branch_isolation_20261117';BASE=LAB/'cunfa_boundary_relations_20261117'
def read(p):return json.loads(Path(p).read_text('utf8'))
def main():
 torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 p=read(ROOT/'protocol.json');r=read(ROOT/'results.json');z=np.load(BASE/'features.npz');saved=torch.load(ROOT/'isolation_models.pt',map_location='cpu',weights_only=False);dense=np.load(ROOT/'dense_outputs.npz');checks={}
 checks['old_artifacts_unchanged']=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in p['old_artifacts'].items())
 for arm,state in saved['states'].items():
  m=IsolatedRoleHead(arm).cuda().eval();m.load_state_dict(state)
  with torch.inference_mode():o=m(torch.from_numpy(z['features']).cuda(),torch.from_numpy(z['roi']).cuda())
  checks[arm+'_role_roundtrip']=o[0].argmax(1).cpu().tolist()==[ROLE_NAMES.index(v['predictions'][arm]) for v in r['records']]
  checks[arm+'_edge_roundtrip']=np.allclose(o[1].sigmoid().cpu().numpy(),dense[arm+'_edge'],atol=1e-6)
 checks['browser146_passed']=read(ROOT/'browser_checks.json')['all146_regions_switch_and_load'];checks['training_selection_preserved']=r['selected_arm']==read(ROOT/'training_selection.json')['selected_arm'];checks['isolated_roles_match_role_only']=all(r['role_only_predictions_exactly_preserved'].values());assert all(checks.values());checks['robot_connected']=False;(ROOT/'postexperiment_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
 status=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(status);assert not any(s['capability_state'].values());s.update(latest_cunfa_result=str(ROOT/'results.json'),latest_cunfa_report=read(ROOT/'server.json')['url'],latest_cunfa_verification=str(ROOT/'verification.json'),latest_cunfa_browser_checks=str(ROOT/'browser_checks.json'),latest_role_context_result=str(ROOT/'results.json'),latest_role_reference_manifest=str(ROOT/'reference_manifest.json'),latest_isolation_research_model=str(ROOT/'isolation_models.pt'),latest_cunfa_status='Gradient routing3arms; shared baseline exact replay; detached roles exactly match role-only,NiZan3to4/7; sparse recall74.3%,FPR49.0%, prior-selected80.3%/50.0%. Known regressions only; not adopted',latest_cunfa_date='2026-11-17',role_context_independent_validation=False,new_model_trained_this_round=True,complete_stroke_labels_added=0,fully_resolved_role_labels_added=0,rejector_retrained_this_round=False,production_replaced=False,robot_connected=False)
 status.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(checks));print(s['latest_cunfa_report'])
if __name__=='__main__':main()
