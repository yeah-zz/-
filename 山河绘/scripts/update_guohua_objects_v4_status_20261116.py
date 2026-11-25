import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
r=LAB/'guohua_objects_v4_sparse_20261116';p=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=json.loads(p.read_text('utf8'));assert not any(s['capability_state'].values())
s.update(latest_object_result=str(r/'results.json'),latest_object_report=json.loads((r/'server.json').read_text('utf8'))['url'],latest_object_verification=str(r/'verification.json'),latest_object_browser_checks=str(r/'browser_checks.json'),latest_object_status='Sparse weak labels tested on13 regions from3 old training works; foliage preserved but14-case regression worse than initial/control; no adoption',latest_object_date='2026-11-16',new_model_trained=True,new_model_trained_this_round=True,new_cunfa_model_trained=False,production_replaced=False,robot_connected=False)
p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8');print(s['latest_object_report'])
