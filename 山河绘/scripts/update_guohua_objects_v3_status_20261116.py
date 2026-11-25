import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
root=LAB/'guohua_objects_v3_20261116';p=LAB/'guohua_hierarchy_20261112/current_milestones.json'
s=json.loads(p.read_text('utf8'));assert not any(s['capability_state'].values())
s.update(latest_object_result=str(root/'results.json'),latest_object_report=json.loads((root/'server.json').read_text('utf8'))['url'],latest_object_verification=str(root/'verification.json'),latest_object_browser_checks=str(root/'browser_checks.json'),latest_object_status='Core/interface loss failed known-work and foliage regression; both candidates rejected',latest_object_date='2026-11-16',new_model_trained=True,new_cunfa_model_trained=False,production_replaced=False,robot_connected=False)
p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8');print(s['latest_object_report'])
