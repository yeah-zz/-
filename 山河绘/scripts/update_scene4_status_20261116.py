import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
r=LAB/'guohua_scene4_20261116';p=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=json.loads(p.read_text('utf8'));assert not any(s['capability_state'].values())
s.update(latest_object_result=str(r/'results.json'),latest_object_report=json.loads((r/'server.json').read_text('utf8'))['url'],latest_object_verification=str(r/'verification.json'),latest_object_browser_checks=str(r/'browser_checks.json'),latest_object_status='Four-class scene candidates trained; known white-mountain cores improve but heldout open-scene recall0.1% and old regression lower; neither adopted',new_model_trained=True,new_model_trained_this_round=True,scene_background_class_trained=True,scene_background_independent_validation=False,new_cunfa_model_trained=False,production_replaced=False,robot_connected=False)
p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8');print(s['latest_object_report'])
