import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
r=LAB/'guohua_scene4_mapping_fix_20261116';p=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=json.loads(p.read_text('utf8'));assert not any(s['capability_state'].values())
s.update(latest_object_result=str(r/'results.json'),latest_object_report=json.loads((r/'server.json').read_text('utf8'))['url'],latest_object_verification=str(r/'verification.json'),latest_object_browser_checks=str(r/'browser_checks.json'),latest_object_status='Source mapping conflict fixed; comparable corrected-taxonomy regression improves, development open/rock cores improve; other objects and some tree regression remain; no production adoption',active_scene_supervision_schema='source14_to_scene4_v2_open1_4_10',legacy_scene4_run_has_supervision_conflict=True,legacy_scene4_results_preserved=True,canonical_four_class_mapper='gongbi_training/source_scene_mapping.py',new_model_trained=True,new_model_trained_this_round=True,new_cunfa_model_trained=False,production_replaced=False,robot_connected=False)
p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8');print(s['latest_object_report'])
