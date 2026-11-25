import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
root=LAB/'guohua_objects_v2_20261113'
path=LAB/'guohua_hierarchy_20261112/current_milestones.json'
state=json.loads(path.read_text('utf8'))
assert not any(state['capability_state'].values())
state.update(latest_object_result=str(root/'results.json'),latest_object_report=json.loads((root/'server.json').read_text('utf8'))['url'],latest_object_verification=str(root/'verification.json'),latest_object_browser_checks=str(root/'browser_checks.json'),latest_object_status='Two expanded-data candidates evaluated; tree regression prevents adoption',source_checked_concept_illustrations=6,concept_publication_groups=2,concept_illustrators=None,concept_illustrator_count_status='HKMA illustrator identity unverified; do not claim six independent authors',new_cunfa_model_trained=False,production_replaced=False,robot_connected=False)
path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(latest=state['latest_object_report'],capability_state=state['capability_state'],production_replaced=False),ensure_ascii=False))
