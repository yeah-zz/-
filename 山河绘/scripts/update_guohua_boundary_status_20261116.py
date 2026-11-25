import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
root=LAB/'guohua_boundary_reference_20261116';p=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=json.loads(p.read_text('utf8'));assert not any(s['capability_state'].values())
s.update(latest_reference_file=str(root/'reviewed_references.json'),latest_reference_report=json.loads((root/'server.json').read_text('utf8'))['url'],latest_reference_diagnostics=str(root/'reviewed_results.json'),latest_reference_status='Four known cases,20 assistant sparse object/corridor references and5 unknown regions; no new independent validation',latest_reference_expert_reviewed=False,new_model_trained_this_round=False,production_replaced=False,robot_connected=False)
p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8');print(s['latest_reference_report'])
