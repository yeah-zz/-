import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
r=LAB/'guohua_priority_check_20261116';p=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=json.loads(p.read_text('utf8'));assert not any(s['capability_state'].values())
s.update(latest_joint_report=json.loads((r/'server.json').read_text('utf8'))['url'],latest_partition_prospective_check=str(r/'results.json'),latest_cunfa_failure_queue=str(r/'style_failure_review_queue.json'),priority_execution_file=str(Path(__file__).resolve().parents[1]/'research/guohua_four_priorities_execution_20261116.json'),main_research_track='cunfa concepts, full rock context, mark/group evidence',partition_scope='Frozen research candidate; critical rock/tree/main-contour errors only; no generic other-class score chasing',previously_reserved_work_groups_status='Both opened for test; unselected crops are not future independent works',object_validation_still_incomplete=True,new_model_trained_this_round=False,new_cunfa_model_trained=False,production_replaced=False,robot_connected=False)
p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8');print(s['latest_joint_report'])
