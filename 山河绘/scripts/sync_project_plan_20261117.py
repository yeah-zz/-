# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.guohua_hierarchy import CapabilityState,allowed_phases
PROJECT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 plan=PROJECT/'PROJECT_PLAN.md';template=PROJECT/'research/ROUND_REPORT_TEMPLATE.md';report=PROJECT/'reports/20261117_project_plan_reorganization.md';lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json')
 baseline_checks={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(baseline_checks.values())
 headings=['推进哪项目标','实际新增什么能力','还有哪些失败','是否通过独立验证'];assert all(h in template.read_text('utf8') and h in report.read_text('utf8') for h in headings)
 p=PROJECT/'research/experiment_goal_alignment_template.json';t=read(p);t.update(project_plan=str(plan),round_report_template=str(template),round_report_file=None,mandatory_report_sections=headings,agent_participation='none_or_retrieval_or_planning_or_tool_orchestration',agent_planning_test_separate_from_perception=True,
  validation_scope='not_tested_or_engineering_checks_or_known_regression_or_independent_works_or_physical',adoption_decision='not_adopted_or_local_or_object_or_full',rollback_baseline=None,failures_and_regressions=[],independent_validation_passed=False,production_replaced=False,robot_connected=False);write(p,t)
 execution=PROJECT/'research/guohua_four_priorities_execution_20261116.json';e=read(execution)
 if 'history_before_project_plan_20261117' not in e:e['history_before_project_plan_20261117']={'current_round':e['current_round']}
 e.update(canonical_project_plan=str(plan),round_report_template=str(template),latest_round_report=str(report),current_round='Project plan and reporting reorganization only; no new model training, no recognition validation, no baseline replacement',agent_track=dict(status='planned_not_implemented',initial_architecture='single planner with source-grounded tools and deterministic validators',allowed_now='offline planning engineering only; given-reference planning separated from perception/end-to-end validation',cannot_replace_goals=[1,2],does_not_unlock_tree_training=True))
 e['experiment_requirements'].update(mandatory_four_question_report=True,round_report_template=str(template),agent_plans_not_perception_evidence=True,verified_modules_integrate_incrementally=True,rollback_and_baseline_hash_required=True)
 assert [r['id'] for r in e['tracks']]==[1,2,3,4];assert next(r for r in e['tracks'] if r['id']==3)['dependencies']==[1,2];write(execution,e)
 machine=dict(version='2026-11-17',goals=[1,2,3,4],main_tracks_now=[1,2],
  stages=[dict(id='P0',name='基线与报告制度',status='engineering_complete'),dict(id='P1',name='可靠数据与知识技能参考',status='priority_in_progress'),dict(id='P2',name='对象与皴法辨识',status='candidates_not_validated',goals=[1,2]),dict(id='P3',name='完整山石与笔触',status='local_interface_available_quality_not_validated',goals=[1,2,4]),dict(id='P4',name='单Agent离线原型',status='planned_not_implemented',engineering_can_start=True),dict(id='P5',name='树木技能',status='waiting_for_goals1and2',requires_goals=[1,2]),dict(id='P6',name='整幅临摹与同风格创作',status='waiting_for_validated_components'),dict(id='P7',name='实体毛笔与闭环',status='not_validated_default_offline')],
  report_questions=headings,plan_file=str(plan),plan_sha256=sha(plan),report_template=str(template),baseline_checks=baseline_checks,agent_implemented=False,production_replaced=False,robot_connected=False)
 write(PROJECT/'research/project_plan_20261117.json',machine)
 current=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(current);before=s['capability_state'].copy();phases=allowed_phases(CapabilityState(**before));assert not any(before.values()) and not phases['tree_skill_training']
 s.update(canonical_user_goals_sha256=sha(PROJECT/'PROJECT_GOALS.md'),canonical_project_plan_file=str(plan),canonical_project_plan_sha256=sha(plan),canonical_project_plan_json=str(PROJECT/'research/project_plan_20261117.json'),round_report_template=str(template),latest_round_report=str(report),project_plan_reorganization_new_model_trained=False,agent_capability_state=dict(implemented=False,planning_independent_validation=False,end_to_end_independent_validation=False),engineering_work_allowed=dict(agent_offline_planning_prototype=True,local_candidate_integration=True),production_replaced=False,robot_connected=False)
 assert s['capability_state']==before;s['allowed_phases']=phases;write(current,s)
 checks=dict(four_goals_preserved=True,four_report_questions_present=True,approved500_hash_unchanged=baseline_checks['approved500'],working489_hash_unchanged=baseline_checks['working489'],capability_flags_unchanged=s['capability_state']==before,tree_gate_preserved=True,agent_still_unimplemented=True,new_model_trained=False,robot_connected=False)
 write(LAB/'guohua_hierarchy_20261112/project_plan_verification_20261117.json',checks);print(json.dumps(checks,ensure_ascii=False))
if __name__=='__main__':main()
