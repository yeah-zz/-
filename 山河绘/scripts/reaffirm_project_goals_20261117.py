# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.guohua_hierarchy import CapabilityState,allowed_phases
PROJECT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 goal=PROJECT/'PROJECT_GOALS.md';execution=PROJECT/'research/guohua_four_priorities_execution_20261116.json';current=LAB/'guohua_hierarchy_20261112/current_milestones.json';e=read(execution);s=read(current);before=s['capability_state'].copy();assert not any(before.values())
 if 'history_before_user_reaffirmation_20261117' not in e:e['history_before_user_reaffirmation_20261117']={k:e[k] for k in ['priority_source','current_round']}
 e.update(priority_source='User explicitly reaffirmed all four goals on2026-11-17; every change must advance them',canonical_goals_file=str(goal),current_round='Persisted four-goal alignment and evidence rules; local21path replay is engineering support only; no new model training this round')
 for t in e['tracks']:
  t['independent_validation_passed']=False
  if t['id']==1:t.update(current_result='Cunfa labels, partial attributes and role/boundary candidates remain unvalidated; local21path image fit does not add recognition',next_evidence='Reliable complete-rock object/structure/cun-ink/group references, explicit pima/fupi and mixed/unknown tests on unseen works')
  if t['id']==2:t['reopen_triggers']=list(dict.fromkeys(t['reopen_triggers']+['内部结构线被错当皴笔','植被/点苔混入山石笔组']))
  if t['id']==3:assert t['dependencies']==[1,2] and not t['new_tree_skill_training']
 e['experiment_requirements']=dict(template=str(PROJECT/'research/experiment_goal_alignment_template.json'),engineering_support_not_capability_pass=True,manual_reference_not_model_output=True,full_stroke_and_group_evidence_required=True,approved500_preserved=True,no_silent_production_replacement=True)
 write(execution,e);state=CapabilityState(**before);phases=allowed_phases(state);assert not phases['tree_skill_training'] and not phases['full_offline_creation_validation']
 s.update(user_goals_reaffirmed_date='2026-11-17',canonical_user_goals_file=str(goal),canonical_user_goals_sha256=hashlib.sha256(goal.read_bytes()).hexdigest(),priority_execution_file=str(execution),experiment_alignment_template=str(PROJECT/'research/experiment_goal_alignment_template.json'),main_research_track='Goals1&2: cunfa evidence/complete rock strokes and critical rock-tree separation',local_patch_scope='Engineering integration and assistant-reviewed phase ordering; no recognition capability validated',allowed_phases=phases,goals_reaffirmation_new_model_trained=False,production_replaced=False,robot_connected=False)
 assert s['capability_state']==before;write(current,s)
 check=dict(goal_ids=[t['id'] for t in e['tracks']],capability_state_unchanged=s['capability_state']==before,tree_skill_still_waiting=True,local_replay_not_recognition=True,robot_connected=False)
 assert check['goal_ids']==[1,2,3,4];write(LAB/'guohua_hierarchy_20261112/goals_reaffirmation_20261117.json',check);print(json.dumps(check,ensure_ascii=False))
if __name__=='__main__':main()
