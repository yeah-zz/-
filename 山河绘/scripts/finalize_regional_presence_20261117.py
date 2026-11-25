# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.guohua_hierarchy import CapabilityState,allowed_phases
ROOT=LAB/'regional_presence_trial_20261117';PROJECT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    verification=read(ROOT/'verification.json');browser=read(ROOT/'browser_checks.json');data=read(ROOT/'report_data.json')
    assert verification['four_candidates_have_frozen_checkpoints'] and browser['six_panel_raw_output_switches']
    report=PROJECT/'reports/20261117_regional_presence_training.md'
    assert all(x in report.read_text('utf8') for x in ['推进哪项目标','实际新增什么能力','还有哪些失败','是否通过独立验证'])
    registry=PROJECT/'research/source_usage_registry_20261117.json'
    sources=read(registry) if registry.exists() else dict(records=[])
    ident='ganjing_video:1gga38p7dgh3iKX2kYUhFzF7k1ns1c'
    sources['records']=[r for r in sources['records'] if r.get('source_identity')!=ident]
    sources['records'].append(dict(source_identity=ident,usage='known_development_object_presence_training',
        all_frames_resolutions_and_crops_one_work=True,eligible_for_independent_test=False,
        usage_change_record=str(ROOT/'split_retirement.json'),historical_artifacts_preserved=True))
    write(registry,sources)
    p=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(p);caps=s['capability_state'].copy()
    assert not any(caps.values()) and not allowed_phases(CapabilityState(**caps))['tree_skill_training']
    s.update(latest_round_report=str(report),latest_region_presence_trial=str(ROOT/'results.json'),
        latest_region_presence_report=str(report),latest_region_presence_url=browser['url'],
        latest_region_presence_status='four_known-case_candidates_not_adopted; positive-bag fit causes rock spillover; classifier-only did not add regional learning',
        latest_region_presence_independent_validation=False,new_model_trained_this_round=True,
        new_object_presence_models_trained_this_round=4,new_cunfa_model_trained_this_round=False,
        source_usage_registry=str(registry),production_replaced=False,robot_connected=False)
    assert s['capability_state']==caps;write(p,s)
    p=PROJECT/'research/guohua_four_priorities_execution_20261116.json';e=read(p)
    e.update(latest_round_report=str(report),current_round='G2 weak region-presence and final-classifier isolation paired training; all4candidates not adopted; G1style and tree skills not trained',
        latest_region_presence_trial=str(ROOT/'results.json'),source_usage_registry=str(registry))
    assert next(x for x in e['tracks'] if x['id']==3)['dependencies']==[1,2];write(p,e)
    p=PROJECT/'research/project_plan_20261117.json';plan=read(p)
    plan.update(latest_round_report=str(report),latest_region_presence_trial=str(ROOT/'results.json'),
                agent_implemented=False,production_replaced=False,robot_connected=False)
    write(p,plan)
    checks=dict(four_report_sections=True,capability_flags_unchanged=True,tree_skill_gate_preserved=True,
        baseline_hashes=verification['baseline_hashes'],all_four_candidate_outcomes_reported=True,
        source_usage_change_registered=True,new_model_trained=True,new_cunfa_model_trained=False,
        independent_validation=False,production_replaced=False,robot_connected=False)
    write(ROOT/'postexperiment_checks.json',checks)
    print(json.dumps(dict(url=browser['url'],report=str(report),checks=checks),ensure_ascii=False))
if __name__=='__main__':main()
