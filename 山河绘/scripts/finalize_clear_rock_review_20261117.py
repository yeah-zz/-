# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.guohua_hierarchy import CapabilityState,allowed_phases
ROOT=LAB/'clear_rock_sources_20261117';PROJECT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    verification=read(ROOT/'verification.json');browser=read(ROOT/'browser_checks.json')
    assert verification['ten_crops_pixel_exact'] and browser['all_22_images_loaded']
    report=PROJECT/'reports/20261117_clear_rock_source_followup.md'
    assert all(x in report.read_text('utf8') for x in ['推进哪项目标','实际新增什么能力','还有哪些失败','是否通过独立验证'])
    p=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(p);before=s['capability_state'].copy()
    assert not any(before.values()) and not allowed_phases(CapabilityState(**before))['tree_skill_training']
    s.update(latest_round_report=str(report),latest_source_followup=str(ROOT/'local_references.json'),
             latest_source_followup_report=str(report),latest_source_followup_url=browser['url'],
             latest_source_followup_new_regions=10,latest_source_followup_new_independent_works=0,
             latest_source_followup_new_complete_strokes=0,
             latest_source_followup_status='known_source_native_resolution_and_regional_references_only; complete_stroke_reference_goal_unmet',
             new_model_trained_this_round=False,production_replaced=False,robot_connected=False)
    assert s['capability_state']==before;write(p,s)
    p=PROJECT/'research/guohua_four_priorities_execution_20261116.json';e=read(p)
    e.update(latest_round_report=str(report),current_round='P1 source audit: existing video supplemented at720p,10new regional context references; no independent work or complete artist stroke acquired',
             latest_source_followup=str(ROOT/'local_references.json'))
    assert next(x for x in e['tracks'] if x['id']==3)['dependencies']==[1,2];write(p,e)
    p=PROJECT/'research/project_plan_20261117.json';plan=read(p)
    plan.update(latest_round_report=str(report),latest_source_followup=str(ROOT/'local_references.json'),
                agent_implemented=False,production_replaced=False,robot_connected=False)
    write(p,plan)
    checks=dict(four_report_sections=True,capabilities_unchanged=True,tree_gate_preserved=True,
                baseline_hashes=verification['baseline_hashes'],new_independent_works=0,
                new_complete_artist_strokes=0,new_model_trained=False,robot_connected=False)
    write(ROOT/'postexperiment_checks.json',checks)
    print(json.dumps(dict(url=browser['url'],report=str(report),checks=checks),ensure_ascii=False))
if __name__=='__main__':main()
