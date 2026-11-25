# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.guohua_hierarchy import CapabilityState,allowed_phases
PROJECT=Path(__file__).resolve().parents[1];ROOT=LAB/'whole_rock_evidence_20261117'
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    verification=read(ROOT/'verification.json');browser=read(ROOT/'browser_checks.json');summary=read(ROOT/'summary.json')
    assert verification['engineering_passed'] and browser['playback_advances']
    report=PROJECT/'reports/20261117_whole_rock_evidence.md'
    assert all(x in report.read_text('utf8') for x in ['推进哪项目标','实际新增什么能力','还有哪些失败','是否通过独立验证'])
    p=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(p);caps=s['capability_state'].copy()
    assert not any(caps.values())
    assert not allowed_phases(CapabilityState(**caps))['tree_skill_training']
    s.update(latest_round_report=str(report),latest_evidence_pack=str(ROOT/'packet.json'),
             latest_evidence_pack_report=str(report),latest_evidence_pack_replay=browser['url'],
             latest_evidence_pack_verification=str(ROOT/'verification.json'),
             latest_evidence_pack_status='known_reference_integration_and_supervision_audit_only',
             latest_evidence_pack_new_complete_strokes=0,latest_evidence_pack_new_independent_works=0,
             new_model_trained_this_round=False,production_replaced=False,robot_connected=False)
    assert caps==s['capability_state'];write(p,s)
    p=PROJECT/'research/guohua_four_priorities_execution_20261116.json';e=read(p)
    e.update(latest_round_report=str(report),current_round='P1 whole-rock evidence integration and task-scoped supervision audit; 35 reused units, no new complete strokes, independent works or trained model',
             latest_evidence_pack=str(ROOT/'packet.json'),latest_evidence_pack_readiness=str(ROOT/'training_readiness.json'))
    assert next(x for x in e['tracks'] if x['id']==3)['dependencies']==[1,2];write(p,e)
    p=PROJECT/'research/project_plan_20261117.json';plan=read(p)
    plan.update(latest_round_report=str(report),latest_evidence_pack=str(ROOT/'packet.json'),
                agent_implemented=False,production_replaced=False,robot_connected=False)
    write(p,plan)
    checks=dict(four_report_sections=True,capability_flags_unchanged=caps==s['capability_state'],
                tree_skill_gate_preserved=True,agent_contracts_only=True,baseline_hashes=summary['baseline_hash_checks'],
                independent_validation_passed=False,new_model_trained=False,robot_connected=False)
    write(ROOT/'postexperiment_checks.json',checks)
    print(json.dumps(dict(url=browser['url'],report=str(report),checks=checks),ensure_ascii=False))
if __name__=='__main__':main()
