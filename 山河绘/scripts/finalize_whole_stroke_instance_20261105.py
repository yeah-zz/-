# -*- coding: utf-8 -*-
"""Derive explicit pass/fail from saved outputs, without regenerating candidates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,shutil,hashlib,copy

ROOT=LAB/'whole_stroke_instance_20261105/comparison'
def read(p):return json.loads(p.read_text('utf8'))
def main():
    diag=read(ROOT/'reference_diagnostic.json')['records'];catalog=read(ROOT/'catalog.json')
    for record in catalog['scenes']:
        file=ROOT/record['id']/'plan.json';plan=read(file)
        plan['stats']['before']=copy.deepcopy(plan['stats']['after']);plan['stats']['accepted_continuations']=0
        file.write_text(json.dumps(plan,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        record['plan_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    (ROOT/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf8')
    index=ROOT/'index.html';index.write_text(index.read_text('utf8').replace('原图支持的断段连接','新模型整笔替换（未采用）'),encoding='utf8')
    accepted=sum(r['controls']['attention']['accepted_whole_replacements'] for r in catalog['scenes'])
    chosen=[r for r in diag if r['model']=='attention']
    gain=all(r['mean_single_curve_coverage_2_native_px']>r['baseline_best_single_path_coverage'] for r in chosen)
    decision=read(ROOT/'decision.json');decision.update(experiment_passed=gain and accepted>0,
        reference_gain_on_both_fixed_records=gain,accepted_whole_replacements=accepted,
        rejection_is_derived_from_saved_outputs=True)
    (ROOT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf8')
    table='<h2>固定真实参考与旧提取器对照</h2><table><tr><th>参考图</th><th>旧提取器最佳单路径覆盖</th><th>联合模型</th><th>归属注意力模型</th></tr>'
    for key in ('R03','R04'):
        a=next(r for r in diag if r['id']==key and r['model']=='first');b=next(r for r in diag if r['id']==key and r['model']=='attention')
        table+=f'<tr><td>{key}</td><td>{a["baseline_best_single_path_coverage"]:.1%}</td><td>{a["mean_single_curve_coverage_2_native_px"]:.1%}</td><td>{b["mean_single_curve_coverage_2_native_px"]:.1%}</td></tr>'
    table+='</table><p>这里是原生2像素容差下，单条路径覆盖已复核参考的比例；模型使用参考短提示与参考框，旧提取器取已生成路径中的最佳者。两者接口不同，不能称为同条件的自动整图准确率比较。本次新模型在这个更容易的提示诊断中仍未超过旧候选。实际500／700笔查询中，整条替换通过数为'+str(accepted)+'。</p>'
    p=ROOT/'report.html';html=p.read_text('utf8')
    if '<!-- saved-comparison-gate -->' not in html:html=html.replace('<h2>固定500／700笔基线上的实际输入</h2>','<!-- saved-comparison-gate -->'+table+'<h2>固定500／700笔基线上的实际输入</h2>')
    p.write_text(html,encoding='utf8')
    shutil.copy2(Path(__file__).with_name('whole_stroke_instance_20261105.py'),ROOT/'source_snapshot/whole_stroke_instance_20261105.py')
    shutil.copy2(Path(__file__).with_name('build_whole_stroke_instance_results_20261105.py'),ROOT/'source_snapshot/build_whole_stroke_instance_results_20261105.py')
    print('EXPERIMENT_PASS',decision['experiment_passed'],'ADOPTED',decision['adopted'],'ACCEPTED',accepted,flush=True)
if __name__=='__main__':main()
