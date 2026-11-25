# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,shutil
ROOT=LAB/'ink_mask_centerline_20261105'
def read(p):return json.loads(p.read_text('utf8'))
def main():
    catalog=read(ROOT/'catalog.json');decision=read(ROOT/'decision.json')
    accepted=sum(r['controls']['model_source']['whole_replacements'] for r in catalog['scenes'])
    metrics=read(ROOT/'reference_check.json')['records']
    fused=[r for r in metrics if r['method']=='model_source']
    beats_old=all(r['mean_single_curve_reference_coverage']>r['old_best_single_path_reference_coverage'] for r in fused)
    decision.update(accepted_whole_replacements=accepted,beats_old_on_both_fixed_reference_coverages=beats_old,
        experiment_passed=False,new_weights_trained=False,
        reason='no reliable whole-image improvement over approved 500/700 baselines; source control also not consistently surpassed')
    (ROOT/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf8')
    page=ROOT/'report.html';html=page.read_text('utf8')
    note='<p class="note" id="result-status"><b>当前结论：局部几何贴合改善，但没有证明超过已认可500／700笔版本，未采用。</b> 固定参考的中心线仍有遗漏和歧义；整幅图查询候选有重复或待定分支，完整替换通过数为'+str(accepted)+'。原方案保持不变。本轮是既有分割模型＋源图约束＋确定性读出，没有新训练，也不是新的神经曲线头效果。</p>'
    if 'id="result-status"' not in html:html=html.replace('<h2>固定R04前三条局部</h2>',note+'<h2>固定R04前三条局部</h2>')
    html=html.replace('再对单个区域做骨架读出','再在已连通区域内读出经过提示附近的完整骨架路径')
    page.write_text(html,encoding='utf8')
    for name in ('ink_mask_centerline_20261105.py','build_ink_mask_centerline_20261105.py','finalize_ink_mask_readout_20261105.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source_snapshot'/name)
    print('MASK_READOUT_ADOPTED',decision['adopted'],'REPLACEMENTS',accepted,flush=True)
if __name__=='__main__':main()
