# -*- coding: utf-8 -*-
from simple_landscape_pairs_r2 import ROOT
from rock_v15_common import LAB,read,jsave,digest
from pathlib import Path
import shutil,json
if __name__=='__main__':
    r1=ROOT.parent;one=read(r1/'training_summary.json');two=read(ROOT/'training_summary.json');evaluation=read(ROOT/'evaluation_summary.json');paper=read(ROOT/'paper_refinement'/'training_summary.json');entry=read(ROOT/'entry_check_result.json')
    assert one['completed_updates']==1200 and two['completed_updates']==1200 and paper['completed_updates']==300
    assert digest(ROOT/'specialist.pt')==evaluation['checkpoint_sha256'] and digest(ROOT/'selection.json')==evaluation['selection_sha256']
    assert read(ROOT/'data_checks.json')['all_passed'] and entry['raw_bitwise_equal'] and entry['selected_bitwise_equal']
    old=read(LAB/'stable_line_extractor_20261028'/'baseline_lock.json');assert all(digest(r['path'])==r['sha256'] for r in old['files'])
    files=['simple_landscape_pairs.py','simple_landscape_pairs_r2.py','train_simple_landscape_specialist.py','train_simple_landscape_specialist_r2.py','evaluate_simple_landscape_specialist.py','evaluate_simple_landscape_specialist_r2.py','prepare_specialist_holdouts.py','prepare_specialist_holdouts_r2.py','check_specialist_data.py','check_specialist_data_r2.py','predict_simple_landscape_specialist.py','run_simple_landscape_specialist.ps1','refine_specialist_paper.py','report_simple_landscape_specialist.py','report_native_specialist.py','approved_reference_quality_gate.py']
    scripts=Path(__file__).parent;source=[]
    for name in files:
        p=scripts/name;dest=ROOT/'source_snapshot'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest);source.append(dict(path=str(p),sha256=digest(p)))
    decision=dict(goal_every_generated_image_matches_reference_not_achieved=True,new_gpu_updates_executed=2700,retained_main_specialist_training_updates=2400,paper_trial_updates=300,paper_trial_retained_step=paper['retained_step'],paper_trial_not_promoted=paper['retained_step']==0,final_imagegen_automatic_rule_pass_count=evaluation['imagegen_automatic_pass_count'],final_imagegen_scene_count=30,three_native_checks_separate=True,quality_rule_pass_not_expert_quality=True,paired_metrics_only_apply_to_procedural_renderer=True,old_weights_and_v4_preserved=True,entry_matches_batch_bitwise=True,source=source,checkpoint_sha256=digest(ROOT/'specialist.pt'),selection_sha256=digest(ROOT/'selection.json'),status='experimental; keep accepted v4; no robot connection')
    jsave(ROOT/'FINAL_VERIFICATION.json',decision)
    conclusion='''<section id="adoption_decision"><h2>当前结论：尚未实现每张生成图稳定达标</h2><p>已完成两阶段共 2400 次 GPU 监督更新。40 张受控程序留出样本的中心线 F1 从 v4 的 0.909 提高到 0.976；这些样本具有精确声明的程序路径，不能将这个数字用作真正绘画图像的准确率。30 个全新 imagegen 场景均触发了预设质检，3 张原生整幅图也未通过；这不是人工判断“全部不好”，但说明当前质检和模型尚不能提供目标稳定性。</p><p>原始网络结果中仍可见纸纹误检、皴线碎片和粗墨交汇问题。追加了 300 次空白纸面负样本修正；按独立纸面与正样本校准规则未选出更好的检查点，保留步数为 0，故未采用。旧 v4 和旧稳定入口均不替换。不得把自动回退后的 v4 图片说成新网络训练成果。</p><p><a href="native_report.html">原生整图与松针局部</a> · <a href="paper_refinement/PAPER_RECHECK.jpg">未采用的纸面修正复测</a> · <a href="FINAL_VERIFICATION.json">最终采用决定与完整性记录</a></p><p>曾尝试仅用三张已认可输出形成参照质检，但缩放会改变闭环统计，且只有少量图像级正例，规则未启用，没有据此放宽新图通过条件。后续应优先补齐真实生成画风的可靠线条参考和纸纹/粗墨反例，再验证泛化。</p></section>'''
    main=ROOT/'index.html';page=main.read_text(encoding='utf-8')
    if 'id="adoption_decision"' not in page:page=page.replace('<h2>40 张程序留出图上的测量</h2>',conclusion+'<h2>40 张程序留出图上的测量</h2>')
    main.write_text(page,encoding='utf-8')
    print('SPECIALIST_FINALIZED',str(main),json.dumps({k:v for k,v in decision.items() if k!='source'}),flush=True)
