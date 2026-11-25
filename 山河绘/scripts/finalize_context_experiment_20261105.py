# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
import json,hashlib,shutil
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from context_curve_data_20261105 import ROOT,write
from context_curve_model_20261105 import ContextCurveNet
from context_curve_apply_20261105 import select
from train_context_center_r2_20261105 import R2,reserve_record

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2)
    r=reserve_record();model=ContextCurveNet().cuda();model.load_state_dict(torch.load(R2/'model.pt',map_location='cpu',weights_only=False)['model'])
    summary=json.loads((R2/'training_summary.json').read_text('utf8'))
    rows=[dict(points=p,module=0,source_curve_ids=[i]) for i,p in enumerate(r['refs'])]
    result=select(model,summary['threshold'],r['rgb'],rows)
    assert not summary['teacher_preservation_passed'] and len(result['rejected'])>0
    source=Image.fromarray(r['rgb']);before=Image.new('RGB',source.size,'white');after=before.copy();b=ImageDraw.Draw(before);a=ImageDraw.Draw(after)
    for i,p in enumerate(r['refs']):
        points=[tuple(v) for v in p];b.line(points,fill='black',width=1)
        a.line(points,fill='#d83727' if i in result['rejected'] else 'black',width=2 if i in result['rejected'] else 1)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    board=Image.new('RGB',(1530,600),'#f4f1e8');d=ImageDraw.Draw(board)
    for k,(im,label) in enumerate([(source,'A06原图'),(before,'直接输入的已复核参考（非提取结果）'),(after,'红：被第二轮误删的真实参考')]):
        im=im.resize((500,500),Image.Resampling.LANCZOS);board.paste(im,(k*510+3,45));d.text((k*510+3,10),label,font=font,fill='black')
    d.text((8,560),f'检查{len(rows)}条真实参考，误删{len(result["rejected"])}条。第二轮未通过保留检查，未接入当前候选。',font=font,fill='black')
    board.save(R2/'rejected_teacher_regression.jpg',quality=95)
    write(R2/'adoption_decision.json',dict(adopted=False,reason='failed checked teacher-contour preservation on reserved A06 image',
        checked_input_curves=len(rows),wrongly_rejected=len(result['rejected']),reference_test_is_not_actual_extraction_accuracy=True,
        prior_A05_test_retired_to_round2_training=True,raw_threshold=summary['threshold'],robot_connected=False))
    joint=ROOT/'joint_calibration.json';j=json.loads(joint.read_text('utf8'));j.update(adopted=False,
        reason='lower joint threshold rejected two checked A05 target curves; raw-network threshold retained')
    write(joint,j)
    outcome=json.loads((ROOT/'results.json').read_text('utf8'));outcome.update(
        center_round2_adopted=False,center_round2_wrongly_removed_reference_curves=len(result['rejected']),
        lower_joint_threshold_adopted=False,release_to_robot=False,
        effect_assessment='small offline changes only; no demonstrated broad reduction of landscape clutter')
    write(ROOT/'results.json',outcome)
    page=(ROOT/'index.html').read_text('utf8')
    note='<p class="note"><strong>本轮结论：尚未取得明显去杂突破。</strong>较保守候选只少量排除了同墨带路径，补提少量松枝细线。更激进的联合阈值误删2条A05参考，未采用。第二轮加入已知墨带侧边反例后，A06的181条参考中仍误删8条，第二轮权重也未采用。以下效果图是离线候选，不能当作已验证的改进版或实画效果。</p>'
    page=page.replace('<h1>整条曲线＋局部图像上下文筛选</h1>','<h1>整条曲线＋局部图像上下文筛选</h1>'+note)
    page=page.replace('当前保护门槛通过：','旧R01–R04局部参考覆盖门槛通过：')
    page+='<h2>未采用的第二轮：真实参考误删诊断</h2><p>这是把已复核曲线直接输入筛选器的保留性检查，不是把这些曲线当作模型实际提取结果。它说明该轮会损伤应保留的轮廓，不能因画面变空就采用。</p><img src="center_round2/rejected_teacher_regression.jpg"><p><a href="center_round2/training_summary.json">第二轮训练与检查</a> · <a href="center_round2/protocol.json">第二轮数据划分</a> · <a href="center_round2/adoption_decision.json">未采用原因</a> · <a href="joint_calibration.json">未采用的低阈值方案</a></p>'
    (ROOT/'index.html').write_text(page,encoding='utf8')
    for name in ['context_curve_model_20261105.py','context_curve_data_20261105.py','context_curve_apply_20261105.py',
        'context_relation_20261105.py','calibrate_context_joint_20261105.py','train_context_center_r2_20261105.py','finalize_context_experiment_20261105.py']:
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source_snapshot'/name)
    print('FINALIZED',outcome['records'],flush=True)

if __name__=='__main__':main()
