# -*- coding: utf-8 -*-
"""Prepare four small native examples for optional user verification."""
import sys,json,shutil
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from stroke_topology_v13 import ROOT,OLD,jsave,digest
from gongbi_training.review_server_v11 import empty_document

def build():
    root=ROOT/'user_review';root.mkdir(parents=True,exist_ok=True)
    if (root/'package_manifest.json').exists():return root
    for name in ('samples','overlays','labels','backups','exports','logs','browser_profile','cache'):(root/name).mkdir(exist_ok=True)
    source=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'));records=[];package_id='clear_stroke_user_review_v13_20261024'
    for i,identity in enumerate(('C02','C03','C04','C05')):
        r=next(x for x in source['records'] if x['id']==identity);key='R0'+str(i+1);dest=root/'samples'/(key+'.png');shutil.copy2(r['image'],dest)
        from PIL import Image
        with Image.open(dest) as im:w,h=im.size
        record=dict(id=key,title=identity+' 清晰山石逐笔复核',focus='补全你希望保留的每一笔；一条标注代表希望机械臂连续走完的一笔。断墨不必断笔；交叉处明确各笔的连接。',image='samples/'+key+'.png',image_sha256=digest(dest),width=w,height=h,source_group=r['source_group'],source_id=identity,source_box=r['source_box'],source_page=r['source_page'],split='train_calibration',overlay=False,independent_test=False)
        d=empty_document(record,package_id);original=json.loads(Path(r['labels']).read_text(encoding='utf-8'));d['strokes']=original['strokes'];d['regions']=[];d['label_provenance']='assistant_initial_visual_trace_not_expert';d['annotation_goal']='complete_desired_robot_strokes_but_completeness_not_assumed';jsave(root/'labels'/(key+'.json'),d);records.append(record)
    jsave(root/'package_manifest.json',dict(package_id=package_id,samples=records,schema='shanhehui.partial_strokes.v11',initial_labels='assistant drafts only, not user confirmed',unlabeled='unknown',independent_test=False,training_status='not_used_in_current_v13_run'))
    repo=Path(__file__).resolve().parents[1];shutil.copy2(repo/'gongbi_training'/'review_server_v11.py',root/'review_app.py');html=(repo/'gongbi_training'/'review_v11.html').read_text(encoding='utf-8')
    html=html.replace('第一批人工复核','清晰山石逐笔复核').replace('先做 A01–A03：每张 3 条轮廓＋3 条皴线＋2 处明确背景。只标确定的内容，未知区域不算背景。','R01–R04 的橙色线是助手草稿，尚未经过你确认。请逐笔改正、补全；拿不准的部分圈紫色。')
    html=html.replace('本批来自同一画作，只作训练校准，不当独立测试。','本批来自两组山石步骤图，只作训练校准，不当独立测试。').replace('本批来自同一画作，只作训练校准，不当独立测试','本批来自两组山石步骤图，只作训练校准，不当独立测试')
    (root/'review.html').write_text(html,encoding='utf-8');launcher=(LAB/'user_review_v11_20261024'/'start_review.ps1').read_text(encoding='utf-8-sig').replace('qianli_user_review_v11_20261024',package_id).replace('8779','8780');(root/'start_review.ps1').write_text(launcher,encoding='utf-8-sig');(root/'start_review.cmd').write_text('@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_review.ps1"\r\nif errorlevel 1 pause\r\n',encoding='ascii')
    guide='''# 清晰山石逐笔复核

双击 start_review.cmd。原图、工具、标注与浏览器独立配置均保存在 E 盘。

这次不是只选几条主线：请在 R01–R04 上改正、补全你希望机械臂画出的完整目标笔画。初始橙色线全部是助手草稿，尚不是你认可的答案。

每一条标注代表你希望机械臂连续走完的一笔；起止点应落在这笔的起止处。交叉处应把属于同一笔的两边放在同一条标注里。原图的干笔、飞白虽断墨，不一定需要拆成两笔。不要猜测无法从图片确定的原作者真实抬笔动作，只确认我们希望机器人采用的分笔。

可以选中旧标注删除，再重新描；也可以通过“应用到选中项”修改属性。拿不准的位置圈紫色，明确不该画的空白圈绿色。粗细浓淡、皴法名暂不要求填写。

每张完成后点“本张已检查”，再告诉我哪些 R 编号已经复核，以及是否已包含你希望绘制的全部笔画。工具始终保留未标处为未知；不会仅凭点击检查就自动宣称整张已完整标注。

本批只有两组示例的若干阶段，不计为四幅独立画作。这些新复核结果尚未用于当前 v13 训练；收到你的确认后另行纳入。
'''
    (root/'HOW_TO_REVIEW.md').write_text(guide,encoding='utf-8');print('REVIEW_V13_READY',root,flush=True);return root
if __name__=='__main__':build()
