# -*- coding: utf-8 -*-
import sys,json,html,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from two_work_bands_20261118 import ROOT,PRIOR,OLD,LAB,PROJECT,FONT,read,sha,write
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
    results=read(ROOT/'results.json');ds=read(ROOT/'dataset.json');protocol=read(ROOT/'protocol.json');freeze=read(ROOT/'checkpoint_freeze.json')
    assert sha(ROOT/'dataset.json')==protocol['dataset_sha256'];assert sha(OLD/'flat_roles.pt')==protocol['teacher_sha256']
    assert all(sha(ROOT/(a+'.pt'))==h for a,h in freeze.items());assert all(sha(r['path'])==r['sha256'] for r in ds['records'])
    reviewed=read(ROOT/'reviewed.json');assert sha(ROOT/'reviewed.json')==ds['reviewed_sha256'];assert sha(reviewed['source'])==reviewed['source_sha256']
    labels=['原图','冻结旧模型','核心对照','核心＋纸面'];arms=['teacher','core_control','band_paper'];cards=[]
    summary='<table><tr><th>范围</th>'+''.join('<th>'+s+'</th>' for s in labels[1:])+'</tr>'
    for r in results['records']:
        image=np.load(ROOT/'predictions'/(r['id']+'_teacher.npz'))['image'];views=[image]
        for a in arms:
            p=np.load(ROOT/'predictions'/(r['id']+'_'+a+'.npz'))['prediction'];v=image.copy();v[p>0]=(.35*v[p>0]+.65*np.array([32,127,137])).astype('uint8');views.append(v)
        box={'L':[22,60,179,218],'F':[0,0,210,220],'H':[0,0,410,265]}[r['id']]
        board=Image.new('RGB',(1320,390),'#f6f3eb');d=ImageDraw.Draw(board)
        for i,(im,label) in enumerate(zip(views,labels)):
            p=Image.fromarray(im).crop(box);scale=min(318/p.width,295/p.height);p=p.resize((round(p.width*scale),round(p.height*scale)),Image.Resampling.NEAREST)
            board.paste(p,(i*330+6,40));d.text((i*330+6,8),label,font=ImageFont.truetype(FONT,21),fill='#294136')
        d.text((8,350),r['id']+'：青色为原始模型预测墨迹范围；非皴法类别。'+('已知案例回归，非独立测试' if r['id']=='H' else '训练作品'),font=ImageFont.truetype(FONT,19),fill='#7c573b')
        board.save(ROOT/'views'/(r['id']+'_comparison.png'))
        summary+='<tr><td>'+r['id']+' '+('已知回归' if r['id']=='H' else '训练参考')+'</td>'
        for a in arms:
            v=r['scores'][a];summary+=f'<td>墨迹保留 {v["ink_retained"]}/{v["ink_pixels"]}<br>近邻纸面误检 {v["paper_false_ink"]}/{v["paper_pixels"]}</td>'
        summary+='</tr>';cards.append('<h2>'+r['id']+' 对照</h2><img src="views/'+r['id']+'_comparison.png" alt="原图与三组原始墨迹输出">')
    summary+='</table>'
    write('suppressed_reference_review.json',dict(new_suppressed_legacy_points=[[109,202],[110,202],[158,208],[158,209],[112,213]],
        review='Source enlargement shows pale/edge/possibly blank pixels; not all confirmed true missing ink. Preserve legacy scores, do not relabel or train on H14.',
        actual_stroke_loss_verified=False,training_allowed=False,legacy_scores_unchanged=True))
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json');bc={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(bc.values())
    checks=dict(two_work_reference_hashes_match=True,second_source_preexisting_training_development=True,
        protected_clpd_works_not_used=True,H14_no_gradients=True,checkpoint_hashes_match=True,
        final_fixed_step_not_selected_on_H14=True,conditional_role_preserved=True,not_style_training=True,
        no_reference_mask_at_inference=True,baseline_hashes=bc,new_model_trained=True,
        independent_validation=False,adopted=False,robot_connected=False)
    write('verification.json',checks)
    shutil.copyfile(PRIOR/'reference_overview.png',ROOT/'first_source_reference.png')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双作品墨带监督对照</title><style>body{font:17px/1.7 system-ui;max-width:1400px;margin:auto;padding:24px;background:#f6f3eb;color:#294136}img{max-width:100%;height:auto;display:block;margin:20px 0}.notice{padding:18px;background:#e8ddc7}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #bbb;white-space:nowrap}.scroll{overflow:auto}a{color:#17694f}</style><h1>两件作品：墨迹核心与近邻纸面监督对照</h1><p class="notice">已补第2件既有训练作品的一段山石轮廓，合计两件作品、6段部分墨带参考。新训练2个前景范围候选，各固定80步；角色条件概率保持冻结，没有训练皴法分类或树木技能。所有参考均非专家完整真值。</p><p>两组使用相同初始化、旧核心回放与新增墨迹核心，差别仅为第二组增加近邻纸面监督。没有依靠H14选择迭代次数或模型；H14是已反复查看的已知回归作品，不是独立测试。</p><div class="scroll">SUMMARY</div><p>加入纸面监督后，训练侧近邻纸面误检从对照组73处降至22处，358个确定墨迹像素都保留。H14所选墨带核心仍为123/125，近邻纸面误检29→7。上述统计是已标参考像素，不是笔画数。</p><p>更广的旧H14角色参考对应出现退步；新增受抑制的5处位置需要核查，放大后部分是淡痕/边缘或可能留白，不能直接称为漏掉5条皴笔。未证明跨作品泛化，本轮不采用新模型。</p>CARDS<h2>新增第二作品参考（非模型输出）</h2><img src="views/second_source_review.png"><details><summary>第一作品的前后阶段参考</summary><img src="first_source_reference.png"></details><h2>新增抑制位置复核</h2><img src="views/new_missing_ink.png"><p>保留原始计分，不用回归图修正模型或回灌标签；皴法命名、真实分笔和物理参数仍未知。</p><p><a href="reviewed.json">第二来源参考</a> · <a href="dataset.json">两作品分组</a> · <a href="protocol.json">训练前冻结方案</a> · <a href="results.json">全部结果</a> · <a href="checkpoint_freeze.json">权重哈希</a> · <a href="suppressed_reference_review.json">待复核位置</a> · <a href="verification.json">完整性检查</a></p></html>'''
    (ROOT/'index.html').write_text(page.replace('SUMMARY',summary).replace('CARDS',''.join(cards)),encoding='utf8')
    report=PROJECT/'reports/20261118_two_work_bands.md'
    report.write_text('''# 双作品墨带监督训练对照

## 推进哪项目标
推进G1/G2前置的可靠墨迹范围监督，将范围与角色分开；未新增皴法分类或树木技能。

## 实际新增什么能力
补充第2件既有训练开发作品的1段轮廓墨带，累计2作品6段部分参考；新增77个墨迹核心和222个纸面参考像素。第二来源周围内部纹理、树木及起收笔不监督。两组相同初始化和回放，各固定80步，只改变是否增加近邻纸面损失，训练前冻结方案，条件角色概率保持原值。
训练参考358个墨迹核心均保留，近邻纸面误检：冻结旧模型92、核心对照73、核心加纸面22。H14已知局部墨迹保留均为123/125；纸面误检分别为53、29、7。这是局部候选进展，尚无通过独立验收的新能力。

## 还有哪些失败
旧H14更广角色参考中，内部笔迹类对应从80.6%降至79.8%；这个数字是旧参考像素对应率，不能称为真实皴法准确率。新增5个旧墨迹参考点被抑制，源图放大后有淡痕、边缘及可能留白位置，尚未确认均为真实漏线。原始标签和分数保留，不用H14改标签再训练。
具体皴法、完整分笔与笔顺仍未解决。第二作品缺少同等可靠的内部皴笔角色与具名皴法参考，不能据此宣称跨作品结构/皴笔或皴法识别通过。

## 是否通过独立验证
未进行。两件训练作品都已使用过，H14也是已知回归。没有选择测试最优检查点，不读取受保护CLPD新图训练；原图、数据和模型哈希、500/489笔基线已核查。新候选不采用，机械臂离线。

## 下一步
核查另一作品中可靠的内部笔迹角色和皴法命名依据，同时补淡墨与真实边缘参考；固定本轮候选后再做独立作品验证。不能继续在H14上迭代选模，也不提前进入树木画法训练。
''',encoding='utf8')
    p=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(p);assert not any(s['capability_state'].values())
    s.update(latest_round_report=str(report),latest_two_work_band_trial=str(ROOT/'results.json'),new_model_trained_this_round=True,production_replaced=False,robot_connected=False)
    p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
    for name in ['project_plan_20261117.json','guohua_four_priorities_execution_20261116.json']:
        p=PROJECT/'research'/name;s=read(p);s.update(latest_round_report=str(report),latest_two_work_band_trial=str(ROOT/'results.json'))
        p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(checks,ensure_ascii=False))

if __name__=='__main__':main()
