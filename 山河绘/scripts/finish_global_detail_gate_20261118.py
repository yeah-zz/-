# -*- coding: utf-8 -*-
import sys,json,hashlib,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.guohua_hierarchy import CapabilityState,allowed_phases
ROOT=LAB/'global_detail_gate_20261118';PROJECT=Path(__file__).resolve().parents[1]
ARMS=['global256','detail512','fixed_mean','learned_gate'];NAMES=['冻结256','冻结512','固定平均','学习融合'];COLORS=np.array([[241,237,222],[196,140,62],[70,143,105],[110,112,163]],'uint8')
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    result=read(ROOT/'results.json');p=result['protocol'];train=read(ROOT/'train_manifest.json')
    assert sha(p['frozen_checkpoint'])==p['frozen_checkpoint_sha256']
    assert sha(ROOT/'gate.pt')==read(ROOT/'checkpoint_freeze.json')['gate_sha256']
    assert sha(PROJECT/'gongbi_training/global_detail_gate.py')==p['module_sha256']
    assert not ({r['work_group'] for r in train['records']}&set(train['excluded_known_work_groups']))
    for r in train['records']:
        if 'image_sha256' in r:assert sha(r['image'])==r['image_sha256']
        if 'target_sha256' in r:assert sha(r['target'])==r['target_sha256']
    for arm in ARMS:
        c=sum(np.array(r['confusion'][arm]) for r in result['records'] if arm in r['confusion']);u=c.sum(0)+c.sum(1)-c.diagonal()
        assert abs(np.mean([c[k,k]/u[k] for k in range(4) if u[k]])-result['known14'][arm]['mean_iou'])<1e-12
    (ROOT/'views').mkdir(exist_ok=True);catalog=[]
    for r in result['records']:
        ident=r['id'];im=Image.open(r['image']).convert('RGB');im.save(ROOT/f'views/{ident}_source.png');predictions={}
        for arm in ARMS:
            pred=np.load(ROOT/f'predictions/{ident}_{arm}.npz')['prediction'];assert pred.shape==(im.height,im.width) and pred.max()<4
            path=f'views/{ident}_{arm}.png';Image.fromarray(COLORS[pred]).save(ROOT/path);predictions[arm]=path
        catalog.append(dict(id=ident,source=f'views/{ident}_source.png',predictions=predictions,group='已知案例，非独立测试'))
    write(ROOT/'catalog.json',dict(records=catalog,models=dict(zip(ARMS,NAMES)),reference_used_at_inference=False))
    board=Image.new('RGB',(1800,1140),'#f5f2e9');d=ImageDraw.Draw(board);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    for row,ident in enumerate(['H_14','qiu-ying_fairyland-of-peach-blossoms_8','wang-shimin_clearing-of-rivers-and-mountains-after-snow_6']):
        c=next(c for c in catalog if c['id']==ident)
        for col,(path,title) in enumerate([(c['source'],'原图')]+[(c['predictions'][a],n) for a,n in zip(ARMS,NAMES)]):
            im=Image.open(ROOT/path);s=min(345/im.width,300/im.height);im=im.resize((round(im.width*s),round(im.height*s)),Image.Resampling.NEAREST)
            board.paste(im,(col*360+6,row*380+38));d.text((col*360+6,row*380+5),title,font=font(20),fill='#243d35')
        d.text((8,row*380+350),ident,font=font(17),fill='#78523c')
    board.save(ROOT/'comparison.jpg',quality=94)
    table='<table id="summary"><thead><tr><th>方法</th><th>已知14图四类平均IoU</th></tr></thead><tbody>'+''.join(f'<tr><td>{n}</td><td>{result["known14"][a]["mean_iou"]*100:.1f}%</td></tr>' for a,n in zip(ARMS,NAMES))+'</tbody></table>'
    regional='<table><tr><th>既有核心</th>'+''.join('<th>'+n+'</th>' for n in NAMES)+'</tr>'+''.join('<tr><td>'+html.escape(r['id']+'/'+r['region'])+'</td>'+''.join(f'<td>{r["correct_fraction"][a]*100:.1f}%</td>' for a in ARMS)+'</tr>' for r in result['regional'])+'</table>'
    htmltext='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>全局与细节学习融合</title><style>body{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px;background:#f5f2e9;color:#243d35}h1{font-size:30px}.notice{background:#e8dfc8;padding:16px}.scroll{overflow-x:auto}table{border-collapse:collapse;white-space:nowrap}td,th{padding:9px;border:1px solid #bbb}img{max-width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr));gap:14px}article{padding:12px;background:white}select{font:inherit;max-width:100%}a{color:#265d76}</style><h1>全局与细节学习融合：实际训练与固定对照</h1><p class="notice">新增融合模块已训练，未采用；原对象模型、500笔及489笔基线保持。以下是完整原图的对象分区输出，不是皴法或笔迹重绘。全部17图为已知资料，未通过独立验证。</p><p>只学习一张全局／细节混合权重图，同一权重用于全部类别。输入为原图及冻结模型的两种概率图，无参考框、角色标签或人工指定的融合区域。棕：山石；绿：树木；米白：空景；紫：其他。</p><div class="scroll">SUMMARY</div><p>学习融合未超过固定平均融合；局部找回不能代表整体通过。所有对照及退步完整保留。</p><div class="scroll">REGIONAL</div><img src="comparison.jpg" alt="原图及四组完整对照"><h2>查看全部已知案例</h2><select id="case"></select><p id="scope"></p><div id="gallery" class="grid"></div><p>融合模块使用60张既有训练图和局部场景参考，固定128步。诊断作品组从本轮融合训练中排除，但这些图已被项目多次查看，且原模型可能见过部分作品，所以不能称新作品独立验证。对象区域和粗标签仍不是精确墨迹或专家边界。</p><p><a href="protocol.json">冻结协议</a> · <a href="results.json">完整结果</a> · <a href="verification.json">完整性核查</a> · <a href="decision.json">采用决定</a></p><script>fetch('catalog.json').then(r=>r.json()).then(data=>{const select=document.querySelector('#case');for(const r of data.records){const o=document.createElement('option');o.value=r.id;o.textContent=r.id;select.append(o)}function render(){const r=data.records.find(r=>r.id===select.value);document.querySelector('#scope').textContent=r.group;const grid=document.querySelector('#gallery');grid.innerHTML='';for(const [n,path] of [['原图',r.source],...Object.entries(data.models).map(([k,n])=>[n,r.predictions[k]])]){const a=document.createElement('article'),t=document.createElement('strong'),i=document.createElement('img');t.textContent=n;i.src=path;i.alt=n;a.append(t,i);grid.append(a)}}select.onchange=render;select.value='H_14';render()})</script></html>'''
    (ROOT/'index.html').write_text(htmltext.replace('SUMMARY',table).replace('REGIONAL',regional),encoding='utf8')
    regressions=[dict(id=r['id'],region=r['region'],baseline=r['correct_fraction']['global256'],learned=r['correct_fraction']['learned_gate']) for r in result['regional'] if r['correct_fraction']['learned_gate']<r['correct_fraction']['global256']-.01]
    write(ROOT/'decision.json',dict(adopted=False,reason='Learning did not exceed fixed mean, and Wang rock core still regressed',regressions=regressions,independent_validation=False,robot_connected=False))
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json');bc={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(bc.values())
    write(ROOT/'verification.json',dict(frozen_model_hash_unchanged=True,gate_checkpoint_frozen=True,train_eval_work_groups_disjoint_for_gate=True,
        original_training_image_hashes_checked=True,all_four_arms_reported=True,known14_confusion_recomputed=True,
        original256_and512_predictions_match_previous_run=True,all17_cases_rendered=True,baseline_hashes=bc,
        independent_validation=False,adopted=False,robot_connected=False))
    report=PROJECT/'reports/20261118_global_detail_gate.md'
    report.write_text('''# 全局对象与局部细节学习融合

## 1. 推进哪项目标

推进G2山石与树木区分，支持G1避免树叶误入皴笔。尝试兼顾256全局山体归属和512局部树冠细节；没有训练皴法、树木画法或Agent，机械臂离线。

## 2. 实际新增什么能力

实现并训练一个小型全局／细节融合模块。输入完整原图和冻结对象模型的两种概率图，学习共享于四类的逐位置混合权重；不是由参考框或人工挑选局部结果。原模型权重不变。已有训练图与部分场景参考共60张，固定128步；本轮没有新增独立数据或专家边界。

同一批已知14图四类平均IoU：冻结256为53.5%，冻结512为60.1%，固定平均为60.0%，学习融合为56.4%。学习模块已接通，但尚无通过独立验收的新识别能力，未证明比简单平均更有效。

## 3. 还有哪些失败

学习融合在H_14保住两处淡山体核心100%，前景岩面为98.9%，树叶核心100%；仇英树冠由79.6%至94.1%。但王时敏岩面核心仍从89.9%降至85.7%，整体又低于固定平均。因此没有采用，不能只展示改善位置。

所有核心是既有助手区域，不是专家精确边界。此模块只组合已有对象判断，不能修复两种尺寸都判断错误的位置，也未恢复完整笔触或皴法技能。本轮不追加针对这17图的调参。

## 4. 是否通过独立验证

未进行新的独立验证。17图已被多次查看，原模型可能见过部分作品。本轮按作品组排除这些诊断图对应的组，不用三组核心多边形训练融合模块；这只能说明本轮训练／检查隔离，不能把旧案例重新变成盲测。

已核查训练／诊断作品组隔离、训练图像哈希、权重冻结、四组对照、原256/512预测与上一轮逐像素相同、同口径指标重算，以及500/489笔基线完整性。融合初始状态为固定平均，未知像素没有直接分类监督；空间平滑项作为正则单列。

## 采用决定及下一步

新融合模块不接入绘画版本。认可旧版和默认输入保持，核心能力验收不解锁，树木技能仍等待G1/G2通过。

本轮结论是：这种小型融合模块在当前训练监督上未带来明确收益。下一步应回到可核查的对象边界、结构与皴笔参考，并使用真正未参与选择的新作品验证；不继续只在这些熟悉局部调融合系数。

实验目录：`E:\\Codex_Workspace_Data\\Gongbi_Line_Model_20261022\\global_detail_gate_20261118`。
''',encoding='utf8')
    statepath=LAB/'guohua_hierarchy_20261112/current_milestones.json';state=read(statepath);caps=state['capability_state'].copy();assert not any(caps.values()) and not allowed_phases(CapabilityState(**caps))['tree_skill_training']
    state.update(latest_round_report=str(report),latest_global_detail_gate=str(ROOT/'results.json'),new_model_trained_this_round=True,
        latest_global_detail_gate_adopted=False,latest_global_detail_gate_independent_validation=False,
        production_replaced=False,robot_connected=False);assert state['capability_state']==caps;write(statepath,state)
    for name in ['research/project_plan_20261117.json','research/guohua_four_priorities_execution_20261116.json']:
        path=PROJECT/name;v=read(path);v.update(latest_round_report=str(report),latest_global_detail_gate=str(ROOT/'results.json'));write(path,v)
    # Reuse the prior browser harness with this report's actual DOM counts.
    text=(PROJECT/'scripts/check_regional_presence_view_20261117.py').read_text('utf8')
    for a,b in [("ROOT=LAB/'regional_presence_trial_20261117'","ROOT=LAB/'global_detail_gate_20261118'"),('length===43','length===17'),('document.images.length===7','document.images.length===6'),('document.querySelectorAll("tbody tr").length','document.querySelectorAll("#summary tbody tr").length'),("')==5","')==4"),("')==6","')==5"),("['F1500','H_14','wen-zhengming_peach-blossom-spring_10']","['H_14','qiu-ying_fairyland-of-peach-blossoms_8','wang-shimin_clearing-of-rivers-and-mountains-after-snow_6']"),('five_model_rows=True','four_model_rows=True'),('all43_case_options=True','all17_case_options=True'),('six_panel_raw_output_switches=True','five_panel_raw_output_switches=True')]:text=text.replace(a,b)
    (PROJECT/'scripts/check_global_detail_gate_view_20261118.py').write_text(text,encoding='utf8')
    print(json.dumps(dict(report=str(report),regressions=regressions,baseline_hashes=bc),ensure_ascii=False))
if __name__=='__main__':main()
