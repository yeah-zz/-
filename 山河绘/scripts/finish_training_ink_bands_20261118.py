# -*- coding: utf-8 -*-
"""Inspect two existing frozen outputs on new training-side partial band references."""
import sys,json,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from training_ink_bands_20261118 import ROOT,OLD,LAB,FONT,read,sha,write
from gongbi_training.partial_ink_bands import rasterize_band,merge_bands
import numpy as np
from PIL import Image,ImageDraw,ImageFont
PROJECT=Path(__file__).resolve().parents[1]
PREV=LAB/'cunfa_band_checked_20261112'
ARMS=['v2_frozen','frozen_role_refiner'];NAMES=['旧角色模型','既有范围修正模型']


def font(n):return ImageFont.truetype(FONT,n)


def score(ink,roles,p):
    core=ink==1;paper=ink==0;known=roles!=255
    return dict(ink_pixels=int(core.sum()),ink_retained=int((core&(p!=0)).sum()),
        paper_pixels=int(paper.sum()),paper_false_ink=int((paper&(p!=0)).sum()),
        role_pixels=int(known.sum()),role_corresponding=int((known&(roles==p)).sum()),
        ink_recall=float((p[core]!=0).mean()),paper_false_ink_rate=float((p[paper]!=0).mean()),
        scope='known training-side assistant partial reference only; not new test accuracy')


def overlay(image,p):
    a=image.copy()
    for k,c in [(1,[191,123,40]),(2,[35,128,180])]:
        take=p==k;a[take]=(.35*a[take]+.65*np.array(c)).astype('uint8')
    return a


def panel(images,labels,foot,cw=310,ch=340):
    board=Image.new('RGB',(cw*len(images),ch+70),'#f6f3eb');d=ImageDraw.Draw(board)
    for i,(im,label) in enumerate(zip(images,labels)):
        im=im.copy();scale=min((cw-12)/im.width,(ch-40)/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST)
        board.paste(im,(i*cw+6,40));d.text((i*cw+6,8),label,font=font(18),fill='#293e36')
    d.text((8,ch+18),foot,font=font(18),fill='#7b563c');return board


def main():
    assert not (ROOT/'results.json').exists(),'Keep diagnostic immutable; separate rendering from scoring changes'
    m=read(ROOT/'dataset_manifest.json');refs=read(ROOT/'reviewed_references.json');f=read(ROOT/'reference_freeze.json')
    assert sha(ROOT/'reviewed_references.json')==f['reference_sha256']==m['reference_sha256']
    assert sha(m['data'])==f['data_sha256']==m['data_sha256']
    assert refs['source']['split']=='train' and sha(refs['source']['path'])==refs['source']['sha256']
    assert sha(OLD/'references/L_149_3_input.png')==refs['source_image_sha256']
    assert sha(OLD/'references/L_149_3_before.png')==refs['before_image_sha256']
    freeze=read(PREV/'checkpoint_freeze.json');assert sha(OLD/'flat_roles.pt')==freeze['teacher_sha256'];assert sha(PREV/'foreground_refiner.pt')==freeze['refiner_sha256']
    z=dict(np.load(m['data']));shape=z['image'].shape[:2]
    predictions={a:np.load(PREV/f'predictions/L_149_3_{a}.npz')['prediction'] for a in ARMS}
    assert all(v.shape==shape for v in predictions.values())
    bands=[rasterize_band(shape,r['sections'],r['role'],r['uncertain_rows']) for r in refs['records']]
    recomputed=merge_bands(bands)
    for k in recomputed:assert np.array_equal(z[k],recomputed[k])
    full_overlay=z['image'].copy();full_overlay[z['fringe']]=[218,188,126];full_overlay[z['ink_targets']==0]=[106,172,138]
    full_overlay[z['role_targets']==1]=[192,122,40];full_overlay[z['role_targets']==2]=[35,128,180]
    full=panel([Image.fromarray(z['before']),Image.fromarray(z['image']),Image.fromarray(full_overlay)],
        ['轮廓阶段前帧（已对齐）','当前静态画面','助手部分监督参考'],
        '橙=结构 蓝=后加内部笔迹 绿=纸面 黄=边缘未知；不是模型输出或真实分笔',cw=390,ch=630)
    full.save(ROOT/'reference_overview.png')
    rows=[];cards=[];summary='<table><tr><th>参考</th><th>证据角色</th>'+''.join('<th>'+n+'</th>' for n in NAMES)+'</tr>'
    for r,b in zip(refs['records'],bands):
        scores={a:score(b['ink_targets'],b['role_targets'],p) for a,p in predictions.items()};rows.append(dict(id=r['id'],role_detail=r['role_detail'],scores=scores))
        box=r['view_box'];images=[Image.fromarray(z['image']).crop(box),Image.fromarray(full_overlay).crop(box)]
        images += [Image.fromarray(overlay(z['image'],predictions[a])).crop(box) for a in ARMS]
        p=panel(images,['原图','助手参考（非模型）']+NAMES,r['id']+'：模型完整局部输出，无参考遮罩修饰；只检查已知训练作品')
        p.save(ROOT/'views'/(r['id']+'_comparison.png'))
        summary+='<tr><td>'+r['id']+'</td><td>'+('结构边' if r['role']==1 else '后加内部笔迹')+'</td>'
        for a in ARMS:
            v=scores[a];summary+=f'<td>墨迹保留 {v["ink_retained"]}/{v["ink_pixels"]}<br>近邻纸面误检 {v["paper_false_ink"]}/{v["paper_pixels"]}</td>'
        summary+='</tr>'
        cards.append('<section><h2>'+r['id']+'</h2><p>'+html.escape(r['review_note'])+'</p><a href="views/'+r['id']+'_draft.png">前后阶段与原生像素局部</a><img src="views/'+r['id']+'_comparison.png" alt="原图、助手监督参考和两个既有模型的完整局部输出"></section>')
    summary+='</table>'
    aggregates={a:score(z['ink_targets'],z['role_targets'],p) for a,p in predictions.items()}
    write('results.json',dict(aggregates=aggregates,records=rows,new_model_trained=False,
        expert_verified=False,independent_validation=False,scope='newly reviewed supervision on an existing training work; cannot select a model or claim generalization',adopted=False,robot_connected=False))
    registry=read(PROJECT/'research/source_usage_registry_20261117.json')['records']
    protected=[v for v in registry if v.get('training_allowed') is False]
    assert all(v['source_identity'].split(':')[-1] not in refs['source']['source_family'] for v in protected)
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json');bc={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(bc.values())
    checks=dict(source_is_preexisting_training_work=True,unique_training_works=1,unique_current_images=1,paired_context_frames=1,
        bands=5,reference_hash_matches=True,masks_recomputed=True,source_pixels_unchanged=True,
        model_hashes_unchanged=True,no_pixel_threshold_targets=True,unknown_fringe_not_negative=True,
        role_and_ink_targets_separate=True,before_frame_not_model_input=True,reference_not_model_input=True,
        protected_test_works_not_used=True,baseline_hashes=bc,logic_tests_passed=5,
        assistant_reference=True,expert_verified=False,complete_ink_masks=False,complete_strokes=False,
        named_cunfa_labels=False,new_model_trained=False,independent_validation=False,production_replaced=False,robot_connected=False)
    write('verification.json',checks)
    limitation='本轮没有新训练：5段来自1件训练作品、1张当前帧和1张已对齐阶段前帧。它们是助手复核的部分墨带区间，不是专家完整墨带真值、真实分笔或皴法标签。'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>训练侧墨带与角色参考</title><style>body{font:17px/1.7 system-ui;max-width:1400px;margin:auto;padding:24px;background:#f6f3eb;color:#293e36}.notice{padding:18px;background:#e8ddc6}img{max-width:100%;height:auto;display:block;margin:18px 0}table{border-collapse:collapse}th,td{padding:12px;border:1px solid #bbb;white-space:nowrap}.scroll{overflow:auto}section{margin-top:34px}a{color:#19674c}</style><h1>训练侧：从细线核心到变宽墨带与角色分离</h1><p class="notice">LIMITATION</p><p>新增281个确定墨迹像素（结构199、后加内部笔迹82）、1376个近邻纸面像素；813个边缘过渡像素保留未知。所有区域之外仍未知，两处低对比断续不跨越填补。像素数不等于笔画数，未知边缘不计作漏线或纸面。</p><img src="reference_overview.png" alt="阶段前、静态当前图、助手墨带参考"><h2>既有模型在新增训练参考上的诊断</h2><p>下面两列都是已存在的冻结模型，没有新模型。输入仅当前静态图；曲线、阶段前帧和参考遮罩未用于修饰模型输出。范围修正是否减少纸面误检，需要同时查看真实墨迹是否被删掉。此表只能揭示已知案例问题，不能据此选模型或声称泛化。</p><div class="scroll">SUMMARY</div>CARDS<p><a href="draft_references.json">原始人工截面</a> · <a href="reviewed_references.json">源图复核</a> · <a href="dataset_manifest.json">可训练数据清单</a> · <a href="reference_freeze.json">计分前冻结</a> · <a href="results.json">既有模型诊断</a> · <a href="verification.json">完整性检查</a></p><p>下一步先补另一件清晰训练作品的同类墨带和角色参考，并核查可靠皴法命名依据，再做固定方案训练对照。前两项目标尚未通过，不启动树木画法训练或实体绘画。</p></html>'''
    (ROOT/'index.html').write_text(page.replace('LIMITATION',limitation).replace('SUMMARY',summary).replace('CARDS',''.join(cards)),encoding='utf8')
    a=aggregates[ARMS[0]];b=aggregates[ARMS[1]]
    report=PROJECT/'reports/20261118_training_ink_bands.md'
    report.write_text(f'''# 训练侧墨带边界与角色参考

## 推进哪项目标

推进G1所需的结构边/内部笔迹与墨带监督，为G2对象归属之后的笔迹分析准备接口。没有新增皴法分类、树石识别或树木技能能力。

## 实际新增什么能力

建立变量宽度嵌套截面监督接口：分别记录确定墨迹、可能墨带边缘和邻近纸面，并单独保存角色标签。未知边缘、端帽和两处断续不充当负例。5项逻辑测试通过。

复核5段训练参考（2段外轮廓、1段结构边、2段后加内部笔迹），来自既有1件作品、1张当前帧与1张已对齐阶段前帧。确定墨迹281像素、纸面1376像素，边缘813像素未知；不是5件作品或5条已恢复完整笔画。与旧标签已知重叠处冲突为0。

## 还有哪些失败

新增参考仍为助手部分区间，不是专家完整墨带真值。小石侧边属于独立对象边缘还是山面转折未确认；内部笔迹虽有阶段证据，但不能直接作为具名皴法训练答案。真实笔序、起收笔和物理参数未知。

既有冻结模型在这批新增训练侧参考上：旧角色模型墨迹保留{a['ink_retained']}/281，近邻纸面误检{a['paper_false_ink']}/1376；既有范围修正模型分别为{b['ink_retained']}/281、{b['paper_false_ink']}/1376。两者都仍有{a['role_pixels']-a['role_corresponding']}个确定墨迹像素的角色未对应，范围收窄没有修复角色判断。这是已知训练作品上的诊断，不是模型改进的独立证据。

## 是否通过独立验证

未进行。数据格式、冻结哈希、掩膜重算、5项逻辑检查以及500/489笔基线完整性通过；不代表新识别能力验收。没有新训练或模型替换，留出图没有回灌训练，机械臂离线。

## 下一步

补另一件清晰训练作品的墨带与角色参考，并单独核查具体皴法命名及证据，再启动固定方案训练对照。避免用这一个教程的局部拟合替代跨作品能力。G1/G2通过前不启动G3树木技能训练。
''',encoding='utf8')
    statepath=LAB/'guohua_hierarchy_20261112/current_milestones.json';state=read(statepath);assert not any(state['capability_state'].values())
    state.update(latest_round_report=str(report),latest_training_band_reference=str(ROOT/'dataset_manifest.json'),
        latest_training_band_diagnostic=str(ROOT/'results.json'),new_model_trained_this_round=False,production_replaced=False,robot_connected=False)
    statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
    for name in ['research/project_plan_20261117.json','research/guohua_four_priorities_execution_20261116.json']:
        p=PROJECT/name;v=read(p);v.update(latest_round_report=str(report),latest_training_band_reference=str(ROOT/'dataset_manifest.json'))
        if 'execution' in name:v['current_round']='G1 partial ink/role supervision:5 reviewed variable-band envelopes from1 preexisting training work; no new model or independent acceptance'
        p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(aggregates=aggregates,checks=checks),ensure_ascii=False))


if __name__=='__main__':main()
