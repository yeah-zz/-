# -*- coding: utf-8 -*-
"""Archive supplied-run ownership diagnosis with full, unmasked predictions."""
import sys, json, html
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ink_run_review_20261118 import ROOT, SOURCE, LAB, PROJECT, ARMS, NAMES, COLORS, FONT, read, sha, write
from gongbi_training.ink_path_diagnostics import score_visible_runs
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size): return ImageFont.truetype(FONT, size)


def comparison(r):
    box=r['view_box'];im=Image.open(r['image']).convert('RGB');crop=im.crop(box)
    cw,ch=220,210;board=Image.new('RGB',(cw*6,ch+75),'#f6f3eb');draw=ImageDraw.Draw(board)
    views=[crop.copy(),crop.copy()];refdraw=ImageDraw.Draw(views[1])
    for run in r['runs']:
        refdraw.line([(x-box[0],y-box[1]) for x,y in run],fill='#00cce5',width=1)
    for arm in ARMS:
        pred=np.load(SOURCE/'predictions'/(r['image_id']+'_'+arm+'.npz'))['prediction']
        full=np.zeros((*pred.shape,3),np.uint8)
        for cls,color in COLORS.items():full[pred==cls]=tuple(bytes.fromhex(color[1:]))
        # Entire crop, including mistakes and blank areas, no reference mask.
        views.append(Image.blend(crop,Image.fromarray(full).crop(box),0.47))
    for i,(label,v) in enumerate(zip(['原图','助手可见段（非模型）']+NAMES,views)):
        scale=min((cw-12)/v.width,(ch-12)/v.height)
        v=v.resize((round(v.width*scale),round(v.height*scale)),Image.Resampling.NEAREST)
        board.paste(v,(i*cw+6,40));draw.text((i*cw+6,8),label,font=font(15),fill='#253f36')
        if i>=2:
            score=r['outputs'][ARMS[i-2]]
            text='待定，不计分' if score['coverage'] is None else f'参考长度对应 {score["coverage"]:.1%}'
            draw.text((i*cw+6,ch+42),text,font=font(14),fill='#253f36')
    board.save(ROOT/'views'/(r['id']+'_comparison.png'))
    return board


def main():
    result=read(ROOT/'results.json');refs=read(ROOT/'reviewed_references.json');f=read(ROOT/'reference_freeze.json')
    assert sha(ROOT/'reviewed_references.json')==f['reference_sha256']
    assert f['source_and_predictions_previously_seen'] and not f['independent_validation']
    registry=read(PROJECT/'research/source_usage_registry_20261117.json')['records']
    entry=next(r for r in registry if r['source_identity']=='clpd_filename_group:zouzhe_xiangjiaoqiuwutu');assert entry['training_allowed'] is False
    protocol=read(SOURCE/'protocol.json')
    assert sha(protocol['checkpoint'])==protocol['checkpoint_sha256'] and sha(protocol['gate'])==protocol['gate_sha256']
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json');bc={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(bc.values())
    failures=[];cards=[];thumbs=[]
    table='<table id="summary"><tr><th>参考段</th><th>归属</th><th>长度/原图px</th>'+''.join('<th>'+n+'</th>' for n in NAMES)+'</tr>'
    for r in result['records']:
        assert sha(r['image'])==r['source_sha256'];assert not r['training_allowed'] and r['cunfa']=='unknown' and r['artist_stroke_identity'] is None
        expected={'rock':1,'tree':2,'unknown':None}[r['object_type']]
        for arm in ARMS:
            pred=np.load(SOURCE/'predictions'/(r['image_id']+'_'+arm+'.npz'))['prediction']
            current=score_visible_runs(pred,r['runs'],expected);saved=r['outputs'][arm]
            assert current['coverage']==saved['coverage']
            assert current['runs']==saved['runs']
            if expected is not None:
                for run_index,run in enumerate(saved['runs']):
                    for interval in run['intervals']:
                        if not interval['correct']:
                            failures.append(dict(id=r['id'],arm=arm,visible_component=run_index,interval=interval,
                                                expert_verified=False,training_allowed=False,diagnostic_only=True))
        board=comparison(r)
        if r['id'] in ['T_A','T_C']:thumbs.append(board)
        length='不计分' if expected is None else f'{r["outputs"][ARMS[0]]["length_px"]:.1f}'
        table+='<tr><td>'+r['id']+'</td><td>'+{'rock':'岩面墨迹','tree':'树木墨迹','unknown':'轨迹待定'}[r['object_type']]+'</td><td>'+length+'</td>'
        for arm in ARMS:
            v=r['outputs'][arm]
            table+='<td>待定</td>' if expected is None else f'<td>{v["coverage"]:.1%}<br>最长错分 {v["longest_wrong_px"]:.1f}px<br>±1px: {v["one_pixel_shift_coverage_range"][0]:.1%}–{v["one_pixel_shift_coverage_range"][1]:.1%}</td>'
        table+='</tr>'
        cards.append('<section><h2>'+r['id']+' · '+html.escape(r['review_note'])+'</h2><a href="views/'+r['id']+'_source_review.png">放大源图与助手参考</a><img src="views/'+r['id']+'_comparison.png" alt="源图、助手参考及四组完整局部分区输出"></section>')
    table+='</table>'
    panel=Image.new('RGB',(1320,sum(v.height for v in thumbs)+72),'#f6f3eb');d=ImageDraw.Draw(panel)
    d.text((12,8),'连续墨迹归属诊断：参考为助手提供，彩色区域为既有模型完整输出',font=font(23),fill='#253f36')
    d.text((12,40),'棕=山石 绿=树木 灰=空景 紫=其他；不是自动提取长笔触，也不是皴法识别结果',font=font(18),fill='#734d3a')
    y=72
    for v in thumbs:panel.paste(v,(0,y));y+=v.height
    panel.save(ROOT/'comparison.png')
    # Preserve unknown context as separate evidence. Never convert it into paper negatives.
    for u in result['unknown_regions']:
        im=Image.open(SOURCE/'raw'/(u['image_id']+'.jpg')).convert('RGB').crop(u['box']);im=im.resize((im.width*5,im.height*5),Image.Resampling.NEAREST);im.save(ROOT/'views'/(u['id']+'.png'))
        cards.append('<section><h2>'+u['id']+' · '+html.escape(u['reason'])+'</h2><img src="views/'+u['id']+'.png" alt="交叠区域：分笔归属待定"></section>')
    write('review_queue.json',dict(failures=failures,excluded_gap=refs['records'][0].get('excluded_uncertain_gap'),
          scope='posthoc review queue, not training targets',training_allowed=False))
    checks=dict(reference_freeze_matches=True,source_hashes_unchanged=True,weights_unchanged=True,
        recomputed_scores_match=True,baseline_hashes=bc,known_visible_candidates=5,unknown_candidate=1,
        unknown_regions=2,uncertain_gap_excluded=True,gaps_not_bridged=True,
        adjacent_context_not_negative_labels=True,full_unmasked_object_output_displayed=True,
        expert_verified=False,independent_validation=False,training_allowed=False,new_model_trained=False,
        complete_artist_strokes_recovered=False,cunfa_recognition_added=False,adopted=False,robot_connected=False)
    write('verification.json',checks)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>连续墨迹归属复核</title><style>body{font:17px/1.7 system-ui;max-width:1420px;padding:24px;margin:auto;background:#f6f3eb;color:#253f36}.notice{background:#e9ddc6;padding:18px}img{display:block;max-width:100%;height:auto;margin:18px 0}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #b8b6ac;white-space:nowrap}.scroll{overflow:auto}section{margin-top:35px}a{color:#11694e}</style><h1>从单点推进到连续墨迹归属检查</h1><p class="notice">本轮新增诊断接口和助手可见段参考，没有训练新模型。5条可核查候选（其中1条含2个分开计分的可见片段）；1条轨迹、2处交叠区域和1个淡断口保留待定。这里不是模型自动提取的完整笔画，也没有皴法类别或画家笔序真值。</p><p>原图仍是前轮已查看的2张裁片、同一个文件名作品候选组。所有参考禁止训练；作者和完整作品归属未经外部认证。沿参考长度的对应率只能定位错误，不能作为新图准确率或完整提取通过证据。</p><img src="comparison.png" alt="两处树枝错误对照"><h2>各参考段的完整结果</h2><p>对应率按原图弧长加权，重采样点不算独立样本；±1像素只检查定位敏感性。不确定间隙不续接。紫色邻近对照仅用于观察，不是纸面负标签。</p><div class="scroll">SUMMARY</div><p>两条岩面段、树干段局部对应良好；细枝仍有成段错分。冻结256与学习融合在5条候选上对应正确，也不能推翻此前整幅新图检查未通过的结论。输入参考轨迹与对象分区正确，不等于模型恢复了完整笔触。</p>CARDS<p><a href="draft_references.json">初稿</a> · <a href="reviewed_references.json">源图复核与修订</a> · <a href="reference_freeze.json">计分前冻结</a> · <a href="results.json">全部计分</a> · <a href="review_queue.json">错分区段</a> · <a href="verification.json">完整性检查</a></p></html>'''
    (ROOT/'index.html').write_text(page.replace('SUMMARY',table).replace('CARDS',''.join(cards)),encoding='utf8')
    report=PROJECT/'reports/20261118_ink_run_review.md'
    report.write_text('''# 连续墨迹归属与错分区段诊断

## 推进哪项目标

推进G2树石区分，给G1排除树枝、树叶误入皴笔提供诊断。由核心点检查推进到可见片段弧长与连续错分区段检查；没有开展G3树木技能训练。

## 实际新增什么能力

实现可复用的连续参考计分接口：保留曲线拐点、分开计量断开的可见段、按弧长加权、记录最长错分区段及一像素位置敏感性。5项逻辑测试通过。没有新增模型识别能力。

6条助手初稿经过源图检查，其中5条可核查（岩面2、树木3），1条淡岩面轨迹保持待定；1个断口不跨越连接，2处叶团/交汇分笔待定。邻近对照不是纸面真值或对象负例。所有参考均为已看过预测作品上的事后诊断，禁止训练。

## 还有哪些失败

T_A树枝可见参考长度64.4原图像素：512配置沿参考正确归属17.4%，最长连续错分53.2像素；参考平移一像素后对应率15.8%–22.6%，说明问题不只是单个坐标舍入。T_C分枝在512配置为87.3%，最长错分4.9像素；固定平均为97.5%，最长错分约0.5像素且定位敏感。

岩面两段和树干段局部对应正确。256与学习融合在这5条候选上对应正确，不代表模型提取了这5笔：曲线由助手提供，模型只输出对象分区。尚未恢复墨带边界、画家完整单笔、角色、皴法和笔序。

## 是否通过独立验证

没有。本轮来自同一个文件名作品候选组、2张已知裁片，非专家真值、非新作品测试。没有替换权重、生产模块或500/489笔基线，没有连接机械臂。原图、模型、参考冻结、分开计分及基线哈希均核查。

## 下一步

不根据这些留出图调参或把助手曲线当作训练答案。优先在训练侧的清晰单块山石/树石对照中核查完整墨带边界、轮廓和内部笔迹角色；以本轮错例定义需要检验的细枝保护和成段错分问题。来源可靠、边界可核查后再训练，G1皴法证据与G2对象归属仍分别验收。
''',encoding='utf8')
    statepath=LAB/'guohua_hierarchy_20261112/current_milestones.json';state=read(statepath);assert not any(state['capability_state'].values())
    state.update(latest_round_report=str(report),latest_ink_run_review=str(ROOT/'results.json'),new_model_trained_this_round=False,production_replaced=False,robot_connected=False)
    statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
    for name in ['research/project_plan_20261117.json','research/guohua_four_priorities_execution_20261116.json']:
        path=PROJECT/name;v=read(path);v.update(latest_round_report=str(report),latest_ink_run_review=str(ROOT/'results.json'))
        if 'execution' in name:v['current_round']='G2 visible-run posthoc ownership diagnosis:5 assistant candidates,1 uncertain trace,2 unknown junction regions; no new model'
        path.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':main()
