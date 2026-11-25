# -*- coding: utf-8 -*-
"""Frozen model mask -> explicit source constraint -> deterministic readout.

Raw masks, source-only control, pixel-vectorization outputs and final guarded
candidates are saved separately. No new weights and no robot connection.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import copy,json,hashlib,shutil
from collections import Counter
import numpy as np,cv2,torch
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from module_stroke_core_20261104 import samples,length
from ink_mask_centerline_20261105 import read_instance,CONFIG
from local_curve_replacement_20261105 import whole_guard
from hierarchical_paint_core_20261105 import hierarchy_schedule,order_metrics
import whole_stroke_instance_20261105 as instance

ROOT=LAB/'ink_mask_centerline_20261105';CACHE=LAB/'whole_stroke_instance_20261105/comparison'
BASE=LAB/'drawing_budgets_20261105';MODES=('model_only','source_only','model_source')
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf8')
def draw(rows,color=(25,95,190)):
    image=np.full((992,992,3),255,np.uint8)
    for r in rows:
        if len(r['points'])>1:cv2.polylines(image,[np.rint(np.asarray(r['points'])*2).astype(np.int32)],False,color,2,cv2.LINE_AA)
    return Image.fromarray(image)
def valid_proposals(rows):return [r for r in rows if r.get('eligible_for_replacement') and len(r['points'])>=2]
def compact(out):return {k:v for k,v in out.items() if k not in ('fields','binary')}

def extract(rgb,raw,masks):
    outputs={mode:[] for mode in MODES};allfields=[]
    for i,(r,prob) in enumerate(zip(raw['proposals'],masks)):
        q=r['query']
        for mode in MODES:
            if r['presence']<.5 and mode!='source_only':
                out=dict(points=np.empty((0,2)),status='low_model_presence',eligible_for_replacement=False)
            else:out=read_instance(rgb,prob,r['origin'],r['side'],q['back'],q['front'],mode)
            row=dict(**compact(out),query_index=i,query=q,model_presence=r['presence'])
            outputs[mode].append(row)
            if mode=='model_source':allfields.append(out)
    return outputs,allfields

def refined_overlay(rgb,fields):
    overlay=(rgb*.65+255*.35).astype(np.uint8);mask=np.zeros(rgb.shape[:2],np.uint8)
    for out in fields:
        if 'fields' not in out or 'binary' not in out:continue
        f=out['fields'];binary=out['binary'];x,y=f['origin'].astype(int);scale=f['scale']
        h,w=binary.shape;small=cv2.resize(binary.astype(np.uint8),(w//scale,h//scale),interpolation=cv2.INTER_NEAREST)
        mask[y:y+small.shape[0],x:x+small.shape[1]]|=small
    overlay[mask>0]=np.uint8(overlay[mask>0]*.4+np.array([25,155,80])*.6)
    return Image.fromarray(overlay)

def build_scene(ident):
    folder=ROOT/ident;folder.mkdir(exist_ok=True)
    raw=read(CACHE/ident/'attention_raw_curves.json');z=np.load(CACHE/ident/'attention_ownership.npz');masks=z['masks']
    rgb=np.array(Image.open(BASE/(ident+'_700')/'source.png').convert('RGB'))
    assert len(masks)==len(raw['proposals'])
    outputs,fields=extract(rgb,raw,masks)
    for mode in MODES:
        save(folder/(mode+'.json'),dict(proposals=outputs[mode],model_weights_unchanged=True,
            model_mask_cache_sha256=sha(CACHE/ident/'attention_ownership.npz'),source_sha256=sha(BASE/(ident+'_700')/'source.png'),
            mode=mode,deterministic_mask_vectorization=True,not_neural_coordinate_head=True,reference_labels_read_at_inference=False,
            statuses=dict(Counter(r['status'] for r in outputs[mode]))))
    results=[]
    for budget in (500,700):
        key=f'{ident}_{budget}';dest=ROOT/key;dest.mkdir(exist_ok=True);parent=BASE/key/'plan.json';digest=sha(parent);old=read(parent)
        controls={};allguards={};allrows={}
        for mode in MODES:
            rows,guard=whole_guard(rgb,old['curves'],valid_proposals(outputs[mode]))
            controls[mode]=dict(readout_paths=sum(len(r['points'])>1 for r in outputs[mode]),
                nonambiguous_inside_crop=len(valid_proposals(outputs[mode])),
                whole_replacements=len(guard['selected_proposals']),candidate_count=len(rows),reasons=guard['reasons'])
            allguards[mode]=guard;allrows[mode]=rows
        rows=allrows['model_source'];route,order=hierarchy_schedule(rows,old['modules'])
        tree=cKDTree(np.concatenate([samples(np.asarray(r['points']),.3) for r in rows]))
        dd=tree.query(np.concatenate([samples(np.asarray(r['points']),.3) for r in old['curves']]))[0]
        assert dd.max()<=1.3
        candidate=copy.deepcopy(old);candidate.update(parent=str(parent),parent_sha256=digest,curves=rows,actions=route,module_order=order,
            adopted=False,ready_for_robot=False,new_weights_trained=False,mask_vectorization_after_neural_prediction=True,
            baseline_sample_max_distance_px=float(dd.max()),readout_controls=controls)
        save(dest/'guarded_candidate.json',candidate);save(dest/'replacement_diagnostics.json',allguards)
        # Active viewer remains the accepted baseline; experimental candidate is
        # kept separate pending evidence of reliable whole-image improvement.
        active=copy.deepcopy(old);active.update(parent=str(parent),parent_sha256=digest,adopted=False,ready_for_robot=False,
            new_weights_trained=False,readout_controls=controls)
        active['stats']={**old['stats'],'before':old['stats']['after'],'after':old['stats']['after'],'accepted_continuations':0}
        save(dest/'plan.json',active)
        for name in ('source.png','modules.png'):shutil.copy2(BASE/key/name,dest/name)
        panels=[(Image.fromarray(rgb),'原图'),(draw(old['curves'],(30,43,35)),f'已认可{len(old["curves"])}笔方案'),
            (refined_overlay(rgb,fields),'分割区域＋原图墨迹约束'),(draw(outputs['model_source']),'沿墨带读出的中心线候选')]
        board=Image.new('RGB',(2080,605),'#f4f1e8');d=ImageDraw.Draw(board)
        for j,(im,title) in enumerate(panels):board.paste(im.resize((496,496),Image.Resampling.LANCZOS),(j*520+10,43));d.text((j*520+10,10),title,font=FONT,fill='#263e34')
        d.text((10,551),'蓝线由预测分割区域经过原图约束和骨架读出得到；不是神经网络直接预测坐标，也不是整幅最终线稿。',font=SMALL,fill='#263e34')
        d.text((10,580),'候选来自80个旧线段短提示，包含重复或待定分支；原500／700笔方案保持不变，未上机。',font=SMALL,fill='#263e34')
        board.save(dest/'comparison.jpg',quality=96);shutil.copy2(dest/'comparison.jpg',dest/'sequence_comparison.jpg')
        assert sha(parent)==digest
        results.append(dict(id=key,label=f'{key} · 墨带中心线读出对照',plan_sha256=sha(dest/'plan.json'),controls=controls))
        print('MASK_READOUT',key,controls,flush=True)
    return results

def reference_check():
    from context_curve_data_20261105 import source_records
    instance.ROOT=LAB/'whole_stroke_instance_20261105/ownership_attention';net=instance.load(True)
    records=[];visual=[]
    for record in source_records():
        if record['id'] not in ('R03','R04'):continue
        queries=[]
        for p in record['refs']:
            q=instance.arc_points(p);lo=np.asarray(p).min(0);hi=np.asarray(p).max(0)
            queries.append(dict(center=(lo+hi)/2,side=max(28,float(max(hi-lo))*1.4),back=q[14],front=q[18]))
        pred=instance.infer_queries(net,record['rgb'],queries);method={mode:[] for mode in MODES};head=[];old=[]
        for index,(p,q,target) in enumerate(zip(pred,queries,record['refs'])):
            truth=samples(target,.3);reference_tree=cKDTree(truth)
            def metrics(points):
                if len(points)<2:return dict(coverage=0.,precision=0.)
                sampled=samples(points,.3);return dict(coverage=float((cKDTree(sampled).query(truth)[0]<=2).mean()),
                    precision=float((reference_tree.query(sampled)[0]<=2).mean()))
            head.append(metrics(p['points']));old.append(max([metrics(v)['coverage'] for v in record['paths']]+[0.]))
            cases={}
            for mode in MODES:
                out=read_instance(record['rgb'],p['mask'],p['origin'],p['side'],q['back'],q['front'],mode)
                method[mode].append(dict(**metrics(out['points']),status=out['status'],eligible=out['eligible_for_replacement']))
                cases[mode]=out
            if record['id']=='R04' and index<3:visual.append((record['rgb'],p,q,target,cases))
        for mode,rows in method.items():
            records.append(dict(id=record['id'],method=mode,count=len(rows),mean_single_curve_reference_coverage=float(np.mean([r['coverage'] for r in rows])),
                mean_curve_near_reference=float(np.mean([r['precision'] for r in rows])),
                whole_curve_80pct_coverage_fraction=float(np.mean([r['coverage']>=.8 for r in rows])),
                eligible_fraction=float(np.mean([r['eligible'] for r in rows])),
                previous_neural_head_reference_coverage=float(np.mean([r['coverage'] for r in head])),
                old_best_single_path_reference_coverage=float(np.mean(old)),
                reference_short_prompt_and_bbox_used=True,not_autonomous_full_image_accuracy=True))
    save(ROOT/'reference_check.json',dict(records=records,posthoc_reused_reference_diagnostic=True,not_independent_new_artworks=True))
    board=Image.new('RGB',(2080,1235),'#f4f1e8');d=ImageDraw.Draw(board)
    for i,(rgb,p,q,target,cases) in enumerate(visual):
        im,origin=instance.crop_query(rgb,q['center'],q['side'],q['back'],q['front']);source=im[:3].transpose(1,2,0)
        probability=np.uint8(np.repeat(p['mask'][:,:,None],3,axis=2)*255)
        out=cases['model_source'];mask=out.get('binary',np.zeros((96,96),np.uint8));maskim=np.repeat(np.uint8(mask[:,:,None])*255,3,axis=2)
        overlay=source.copy();points=out['points']
        if len(points)>1:
            z=(points-origin)/q['side']*96-.5;cv2.polylines(overlay,[np.rint(z).astype(np.int32)],False,(30,90,220),1,cv2.LINE_AA)
        for j,(picture,title) in enumerate([(source,f'固定R04局部 {i+1} · 原图'),(probability,'模型原始归属概率'),(maskim,'原图约束后的墨带'),(overlay,'蓝色：墨带内读出的中心线')]):
            d.text((j*520+10,i*400+10),title,font=FONT,fill='#263e34')
            board.paste(Image.fromarray(picture).resize((390,340),Image.Resampling.NEAREST),(j*520+10,i*400+45))
    d.text((10,1205),'固定前三条参考，未挑成功案例；局部使用参考短提示和参考框，不代表自动整幅提取。',font=SMALL,fill='#263e34')
    board.save(ROOT/'reference_readout.jpg',quality=96)
    return records

def main():
    ROOT.mkdir(exist_ok=True);cv2.setNumThreads(2);torch.set_num_threads(4)
    save(ROOT/'protocol.json',dict(configuration=CONFIG,new_weights_trained=False,
        fixed_model_sha256=sha(LAB/'whole_stroke_instance_20261105/ownership_attention/instance_curve.pt'),
        modes=list(MODES),no_component_gap_bridging=True,ambiguous_branches_not_replaced=True,
        source_only_control_uses_same_queries=True,source_control_does_not_use_model_in_path_scoring=True,
        neural_masks_and_deterministic_readout_separate=True,
        parameter_changes_after_inspecting_this_experiment=True,posthoc_fix='restore dark cores and small dark holes; tolerate 1.25px ownership localization while still requiring measured source ink',
        real_mask_pixel_ground_truth_available=False))
    reference=reference_check();scenes=[]
    for ident in ('G03_01','G05_03','T02'):scenes+=build_scene(ident)
    save(ROOT/'catalog.json',dict(scenes=scenes,default='G03_01_500',robot_connected=False))
    save(ROOT/'execution_binding.json',dict(scene=None,journal=None,plan_sha256=None))
    model=[r for r in reference if r['method']=='model_source'];control=[r for r in reference if r['method']=='source_only']
    gain=all(m['mean_single_curve_reference_coverage']>c['mean_single_curve_reference_coverage'] for m,c in zip(model,control))
    save(ROOT/'decision.json',dict(adopted=False,model_better_than_source_control_on_both_fixed_records=gain,
        real_autonomous_full_image_gain_established=False,new_model_weights_trained=False,
        baseline_500_700_unchanged=True,reason='exploratory mask vectorization only; no full-image expert validation'))
    page=Path(__file__).with_name('hierarchical_paint_platform_20261105.html').read_text('utf8')
    page=page.replace("o.textContent=({G03_01:","o.textContent=s.label||({G03_01:").replace('plan=p;scene=id;sourceImage',"plan=p;scene=id;$('scene').value=id;sourceImage")
    page=page.replace('山河绘 · 笔序工作台','山河绘 · 墨带中心线对照').replace('输入细段 → 连续候选','固定基线（实验未采用）').replace('原图支持的断段连接','自动采用的实验替换')
    page=page.replace('<footer>','<footer><p><a href="/report.html">查看墨迹分割、中心线读出与源图对照</a></p>')
    (ROOT/'index.html').write_text(page,encoding='utf8')
    shutil.copy2(ROOT/'reference_readout.jpg',ROOT/'G03_01_500/rock.jpg')
    report='''<!doctype html><meta charset="utf-8"><title>墨迹分割与中心线读出</title><style>body{font:17px/1.65 system-ui;max-width:1800px;margin:auto;padding:25px;background:#f4f1e8;color:#263e34}img{width:100%}td,th{padding:9px;border:1px solid #b8c6b5}table{border-collapse:collapse}.note{background:#e4ebdc;padding:15px}a{color:#346943}</style><h1>先定位墨带，再恢复中心线</h1><a href="/">返回固定基线回放</a>'''
    report+='<p class="note">本轮没有重新训练权重。保留上一轮模型的原始归属概率，把原图的局部对比、共同颜色变暗和横截面暗脊作为墨带约束，再在已连通区域内读出经过提示附近的完整骨架路径。蓝线是确定性向量化结果，不是神经网络直接预测坐标。不同连通区域不跨亮缝连接，分支不确定时标为待定。</p>'
    report+='<h2>固定R04前三条局部</h2><img src="G03_01_500/rock.jpg"><p>固定参考提示及参考框是诊断条件，并非自动整幅提取。模型原始概率、原图约束后的墨带和读出的中心线分别展示；不是专家逐像素真值。</p>'
    table=''
    for r in reference:table+=f'<tr><td>{r["id"]}</td><td>{r["method"]}</td><td>{r["mean_single_curve_reference_coverage"]:.1%}</td><td>{r["mean_curve_near_reference"]:.1%}</td><td>{r["eligible_fraction"]:.1%}</td></tr>'
    report+='<h2>同提示／同框的局部读出对照</h2><table><tr><th>参考</th><th>方法</th><th>参考覆盖／原生2px</th><th>输出靠近参考／2px</th><th>无歧义且未触框比例</th></tr>'+table+'</table><p>model_only：仅模型概率骨架；source_only：同一局部的源图暗脊，不使用模型概率；model_source：模型区域与源图证据相结合。不能把提升全部归因于学习，也不能把参考覆盖当成整幅准确率。资料和参考已在过去实验使用，属于事后开发诊断。</p>'
    report+='<h2>整幅输入上的中心线候选</h2><p>以下提示来自旧700笔候选的短片段及64／128像素固定窗口，没有读取参考标注。右栏只是这些查询的读出，不是完整线稿。当前500／700笔正式对照保持不变。</p>'
    for ident in ('G03_01','G05_03','T02'):report+=f'<h3>{ident}</h3><img src="{ident}_500/comparison.jpg">'
    report+='<p>所有整条替换候选另存guarded_candidate.json，未自动采用。局部中心线贴合原图不能证明整笔归属正确，墨带漏检或歧义仍可能导致漏线和分笔。未连接机械臂，未标定真实毛笔笔宽、墨量、下压量或实画用时。</p>'
    (ROOT/'report.html').write_text(report,encoding='utf8')
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ('ink_mask_centerline_20261105.py','build_ink_mask_centerline_20261105.py'):shutil.copy2(Path(__file__).with_name(name),snap/name)
    print('INK_READOUT_REPORT',ROOT/'report.html',flush=True)
if __name__=='__main__':main()
