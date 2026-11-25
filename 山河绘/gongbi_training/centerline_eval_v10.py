# -*- coding: utf-8 -*-
"""Controlled before/after evaluation with the same ink input and planner."""
import json,math,hashlib,shutil
from pathlib import Path
import torch,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from .centerline_only_v10 import model_new,infer,input_from_ink,geometry_target
from .stroke_train_v08 import model_new as old_model,infer as old_infer
from .stroke_streamline_v08 import plan
from .model import GongbiLineNet
from .rock_multitask_v09 import synthetic
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
def write(root,name,obj):(root/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def load(path,kind):
    m=kind().cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval();return m
def pack_geometry(p,old):
    q=old.copy();q[:3]=p[:3]
    assert np.array_equal(q[3:],old[3:])
    return q
def line_image(info):
    h,w=info['image_height'],info['image_width'];out=np.full((h,w,3),255,np.uint8)
    for r in info['strokes']:cv2.polylines(out,[np.rint(r['xy_pixels']).astype(np.int32)],False,(0,0,0),1,cv2.LINE_AA)
    return out
def metrics(info,r,probability):
    p=[np.asarray(s['xy_pixels'],float) for s in info['strokes']];points=np.concatenate(p) if p else np.zeros((0,2));ids=np.concatenate([np.full(len(a),i) for i,a in enumerate(p)]) if p else np.zeros(0,int);trees=[cKDTree(a) for a in p];tree=cKDTree(points) if p else None;total=covered=0;best=[];frags=[]
    for stroke in r['dense_paths']:
        q=np.asarray(stroke['points']);total+=len(q)
        if tree is None:best.append(0);frags.append(0);continue
        distance,index=tree.query(q);good=distance<=4;covered+=int(good.sum());best.append(max(float((t.query(q)[0]<=4).mean()) for t in trees));counts=np.bincount(ids[index[good]],minlength=len(p));frags.append(int((counts>=max(3,len(q)*.04)).sum()))
    z=np.load(r['labels']);target=z['target'][0];confidence=z['confidence'][0];negative=(confidence>=.25)&(target<.05);positive=(confidence>.1)&(target>.5)
    return dict(reference_point_coverage=covered/max(1,total),best_single_path_coverage=float(np.mean(best)),fragments_per_reference=float(np.mean(frags)),heatmap_recall_on_marked_positive=float((probability[positive]>.35).mean()),heatmap_false_positive_on_marked_paper=float((probability[negative]>.35).mean()) if negative.any() else None,planned_paths=len(info['strokes']),reference_status='partial assistant initial traces, not expert gold; unmarked dark areas unscored')
def probability_metric(r,probability):
    z=np.load(r['labels']);t=z['target'][0];c=z['confidence'][0];positive=(c>.1)&(t>.5);negative=(c>=.25)&(t<.05)
    return dict(marked_positive_recall=float((probability[positive]>.35).mean()),marked_paper_false_positive=float((probability[negative]>.35).mean()) if negative.any() else None)
def paste(board,array,x,y,w,h):
    im=Image.fromarray(array) if isinstance(array,np.ndarray) else array.copy();im=im.convert('RGB');im.thumbnail((w,h));board.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))
def synthetic_test(student,initial,old,teacher):
    rows=[];seed=630000
    with torch.no_grad():
        while len(rows)<32:
            rgb,t,c,ids=synthetic(np.random.default_rng(seed));current=seed;seed+=1
            if (t[0]>.5).sum()<750:continue
            pad=cv2.copyMakeBorder(rgb,32,32,32,32,cv2.BORDER_REPLICATE);ink=teacher(torch.from_numpy(pad.transpose(2,0,1).copy())[None].cuda().float()/255).sigmoid()[0,:,32:-32,32:-32].cpu().numpy();x=torch.from_numpy(input_from_ink(ink))[None].cuda();a=student(x)[0,0].sigmoid().cpu().numpy();b=initial(x)[0,0].sigmoid().cpu().numpy();oldx=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].cuda().float()/255;o=old(oldx)[0,0].sigmoid().cpu().numpy();truth=t[0]>.35
            def score(prob):
                p=prob>.35;tp=int((p&truth).sum());return dict(precision=tp/max(1,int(p.sum())),recall=tp/max(1,int(truth.sum())),f1=2*tp/max(1,int(p.sum()+truth.sum())))
            rows.append(dict(seed=current,previous_rgb=score(o),adapted_initial=score(b),trained=score(a)))
    macro={name:{k:float(np.mean([r[name][k] for r in rows])) for k in ('precision','recall','f1')} for name in ('previous_rgb','adapted_initial','trained')}
    return dict(examples=32,interpretation='known procedural geometry only; not real painting accuracy',macro=macro,per_image=rows)

def evaluate(root,v08,v09,ink_root):
    torch.set_num_threads(4);cv2.setNumThreads(2);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));protocol=json.loads((root/'evaluation_protocol.json').read_text(encoding='utf-8'));config=json.loads((root/'config.json').read_text(encoding='utf-8'));assert config['manifest_sha256']==hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    new=load(root/'best.pt',model_new);initial=load(root/'initial.pt',model_new);old=load(v08/'best.pt',old_model);teacher=load(ink_root/'best.pt',GongbiLineNet);known=synthetic_test(new,initial,old,teacher);write(root,'evaluation/known_synthetic_metrics.json',known);del teacher;torch.cuda.empty_cache()
    tests=[r for r in manifest['records'] if r['split']=='test'];results=[];images={};plans=[]
    for r in tests:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));ink=np.load(r['ink_cache'])['ink'].astype(np.float32);before=old_infer(old,rgb);adapted=infer(initial,ink);after=infer(new,ink);prediction=pack_geometry(after,before)
        bp,bv,br,_=plan(ink,before,strict=True);ap,av,ar,_=plan(ink,prediction,strict=True);old_lines=line_image(bp);new_lines=line_image(ap);overlay=rgb.copy();line_mask=(new_lines[:,:,0]<200).astype(np.uint8);line_mask=cv2.dilate(line_mask,np.ones((2,2),np.uint8))>0;overlay[line_mask]=np.clip(overlay[line_mask]*.35+np.array([245,30,30])*.65,0,255).astype(np.uint8)
        folder=root/'evaluation'/r['id'];folder.mkdir(exist_ok=True)
        for name,a in [('original',rgb),('before_centerlines',old_lines),('after_centerlines',new_lines),('overlay',overlay),('colored_paths',av),('width_replay_unchanged_estimator',ar),('center_probability',np.uint8(np.clip(after[0]*255,0,255))),('initial_probability',np.uint8(np.clip(adapted[0]*255,0,255)))]:Image.fromarray(a).save(folder/(name+'.png'))
        np.savez_compressed(folder/'probabilities.npz',previous=before,initial=adapted,trained=after,frozen_ink=ink)
        (folder/'planned_trajectory.json').write_text(json.dumps(ap,ensure_ascii=False),encoding='utf-8');plans.append(dict(id=r['id'],**ap));svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{r["width"]}" height="{r["height"]}" viewBox="0 0 {r["width"]} {r["height"]}"><g fill="none" stroke="black" stroke-width="1">']
        for s in ap['strokes']:svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in s['xy_pixels'])+'"/>')
        svg.append('</g></svg>');(folder/'centerlines.svg').write_text('\n'.join(svg),encoding='utf-8')
        row=dict(id=r['id'],source_group=r['source_group'],new_reference=r['id'].startswith('v09_'),previous=metrics(bp,r,before[0]),trained=metrics(ap,r,after[0]),adapted_initial_probability=probability_metric(r,adapted[0]));results.append(row);images[r['id']]=[rgb,old_lines,new_lines,overlay]
        print('FIXED_TEST',r['id'],flush=True)
    write(root,'evaluation/per_image_metrics.json',results)
    def aggregate(rows):
        return {stage:{k:float(np.mean([r[stage][k] for r in rows if r[stage][k] is not None])) for k in ('reference_point_coverage','best_single_path_coverage','fragments_per_reference','heatmap_recall_on_marked_positive','heatmap_false_positive_on_marked_paper')} for stage in ('previous','trained')}
    summary=dict(all_image_macro=aggregate(results),legacy_image_macro=aggregate([r for r in results if not r['new_reference']]),new_image_macro=aggregate([r for r in results if r['new_reference']]),source_groups={g:aggregate([r for r in results if r['source_group']==g]) for g in sorted({r['source_group'] for r in results})},synthetic=known['macro']);write(root,'evaluation/summary.json',summary)
    labels=['原图','旧版中心线（相同墨迹/规划器）','本轮中心线（相同墨迹/规划器）','本轮中心线叠加原图']
    for name,ids in [('new_references',[r['id'] for r in results if r['new_reference']]),('legacy_references',[r['id'] for r in results if not r['new_reference']])]:
        board=Image.new('RGB',(1600,350*len(ids)),'#f4f4ed');draw=ImageDraw.Draw(board)
        for i,identity in enumerate(ids):
            draw.text((8,i*350+3),identity+' | 未参与本轮训练；仅部分助手初始标注',font=FONT,fill='black')
            for j,label in enumerate(labels):draw.text((j*400+7,i*350+31),label,font=FONT,fill='black');paste(board,images[identity][j],j*400+9,i*350+65,382,277)
        board.save(root/'evaluation'/(name+'.jpg'),quality=95)
    # All evaluation paths use exactly the same frozen planner, ink arrays and
    # width/density estimators. Projection offsets are an auxiliary learned task.
    html=(v09/'trajectory_viewer.html').read_text(encoding='utf-8');begin=html.index('const data=')+len('const data=');end=html.index(';let running=',begin);html=html[:begin]+json.dumps(plans,ensure_ascii=False).replace('</','<\\/')+html[end:];html=html.replace('v09','v10').replace('路线来自v10中心线、方向、墨色与宽度输出。','本轮只训练中心线和方向；墨迹、宽度与规划器固定旧版。').replace('id="wide" type="checkbox" checked','id="wide" type="checkbox"');(root/'trajectory_viewer.html').write_text(html,encoding='utf-8')
    report(root,summary);print('EVALUATION_COMPLETE',json.dumps(summary['new_image_macro']),flush=True)

def report(root,metrics):
    training=json.loads((root/'training_summary.json').read_text(encoding='utf-8'));audit=json.loads((root/'data_audit.json').read_text(encoding='utf-8'));s=metrics['synthetic'];r=metrics['new_image_macro'];legacy=metrics['legacy_image_macro']
    text=f'''# v10 中心线专项实验（2026-10-24）

## 本轮范围

固定v06墨迹模型和v08严格方向场规划器，只训练中心线、局部方向及辅助轴投影。宽度和角色候选仍取自冻结v08，浓淡不再重训，也不重训皴法分类。推理时只替换旧模型的前三个几何通道，辅助投影输出没有应用到规划器，因此没有隐含额外平滑或连线策略。

中心线模块的输入由RGB改为冻结墨迹的密度、覆盖和轻度平滑视图；这属于输入表示和训练的共同变化，不能把全部收益归因于梯度更新。保存initial.pt和初始概率图，用于检查未训练适配器及训练后的差异。

## 数据与标注状态

本轮工作集20张图：训练10张88条初始参考线、验证3张22条、对照7张61条。新增8张训练/验证来源图和4张对照图的描线均由助手逐条看图制作。全部未经人工画家确认；不确定的未标注暗线保持未知，不自动判为背景。参考线并不必然等于原作者一次落笔。

已知来源组不跨本轮划分；验证图可能被父模型见过，同书或跨站转载关系未完全排查。7张对照图中，原3张是反复查看的旧开发图；其余4张虽未参与本轮训练，但上轮已用作开发测试。因此这不是严格独立盲测，也不是专家准确率。

不再使用大批自动脊线作为正例答案。训练混合助手部分描线与200个已知几何程序样本，输入均经过同一冻结墨迹模型。程序样本不计为真实画作。所有未标注区域和交点方向歧义均通过置信权重控制。

## 实际训练

两阶段，共{training['successful_updates']}次有效更新，跳过{training['skipped_updates']}次，训练循环耗时{training['elapsed_seconds']:.1f}秒（不含程序样本及冻结输入准备）。按验证损失选第{training['selected']['phase']}阶段第{training['selected']['step']}步。旧权重和旧绘画流程未改动。

## 已知程序几何（32个未训练种子，非真实画作准确率）

| 模块 | 中心线像素F1 |
|---|---|
| 旧v08 RGB输入 | {s['previous_rgb']['f1']:.4f} |
| 适配墨迹输入但未训练 | {s['adapted_initial']['f1']:.4f} |
| 本轮训练后 | {s['trained']['f1']:.4f} |

## 4张新增参考图的部分描线对照

同一冻结墨迹、同一规划器；参考线匹配容差4像素。标注只覆盖部分线条，无法计算全图的真实精确率。纸面误检只在自动判作纸面的局部区域统计（仍可能误判），未标注暗线不计入。

| 指标 | 旧版 | 本轮 |
|---|---|---|
| 参考线点覆盖 | {r['previous']['reference_point_coverage']:.4f} | {r['trained']['reference_point_coverage']:.4f} |
| 最佳单路径覆盖 | {r['previous']['best_single_path_coverage']:.4f} | {r['trained']['best_single_path_coverage']:.4f} |
| 每条参考线对应片段 | {r['previous']['fragments_per_reference']:.2f} | {r['trained']['fragments_per_reference']:.2f} |
| 已标纸面局部概率误检率 | {r['previous']['heatmap_false_positive_on_marked_paper']:.4f} | {r['trained']['heatmap_false_positive_on_marked_paper']:.4f} |

原3张开发图参考点覆盖：{legacy['previous']['reference_point_coverage']:.4f} → {legacy['trained']['reference_point_coverage']:.4f}。完整逐图、来源组及初始适配器指标保存在evaluation/目录。

## 查看结果

- evaluation/new_references.jpg：四张未参与本轮训练的参考图，原图/旧线/新线/红色叠加，全量展示。
- evaluation/legacy_references.jpg：原三张开发图回归。
- review/annotation_pack_*.jpg：20张草标原图对照，便于复核。
- trajectory_viewer.html：离线中心线播放，默认细线显示，可切换旧估计笔宽。
- 各图子目录保存原始概率、PNG中心线、SVG和轨迹JSON，没有手工美化预测图。

没有连接、使能或移动机械臂。轨迹中物理Z、速度和纸面映射为空，ready_for_robot=false。细线预览是候选中心线，不是完整国画墨迹、真实毛笔动作或实画验证。模型是否可靠仍需要独立、经复核的更多图样评估。
'''
    (root/'REPORT.md').write_text(text,encoding='utf-8')
