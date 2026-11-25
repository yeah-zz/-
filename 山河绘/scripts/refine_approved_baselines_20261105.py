# -*- coding: utf-8 -*-
"""Conservative source joins on approved budgets; failed neural trials separate."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,copy,hashlib,shutil
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from long_stroke_recovery_20261105 import recover_long_strokes
from hierarchical_paint_core_20261105 import hierarchy_schedule,order_metrics
from module_stroke_core_20261104 import samples,length

ROOT=LAB/'approved_baseline_refinement_20261105';BASE=LAB/'drawing_budgets_20261105'
NEURAL=LAB/'local_curve_policy_20261105';FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf8')
def drawing(rows):
    im=np.full((992,992,3),255,np.uint8)
    for r in rows:cv2.polylines(im,[np.rint(np.asarray(r['points'])*2).astype(np.int32)],False,(30,43,35),2,cv2.LINE_AA)
    return im

def build(key):
    folder=ROOT/key;folder.mkdir(exist_ok=True);path=BASE/key/'plan.json';digest=sha(path);old=read(path)
    rgb=np.array(Image.open(BASE/key/'source.png').convert('RGB'));new,diag=recover_long_strokes(rgb,old['curves'])
    for r in new:
        ids=r['complete_input_ids'];phase=min(old['curves'][i]['phase'] for i in ids)
        ref=next(old['curves'][i] for i in ids if old['curves'][i]['phase']==phase)
        r.update(phase=phase,phase_label=ref['phase_label'],role_verified=False,length_px=length(r['points']),
            source_support=float(np.mean([old['curves'][i]['source_support'] for i in ids])),
            source_curve_ids=[j for i in ids for j in old['curves'][i]['source_curve_ids']])
    route,order=hierarchy_schedule(new,old['modules'])
    oldq=np.concatenate([samples(np.asarray(r['points']),.25) for r in old['curves']]);tree=cKDTree(np.concatenate([samples(r['points'],.25) for r in new]))
    dd=tree.query(oldq)[0]
    if dd.max()>.8:raise ValueError('Input geometry lost')
    stats={**old['stats'],'before':old['stats']['after'],
        'after':dict(count=len(new),short_under_5px=sum(length(r['points'])<5 for r in new)),
        'accepted_continuations':len(diag['accepted']),'new_joins':len(diag['accepted']),'join_types':diag['types'],
        'baseline_order':order_metrics(old['actions']),'hierarchy_order':order_metrics(route),
        'baseline_max_sample_distance_px':float(dd.max()),'detail_omission_this_round':0,
        'new_neural_model_applied':False,'new_neural_model_not_adopted':True,'physical_improvement_verified':False}
    plan=copy.deepcopy(old);plan.update(parent=str(path),parent_sha256=digest,curves=new,actions=route,module_order=order,stats=stats,
        ready_for_robot=False,new_neural_model_applied=False,no_additional_detail_omission=True,
        accepted_baseline_original_unchanged=True)
    for m in plan['modules']:
        m['planned_strokes']=sum(r['module']==m['id'] for r in new);m['order']=order.index(m['id'])+1
    save(folder/'plan.json',plan);save(folder/'diagnostics.json',diag)
    for name in ['source.png','modules.png']:shutil.copy2(BASE/key/name,folder/name)
    change=drawing(new)
    for r in diag['accepted']:
        p=np.asarray(r['bridge'])*2;cv2.polylines(change,[np.rint(p).astype(np.int32)],False,(20,170,70),3,cv2.LINE_AA)
        cv2.circle(change,tuple(np.rint(p[len(p)//2]).astype(int)),4,(20,170,70),1,cv2.LINE_AA)
    board=Image.new('RGB',(2080,600),'#f4f1e8');draw=ImageDraw.Draw(board)
    for i,(im,title) in enumerate([(rgb,'原图'),(drawing(old['curves']),f'已认可方案 · {len(old["curves"])}笔'),
        (drawing(new),f'原图支持续接 · {len(new)}笔'),(change,'绿色：本轮续接位置')]):
        board.paste(Image.fromarray(im).resize((496,496),Image.Resampling.LANCZOS),(i*520+10,43));draw.text((i*520+10,10),title,font=FONT,fill='#263e34')
    draw.text((10,551),'本轮没有继续省略细节；这是几何与原图支持的连接候选，新训练模型尚未采用。非实画。',font=FONT,fill='#263e34')
    board.save(folder/'comparison.jpg',quality=96);shutil.copy2(folder/'comparison.jpg',folder/'sequence_comparison.jpg')
    assert sha(path)==digest
    print('APPROVED_REFINEMENT',key,len(old['curves']),len(new),stats['join_types'],flush=True)
    return dict(id=key,label=f'{key} · 连续性候选（{len(new)}笔）',plan_sha256=sha(folder/'plan.json'),stats=stats)

def main():
    ROOT.mkdir(exist_ok=True);cv2.setNumThreads(2)
    records=[build(f'{ident}_{budget}') for ident in ('G03_01','G05_03','T02') for budget in (500,700)]
    save(ROOT/'catalog.json',dict(scenes=records,default='G03_01_500',robot_connected=False))
    save(ROOT/'execution_binding.json',dict(scene=None,journal=None,plan_sha256=None))
    page=Path(__file__).with_name('hierarchical_paint_platform_20261105.html').read_text('utf8')
    page=page.replace("o.textContent=({G03_01:","o.textContent=s.label||({G03_01:").replace('plan=p;scene=id;sourceImage',"plan=p;scene=id;$('scene').value=id;sourceImage")
    page=page.replace('山河绘 · 笔序工作台','山河绘 · 保守连续性改进').replace('输入细段 → 连续候选','已认可方案 → 续接候选')
    page=page.replace('<footer>','<footer><p><a href="/report.html">查看实际续接与两轮新模型的结果</a></p>')
    (ROOT/'index.html').write_text(page,encoding='utf8')
    first=read(NEURAL/'training_summary.json');second=read(NEURAL/'recovery_round2/training_summary.json')
    table=''.join(f'<tr><td>{r["id"]}</td><td>{r["stats"]["before"]["count"]}</td><td>{r["stats"]["after"]["count"]}</td><td>{r["stats"]["new_joins"]}</td></tr>' for r in records)
    report='''<!doctype html><meta charset="utf-8"><title>保守续接与整笔模型实验</title><style>body{font:17px/1.65 system-ui;max-width:1800px;margin:auto;padding:25px;color:#263e34;background:#f4f1e8}img{width:100%}td,th{border:1px solid #b4c3af;padding:10px}table{border-collapse:collapse}.note{background:#e4ebdc;padding:16px}a{color:#346943}</style><h1>保留认可画面，继续改善连续性</h1><p><a href="/">播放候选笔序</a></p>'''
    report+='<p class="note">本页实用候选采用上一轮已测试的原图支持连接算法，应用到已认可500／700笔方案。完整原稿、两档原文件均未修改；本轮不再省略细节。连接仍未经逐笔专家验证，未上机。</p><table><tr><th>方案</th><th>之前</th><th>候选</th><th>续接处</th></tr>'+table+'</table>'
    for budget in (500,700):report+=f'<h2>G03 · {budget}笔方案</h2><img src="G03_01_{budget}/comparison.jpg">'
    report+='<h2>新模型结果：未采用</h2><p>另完成两轮各2200次GPU更新。第一轮在参考位置和方向附近预测下一点，整图自由追踪却产生悬空长弧。第二轮加入位置／方向偏差、历史轨迹漂移和已知离墨状态，让模型学习纠偏。第二轮参考纠偏位置的平均下一点误差为'+f'{second["test"]["real_reference_seeded"]["mean_next_point_error_px"]:.3f}px；该任务比第一轮更难，两个误差不能直接比较。'
    report+='</p><p>两轮纯模型直接追踪都没有候选达到整笔替换门槛。第二轮另做“模型＋逐步图像约束”和“不用模型的同条件几何追踪”对照，也未证明能可靠替换至少两条完整旧曲线。没有降低门槛强行采用，没有逐点切碎来隐藏失败。当前可见改动不来自新模型。</p>'
    for roundname in ['round1','round2']:
        src=NEURAL/('G03_01_500/comparison.jpg' if roundname=='round1' else 'recovery_round2/G03_01_500/comparison.jpg')
        # Copy to whitelisted scene image names for this read-only platform.
        target=ROOT/'G03_01_500'/('rock.jpg' if roundname=='round1' else 'pavilion.jpg');shutil.copy2(src,target)
        report+=f'<h3>{roundname}：原图／基线／纯模型直接输出／核验结果</h3><img src="G03_01_500/{target.name}">'
    report+='<p>训练仅使用程序渲染和之前的显式复核曲线，没有新增独立整幅画真值。真实参考末端没有当作停笔负例，未标区域没有当背景。下一点误差不是整笔或整幅准确率。网络尚未识别传统皴法类别，对象分块也仍有待定区域。</p>'
    report+='<p>总体仍是小幅连续性改善，不能宣称长笔触提取突破。实际笔宽、含墨量和起收笔尚未验证；平台为离线笔序，不是摄像头画面或实机运行。</p>'
    (ROOT/'report.html').write_text(report,encoding='utf8')
    save(ROOT/'neural_adoption.json',dict(round1_model_sha256=first['model_sha256'],round2_model_sha256=second['model_sha256'],
        adopted=False,reason='No whole-curve replacement passed on tested baseline scenes',
        practical_candidate_is_geometric_source_supported_only=True,new_model_updates=4400,robot_connected=False))
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__),snap/Path(__file__).name)
if __name__=='__main__':main()
