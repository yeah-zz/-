# -*- coding: utf-8 -*-
"""Raw neural proposals beside frozen 500/700 baselines and guarded candidates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import copy,json,hashlib,shutil
import numpy as np,cv2,torch
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from local_curve_policy_20261105 import ROOT,load,trace_batch,constrained_trace_batch,arc_points,STOP,GRID,SPACING
from local_curve_replacement_20261105 import whole_guard
from hierarchical_paint_core_20261105 import hierarchy_schedule,order_metrics
from module_stroke_core_20261104 import samples,length

BASE=LAB/'drawing_budgets_20261105'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf8')

def source_seeds(rows,max_count=64):
    chosen=[];counts={}
    for i in sorted(range(len(rows)),key=lambda i:(rows[i]['phase'],-length(np.asarray(rows[i]['points'])))):
        r=rows[i];p=np.asarray(r['points']);arc=length(p)
        if arc<8 or np.linalg.norm(p[0]-p[-1])<.1:continue
        m=r['module']
        if counts.get(m,0)>=4:continue
        q=arc_points(p,max(9,int(arc/.5)+1));k=len(q)//2;u=q[min(len(q)-1,k+3)]-q[max(0,k-3)]
        u/=max(np.linalg.norm(u),1e-8)
        chosen.append(dict(curve=i,module=m,seed=q[k],u=u,
            history_forward=q[max(0,k-7):k:2].tolist(),history_backward=q[k+1:min(len(q),k+8):2][::-1].tolist()))
        counts[m]=counts.get(m,0)+1
        if len(chosen)>=max_count:break
    return chosen

def draw(rows,color=(30,43,35)):
    im=np.full((992,992,3),255,np.uint8)
    for r in rows:
        p=np.asarray(r['points']);cv2.polylines(im,[np.rint(p*2).astype(np.int32)],False,color,2,cv2.LINE_AA)
    return Image.fromarray(im)

def scene(ident,model):
    folder=ROOT/ident;folder.mkdir(exist_ok=True)
    parent=BASE/(ident+'_700')/'plan.json';full=read(parent);rgb=np.array(Image.open(BASE/(ident+'_700')/'source.png').convert('RGB'))
    seeds=source_seeds(full['curves']);positions=[r['seed'] for r in seeds];directions=[r['u'] for r in seeds]
    forward=trace_batch(model,rgb,positions,directions,max_steps=80,histories=[r['history_forward'] for r in seeds])
    backward=trace_batch(model,rgb,positions,[-u for u in directions],max_steps=80,histories=[r['history_backward'] for r in seeds])
    raw=[]
    for seed,a,b in zip(seeds,forward,backward):
        raw.append(dict(points=np.vstack([b['points'][::-1],a['points'][1:]]),seed_curve=seed['curve'],
            stop_forward=a['stop'],stop_backward=b['stop'],seed_from_baseline_not_reference=True))
    save(folder/'raw_neural_proposals.json',dict(proposals=raw,model_sha256=sha(ROOT/'policy.pt'),seeds=seeds,
        source_sha256=sha(BASE/(ident+'_700')/'source.png'),reference_labels_read_at_inference=False,
        direct_model_output_not_final_extraction=True))
    print('RAW_TRACE',ident,len(raw),flush=True)
    guided=[];geometric=[]
    if '--constrained' in sys.argv:
        for use_model,store in [(True,guided),(False,geometric)]:
            a=constrained_trace_batch(model,rgb,positions,directions,max_steps=80,
                histories=[r['history_forward'] for r in seeds],use_model=use_model)
            b=constrained_trace_batch(model,rgb,positions,[-u for u in directions],max_steps=80,
                histories=[r['history_backward'] for r in seeds],use_model=use_model)
            for s,f,rev in zip(seeds,a,b):store.append(dict(points=np.vstack([rev['points'][::-1],f['points'][1:]]),
                seed_curve=s['curve'],stop_forward=f['stop'],stop_backward=rev['stop']))
        save(folder/'source_constrained_proposals.json',dict(guided=guided,geometric_control=geometric,
            pure_neural_output_is_separate=True,reference_labels_read_at_inference=False))
    scene_records=[]
    for budget in (500,700):
        key=f'{ident}_{budget}';dest=ROOT/key;dest.mkdir(exist_ok=True);path=BASE/key/'plan.json';digest=sha(path);baseline=read(path)
        pure_rows,pure_guard=whole_guard(rgb,baseline['curves'],raw)
        rows,guard=whole_guard(rgb,baseline['curves'],guided if guided else raw)
        control_rows,control_guard=whole_guard(rgb,baseline['curves'],geometric) if geometric else (baseline['curves'],dict(selected_proposals=[]))
        actions,order=hierarchy_schedule(rows,baseline['modules']);plan=copy.deepcopy(baseline)
        tree=cKDTree(np.concatenate([samples(np.asarray(r['points']),.3) for r in rows]))
        dd=tree.query(np.concatenate([samples(np.asarray(r['points']),.3) for r in baseline['curves']]))[0]
        if dd.max()>1.3:raise ValueError('Guarded output loses baseline beyond tolerance')
        stat={**baseline['stats'],'before':{**baseline['stats']['after']},
            'after':dict(count=len(rows),short_under_5px=sum(length(np.asarray(r['points']))<5 for r in rows)),
            'raw_neural_proposals':len(raw),'whole_neural_replacements':len(guard['selected_proposals']),
            'pure_neural_replacements':len(pure_guard['selected_proposals']),
            'geometric_control_replacements':len(control_guard['selected_proposals']),
            'geometric_control_output_count':len(control_rows),
            'source_constrained_neural_guidance':bool(guided),
            'replaced_baseline_fragments':len(guard['replaced_input_ids']),
            'baseline_sample_max_distance_px':float(dd.max()),'guard_reasons':guard['reasons'],
            'accepted_continuations':len(guard['selected_proposals']),
            'baseline_order':order_metrics(baseline['actions']),'hierarchy_order':order_metrics(actions),
            'new_model_weights_trained':True,'real_complete_stroke_accuracy_verified':False}
        for r in rows:r['points']=np.asarray(r['points'])
        plan.update(id=key,curves=rows,actions=actions,module_order=order,stats=stat,parent=str(path),parent_sha256=digest,
            neural_model_sha256=sha(ROOT/'policy.pt'),raw_neural_output_displayed_separately=True,
            ready_for_robot=False,approved_baseline_unchanged=True)
        for m in plan['modules']:
            m['planned_strokes']=sum(r['module']==m['id'] for r in rows);m['order']=order.index(m['id'])+1
        save(dest/'plan.json',plan);save(dest/'replacement_diagnostics.json',guard)
        save(dest/'control_diagnostics.json',dict(pure_neural=pure_guard,geometric_control=control_guard))
        shutil.copy2(BASE/key/'source.png',dest/'source.png');shutil.copy2(BASE/key/'modules.png',dest/'modules.png')
        panels=[(Image.fromarray(rgb),'原图'),(draw(baseline['curves']),f'已认可{budget}笔基线'),
                (draw(raw,(190,65,55)),f'模型直接追踪 · {len(raw)}条候选'),(draw(rows),f'整条核验后 · {len(rows)}笔')]
        board=Image.new('RGB',(2080,610),'#f4f1e8');d=ImageDraw.Draw(board)
        for j,(im,title) in enumerate(panels):board.paste(im.resize((496,496),Image.Resampling.LANCZOS),(j*520+10,43));d.text((j*520+10,10),title,font=FONT,fill='#263e34')
        d.text((10,551),f'{ident}｜模型种子来自旧候选，未读取参考线；第三栏只是直接候选，不是整幅提取结果。',font=SMALL,fill='#263e34')
        d.text((10,580),'核验只整条接纳或整条拒绝，不逐点截成碎线；已认可方案原文件保持不变。未实画。',font=SMALL,fill='#263e34')
        board.save(dest/'comparison.jpg',quality=96);shutil.copy2(dest/'comparison.jpg',dest/'sequence_comparison.jpg')
        assert sha(path)==digest
        scene_records.append(dict(id=key,label=f'{ident} · {budget}笔基线上的整笔实验（{len(rows)}笔）',plan_sha256=sha(dest/'plan.json'),stats=stat))
        print('WHOLE_REPLACEMENT',key,len(baseline['curves']),len(rows),guard['reasons'],flush=True)
    return scene_records

def constant_control():
    z=np.load(ROOT/'test.npz');y=z['y'];orig=z['origin'];real=orig=='explicit_real_next_point';mask=y!=STOP
    target=np.column_stack([y%GRID,y//GRID]);constant=np.array([GRID//2+4,GRID//2])
    error=np.linalg.norm(target-constant,axis=1)*SPACING
    result=dict(description='fixed 2px forward prediction, no image input',
        real_reference_mean_error_px=float(error[real].mean()),real_reference_within_1px=float((error[real]<=1).mean()),
        does_not_learn_stop_or_trace_curve=True)
    save(ROOT/'fixed_step_control.json',result);return result

def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);model=load();control=constant_control();records=[]
    for ident in ('G03_01','G05_03','T02'):records+=scene(ident,model)
    save(ROOT/'catalog.json',dict(scenes=records,default='G03_01_500',robot_connected=False))
    save(ROOT/'execution_binding.json',dict(scene=None,journal=None,plan_sha256=None))
    page=Path(__file__).with_name('hierarchical_paint_platform_20261105.html').read_text('utf8')
    page=page.replace("o.textContent=({G03_01:","o.textContent=s.label||({G03_01:")
    page=page.replace('plan=p;scene=id;sourceImage',"plan=p;scene=id;$('scene').value=id;sourceImage")
    page=page.replace('山河绘 · 笔序工作台','山河绘 · 整笔模型实验')
    page=page.replace('输入细段 → 连续候选','已认可基线 → 整条核验候选').replace('原图支持的断段连接','模型整条替换候选').replace('查看同线集排序对照','查看模型直接输出与核验对照')
    page=page.replace('<footer>','<footer><p><a href="/report.html">查看新模型训练结果与局限</a></p>')
    (ROOT/'index.html').write_text(page,encoding='utf8')
    train=read(ROOT/'training_summary.json');real=train['test']['real_reference_seeded']
    report='''<!doctype html><meta charset="utf-8"><title>整笔局部策略模型实验</title><style>body{font:17px/1.65 system-ui;max-width:1800px;margin:auto;padding:25px;color:#263e34;background:#f4f1e8}img{width:100%}.note{padding:15px;background:#e5ecde}a{color:#346943}td,th{padding:10px;border:1px solid #b4c3af}table{border-collapse:collapse}</style><h1>保留500／700笔基线的整笔模型实验</h1><a href="/">进入笔序回放</a>'''
    report+=f'<p class="note">新模型在{train["gpu"]}完成{train["updates"]}次更新，保留第{train["retained_step"]}次权重。每走一步重新查看RGB、局部墨迹对比与最近走过的轨迹，预测下一点与停止信号。曲线种子来自原候选，推理不读取参考标注。没有重新训练对象分割或传统皴法类别。</p>'
    report+=f'<p>给定参考位置和方向的真实局部检查：平均下一点误差{real["mean_next_point_error_px"]:.3f}px，固定向前2px控制为{control["real_reference_mean_error_px"]:.3f}px。此检查每步使用正确参考位置，不能代表自由连续追踪、完整分笔或整幅提取准确率。图像来自旧复核资料，同源画作可能跨图片，不是新画作盲测。</p>'
    report+='<table><tr><th>图／预算</th><th>基线笔数</th><th>核验后</th><th>整条替换候选</th></tr>'
    for r in records:report+=f'<tr><td>{r["id"]}</td><td>{r["stats"]["before"]["count"]}</td><td>{r["stats"]["after"]["count"]}</td><td>{r["stats"]["whole_neural_replacements"]}</td></tr>'
    report+='</table><p>第三栏展示同样种子的全部模型直接追踪，不是完整线稿。第四栏是整条候选来源核验后的输出；未通过者整条拒绝，未用逐点截取隐藏错误。原500／700笔文件哈希逐一核对保持不变。当前核验未经过完整专家真值验证，不自动部署。</p>'
    for ident in ('G03_01','G05_03','T02'):report+=f'<h2>{ident} · 500笔基线</h2><img src="{ident}_500/comparison.jpg">'
    report+='<p>训练数据含程序渲染及显式复核曲线的下一点监督。真实参考末端没有当作停笔答案，未标区域没有当成背景；停止监督主要来自已知渲染曲线，因此真实停笔与交汇处理仍需独立检验。全部为离线中心线预览，未标定真实笔宽、墨色或笔压，未连接机械臂。</p>'
    (ROOT/'report.html').write_text(report,encoding='utf8')
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ('local_curve_policy_20261105.py','local_curve_replacement_20261105.py','build_local_curve_policy_results_20261105.py'):
        shutil.copy2(Path(__file__).with_name(name),snap/name)
if __name__=='__main__':
    if '--round2' in sys.argv:
        import local_curve_policy_20261105 as policy
        ROOT=ROOT/'recovery_round2';policy.ROOT=ROOT
    main()
