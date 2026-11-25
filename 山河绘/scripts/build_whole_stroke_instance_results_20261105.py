# -*- coding: utf-8 -*-
"""Expose ownership, direct full curves and retained baseline separately."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import copy,json,hashlib,shutil
import numpy as np,cv2,torch
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
import whole_stroke_instance_20261105 as instance
from whole_stroke_instance_20261105 import infer_queries,SIZE,arc_points
from local_curve_replacement_20261105 import whole_guard
from hierarchical_paint_core_20261105 import hierarchy_schedule,order_metrics
from module_stroke_core_20261104 import samples,length
from build_local_curve_policy_results_20261105 import source_seeds

TRAIN=instance.ROOT;ROOT=TRAIN/'comparison';BASE=LAB/'drawing_budgets_20261105'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf8')

def drawing(rows,color=(30,43,35)):
    im=np.full((992,992,3),255,np.uint8)
    for r in rows:cv2.polylines(im,[np.rint(np.asarray(r['points'])*2).astype(np.int32)],False,color,2,cv2.LINE_AA)
    return Image.fromarray(im)

def query_set(plan):
    seeds=source_seeds(plan['curves'],40);queries=[]
    for seed in seeds:
        r=plan['curves'][seed['curve']];p=np.asarray(r['points']);q=arc_points(p,33)
        back=q[14];front=q[18];center=q[16]
        for side in (64.,128.):queries.append(dict(center=center,side=side,back=back,front=front,
            seed_curve=seed['curve'],module=seed['module'],query_uses_reference=False))
    return queries

def ownership_overlay(rgb,proposals):
    field=np.zeros(rgb.shape[:2],np.float32);yy,xx=np.mgrid[:rgb.shape[0],:rgb.shape[1]]
    for r in proposals:
        if r['presence']<.5:continue
        coords=(np.stack([xx,yy],-1)-r['origin'])/r['side']*SIZE-.5
        values=cv2.remap(r['mask'],coords[:,:,0].astype(np.float32),coords[:,:,1].astype(np.float32),cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        field=np.maximum(field,values)
    image=(rgb*.65+255*.35).astype(np.uint8);mask=field>.5
    image[mask]=np.uint8(image[mask]*.45+np.array([25,155,80])*.55)
    return Image.fromarray(image)

def run_scene(ident,models):
    rgb=np.array(Image.open(BASE/(ident+'_700')/'source.png').convert('RGB'))
    full=read(BASE/(ident+'_700')/'plan.json');queries=query_set(full);predictions={};folder=ROOT/ident;folder.mkdir(exist_ok=True)
    for name,net in models.items():
        pred=infer_queries(net,rgb,queries);predictions[name]=pred
        np.savez_compressed(folder/(name+'_ownership.npz'),masks=np.stack([r['mask'] for r in pred]))
        save(folder/(name+'_raw_curves.json'),dict(queries=queries,proposals=[{k:v for k,v in r.items() if k!='mask'} for r in pred],
            reference_labels_read_at_inference=False,whole_curve_predicted_simultaneously=True,postprocessing_stitching_used=False))
    results=[]
    for budget in (500,700):
        key=f'{ident}_{budget}';dest=ROOT/key;dest.mkdir(exist_ok=True);parent=BASE/key/'plan.json';digest=sha(parent);old=read(parent)
        controls={};guarded={}
        for name,pred in predictions.items():
            selected=[r for r in pred if r['presence']>=.5]
            rows,guard=whole_guard(rgb,old['curves'],selected);controls[name]=dict(proposals_above_presence=len(selected),
                accepted_whole_replacements=len(guard['selected_proposals']),output_count=len(rows),reasons=guard['reasons'])
            guarded[name]=(rows,guard)
        # Keep the recognized baselines as the released/displayed execution plan.
        # Experimental whole replacements are exported separately, not auto-adopted.
        plan=copy.deepcopy(old);plan.update(parent=str(parent),parent_sha256=digest,
            new_instance_model_applied=False,adopted=False,ready_for_robot=False,
            experimental_controls=controls,approved_baseline_unchanged=True)
        plan['stats']={**old['stats'],'before':copy.deepcopy(old['stats']['after']),
            'after':copy.deepcopy(old['stats']['after']),'accepted_continuations':0}
        save(dest/'plan.json',plan)
        for name,(rows,guard) in guarded.items():
            save(dest/(name+'_guarded_candidate.json'),dict(curves=rows,guard=guard,adopted=False,ready_for_robot=False))
        for name in ('source.png','modules.png'):shutil.copy2(BASE/key/name,dest/name)
        panels=[(Image.fromarray(rgb),'原图'),(drawing(old['curves']),f'已认可基线 · {len(old["curves"])}笔'),
            (ownership_overlay(rgb,predictions['attention']),'模型预测整笔归属（合并显示）'),
            (drawing([r for r in predictions['attention'] if r['presence']>=.5],(180,55,45)),'模型直接完整曲线 · 实验输出')]
        board=Image.new('RGB',(2080,605),'#f4f1e8');draw=ImageDraw.Draw(board)
        for j,(im,label) in enumerate(panels):board.paste(im.resize((496,496),Image.Resampling.LANCZOS),(j*520+10,43));draw.text((j*520+10,10),label,font=FONT,fill='#263e34')
        draw.text((10,551),'每个查询只使用旧线稿中的一小段提示；后两栏是模型直接输出，不是几何续接，也不是完整提取结果。',font=SMALL,fill='#263e34')
        draw.text((10,579),'旧500／700笔方案保留；候选没有自动上机。模型归属图尚无真实像素级专家真值，非毛笔实画。',font=SMALL,fill='#263e34')
        board.save(dest/'comparison.jpg',quality=96);shutil.copy2(dest/'comparison.jpg',dest/'sequence_comparison.jpg')
        assert sha(parent)==digest
        results.append(dict(id=key,label=f'{key} · 固定基线与整笔实例实验',plan_sha256=sha(dest/'plan.json'),controls=controls))
        print('INSTANCE_ACTUAL',key,controls,flush=True)
    return results

def reference_diagnostic(models):
    from context_curve_data_20261105 import source_records
    records=[];views=[]
    for rec in source_records():
        if rec['id'] not in ('R03','R04'):continue
        queries=[];truth=[]
        for p in rec['refs']:
            q=arc_points(p,33);lo=np.asarray(p).min(0);hi=np.asarray(p).max(0);side=max(28,float(max(hi-lo))*1.4)
            queries.append(dict(center=(lo+hi)/2,side=side,back=q[14],front=q[18]));truth.append(np.asarray(p))
        for name,net in models.items():
            pred=infer_queries(net,rec['rgb'],queries);coverage=[];precision=[];baseline=[]
            for p,target in zip(pred,truth):
                q=samples(target,.4);r=samples(p['points'],.4)
                coverage.append(float((cKDTree(r).query(q)[0]<=2).mean()))
                precision.append(float((cKDTree(q).query(r)[0]<=2).mean()))
                baseline.append(max([float((cKDTree(samples(v,.4)).query(q)[0]<=2).mean()) for v in rec['paths']]+[0.]))
            row=dict(id=rec['id'],model=name,mean_single_curve_coverage_2_native_px=float(np.mean(coverage)),
                mean_output_near_reference_2px=float(np.mean(precision)),baseline_best_single_path_coverage=float(np.mean(baseline)),
                whole_reference_80pct_coverage_fraction=float((np.asarray(coverage)>=.8).mean()),
                reference_short_query_and_reference_bbox_used=True,not_autonomous_extraction_accuracy=True)
            records.append(row)
            if rec['id']=='R04' and name=='attention':
                for j in range(min(3,len(pred))):views.append((rec['rgb'],truth[j],pred[j],queries[j]))
    board=Image.new('RGB',(1560,1230),'#f4f1e8');d=ImageDraw.Draw(board)
    for i,(rgb,target,pred,query) in enumerate(views):
        x,origin=instance.crop_query(rgb,query['center'],query['side'],query['back'],query['front'])
        source=x[:3].transpose(1,2,0);mask=pred['mask'];ownership=np.uint8(np.repeat(mask[:,:,None],3,2)*255)
        canvas=source.copy();tf=lambda q:(np.asarray(q)-origin)/query['side']*SIZE-.5
        cv2.polylines(canvas,[np.rint(tf(target)).astype(np.int32)],False,(30,160,55),1,cv2.LINE_AA)
        cv2.polylines(canvas,[np.rint(tf(pred['points'])).astype(np.int32)],False,(215,45,35),1,cv2.LINE_AA)
        for j,(im,label) in enumerate([(source,f'固定R04参考 {i+1} · 短提示与参考框'),(ownership,'模型整笔归属概率'),(canvas,'绿：参考／红：模型直接曲线')]):
            board.paste(Image.fromarray(im).resize((380,340),Image.Resampling.NEAREST),(j*520+10,i*400+45))
            d.text((j*520+10,i*400+10),label,font=FONT,fill='#263e34')
    board.save(ROOT/'reference_diagnostic.jpg',quality=96)
    save(ROOT/'reference_diagnostic.json',dict(records=records,reference_guided_diagnostic_only=True,
        old_images_and_source_families_reused=True,fixed_first_three_references_not_selected_for_success=True))
    return records

def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(2)
    instance.ROOT=TRAIN;first=instance.load(False)
    instance.ROOT=TRAIN/'ownership_attention';attention=instance.load(True)
    models=dict(first=first,attention=attention)
    diagnostic=reference_diagnostic(models);scenes=[]
    for ident in ('G03_01','G05_03','T02'):scenes+=run_scene(ident,models)
    save(ROOT/'catalog.json',dict(scenes=scenes,default='G03_01_500',robot_connected=False))
    save(ROOT/'execution_binding.json',dict(scene=None,journal=None,plan_sha256=None))
    first_meta=read(TRAIN/'training_summary.json');second=read(TRAIN/'ownership_attention/training_summary.json')
    # No automatic promotion based on masks/seeded diagnostics. The fixed
    # recognized drawings remain active until a full-image gain is established.
    reference_rows=[r for r in diagnostic if r['model']=='attention']
    reference_gain=all(r['mean_single_curve_coverage_2_native_px']>r['baseline_best_single_path_coverage'] for r in reference_rows)
    accepted=sum(r['controls']['attention']['accepted_whole_replacements'] for r in scenes)
    save(ROOT/'decision.json',dict(adopted=False,experiment_passed=reference_gain and accepted>0,
        reference_gain_on_both_fixed_records=reference_gain,accepted_whole_replacements=accepted,
        criterion='must demonstrate reliable complete-stroke improvement over recognized 500/700 baselines',
        real_full_image_gain_established=False,reason='whole-curve evidence is insufficient; see direct outputs and fixed diagnostics',
        no_geometric_stitching_used_to_claim_learning_progress=True,baseline_hashes_verified=True,
        model_updates=first_meta['updates']+second['updates'],source_families_not_independent=True))
    page=Path(__file__).with_name('hierarchical_paint_platform_20261105.html').read_text('utf8')
    page=page.replace("o.textContent=({G03_01:","o.textContent=s.label||({G03_01:").replace('plan=p;scene=id;sourceImage',"plan=p;scene=id;$('scene').value=id;sourceImage")
    page=page.replace('山河绘 · 笔序工作台','山河绘 · 整笔实例实验').replace('输入细段 → 连续候选','固定基线（新模型未采用）')
    page=page.replace('原图支持的断段连接','新模型整笔替换（未采用）')
    page=page.replace('<footer>','<footer><p><a href="/report.html">查看模型直接输出、整笔诊断与未采用原因</a></p>')
    (ROOT/'index.html').write_text(page,encoding='utf8')
    report='''<!doctype html><meta charset="utf-8"><title>整笔归属＋完整曲线实验</title><style>body{font:17px/1.65 system-ui;max-width:1800px;margin:auto;padding:25px;background:#f4f1e8;color:#263e34}img{width:100%}td,th{border:1px solid #b6c4b3;padding:10px}table{border-collapse:collapse}.note{background:#e4ebdc;padding:15px}a{color:#346943}</style><h1>整笔墨迹归属＋完整曲线预测</h1><a href="/">进入固定基线回放</a>'''
    report+='<p class="note">本轮结论：未证明超过已认可500／700笔版本，新模型不采用。下面展示实际神经网络输出，不用几何续接充当学习成果。两种变体分别训练2400次GPU更新：第一种联合预测归属图与33个整曲线点；第二种让33个点同时关注归属图和图像特征。无逐步追踪或端点拼接。</p>'
    rows=''
    for name,t in [('联合模型',first_meta),('归属注意力变体',second)]:
        m=t['test']['real_reference_seeded'];rows+=f'<tr><td>{name}</td><td>{t["retained_step"]}</td><td>{m["mean_ordered_curve_error_normalized_px"]:.2f}</td><td>{m["mean_reference_point_coverage_2px"]:.1%}</td><td>{m["whole_curve_80pct_coverage_fraction"]:.1%}</td></tr>'
    report+='<h2>参考提示与参考框的局部检查（不是自动整图准确率）</h2><table><tr><th>模型</th><th>保留步数</th><th>平均曲线点误差／96px归一化框</th><th>参考点覆盖／2px</th><th>至少80%参考点覆盖的整曲线比例</th></tr>'+rows+'</table>'
    report+='<p>真实资料来自以前的部分复核描线，同一画作来源可能跨图片。第二个变体是在看到第一种变体诊断后设计，不能称为新的独立盲测。真实归属监督是参考中心线扩成的2px代理带，并非专家逐像素墨迹归属；未标区域不作为负例，真实端点热图没有监督。</p>'
    report+='<img src="G03_01_500/rock.jpg"><p>上图固定展示R04前三条参考，未按成功程度挑选；参考短提示与参考框使它比自动整图任务容易，仍不能证明自主完整提取。</p>'
    report+='<h2>固定500／700笔基线上的实际输入</h2><p>下面后两栏由原候选短片段和固定64／128像素窗口查询产生，没有读取参考线或参考框。绿色是多条查询的归属概率合并显示，红色是直接完整曲线；都不是完整提取稿，也不是实画。</p>'
    for ident in ('G03_01','G05_03','T02'):report+=f'<h3>{ident} · 500笔基线</h3><img src="{ident}_500/comparison.jpg">'
    report+='<p>整条替换门槛没有放宽，没有把错误候选逐点裁成碎段，也没有把代理归属IoU或局部误差当成整幅画准确率。旧文件全部按哈希核对。未连接、使能或移动机械臂；没有完成皴法类别或真实笔宽、墨色复现。</p>'
    (ROOT/'report.html').write_text(report,encoding='utf8')
    shutil.copy2(ROOT/'reference_diagnostic.jpg',ROOT/'G03_01_500/rock.jpg')
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ('whole_stroke_instance_20261105.py','build_whole_stroke_instance_results_20261105.py'):
        shutil.copy2(Path(__file__).with_name(name),snap/name)
    print('INSTANCE_REPORT',ROOT/'report.html',flush=True)
if __name__=='__main__':main()
