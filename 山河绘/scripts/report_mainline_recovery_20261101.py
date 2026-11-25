# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from mountain_mainline_recovery_20261101 import *
from train_mountain_candidate_recovery_20261101 import LABELS
from html.parser import HTMLParser
import shutil,html

FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)

def board(ident,detail=False):
    f=ROOT/'outputs'/ident;roi=[150,170,350,420] if ident=='G05_01' else None
    fields=[('input.png','原图'),('baseline.png','v4＋既有短线补回'),('previous.png','上一轮结构简稿'),('lines.png','本轮主线恢复候选')]
    if detail:fields.append(('changes.png','蓝：找回；黑：上一轮保留'))
    edge=400 if detail else 496;height=500 if detail else 496
    canvas=Image.new('RGB',(edge*len(fields),height+62),'#f3f1eb');draw=ImageDraw.Draw(canvas)
    for i,(name,title) in enumerate(fields):
        im=Image.open(f/name).convert('RGB')
        if detail and roi:im=im.crop(roi)
        if detail and roi:im=im.resize((edge,height),Image.Resampling.LANCZOS if name=='input.png' else Image.Resampling.NEAREST)
        else:im.thumbnail((edge,height))
        draw.text((i*edge+5,7),title,font=FONT,fill='black');canvas.paste(im,(i*edge,50))
    canvas.save(f/('detail.jpg' if detail else 'comparison.jpg'),quality=96)

def reference_eval():
    refs=read(ROOT/'partial_references.json')['references'];records=[]
    for ident in LABELS:
        decisions=read(ROOT/'outputs'/ident/'decisions.json');rows=read(ROOT/'catalog'/f'{ident}.json')['candidates'];subset=[r for r in refs if r['image']==ident and r['label'] is not None];out=[]
        old=np.asarray(Image.open(ROOT/'outputs'/ident/'previous.png'))<128;new=np.asarray(Image.open(ROOT/'outputs'/ident/'lines.png'))<128
        for ref in subset:
            row=rows[ref['candidate']];d=decisions[ref['candidate']];raw=d['score']>=.8 and d['ensemble_std']<=.15;geometry=qualifies(row,1.,0.);combined=qualifies(row,d['score'],d['ensemble_std']);p=np.asarray(row['points']);q=np.rint(p).astype(int)
            out.append(dict(candidate=ref['candidate'],label=ref['label'],raw_model_accept=raw,geometry_only_accept=geometry,combined_accept=combined,final_recovered=d['decision']=='restore',old_point_coverage=float(old[q[:,1],q[:,0]].mean()),new_point_coverage=float(new[q[:,1],q[:,0]].mean()),final_reason=d['reason']))
        def metric(field):return dict(positive_accepted=sum(r['label']==1 and r[field] for r in out),negative_accepted=sum(r['label']==0 and r[field] for r in out))
        records.append(dict(image=ident,split=subset[0]['split'],positive=sum(r['label']==1 for r in subset),negative=sum(r['label']==0 for r in subset),raw_model=metric('raw_model_accept'),geometry_only=metric('geometry_only_accept'),combined_gate=metric('combined_accept'),final_recovery=metric('final_recovered'),reference_points_coverage_before=float(np.mean([r['old_point_coverage'] for r in out if r['label']==1])),reference_points_coverage_after=float(np.mean([r['new_point_coverage'] for r in out if r['label']==1])),details=out))
    doc=dict(records=records,scope='Biased small assistant candidate-label diagnostic, NOT pixel centerline accuracy, not expert or blind test',unknown_not_negative=True,model_improvement_over_geometry_established=False)
    jsave(ROOT/'reference_diagnostic.json',doc);return doc

def behavior_checks():
    checks=[]
    def ck(name,ok):checks.append(dict(name=name,passed=bool(ok)));assert ok,name
    shape=(120,140);rgb=np.full((*shape,3),245,np.uint8);cv2.line(rgb,(20,60),(100,60),(30,30,30),3);line=np.zeros(shape,np.uint8);cv2.line(line,(20,60),(58,60),1,1);cv2.line(line,(61,60),(100,60),1,1);mask=dict(interior=np.ones(shape,bool),protect=np.zeros(shape,bool))
    # Fixed pixel-size helper fixture uses the production scale (496 long edge).
    rgb=cv2.copyMakeBorder(rgb,0,376,0,356,cv2.BORDER_CONSTANT,value=(245,245,245));line=cv2.copyMakeBorder(line,0,376,0,356,cv2.BORDER_CONSTANT,value=0);mask={k:np.full((496,496),k=='interior',bool) for k in mask}
    res,added,events=bridge_gaps(rgb,line>0,mask);ck('supported_collinear_gap_recovered',added.sum()==2 and res[60,59] and res[60,60]);ck('bridge_has_source_ink',not (added&~v4.source_ink(rgb)[0]).any())
    blank=rgb.copy();blank[58:63,59:61]=245;res2,added2,_=bridge_gaps(blank,line>0,mask);ck('blank_gap_not_hallucinated',not added2.any())
    protected={k:v.copy() for k,v in mask.items()};protected['protect'][60,59]=True;_,a,_=bridge_gaps(rgb,line>0,protected);ck('protected_tree_gap_not_modified',not a.any())
    parallel=np.zeros((496,496),np.uint8);cv2.line(parallel,(30,40),(30,90),1,1);cv2.line(parallel,(33,40),(33,90),1,1);_,a,_=bridge_gaps(rgb,parallel>0,mask);ck('near_parallel_strokes_not_joined',not a.any())
    c=np.zeros((496,496),np.uint8);cv2.rectangle(c,(20,50),(100,100),1,1);c[50,59:61]=0;_,a,_=bridge_gaps(rgb,c>0,mask);ck('open_loop_not_closed',not a.any())
    row=dict(fully_interior=True,protected=False,features=dict(length=30,span_ratio=.9,support012=.9,support006=.95,contrast_median=.04));ck('source_supported_long_candidate_eligible',qualifies(row,.99,.02))
    row['features']['span_ratio']=.2;ck('curled_candidate_deferred_despite_high_model_score',not qualifies(row,.999,.001));row['features']['span_ratio']=.9;row['features']['support012']=.3;ck('weak_ink_candidate_deferred',not qualifies(row,.999,.001))
    ck('unknown_reference_not_supervised',read(ROOT/'fit/evaluation.json')['unknown_excluded']==50)
    jsave(ROOT/'behavior_checks.json',dict(checks=checks,count=len(checks),all_passed=True));return checks

def verify(records):
    checks=[]
    def ck(name,ok):checks.append(dict(name=name,passed=bool(ok)));assert ok,name
    protocol=read(ROOT/'combined_policy.json');ck('frozen_inference_code',digest(Path(__file__).with_name('mountain_mainline_recovery_20261101.py'))==protocol['code_sha256']);ck('frozen_catalog_code',digest(Path(__file__).with_name('mountain_recovery_catalog_20261101.py'))==protocol['catalog_sha256']);ck('frozen_weights',digest(ROOT/'fit/model.pt')==protocol['model_sha256']);ck('31_reused_images',len(records)==31)
    for r in records:
        ident=r['id'];f=ROOT/'outputs'/ident;new=np.asarray(Image.open(f/'lines.png'))<128;old=np.asarray(Image.open(f/'previous.png'))<128;base=np.asarray(Image.open(f/'baseline.png'))<128;core=np.asarray(Image.open(f/'structure.png'))<128;opt=np.asarray(Image.open(f/'adjustable_cun.png'))<128;unk=np.asarray(Image.open(f/'unknown.png'))<128;bridges=np.asarray(Image.open(f/'bridges.png'))<128;z=np.load(OLD/'outputs'/ident/'routing_masks.npz');doc=read(f/'paths.json')
        ck(ident+'_old_pixels_preserved',not (old&~new).any());ck(ident+'_tree_protection_exact',np.array_equal(new&z['protect'],old&z['protect']));ck(ident+'_outside_interior_exact',np.array_equal(new&~z['interior'],old&~z['interior']));ck(ident+'_three_layers_partition',np.array_equal(core|opt,new) and not (core&opt).any() and not (unk&new).any() and np.array_equal(unk,base&~new));ck(ident+'_new_pixels_only_verified_bridges',np.array_equal(new&~base,bridges&~base));ck(ident+'_canonical_export_matches',np.array_equal(new,v4.raster(new.shape,[np.asarray(p['points']) for p in doc['paths']])));ck(ident+'_offline',doc['physical_z'] is None and doc['physical_speed'] is None and doc['ready_for_robot'] is False and doc['actually_drawn'] is False);ck(ident+'_source_unchanged',doc['source_sha256']==digest(OLD/'outputs'/ident/'input.png'));ck(ident+'_previous_paths_unchanged',doc['previous_paths_sha256']==digest(OLD/'outputs'/ident/'paths.json'))
        edges=set();duplicate=0
        for path in doc['paths']:
            for a,b in zip(path['points'][:-1],path['points'][1:]):
                if a==b:continue
                key=tuple(sorted([tuple(a),tuple(b)]));duplicate+=int(key in edges);edges.add(key)
        ck(ident+'_no_duplicate_export_edges',duplicate==0)
        if bridges.any():
            rgb=np.asarray(Image.open(f/'input.png').convert('RGB'));ck(ident+'_bridge_on_actual_source_ink',not (bridges&~v4.source_ink(rgb)[0]).any())
        rows=read(f/'decisions.json');source=read(LAB/'v4_local_repairs_20261029/candidate'/ident/'paths.json')['paths'] if ident!='N02' else []
        for d in rows:
            if d['decision']=='restore' and d['candidate']<len(source):
                p=np.rint(np.asarray(source[d['candidate']]['points'])).astype(int);ck(ident+'_whole_original_candidate_'+str(d['candidate']),bool(new[p[:,1],p[:,0]].all()))
    for r in read(OLD/'summary.json')['tree_roi_checks']:
        f=ROOT/'outputs'/r['id'];a=np.asarray(Image.open(f/'previous.png'));b=np.asarray(Image.open(f/'lines.png'));x,y,xx,yy=r['roi'];ck(r['id']+'_fixed_tree_roi_exact',np.array_equal(a[y:yy,x:xx],b[y:yy,x:xx]))
    entry=ROOT/'entry_check/recovery';ck('independent_arbitrary_image_entry_matches_batch',np.array_equal(np.asarray(Image.open(entry/'lines.png')),np.asarray(Image.open(ROOT/'outputs/G05_01/lines.png'))))
    for name in ['structure','adjustable_cun','unknown','bridges']:
        ck('independent_entry_layer_'+name,np.array_equal(np.asarray(Image.open(entry/(name+'.png'))),np.asarray(Image.open(ROOT/'outputs/G05_01'/(name+'.png')))))
    frozen=read(OLD/('frozen_protocol_'+CFG['version']+'.json'));ck('previous_r7_code_unchanged',digest(Path(__file__).with_name('mountain_structure_v4_20261101.py'))==frozen['source_code_sha256'])
    for r in read(LAB/'stable_line_extractor_20261028/baseline_lock.json')['files']:ck('stable_'+Path(r['path']).name,digest(r['path'])==r['sha256'])
    ck('frozen_v4_unchanged',digest(Path(__file__).with_name('split_detail_v4_frozen.py'))==read(LAB/'v4_local_repairs_20261029/baseline_manifest.json')['baseline_code_sha256'])
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['src','href'] and v and not v.startswith(('http:','https:','#')):ck('report_'+v,(ROOT/v).exists())
    Links().feed((ROOT/'index.html').read_text(encoding='utf-8'));jsave(ROOT/'FINAL_VERIFICATION.json',dict(count=len(checks),all_passed=True,checks=checks,accuracy_established=False,robot_connected=False));return checks

def main():
    records=read(ROOT/'results_all.json')['records'];refs=reference_eval();behavior_checks()
    for r in records:board(r['id'])
    board('G05_01',True);shutil.copy2(ROOT/'outputs/G05_01/comparison.jpg',ROOT/'MAINLINE_PREVIEW.jpg');shutil.copy2(ROOT/'outputs/G05_01/detail.jpg',ROOT/'MAINLINE_DETAIL.jpg')
    summary=dict(images=len(records),training=dict(updates=900,device=read(ROOT/'fit/evaluation.json')['device'],examples=44,source_parent_groups=2,reviewed_candidates=138,known_labels=88,unknown_excluded=50,scope='small feature MLP ensemble, not a retrained pixel extractor'),recovered_candidates=sum(r['restored_whole_candidates'] for r in records),recovered_structure_candidates=sum(r['restored_structure_candidates'] for r in records),recovered_optional_candidates=sum(r['restored_optional_candidates'] for r in records),recovered_pixels=sum(r['restored_pixels'] for r in records),bridge_pixels=sum(r['bridge_pixels'] for r in records),images_with_recovery=sum(r['restored_pixels']>0 for r in records),previous_deleted_pixels=sum(r['previous_deleted_pixels'] for r in records),protected_exact=all(r['protected_detail_exact'] for r in records),outside_interior_exact=all(r['outside_interior_exact'] for r in records),full_image_accuracy=None,model_better_than_geometry_established=False,default_entry_unchanged=True,robot_connected=False,all_images_reused_not_blind=True,actual_stroke_identity_verified=False,scope='Restoration of existing candidate lines, not proof of complete main contours',limits=['Nonexpert labels on four reused scenes, chosen by length/support/curvature strata; not complete line truth.','Model alone still accepts bad candidates; combined rules are needed.','Three layers have provisional roles, not learned semantic stroke classes.','Strong curves, faint lines and missing upstream proposals remain unresolved.','Some added real lines increase graph cycles; cycle counts are not error labels.','Native N02 with no confirmed interior is preserved, not improved.'])
    jsave(ROOT/'summary.json',summary)
    metricrows=[]
    for r in refs['records']:
        metricrows.append(f'<tr><td>{r["image"]} ({r["split"]})</td><td>{r["positive"]}/{r["negative"]}</td><td>{r["raw_model"]["positive_accepted"]}/{r["raw_model"]["negative_accepted"]}</td><td>{r["geometry_only"]["positive_accepted"]}/{r["geometry_only"]["negative_accepted"]}</td><td>{r["combined_gate"]["positive_accepted"]}/{r["combined_gate"]["negative_accepted"]}</td><td>{r["final_recovery"]["positive_accepted"]}/{r["final_recovery"]["negative_accepted"]}</td></tr>')
    panels=[]
    for r in records:
        i=r['id'];panels.append(f'<section><h2>{i}</h2><img loading="lazy" src="outputs/{i}/comparison.jpg"><p>恢复候选 {r["restored_whole_candidates"]} 条（结构候选 {r["restored_structure_candidates"]}、可调皴线候选 {r["restored_optional_candidates"]}）；恢复 {r["restored_pixels"]} 个原有线像素，续接 {r["bridge_pixels"]} 个像素。上一轮已保留线不删除。</p><p><a href="outputs/{i}/changes.png">新增位置</a> · <a href="outputs/{i}/layers.png">黑色结构／蓝色皴线／灰色待定</a> · <a href="outputs/{i}/structure.png">结构保留层</a> · <a href="outputs/{i}/adjustable_cun.png">可调皴线层</a> · <a href="outputs/{i}/unknown.png">待定证据层</a> · <a href="outputs/{i}/paths.json">像素轨迹</a> · <a href="outputs/{i}/lines.svg">SVG</a> · <a href="outputs/{i}/decisions.json">逐候选决定</a></p></section>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>主线恢复与三层输出实验</title><style>body{font:17px system-ui;line-height:1.75;max-width:2020px;margin:24px auto;padding:0 18px;background:#f3f1eb;color:#222}img{width:100%;background:white}aside{background:#fff0cd;padding:18px}section{margin:32px 0;border-top:1px solid #ccc}table{border-collapse:collapse;background:white}td,th{padding:9px 14px;border:1px solid #bbb}a{color:#175b80}</style>
<h1>找回被误删的山面结构，保留三层证据</h1>
<aside>本轮实际在 RTX 5060 Laptop GPU 上完成 900 次更新：三个小候选筛选网络各300步；44个已标候选来自2张相关母图。不是重新训练整个像素提取网络。138个局部候选经助手对照RGB分为88个已知标签和50个待定；不是画家逐笔真值。没有把上一轮简稿当整幅训练答案。</aside>
<h2>这次具体改了什么</h2><p>上一轮长度门槛和线数预算删掉了部分有原图依据的转折与长皴线。本轮对完整候选检查原图横截面、长度、走向、近邻重复及小模型分数，再恢复满足条件的线。20像素以上的直顺结构候选不受原皴线追加数量上限限制；较短候选留在可调皴线层，仍受疏密约束。这些角色是几何推断，不是可靠语义识别。</p>
<p>保持上一轮已选线、山体分区外像素和细节保护区原样。只在3像素以内、相互唯一且走向一致、原图确有墨迹的端点间尝试续接；不跨空白猜线。尚未选中的证据继续待定，低分候选也不作为背景像素标签。</p>
<h2>固定问题图：原图 → v4 → 上轮简稿 → 本轮恢复</h2><img src="MAINLINE_PREVIEW.jpg"><h2>同一山体局部与找回位置</h2><img src="MAINLINE_DETAIL.jpg">
<h2>模型本身有没有证明有效</h2><p>还没有证明优于源图与几何规则。模型单独会放过回折和短横刺，原先宽间隔阈值校准甚至只能关闭自动接受；这个失败保留在 fit/evaluation.json。当前结果使用单独冻结的“模型＋源图＋几何”组合门槛。下表把模型单独、几何单独、组合门槛和最终加入分开展示，最后一步还有重复与疏密约束；不能把组合收益全算给深度学习。</p>
<p>表内均为“已标正例通过数／已标反例误收数”。反例针对整条轨迹（例如墨点周边的小圈），不等于该处没有墨迹。四张都是已有图；按母图分开训练、校准与回归，但不是新画作盲测。</p><table><tr><th>图片／用途</th><th>正例／反例总数</th><th>模型单独</th><th>几何单独</th><th>组合门槛</th><th>最终补回</th></tr>'''+''.join(metricrows)+'''</table>
<h2>局限与下一步</h2><p>能找回一部分真实候选，但还不能宣称主轮廓完整。大弯折、淡墨漏提、粗墨交汇的单笔身份仍有疑难；上游没有产生的线，本轮也不能凭空恢复。闭环数可能因恢复真实交接而增加，不能用线少或环少代替准确性。树木只保持既有结果，并未解决所有树木漏提。原生N02没有可靠内部区域，继续原样保留。</p>
<p>正式入口和旧权重未替换；所有结果是离线候选，没有机械臂实画。像素线宽也不是毛笔实际宽度。下一步应补高曲率主线与淡墨的局部参考，验证候选筛选是否真正优于规则，再扩大新图验证。</p>
<p><a href="summary.json">汇总与限制</a> · <a href="training_protocol.json">训练协议</a> · <a href="combined_policy.json">实际推理协议</a> · <a href="reference_diagnostic.json">局部标签诊断</a> · <a href="fit/evaluation.json">模型原始校准失败记录</a> · <a href="partial_references.json">全部部分参考与待定</a> · <a href="behavior_checks.json">连接反例检查</a></p>
<h2>全部31张旧图复测</h2>'''+''.join(panels)
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    names=['mountain_recovery_catalog_20261101.py','train_mountain_candidate_recovery_20261101.py','mountain_mainline_recovery_20261101.py','report_mainline_recovery_20261101.py','extract_mainline_recovery_20261101.py']
    for n in names:shutil.copy2(Path(__file__).with_name(n),snap/n)
    jsave(ROOT/'source_snapshot_manifest.json',{n:digest(snap/n) for n in names});checks=verify(records);print('MAINLINE_REPORT',json.dumps(summary,ensure_ascii=False),'VERIFIED',len(checks),flush=True)
if __name__=='__main__':main()
