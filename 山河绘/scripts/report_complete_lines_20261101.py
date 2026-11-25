# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from mountain_complete_lines_20261101 import *
from html.parser import HTMLParser
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)

def compare(ident):
    folder=ROOT/'outputs'/ident;board=Image.new('RGB',(1984,552),'#f3f1eb');d=ImageDraw.Draw(board)
    for j,(name,title) in enumerate([('input','原图'),('previous','上一轮主线恢复'),('complete','本轮完整提取模式（候选）'),('sparse','可选简稿')]):
        im=Image.open(folder/(name+'.png')).convert('RGB');im.thumbnail((496,496));d.text((j*496+4,8),title,font=FONT,fill='black');board.paste(im,(j*496,50))
    board.save(folder/'comparison.jpg',quality=96)

def local_cases():
    references=read(ROOT/'partial_references.json')['references'];cases=[('G05_01','ridge_353','淡墨局部（训练参考）'),('G05_01','ridge_880','淡墨弯线（训练参考）'),('G03_02','ridge_1380','岩石弯曲轮廓（训练参考）'),('G04_01','v4_730','弯折局部（旧图回归参考）')];board=Image.new('RGB',(1200,1160),'#f3f1eb');d=ImageDraw.Draw(board);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    for j,(ident,key,title) in enumerate(cases):
        ref=next(r for r in references if r['image']==ident and r['key']==key);p=np.asarray(ref['points']);x,y=np.maximum(0,np.floor(p.min(0)-12)).astype(int);xx,yy=np.minimum([496,496],np.ceil(p.max(0)+13)).astype(int);folder=ROOT/'outputs'/ident;orig=Image.open(folder/'input.png').convert('RGB').crop((x,y,xx,yy));position=orig.copy();dr=ImageDraw.Draw(position);dr.line([tuple(q-[x,y]) for q in p],fill=(35,125,220),width=1);before=Image.open(folder/'previous.png').convert('RGB').crop((x,y,xx,yy));after=Image.open(folder/'complete.png').convert('RGB').crop((x,y,xx,yy));d.text((4,j*290+3),ident+' '+key+' '+title,font=FONT,fill='black')
        for k,(im,label) in enumerate([(orig,'原图'),(position,'蓝：非专家参考候选'),(before,'上一轮'),(after,'完整模式候选')]):
            d.text((k*300+4,j*290+31),label,font=small,fill='black');factor=min(292/im.width,224/im.height,5.);im=im.resize((round(im.width*factor),round(im.height*factor)),Image.Resampling.NEAREST);board.paste(im,(k*300+4,j*290+57))
    board.save(ROOT/'LOCAL_CASES.jpg',quality=96)

def partial_diagnostic():
    refs=read(ROOT/'partial_references.json')['references'];old_decisions={i:{r['candidate']:r for r in read(PREVIOUS/'reference_diagnostic.json')['records'] if r['image']==i for r in r['details']} for i in ['G01_01','G02_01','G04_01','G05_01']};records=[];budget=[]
    for ident in sorted({r['image'] for r in refs}):
        out=ROOT/'outputs'/ident;old=np.asarray(Image.open(out/'previous.png'))<128;new=np.asarray(Image.open(out/'complete.png'))<128;do=distance_transform_edt(~old);dn=distance_transform_edt(~new);scale=max(old.shape)/496.;decisions={r['key']:r for r in read(out/'decisions.json')};z=np.load(ROOT/'prepared'/ident/'field.npz');field={k:z[k] for k in z.files};rgb=np.asarray(Image.open(out/'input.png').convert('RGB'));rows=[]
        for ref in [r for r in refs if r['image']==ident and r['label'] is not None]:
            p=np.asarray(ref['points']);q=np.rint(p).astype(int);x,y=q[:,0],q[:,1];f=arc_profile(rgb,p,field);before=float((do[y,x]<=1.5*scale).mean());after=float((dn[y,x]<=1.5*scale).mean());a=decisions.get(ref['key'],{});rows.append(dict(key=ref['key'],label=ref['label'],before_coverage=before,after_coverage=after,explicitly_added=a.get('accepted',False),contrast=f['contrast_median'],curve_turn=f['total_turn'],scope='Candidate path proximity, not confirmed accurate centerline'))
            if ref['label']==1 and ref['key'].startswith('v4_') and ident in old_decisions:
                prev=old_decisions[ident].get(int(ref['key'][3:]))
                if prev and prev['final_reason']=='optional_density_budget_keep_unknown':budget.append(dict(image=ident,key=ref['key'],before=before,after=after,covered_after=after>=.9))
        positives=[r for r in rows if r['label']==1];faint=[r for r in positives if r['contrast']<.03];curves=[r for r in positives if r['curve_turn']>=1.5]
        records.append(dict(image=ident,positive=len(positives),negative=sum(r['label']==0 for r in rows),mean_reference_path_coverage_before=float(np.mean([r['before_coverage'] for r in positives])),mean_reference_path_coverage_after=float(np.mean([r['after_coverage'] for r in positives])),negative_candidates_explicitly_added=sum(r['label']==0 and r['explicitly_added'] for r in rows),faint_positive=len(faint),faint_covered_before=sum(r['before_coverage']>=.9 for r in faint),faint_covered_after=sum(r['after_coverage']>=.9 for r in faint),curved_positive=len(curves),curved_covered_before=sum(r['before_coverage']>=.9 for r in curves),curved_covered_after=sum(r['after_coverage']>=.9 for r in curves),details=rows))
    doc=dict(records=records,previous_budget_blocked=dict(count=len(budget),covered_before=sum(r['before']>=.9 for r in budget),covered_after=sum(r['covered_after'] for r in budget),details=budget),reference_scope='Biased partial nonexpert candidate reference, selected from proposals on reused images. Never entire-image accuracy.',full_image_accuracy=None);jsave(ROOT/'partial_diagnostic.json',doc);return doc

def check_integrity(records):
    checks=[]
    def ck(name,value):checks.append(dict(name=name,passed=bool(value)));assert value,name
    protocol=read(ROOT/'inference_protocol_r2_bucket.json');ck('frozen_inference_source',digest(Path(__file__).with_name('mountain_complete_lines_20261101.py'))==protocol['inference_code_sha256']);ck('frozen_evidence_source',digest(Path(__file__).with_name('mountain_complete_evidence_20261101.py'))==protocol['evidence_code_sha256']);ck('frozen_weights',digest(FIT/'model.pt')==protocol['model_sha256']);ck('frozen_calibration',digest(ROOT/'bucket_calibration.json')==protocol['calibration_sha256']);ck('31_old_images',len(records)==31)
    for r in records:
        ident=r['id'];folder=ROOT/'outputs'/ident;old=np.asarray(Image.open(folder/'previous.png'))<128;full=np.asarray(Image.open(folder/'complete.png'))<128;sparse=np.asarray(Image.open(folder/'sparse.png'))<128;before=np.asarray(Image.open(folder/'before_gap.png'))<128;bridges=np.asarray(Image.open(folder/'bridges.png'))<128;z=np.load(ROOT/'prepared'/ident/'field.npz');new=full&~old;rgb=np.asarray(Image.open(folder/'input.png').convert('RGB'));ck(ident+'_old_selected_preserved',not (old&~full).any());ck(ident+'_sparse_subset_complete',not (sparse&~full).any());ck(ident+'_protect_exact',np.array_equal(full&z['protect'],old&z['protect']));ck(ident+'_new_pixels_in_source_search',z['area'][new].all());ck(ident+'_bridges_only_explain_continuation',np.array_equal(full,before|bridges));ck(ident+'_bridges_no_new_cycles',graph_cycles(full)==graph_cycles(before))
        ck(ident+'_source_unchanged',digest(STRUCTURE/'outputs'/ident/'input.png')==read(ROOT/'prepared'/ident/'candidates.json')['source_sha256']);ck(ident+'_previous_output_unchanged',digest(PREVIOUS/'outputs'/ident/'paths.json')==read(ROOT/'prepared'/ident/'candidates.json')['previous_sha256']);ck(ident+'_no_count_rejection',not any(not d['accepted'] and 'budget' in d['reason'] for d in read(folder/'decisions.json')))
        for name in ['complete','sparse']:
            doc=read(folder/(name+'_paths.json'));m=full if name=='complete' else sparse;ck(ident+'_'+name+'_export_matches',np.array_equal(v4.raster(m.shape,[np.asarray(p['points']) for p in doc['paths']]),m));ck(ident+'_'+name+'_offline',doc['physical_z'] is None and doc['physical_speed'] is None and doc['ready_for_robot'] is False and doc['actually_drawn'] is False);edges=set();duplicates=0
            for p in doc['paths']:
                for a,b in zip(p['points'][:-1],p['points'][1:]):
                    if a==b:continue
                    key=tuple(sorted([tuple(a),tuple(b)]));duplicates+=int(key in edges);edges.add(key)
            ck(ident+'_'+name+'_no_duplicate_edges',duplicates==0)
        for event in read(folder/'bridge_events.json'):
            if not event['accepted']:continue
            p=np.asarray(event['points']);q=np.rint(p).astype(int);support=z['weak'][q[:,1],q[:,0]];ck(ident+'_bridge_supported_'+str(len(checks)),float(support.mean())>=POLICY['bridge_support_fraction'] and longest_false(support)<=POLICY['bridge_max_unsupported_run'])
    for name in ['complete','sparse','bridges']:
        ck('independent_entry_'+name,np.array_equal(np.asarray(Image.open(ROOT/'entry_check/complete'/(name+'.png'))),np.asarray(Image.open(ROOT/'outputs/G05_01'/(name+'.png')))))
    for r in read(LAB/'stable_line_extractor_20261028/baseline_lock.json')['files']:ck('stable_'+Path(r['path']).name,digest(r['path'])==r['sha256'])
    ck('original_v4_unchanged',digest(Path(__file__).with_name('split_detail_v4_frozen.py'))==read(LAB/'v4_local_repairs_20261029/baseline_manifest.json')['baseline_code_sha256']);ck('previous_mainline_code_unchanged',digest(Path(__file__).with_name('mountain_mainline_recovery_20261101.py'))==read(PREVIOUS/'combined_policy.json')['code_sha256']);ck('behavior_tests_passed',read(ROOT/'behavior_checks.json')['all_passed'])
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src'] and v and not v.startswith(('http:','https:','#','data:')):ck('report_'+v,(ROOT/v).exists())
    Links().feed((ROOT/'index.html').read_text(encoding='utf-8'));jsave(ROOT/'FINAL_VERIFICATION.json',dict(count=len(checks),all_passed=True,checks=checks,accuracy_established=False,robot_connected=False));return len(checks)

def main():
    records=read(ROOT/'results_all.json')['records'];diag=partial_diagnostic();fit=read(FIT/'evaluation.json');cal=read(ROOT/'bucket_calibration.json');entries=[];local_cases()
    for r in records:
        ident=r['id'];compare(ident);entries.append(f'<section><h2>{ident}</h2><img loading="lazy" src="outputs/{ident}/comparison.jpg"><p>原选线 {r["previous_pixels"]} → 完整模式 {r["complete_pixels"]}；新增 {r["accepted_candidates"]} 个候选，源图续接 {r["accepted_connections"]} 处。数量增加不是准确率。</p><p><a href="outputs/{ident}/changes.png">蓝：补回；紫：续接</a> · <a href="outputs/{ident}/unknown.png">待定证据</a> · <a href="outputs/{ident}/complete.svg">完整SVG</a> · <a href="outputs/{ident}/sparse.svg">简稿SVG</a> · <a href="outputs/{ident}/complete_paths.json">完整像素轨迹</a> · <a href="outputs/{ident}/decisions.json">逐候选决定</a> · <a href="outputs/{ident}/bridge_events.json">续接与拒绝原因</a></p></section>')
    shutil.copy2(ROOT/'outputs/G05_01/comparison.jpg',ROOT/'COMPLETE_PREVIEW.jpg');roi=(145,170,350,420);board=Image.new('RGB',(2000,560),'#f3f1eb');d=ImageDraw.Draw(board)
    for j,(name,title) in enumerate([('input','原图局部'),('previous','上一轮'),('complete','完整模式候选'),('sparse','可选简稿'),('changes','蓝：补回；紫：续接')]):
        im=Image.open(ROOT/'outputs/G05_01'/(name+'.png')).convert('RGB').crop(roi).resize((400,488),Image.Resampling.LANCZOS if name=='input' else Image.Resampling.NEAREST);d.text((j*400+4,6),title,font=FONT,fill='black');board.paste(im,(j*400,50))
    board.save(ROOT/'MOUNTAIN_DETAIL.jpg',quality=96)
    treechecks=[]
    for r in read(STRUCTURE/'summary.json')['tree_roi_checks']:
        ident=r['id'];x,y,xx,yy=r['roi'];a=np.asarray(Image.open(ROOT/'outputs'/ident/'previous.png'))[y:yy,x:xx];b=np.asarray(Image.open(ROOT/'outputs'/ident/'complete.png'))[y:yy,x:xx];treechecks.append(dict(id=ident,box=r['roi'],exactly_preserved=bool(np.array_equal(a,b)),changed_pixels=int((a!=b).sum()),scope='Audit box may also include neighboring mountain, not an inference mask'))
    summary=dict(images=len(records),gpu_updates_executed=2400,retained_fit_updates=1200,fit_device=fit['device'],original_training_labels=fit['original_training_labels'],training_rows_with_fading=fit['training_rows'],reviewed_candidates=fit['reviewed_candidates'],unknown_excluded=fit['unknown_excluded'],all_images_reused=True,complete_count_budget=None,added_candidates=sum(r['accepted_candidates'] for r in records),new_ridge_candidates=sum(r['accepted_new_ridges'] for r in records),connections=sum(r['accepted_connections'] for r in records),bridge_pixels=sum(r['bridge_pixels'] for r in records),old_deleted_pixels=sum(r['previous_deleted_pixels'] for r in records),protected_exact=all(r['protected_exact'] for r in records),tree_roi_audit=treechecks,previous_budget_reference_diagnostic=diag['previous_budget_blocked'],full_image_accuracy=None,default_entry_unchanged=True,robot_connected=False,limits=['Complete mode removes count and straightness budgets; it is not a guarantee all contours are complete.','Partial references are assistant nonexpert reviews of existing proposals on reused scenes.','Two trajectory-negative labels revised to unknown after source review; original failure and revisions preserved.','Faint calibration has only one eligible negative and no eligible positive: sensitivity not established.','Some real faded lines, sharp junction identities and closed-contour gaps remain unresolved.','New duplicate/misconnected lines may remain outside annotated regions.','No learned artist-stroke relation identity head; curve connections use evidence rules.']);jsave(ROOT/'summary.json',summary)
    tab=[]
    for r in diag['records']:tab.append(f'<tr><td>{r["image"]}</td><td>{r["positive"]}</td><td>{r["mean_reference_path_coverage_before"]:.1%} → {r["mean_reference_path_coverage_after"]:.1%}</td><td>{r["curved_covered_before"]} → {r["curved_covered_after"]} / {r["curved_positive"]}</td><td>{r["faint_covered_before"]} → {r["faint_covered_after"]} / {r["faint_positive"]}</td><td>{r["negative_candidates_explicitly_added"]}</td></tr>')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>山体完整提取与可选简稿</title><style>body{font:17px system-ui;line-height:1.75;max-width:2020px;margin:25px auto;padding:0 18px;background:#f3f1eb;color:#222}img{width:100%;background:white}aside{background:#fff0cf;padding:18px}section{margin:32px 0;border-top:1px solid #ccc}table{border-collapse:collapse;background:white}td,th{padding:9px;border:1px solid #bbb}a{color:#175b80}</style>
<h1>完整提取候选与疏朗简稿分开</h1><aside>“完整模式”指不再按皴线数量和全局直顺程度删线，不代表轮廓已全部正确。所有31张都是旧图复测，不是新图盲测；正式入口未替换，未连接机械臂。</aside>
<h2>本轮改动</h2><p>取消完整提取阶段的皴线数量硬上限；整条曲线以局部走向和双侧墨迹检查，不要求弯折主线的端点距离接近线长。保留已选线和树木保护区，新增墨迹支撑不足、窄回折或重复中心线继续待定。</p><p>加入按局部噪声估计的多尺度灰度/RGB墨带中心候选；这是独立候选检测，不能把其输出直接当标准答案。沿原图墨迹寻找12像素以内的方向连续短曲线，要求端点相互唯一，拒绝触碰旁边笔画、长空白和新增闭环；仍不能确认画家单笔身份。数量限制只出现在可选简稿。</p>
<h2>同图四列对照</h2><img src="COMPLETE_PREVIEW.jpg"><h2>固定山体局部与补线位置</h2><img src="MOUNTAIN_DETAIL.jpg"><h2>淡线与弯折的局部核对</h2><p>前三处为训练参考，最后一处为旧图回归参考，均不是新图盲测；蓝线标出助手非专家参考候选的位置，不代表已确认的精确笔触中心。</p><img src="LOCAL_CASES.jpg">
<h2>训练与失败记录</h2><p>RTX 5060 Laptop实际执行两次专项拟合，各1200次更新，共2400次。最终保留第二次1200次更新的三个小候选网络；91个原始训练标签来自3张母图，加两档RGB淡化变体后273行，不是273幅新画，也不是完整像素网络重训。参考共245个候选，78个未知不进损失。</p><p>首轮因误收候选，校准只能关闭自动接受，保存在 iterations/r1。复核发现G05 ridge_238与G02 ridge_783的真实墨迹存在，但是否一笔有歧义，原先直接判负过于确定，已改为未知；没有改成正例，原标签和修订理由完整保留。因此不能把新旧标签下的误检数直接当改善幅度。</p><p>最终按对比度分别校准普通线与淡线。淡线组只有1个通过几何检查的反例、没有通过的正例，尚未建立淡线灵敏度；淡线输出仍是实验候选。模型单独高分依旧不能当正确线条保证，整条曲线还需源图/几何检查。</p>
<h2>部分参考诊断</h2><p>覆盖是参考候选点在1.5像素缩放容差内靠近输出的比例；不是精确中心线准确率。参考从候选中选择，有偏差；没有覆盖的真实线仍可能很多。弯曲参考以累计转向划分；淡线参考以原图横截面对比度小于0.03划分。表内不确定项没有当反例。</p><table><tr><th>旧图</th><th>正参考数</th><th>平均路径覆盖</th><th>弯曲参考覆盖≥90%</th><th>淡线参考覆盖≥90%</th><th>明确加入的已标反例</th></tr>'''+''.join(tab)+'''</table>
<h2>仍未完成</h2><p>大转折、淡墨、粗墨交汇与闭合轮廓仍可能漏提或误接。全图误线没有完整真值，不能用线变多/变少、闭环数或局部参考通过来宣布已解决。改善来自候选检测、筛选约束与模型配合，尚未证明模型相对新规则的独立增益。原生N02没有可靠山体分区，继续保持原样。树木保护区不变不等于所有树木已完整识别。笔宽、浓淡和实机效果尚未标定。</p>
<p><a href="summary.json">汇总与限制</a> · <a href="training_protocol.json">最终训练协议</a> · <a href="inference_protocol_r2_bucket.json">最终推理协议</a> · <a href="bucket_calibration.json">淡线校准样本不足说明</a> · <a href="partial_diagnostic.json">局部覆盖明细</a> · <a href="partial_references.json">部分参考</a> · <a href="reference_revision_audit.json">标签修订</a> · <a href="iterations/r1/fit/evaluation.json">首轮失败记录</a> · <a href="behavior_checks.json">弯曲/淡线/空白/平行线检查</a></p>
<h2>全部31张已有图</h2>'''+''.join(entries)
    (ROOT/'index.html').write_text(page,encoding='utf-8');snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True);names=['mountain_complete_evidence_20261101.py','train_complete_candidates_20261101.py','calibrate_complete_candidates_20261101.py','mountain_complete_lines_20261101.py','extract_complete_lines_20261101.py','check_complete_lines_20261101.py','report_complete_lines_20261101.py']
    for n in names:shutil.copy2(Path(__file__).with_name(n),snap/n)
    jsave(ROOT/'source_snapshot_manifest.json',{n:digest(snap/n) for n in names});n=check_integrity(records);print('COMPLETE_REPORT',json.dumps({k:v for k,v in summary.items() if k!='previous_budget_reference_diagnostic'},ensure_ascii=False),'VERIFIED',n,flush=True)
if __name__=='__main__':main()
