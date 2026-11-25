# -*- coding: utf-8 -*-
from v34_next_experiment import *
import base64,io,subprocess,ast
POOL_CACHE={}

def data_url(a):
    im=Image.fromarray(a);im.thumbnail((650,480));b=io.BytesIO();im.save(b,format='JPEG',quality=93);return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()

def draw_paths(rgb,paths,color,width=1):
    out=rgb.copy()
    for p in paths:
        if len(p)>1:cv2.polylines(out,[np.rint(p).astype(np.int32)],False,color,width,cv2.LINE_AA)
    return out

def diagnostic_panels(r,row):
    rgb=np.asarray(Image.open(r['raw']).convert('RGB'));refs=read(r['reference_file'])['paths'];ref=np.asarray(refs[row['reference_id']],np.float32);d=row['tolerances']['3'];cfg=read(CALIBRATION)['selected']['config'];current=read(RESULTS/cfg['name']/'pooled'/r['id']/'trajectory.json')['paths'];group='post' if d['category']=='selector_gap' else 'raw';chosen=d['groups'][group]['best_candidate'] or d['groups'][group]['liberal_best']
    if r['id'] not in POOL_CACHE:POOL_CACHE[r['id']]=load_pool(r['id'])
    pool=POOL_CACHE[r['id']];candidate=next(c for c in pool if c['source']==chosen['source'] and c['index']==chosen['index']);lo,hi=chosen['span'];curve=candidate['points'][lo:hi+1];boxlo=np.maximum(0,np.floor(ref.min(0)-20)).astype(int);boxhi=np.minimum(np.array(rgb.shape[1::-1]),np.ceil(ref.max(0)+21)).astype(int);box=tuple([*boxlo,*boxhi]);views=[rgb,draw_paths(draw_paths(rgb,current,(50,100,220)),[ref],(235,125,20),1),draw_paths(draw_paths(rgb,[curve],(30,155,65)),[ref],(235,125,20),1)];views=[np.asarray(Image.fromarray(a).crop(box)) for a in views]
    return views,dict(candidate_source=chosen['source'],candidate_index=chosen['index'],candidate_scores=chosen,category=d['category'],category_label=CATEGORY[d['category']],reference_selection_used=True,not_automatic_prediction=True)

def finish():
    rq=read(ROOT34/'reference_quality_audit.json')
    jsave(ROOT34/'reference_usage_policy.json',dict(source_labels_unchanged=True,reference_chains=70,user_reviewed_partial=21,assistant_partial_not_expert=49,assistant_references_not_exact_centerline_gold=True,partial_user_review_not_historical_pen_lift_gold=True,all_existing_oracle_counts_provisional=True,do_not_train_on_oracle_selected_curves_as_expert_truth=True,allowed_use='coarse/partial geometry diagnosis with explicit uncertainty',visual_uncertainties=[dict(id='H01',reference_id=7,issue='rough contour polyline may cross pale or blank regions',reviewer='assistant_visual_not_expert',not_confirmed_label_error=True)],simple_alpha_support_test_does_not_establish_reference_precision=True))
    diagnosis=read(ROOT34/'diagnosis_summary.json');oracle=read(TRIAL/'oracle_summary.json');dev=read(TRIAL/'development_evaluation.json');cal=read(TRIAL/'calibration_choice.json');mf=read(V31/'manifest.json');lookup={r['id']:r for split in ['calibration','development'] for r in mf[split]};items=[];records=[]
    for r in mf['development']:
        doc=read(ROOT34/'cases'/(r['id']+'.json'))
        for row in doc['records']:
            records.append((r,row))
            views,meta=diagnostic_panels(r,row);items.append(dict(id=r['id'],reference_id=row['reference_id'],label=f"{r['id']} / 参考{row['reference_id']} / {meta['category_label']}",images=[data_url(a) for a in views],diagnosis=row,meta=meta))
    representative=[]
    for category in ['selector_gap','projection_loss','fragmented_candidates','missing_pixels']:
        group=[(r,a) for r,a in records if a['tolerances']['3']['category']==category]
        if group:representative.append(max(group,key=lambda x:x[1]['length_px']))
    board=Image.new('RGB',(1500,330*len(representative)+45),'#f3f0e8');draw=ImageDraw.Draw(board);draw.text((5,5),'v34 参考辅助的候选诊断：橙=已有参考，蓝=当前输出，绿=选出的候选；含参考答案，不能当新模型结果',font=FONT,fill='black')
    for i,(r,row) in enumerate(representative):
        views,meta=diagnostic_panels(r,row)
        for j,(title,a) in enumerate(zip(['原图局部','当前输出＋参考','候选上限＋参考'],views)):
            draw.text((j*500+5,40+i*330),r['id']+' '+title,font=SMALL,fill='black');im=Image.fromarray(a);s=min(490/im.width,260/im.height);im=im.resize((max(1,round(im.width*s)),max(1,round(im.height*s))),Image.Resampling.NEAREST);board.paste(im,(j*500+5+(490-im.width)//2,65+i*330+(260-im.height)//2))
        draw.text((5,310+i*330),meta['category_label']+'；参考链编号 '+str(row['reference_id']),font=SMALL,fill='black')
    board.save(ROOT34/'ORACLE_DIAGNOSTIC.jpg',quality=97)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>v34 候选诊断</title><style>body{font:16px system-ui;margin:24px;background:#f3f0e8;color:#222}p{max-width:1200px;line-height:1.6}select{font:inherit;padding:8px;max-width:98%}section{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}section>div{overflow:auto}img{max-width:100%;background:white}pre{white-space:pre-wrap;line-height:1.5}</style><h2>参考辅助的候选诊断——不是新模型预测</h2><p>橙色为已有参考链，蓝色为v33输出，绿色候选由参考答案辅助选取，只用于定位瓶颈。参考并不都是专家真值或真实落笔顺序；未标区域不能当作背景。满足几何条件不等于确认画家的笔画身份。本页的候选上限不能导入机器人执行。</p><select id="select"></select><section><div><p>原图局部</p><img id="a"></div><div><p>当前输出＋参考</p><img id="b"></div><div><p>候选上限＋参考</p><img id="c"></div></section><pre id="detail"></pre><script>const rows=DATA;const s=document.getElementById('select');rows.forEach((r,i)=>s.add(new Option(r.label,i)));function update(){const r=rows[s.value||0];['a','b','c'].forEach((k,i)=>document.getElementById(k).src=r.images[i]);document.getElementById('detail').textContent=JSON.stringify({case:r.id,reference:r.reference_id,diagnostic:r.meta,all_tolerances:r.diagnosis.tolerances},null,2)}s.onchange=update;update();</script></html>'''.replace('DATA',json.dumps(items,ensure_ascii=False));(ROOT34/'INTERACTIVE_DIAGNOSIS.html').write_text(page,encoding='utf-8');(ROOT34/'viewer_script.js').write_text(page.split('<script>',1)[1].split('</script>',1)[0],encoding='utf-8');node=subprocess.run(['node','--check',str(ROOT34/'viewer_script.js')],capture_output=True,text=True,encoding='utf-8');jsave(ROOT34/'viewer_syntax_check.json',dict(passed=node.returncode==0,stderr=node.stderr))
    if node.returncode:raise RuntimeError(node.stderr)
    page=page.replace('<select id="select">','<p><strong>参考来源审计：</strong>70条链中49条来自助手非专家参考，21条来自用户部分复核；不是逐像素中心线或真实笔顺金标准。部分粗轮廓疑似跨过空白，分类只是暂定诊断，不能将全部差距归因于模型。原标签未改写。</p><select id="select">');(ROOT34/'INTERACTIVE_DIAGNOSIS.html').write_text(page,encoding='utf-8')
    main=diagnosis['results']['development']['summary']['3'];new=oracle['results']['development'];caloracle=oracle['results']['calibration'];sources_doc=read(ROOT34/'new_source_runs'/'manifest.json');brush=read(ROOT34/'brush_parameter_preview'/'manifest.json');sensitivity='| 容差 | 当前满足完整几何规则 | 可用候选未选中 | 原候选在投影后失去 | 线点有但无完整候选 | 候选线点不足 |\n|---|---:|---:|---:|---:|---:|\n'
    for tol in ['2','3','5']:
        c=diagnosis['results']['development']['summary'][tol]['counts'];sensitivity+='| '+tol+'px | '+' | '.join(str(c[k]) for k in CATEGORY)+' |\n'
    table='| 真实自动输出，同一批10张旧开发图 | 参考点覆盖 | 单路径覆盖 | 平均路径数 | 重叠代理 | 背景路径密度 |\n|---|---:|---:|---:|---:|---:|\n'
    labels=dict(v33='冻结v33基线',beam_13='探索：多假设候选，最大间隔13px',beam_24='探索：多假设候选，最大间隔24px',dp_whole='探索：连续法线投影＋旧替换器')
    for key,m in dev['summary'].items():table+=f"| {labels[key]} | {m['point_coverage_3px']:.2%} | {m['mean_best_single_path_coverage']:.2%} | {m['path_count']:.1f} | {m['duplicate_point_fraction']:.2%} | {m['paper_path_density']:.3%} |\n"
    source_table='\n'.join(f"- [{r['id']}]({r['page']})：冻结主线模型输出{r['paths']}条路径；与旧训练图最小dHash距离{r['closest_training_dhash_distance']}，不等同证明作品独立。" for r in sources_doc['records'])
    report=f'''# v34 候选瓶颈诊断、生成对照与笔宽灰度预览

本轮实际完成：逐笔候选诊断；连续法线投影/多假设完整候选对照；三张新馆藏来源的冻结模型外观检查；两条固定机器轨迹的笔宽和表观灰度预览。**新神经训练步数为0，没有用参考答案选出的曲线冒充自动预测，也没有替换正式流程或连接机械臂。**

**重要限制：来源复核发现，70条链中49条为助手非专家参考，21条为用户部分复核，不能统称逐像素中心线或真实分笔金标准。部分粗轮廓/折线疑似跨过浅色或空白区域（例如H01参考7），不适合直接作为精确墨线目标。** 打包参考与源标签、像素和尺寸一致，没有发现打包造成的统一缩放错位；未修改原标注。简单局部墨迹阈值没有触发弱支撑标记，也不能因此证明参考精确。以下分类混合了候选与参考误差，只能作为暂定分析；reference_usage_policy.json记录了使用边界。

## 先定位瓶颈

十张旧开发图中，符合参考长度条件的70条链，在3px容差、覆盖≥90%、裁剪候选双向几何一致度≥90%、平均方向一致度≥0.8和顺序回退限制下：

- 当前输出满足完整几何规则：{main['counts']['current_sufficient']}条。
- 有可用候选却未进入当前路径：{main['counts']['selector_gap']}条。
- 原始候选可用，但在投影或裁剪后失去：{main['counts']['projection_loss']}条。
- 候选集合能覆盖多数参考点，但没有合格完整路径：{main['counts']['fragmented_candidates']}条。
- 候选集合的参考点覆盖也不足：{main['counts']['missing_pixels']}条。

在现有参考假设下，结果倾向候选完整性和投影裁剪损失；但参考精度限制使定量归因尚不充分。候选集合含错误曲线，集合覆盖到参考点不证明墨迹检测已经正确；参考链也不都是专家确认的历史一笔。上述数字不是整图准确率。

{sensitivity}

参考辅助的上限穷举使用已知参考选择最佳候选，完全不同于真实自动输出；不能部署。正式v33单路径覆盖57.62%仍沿用旧评估。这里排除了短参考，并增加方向、顺序和双向几何要求，数字口径不能混用。

## 根据诊断做的候选生成试验

连续法线投影在±2.5px范围内联合寻找整段的平滑偏移，允许最多6px弱支撑间隙，减少逐点硬裁剪。多假设生成保留若干碎片连接方案，最多组合4条旧碎片；自动选择仍要求图像支撑、方向可靠、无第三笔交汇并有证据优势。

已知中心的程序检查发现初版全局墨强截断会饱和，导致强墨中心恢复失败；已保留generation_trial/，修正版generation_trial_v2/按每个横截面归一化。中心恢复、空白拒绝、照片单侧边界拒绝、连续碎片生成、自动连接五项检查通过。不是放宽提取质量门槛。

参考辅助上限：校准组可用完整参考从{caloracle['complete_before']}增加到{caloracle['complete_with_new_pool']}；开发组从{new['complete_before']}/{new['references']}增加到{new['complete_with_new_pool']}/{new['references']}，只新增加{new['newly_complete']}条。即使使用参考答案辅助挑选，新候选也没有明显突破。

{table}

校准选择为 **{cal['selected']}**，合格新方案列表为{cal['eligible']}。虽然多假设组合在开发组单路径覆盖略有改善，不能再根据已经观察的开发组选择它来宣称成功；本轮保留v33基线。新方法尚未证明能稳定恢复复杂整笔。

## 新馆藏来源外观检查

{source_table}

图片下载后先按固定框选取山石，再统一转灰度、最长边512；只运行冻结v28主线模型和冻结v33保守端点对照，没有新训练或逐图调参。此部分不是完整新生成器的独立定量测试。它们没有专家参考，也存在共同画谱风格/来源家族的可能性；因此只能称新来源外观检查，不能给出新图准确率。实际图上仍有墨色边界误提取、碎片和内部皴线不准确的问题。馆藏画谱包含木版印刷，不能等同真实毛笔痕迹数据。

## 笔宽与表观灰度同步预览

复用v31已训练墨带网络，在R04和H01各选一条有图像支撑的机器路径，用GPU估计沿线参数；**中心线不优化，中心偏移输出忽略**。只渲染这一条路径，不是完整画作。

R04图像笔宽范围{brush['records'][0]['width_range_px'][0]:.2f}–{brush['records'][0]['width_range_px'][1]:.2f}px；H01范围{brush['records'][1]['width_range_px'][0]:.2f}–{brush['records'][1]['width_range_px'][1]:.2f}px。交汇和邻笔附近可见宽度过估/膨胀，不能把这些值直接转为机械臂下压；表观灰度也不是墨液浓度、含水量或真实干湿。路径方向不代表画家的历史笔顺。没有实际落笔记录，也没有皴法类别或上色模型的新完成项。

## 下一轮的具体依据

下一轮先规范已有参考用途：用户部分复核继续用于粗几何约束，助手粗描保留为弱监督，未知部分不作负例；不能作为精确中心线与真实分笔金标准。应把训练目标转向有可靠顺序和位置依据的整笔实例，并联合验证完整性、误接和双边重复。程序样例、参考辅助选出的曲线和自动结果不能升级成专家真值。这轮发现候选与参考质量都需要处理，不支持继续按同一粗参考强化精确中心线损失。

## 文件与复现

- ORACLE_DIAGNOSTIC.jpg、INTERACTIVE_DIAGNOSIS.html：带参考答案的逐笔诊断，不能当作新模型提取效果。
- diagnosis_summary.json、cases/：2/3/5px容差敏感性与每条参考的诊断。
- reference_quality_audit.json、reference_usage_policy.json：参考来源与尺寸核对、使用边界，未改写用户标注。
- generation_trial_v2/oracle_summary.json：新候选的参考辅助上限；calibration_choice.json与development_evaluation.json为真实自动输出对照。
- NEW_SOURCE_COMPARISON.jpg、new_source_runs/：三张冻结模型外观检查、原图来源、固定裁剪和机器轨迹。
- BRUSH_PARAMETER_PREVIEW.jpg、brush_parameter_preview/：固定路径的图像参数、几何回放与数值曲线。
- 所有机器轨迹physical_z=null、ready_for_robot=false、actually_drawn=false。所有新数据、缓存、图像、日志在E盘，无新依赖安装。参考、旧模型和权重保持不变。
'''
    (ROOT34/'REPORT.md').write_text(report,encoding='utf-8')
    prot=read(ROOT34/'protocol.json');checks=dict(all_frozen_inputs_unchanged=all(digest(f['path'])==f['sha256'] for r in prot['records'] for f in r['files']),original_user_review_unchanged=digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json')==prot['original_review_sha256'],original_raster_unchanged=digest(LAB/'clean_strokes_v12_20261024'/'final.pt')==prot['original_raster_sha256'],candidate_behavior_checks=read(TRIAL/'checks.json')['passed'],new_method_not_promoted=cal['selected']=='v33',viewer_script_syntax=node.returncode==0,diagnosis_chain_counts_accounted_for=sum(main['counts'].values())==main['reference_chains'],new_source_inputs_have_no_gold=all(not r['expert_gold'] for r in sources_doc['records']),brush_geometry_frozen=True)
    cfg=read(CALIBRATION)['selected']['config'];cases={r['id']:r for r in read(V31/'manifest.json')['development']}
    for row in brush['records']:
        doc=read(ROOT34/'brush_parameter_preview'/row['id']/'image_parameters.json');source=read(RESULTS/cfg['name']/'pooled'/row['id']/'trajectory.json')['paths'][row['path_id']];checks['brush_geometry_frozen'] &= bool(np.allclose(arc(source,.75),np.asarray(doc['paths'][0]),atol=1e-5))
    imgs=list(ROOT34.glob('*.jpg'))+list((ROOT34/'brush_parameter_preview').glob('*/*.png'))+list((ROOT34/'new_source_runs').glob('*/*.png'))
    for p in imgs:
        with Image.open(p) as im:im.verify()
    checks['previews_verified']=bool(imgs);jsave(ROOT34/'verification.json',dict(checks=checks,all_passed=all(checks.values()),images_verified=len(imgs),oracle_outputs_not_prediction=True,new_neural_training_steps=0,default_replaced=False))
    if not all(checks.values()):raise RuntimeError('Integrity verification failed')
    snap=ROOT34/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['v34_candidate_audit.py','v34_next_experiment.py','v34_previews.py','v34_reference_quality.py','v34_report.py','rock_v34.ps1']:
        path=Path(__file__).parent/name;shutil.copy2(path,snap/name)
        if name.endswith('.py'):ast.parse(path.read_text(encoding='utf-8'))
    jsave(ROOT34/'artifact_manifest.json',dict(files=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in [ROOT34/'REPORT.md',ROOT34/'ORACLE_DIAGNOSTIC.jpg',ROOT34/'INTERACTIVE_DIAGNOSIS.html',ROOT34/'NEW_SOURCE_COMPARISON.jpg',ROOT34/'BRUSH_PARAMETER_PREVIEW.jpg']]))
    print('V34_FINISHED',json.dumps(dict(diagnostic=main['counts'],candidate_complete_before=new['complete_before'],candidate_complete_after=new['complete_with_new_pool'],automatic_selected=cal['selected'],verification=checks)),flush=True)

if __name__=='__main__':finish()
