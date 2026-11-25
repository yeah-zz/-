# -*- coding: utf-8 -*-
from v33_experiment import *
import base64,io,subprocess

LABELS=dict(original='原候选',v32='v32核验后输入',filter_only='仅过滤生成杂线',endpoint='过滤＋端点连接',classical='几何完整候选替换＋连接',pooled='神经/几何完整候选替换＋连接')

def url(a):
    im=Image.fromarray(a);b=io.BytesIO();im.save(b,format='JPEG',quality=95);return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()

def colored(shape,paths,members=None):
    out=np.full((*shape,3),255,np.uint8)
    for i,p in enumerate(paths):
        sid=min(members[i]) if members else i;hsv=np.uint8([[[int(sid*41%180),170,175]]]);color=tuple(int(v) for v in cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)[0,0]);cv2.polylines(out,[np.rint(p).astype(np.int32)],False,color,1,cv2.LINE_AA)
    return out

def overlay_changes(rgb,before,out):
    im=rgb.copy();affected=set();removed=[]
    for event in out['audit']:
        if event['event']=='discard_generated_extra':removed.append(event['input_id'])
        elif event['event']=='whole_candidate_replacement':affected.update(event['input_ids'])
        else:affected.update(event['input_ids_a']);affected.update(event['input_ids_b'])
    for i in removed:cv2.polylines(im,[np.rint(before[i]).astype(np.int32)],False,(225,130,20),1,cv2.LINE_AA)
    for p,ids in zip(out['paths'],out['path_input_membership']):
        if set(ids)&affected:cv2.polylines(im,[np.rint(p).astype(np.int32)],False,(30,100,220),1,cv2.LINE_AA)
    return im

def zoom_center(before,out):
    for event in sorted(out['audit'],key=lambda x:x.get('gap',0),reverse=True):
        if event['event']=='endpoint_candidate_join':
            aa=[before[i] for i in event['input_ids_a']];bb=[before[i] for i in event['input_ids_b']];pairs=[(np.linalg.norm(np.asarray(p)[a]-np.asarray(q)[b]),(np.asarray(p)[a]+np.asarray(q)[b])/2) for p in aa for q in bb for a in [0,-1] for b in [0,-1]];return min(pairs,key=lambda x:x[0])[1]
        if event['event']=='whole_candidate_replacement':return np.mean(np.concatenate([np.asarray(before[i]) for i in event['input_ids']]),axis=0)
    for event in out['audit']:
        if event['event']=='discard_generated_extra':return np.mean(before[event['input_id']],axis=0)
    return np.mean(np.concatenate([np.asarray(p) for p in before]),axis=0)

def finish():
    archive=ROOT33/'before_raw_gray_valley_check'
    if not archive.exists():
        archive.mkdir()
        for name in ['REPORT.md','RESULT_COMPARISON.jpg','REPLACEMENT_DETAILS.jpg','INTERACTIVE_COMPARISON.html','verification.json','action_summary.json','deployment_gate.json','artifact_manifest.json']:
            p=ROOT33/name
            if p.exists():shutil.copy2(p,archive/name)
        if (ROOT33/'source_snapshot').exists():shutil.copytree(ROOT33/'source_snapshot',archive/'source_snapshot')
    protocol=read(ROOT33/'protocol.json');cal=read(CALIBRATION);dev=read(ROOT33/'development_raw_band_evaluation.json');cfg=cal['selected']['config'];src=sources();mf=read(V31/'manifest.json');items=[];views={};totals={event:0 for event in ['discard_generated_extra','whole_candidate_replacement','endpoint_candidate_join']};whole_sources={};core_distances=[];membership_ok=True;count_ok=True;image_checks=[];doc_lookup={}
    for partition in ['calibration','development']:
        for r in mf[partition]:
            key=r['id'];rgb=np.asarray(Image.open(r['raw']).convert('RGB'));before=read(src[key]['previous'])['paths'];core=read(src[key]['core'])['paths'];out=read(RESULTS/cfg['name']/'pooled'/key/'trajectory.json');images=dict(original=rgb,original_paths=colored(rgb.shape[:2],core),before=colored(rgb.shape[:2],before),after=colored(rgb.shape[:2],out['paths'],out['path_input_membership']),overlay=overlay_changes(rgb,before,out));dest=ROOT33/'previews'/key;dest.mkdir(parents=True,exist_ok=True)
            for name,a in images.items():Image.fromarray(a).save(dest/(name+'.png'));image_checks.append(dest/(name+'.png'))
            removed=[dict(input_id=e['input_id'],reason=e['reason'],points=before[e['input_id']],status='not_adopted_not_expert_confirmed_noise') for e in out['audit'] if e['event']=='discard_generated_extra'];jsave(dest/'not_adopted_candidates.json',dict(paths=removed,physical_z=None,ready_for_robot=False,actually_drawn=False))
            uncertain=rgb.copy()
            for row in removed:cv2.polylines(uncertain,[np.rint(row['points']).astype(np.int32)],False,(225,130,20),1,cv2.LINE_AA)
            images['uncertain']=uncertain;Image.fromarray(uncertain).save(dest/'not_adopted_overlay.png');image_checks.append(dest/'not_adopted_overlay.png')
            center=zoom_center(before,out);x,y=np.rint(center).astype(int);box=(max(0,x-50),max(0,y-50),min(rgb.shape[1],x+51),min(rgb.shape[0],y+51));board=Image.new('RGB',(1200,390),'#f3f0e8');draw=ImageDraw.Draw(board);draw.text((5,5),key+' 自动选取改动附近局部：原图 / v32输入 / 完整候选替换后',font=FONT,fill='black')
            for j,name in enumerate(['original','before','after']):
                crop=Image.fromarray(images[name]).crop(box).resize((350,350),Image.Resampling.NEAREST);board.paste(crop,(j*400+20,35))
            board.save(dest/'zoom.jpg',quality=97);image_checks.append(dest/'zoom.jpg');views[key]=images;doc_lookup[key]=out
            if partition=='development':
                for event in out['audit']:
                    totals[event['event']]+=1
                    if event['event']=='whole_candidate_replacement':whole_sources[event['source']]=whole_sources.get(event['source'],0)+1
                core_distances.append(out['preservation']);ids=[i for group in out['path_input_membership'] for i in group];membership_ok &= all(ids.count(i)==1 for i in range(len(core))) and len(ids)==len(set(ids));removed=sum(a['event']=='discard_generated_extra' for a in out['audit']);saved=sum(a.get('replaced_path_count',1)-1 for a in out['audit'] if a['event']=='whole_candidate_replacement')+sum(a['event']=='endpoint_candidate_join' for a in out['audit']);count_ok &= len(before)-len(out['paths'])==removed+saved
                row=next(a for a in dev['records'] if a['id']==key);items.append(dict(id=key,images={name:url(a) for name,a in images.items()},zoom=url(np.asarray(board)),metrics=row['metrics'],counts=out['counts'],events=out['audit'],preservation=out['preservation']))
    board=Image.new('RGB',(1600,1200),'#f3f0e8');draw=ImageDraw.Draw(board);draw.text((6,5),'v33 实际离线对照：同一输入旧片段 → 完整候选筛选/替换；蓝=发生替换或连接的输出路径，橙=被过滤生成线',font=FONT,fill='black')
    for i,key in enumerate(['R04','H02','X03','N11']):
        for j,(title,name) in enumerate([('原图','original'),('v32 核验后的输入','before'),('v33 替换后轨迹','after'),('原图叠加改动','overlay')]):
            x=j*400;y=34+i*290;draw.text((x+5,y),key+' '+title,font=SMALL,fill='black');paste(board,views[key][name],x+4,y+25,392,252)
    board.save(ROOT33/'RESULT_COMPARISON.jpg',quality=97);image_checks.append(ROOT33/'RESULT_COMPARISON.jpg')
    details=[]
    for r in mf['development']:
        out=doc_lookup[r['id']]
        events=[e for e in out['audit'] if e['event']=='whole_candidate_replacement']
        if not events:continue
        event=max(events,key=lambda x:x['replaced_path_count']);before=read(src[r['id']]['previous'])['paths'];p=np.concatenate([np.asarray(before[i]) for i in event['input_ids']]);lo=np.maximum(0,np.floor(p.min(0)-12)).astype(int);hi=np.minimum(np.array(views[r['id']]['original'].shape[1::-1]),np.ceil(p.max(0)+13)).astype(int);details.append((r['id'],event,tuple([lo[0],lo[1],hi[0],hi[1]])))
    if details:
        db=Image.new('RGB',(1200,320*min(4,len(details))+36),'#f3f0e8');dd=ImageDraw.Draw(db);dd.text((5,4),'v33 实际接受完整替换的局部；颜色表示路径分组，不表示笔宽或墨色',font=FONT,fill='black')
        for i,(key,event,box) in enumerate(details[:4]):
            for j,(name,title) in enumerate([('original','原图局部'),('before','替换前碎片'),('after','完整候选替换后')]):
                dd.text((j*400+5,36+i*320),key+' '+title,font=SMALL,fill='black');im=Image.fromarray(views[key][name]).crop(box);scale=min(390/im.width,275/im.height);im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.NEAREST);db.paste(im,(j*400+5+(390-im.width)//2,62+i*320+(275-im.height)//2))
        db.save(ROOT33/'REPLACEMENT_DETAILS.jpg',quality=97);image_checks.append(ROOT33/'REPLACEMENT_DETAILS.jpg')
    for path in image_checks:
        with Image.open(path) as im:im.verify()
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>v33 完整候选替换对照</title><style>body{font:16px system-ui;background:#f3f0e8;margin:22px;color:#222}p{max-width:1150px;line-height:1.6}select,input{font:inherit;padding:7px}label{margin-right:15px}section{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:15px}section>div{overflow:auto}canvas{background:#fff;border:1px solid #ccc}pre{white-space:pre-wrap;line-height:1.6}img{max-width:100%}</style><h2>v33 完整候选替换旧碎片：实际离线输出</h2><p>左侧固定为 v32 核验后的输入，右侧为本轮输出。同一旧线尽量沿用原颜色；合并路径使用其最小旧片段编号的颜色。蓝色叠加表示发生替换/连接的整条输出路径，橙色为过滤的生成碎线。未连接机械臂；没有声称恢复画家真实笔顺或正确皴法身份。所有图片都是反复使用的开发图，参考仅覆盖部分线条，指标不是整图准确率。</p><label>图片 <select id="case"></select></label><label>右侧 <select id="view"><option value="after">v33替换后轨迹</option><option value="overlay">改动叠加原图</option><option value="original">原图</option><option value="original_paths">更早的主候选</option></select></label><label>缩放 <input id="zoom" type="range" min="1" max="4" step=".5" value="1"></label><section><div><p>v32输入</p><canvas id="left"></canvas></div><div><p id="title"></p><canvas id="right"></canvas></div></section><pre id="stats"></pre><p>自动选取改动附近的局部放大（原图 / v32输入 / v33输出）；若无连接，可能展示过滤发生处。</p><img id="detail"><script>const rows=DATA;const names=NAMES;const s=document.getElementById('case'),v=document.getElementById('view'),z=document.getElementById('zoom');rows.forEach((r,i)=>s.add(new Option(r.id,i)));async function draw(id,url){const im=new Image();im.src=url;await im.decode();const c=document.getElementById(id);c.width=im.width*z.value;c.height=im.height*z.value;c.getContext('2d').drawImage(im,0,0,c.width,c.height)}function update(){const r=rows[s.value||0];draw('left',r.images.before);draw('right',r.images[v.value]);document.getElementById('title').textContent=r.id+' '+v.selectedOptions[0].text;document.getElementById('detail').src=r.zoom;document.getElementById('stats').textContent=Object.entries(r.metrics).map(([k,m])=>`${names[k]}：参考点覆盖 ${(100*m.point_coverage_3px).toFixed(2)}%，单路径覆盖 ${(100*m.mean_best_single_path_coverage).toFixed(2)}%，路径 ${m.path_count}，背景密度 ${m.paper_path_density===null?'未标注':(100*m.paper_path_density).toFixed(3)+'%'}`).join('\\n')+'\\n阶段路径数：'+JSON.stringify(r.counts)+'\\n改动事件：'+JSON.stringify(r.events)+'\\n原主线几何保留诊断：'+JSON.stringify(r.preservation)}s.onchange=v.onchange=z.oninput=update;update();</script></html>'''
    page=page.replace('DATA',json.dumps(items,ensure_ascii=False)).replace('NAMES',json.dumps(LABELS,ensure_ascii=False));page=page.replace('<option value="original">','<option value="uncertain">未采用候选（未确认全是噪声）</option><option value="original">');(ROOT33/'INTERACTIVE_COMPARISON.html').write_text(page,encoding='utf-8');js=page.split('<script>',1)[1].split('</script>',1)[0];(ROOT33/'viewer_script.js').write_text(js,encoding='utf-8');node=subprocess.run(['node','--check',str(ROOT33/'viewer_script.js')],capture_output=True,text=True,encoding='utf-8');jsave(ROOT33/'viewer_syntax_check.json',dict(passed=node.returncode==0,stderr=node.stderr))
    if node.returncode:raise RuntimeError(node.stderr)
    guard=read(ROOT33/'reference_free_raw_band_check'/'check.json');fresh=read(ROOT33/'reference_free_raw_band_check'/'trajectory.json');checks=dict(synthetic_behavior_checks=read(ROOT33/'synthetic_geometry_checks.json')['all_passed'],all_input_sources_unchanged=all(digest(r['raw'])==r['raw_sha256'] and digest(r['previous'])==r['previous_sha256'] and digest(r['core'])==r['core_sha256'] and all(digest(p['path'])==p['sha256'] for p in r['proposals'].values()) for r in protocol['sources']),human_review_unchanged=digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json')==protocol['original_review_sha256'],original_raster_unchanged=digest(LAB/'clean_strokes_v12_20261024'/'final.pt')==protocol['original_raster_sha256'],v32_neural_weight_unchanged=digest(ROOT/'curve_model.pt')==protocol['v32_model_sha256'],core_membership_preserved_exactly_once=bool(membership_ok),path_count_change_fully_accounted_for=bool(count_ok),annotation_guard_active=guard['guard_active'] and guard['blocked_annotation_opens']==0,reference_free_replay_matches_cached=fresh['paths']==doc_lookup[guard['tested_source_id']]['paths'],reference_free_exercises_actual_replacement=any(a['event']=='whole_candidate_replacement' for a in fresh['audit']),all_previews_verified=len(image_checks)==113+bool(details),viewer_syntax_passed=node.returncode==0)
    jsave(ROOT33/'verification.json',dict(checks=checks,engineering_passed=all(checks.values()),quality_calibration_passed=cal['selected']['quality']['passed'],quality_development_passed=dev['quality']['passed'],core_geometry_diagnostics=core_distances))
    if not all(checks.values()):raise RuntimeError('Integrity check failed')
    jsave(ROOT33/'deployment_gate.json',dict(calibration_passed=cal['selected']['quality']['passed'],development_quality_passed=dev['quality']['passed'],default_replace_allowed=False,robot_connected=False,actually_drawn=False))
    jsave(ROOT33/'action_summary.json',dict(development_totals=totals,accepted_whole_candidate_sources=whole_sources,core_geometry_diagnostics=core_distances,original_vertex_preservation_revision=True,new_neural_updates=0))
    table='| 同一批10张旧开发图 | 参考点覆盖 / 3px | 单路径覆盖 | 平均路径数 | 重叠代理 | 误连代理 | 标注背景内路径密度 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for key in ['original','v32']+METHODS:
        m=dev['summary'][key];table+=f"| {LABELS[key]} | {m['point_coverage_3px']:.2%} | {m['mean_best_single_path_coverage']:.2%} | {m['path_count']:.1f} | {m['duplicate_point_fraction']:.2%} | {m['mixed_reference_paths_flagged']:.2f} | {m['paper_path_density']:.3%} |\n"
    b=dev['summary']['v32'];p=dev['summary']['pooled'];neural_advantage=any(dev['summary']['pooled'][k]!=dev['summary']['classical'][k] for k in KEYS if dev['summary']['pooled'][k] is not None);preserve_min=min(x['core_fraction_within_2px'] for x in core_distances)
    report=f'''# v33 完整候选替换旧碎片实验

本轮完成了候选选择与替换程序及16张固定图对照，**没有新增神经网络训练步数**。候选来自v32已在GPU生成的网络/几何输出；图像证据、路径匹配和选择在CPU运行。选择规则六张校准图评估后冻结，再跑十张旧开发图。当前未替换正式绘画流程、未连接机械臂。

## 实际结果

{table}

相对v32核验后输入，最终候选平均路径数{b['path_count']:.1f}→{p['path_count']:.1f}；参考点覆盖变化{100*(p['point_coverage_3px']-b['point_coverage_3px']):+.2f}个百分点，单路径覆盖变化{100*(p['mean_best_single_path_coverage']-b['mean_best_single_path_coverage']):+.2f}个百分点。

十张共过滤生成碎线{totals['discard_generated_extra']}条，接受完整候选替换{totals['whole_candidate_replacement']}次，接受端点连接{totals['endpoint_candidate_join']}次。减少数量中包含删除，不能全部解释为正确恢复了一笔。接受的完整候选来源计数：{json.dumps(whole_sources,ensure_ascii=False)}。神经/几何混合池与纯几何池的汇总数值存在差异：{neural_advantage}；有差异也不自动代表神经网络更好。

六张校准质量门槛通过：{cal['selected']['quality']['passed']}；开发诊断通过：{dev['quality']['passed']}。校准门槛详情：{json.dumps(cal['selected']['quality']['checks'],ensure_ascii=False)}。没有将仅仅线段减少当作连续性已突破，也没有放宽门槛来部署。

## 替换的依据

1. 保留完整投影候选，不再先剪去靠近旧线的部分。完整曲线必须按同一弧长顺序覆盖至少两条旧碎片，距离、方向和端点都要匹配；沿曲线重叠的区间拒绝合并，防止吞掉近平行笔画。
2. 匹配碎片之间的新路段需要原图墨迹支撑；候选自回绕、其他笔画穿过、多个相近替换方案无明显优势时拒绝替换。
3. 同时设置端点连接对照：两端朝向相对、横向偏移小、间隔受限，并且中间有墨迹时，才构造完整路径。白缝与第三笔交汇拒绝连接。
4. 背景筛选只作用于v32新生成的附加线；原主候选不因背景筛选删除。每条保留、替换或连接后的路径都有旧片段编号与审计记录。
5. 原图灰度必须存在双侧亮于中心的墨带特征（沿法线5/9/14px查找两侧、最低亮度差0.004），防止纸色归一化把照片单侧明暗边缘伪造成墨线。模糊粗墨交汇可能因这项保守检查被拒绝。

所选配置为{cfg['name']}，最大间隔{cfg['gap']}px，方向一致度{cfg['axis']}，匹配距离{cfg['fit']}px。全部自动图像证据，不使用参考描线决定替换。局部墨迹支持仍不能唯一确定画家的笔画身份。

## 如何分清改进来自哪里

表中“仅过滤”说明删除杂线的作用，“端点连接”说明局部续接的作用；完整候选几何池和混合池检验新长曲线有没有额外价值。不能把过滤带来的干净或数量下降，归功于模型学会了正确分笔。跨交汇的真实连续笔画可能因保守拒绝而仍然分开，原来主候选中的错误笔画也没有凭空消失。

主候选的旧编号在输出中均恰好保留一次；各图原主线采样点距最终轨迹2px以内比例最低{preserve_min:.3%}。这是对旧机器路径的几何保留诊断，不是对真实中心线准确度的证明。

## 评估边界与完整性

全部是反复使用的旧校准/开发图，没有新增独立画作或专家复核；参考只覆盖部分可见线，覆盖率不是整幅画准确率。误连、重复和标注背景密度也只是有限范围的代理指标。数据与人工标注、旧基线权重、v32训练权重哈希保持不变。

原灰度墨带核验更保守，也可能排除真实淡线、过宽墨带或邻近笔画。因此未采用的生成候选另存为previews/*/not_adopted_candidates.json，并在页面可切换查看；没有把所有排除项升级成“已确认背景噪声”标签。

初版会重新采样未修改的旧线，已保留在results/与calibration.json；之后改为保留旧顶点，结果保留在results_vertex_preserved/与calibration_vertex_preserved.json。局部放大又揭示一处照片下缘被错误接受为完整候选，随后加入原始灰度双侧墨带核验并重跑；最终结果为results_raw_band_verified/、calibration_raw_band_verified.json和development_raw_band_evaluation.json。质量门槛未放宽，前版预览和报告保留在before_raw_gray_valley_check/。该修正使用了已观察的开发结果，不是新的独立验证，不能据此声称泛化能力。

七组已知几何行为检查通过：连续线可连、真实白缝拒连、错位近平行拒连、第三笔交汇拒连、照片单侧明暗边缘拒连、有序弯曲整候选可替换、生成边框杂线可过滤且主线保留。程序样例通过不代表真实山水分笔已解决。禁止读取标注的审计钩子下已重跑H01推理，实际触发完整候选替换且与缓存结果一致。

未完成实机压力、墨浓、干湿校准，所有轨迹physical_z=null、ready_for_robot=false、actually_drawn=false。训练/模型缓存及大临时目录显式指向E盘，无新依赖安装。

## 查看文件

- RESULT_COMPARISON.jpg：固定R04/H02/X03/N11原图、v32输入、v33输出与改动叠加。蓝色是发生替换/连接的整条输出路径，橙色是过滤掉的生成碎线，不能将蓝色全部理解为新落笔墨迹。
- REPLACEMENT_DETAILS.jpg：确实发生完整候选替换的局部自动截取；颜色表示路径分组，不是墨色或笔宽。展示被接受的操作，不等于专家确认其历史分笔正确。
- INTERACTIVE_COMPARISON.html：十张图放大、局部对照、各阶段指标、输入编号及逐项操作记录；颜色尽量与旧编号一致。
- previews/：全部16张原尺寸预览与局部放大。
- results_raw_band_verified/：最终离线轨迹与审计记录，未启用机械臂。
- action_summary.json、development_raw_band_evaluation.json、calibration_raw_band_verified.json：操作数量、开发评估及冻结选择。
- verification.json：输入/权重/标注哈希、操作计数、主线编号保留、页面语法与图片完整性。

是否继续扩大这条路线，应以真实笔画连续性与误接同时改善为依据。仅凭路径数下降，仍不能宣称已自动复核或准确识别皴法。
'''
    (ROOT33/'REPORT.md').write_text(report,encoding='utf-8');snap=ROOT33/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['v33_replace_core.py','v33_experiment.py','v33_report.py','rock_v33.ps1']:
        source=Path(__file__).parent/name;shutil.copy2(source,snap/name)
        if name.endswith('.py'):ast.parse(source.read_text(encoding='utf-8'))
    jsave(ROOT33/'artifact_manifest.json',dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in [ROOT33/'REPORT.md',ROOT33/'RESULT_COMPARISON.jpg',ROOT33/'INTERACTIVE_COMPARISON.html',CALIBRATION,ROOT33/'development_raw_band_evaluation.json']]))
    print('V33_FINISHED',json.dumps(dict(summary=dev['summary'],totals=totals,whole_sources=whole_sources,engineering_checks=checks,quality_calibration_passed=cal['selected']['quality']['passed'],quality_development_passed=dev['quality']['passed'])),flush=True)
