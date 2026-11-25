# -*- coding: utf-8 -*-
from stable_line_extract import *
import v26_tone_coverage as ev
import xml.etree.ElementTree as ET

def behavior():
    STABLE.mkdir(parents=True,exist_ok=True);tests=[];paper=np.full((90,140,3),245,np.uint8);p=arc([[10,35],[125,35]]);copy=[p.tolist(),p.tolist()];res=process_paths(paper,copy);tests.append(dict(name='exact_retrace_removed',passed=len(res['paths'])==1));res=process_paths(paper,[p.tolist(),(p+[0,2]).tolist()]);tests.append(dict(name='near_parallel_strokes_preserved',passed=len(res['paths'])==2));faint=paper.copy();cv2.line(faint,(10,35),(125,35),(235,235,235),1);res=process_paths(faint,[p.tolist()]);tests.append(dict(name='faint_line_not_deleted',passed=len(res['paths'])==1))
    t=np.linspace(0,2*np.pi,65);ring=np.c_[65+5*np.cos(t),40+5*np.sin(t)];hollow=paper.copy();cv2.polylines(hollow,[np.rint(ring).astype(np.int32)],True,(15,15,15),1);filled=paper.copy();cv2.circle(filled,(65,40),6,(15,15,15),-1);tests.append(dict(name='white_center_rock_outline_preserved',passed=ink_loop_candidate(ring,hollow) is None));tests.append(dict(name='filled_small_ink_loop_flagged_only_in_trial',passed=ink_loop_candidate(ring,filled) is not None and len(process_paths(filled,[ring.tolist()])['paths'])==1));res=process_paths(paper,[p.tolist()]);tests.append(dict(name='original_endpoints_preserved',passed=np.allclose(np.asarray(res['paths'][0])[[0,-1]],p[[0,-1]])));jsave(STABLE/'behavior_checks.json',dict(tests=tests,passed=all(t['passed'] for t in tests)))
    if not all(t['passed'] for t in tests):raise RuntimeError('Conservative behavior check failed')
    print('STABLE_BEHAVIOR',tests,flush=True)

def metric_case(r,rgb,paths):
    refs=read(r['reference_file'])['paths'];z=np.load(r['mask_file']);return ev.evaluate_paths(refs,[arc(p) for p in paths],z['valid'],z['bg'],rgb)

def benchmark():
    behavior();mf=read(V31/'manifest.json');parts={};input_hashes=[]
    for split in ['calibration','development']:
        records=[]
        for r in mf[split]:
            rgb=np.asarray(Image.open(r['raw']).convert('RGB'));source=V31/'initial_paths'/(r['id']+'.json');base=read(source)['paths'];rec=dict(id=r['id'],metrics={},actions={});input_hashes.append(dict(file=str(source),sha256=digest(source)))
            for mode in ['baseline','compact','ink_marks_trial']:
                res=process_paths(rgb,base,mode);rec['metrics'][mode]=metric_case(r,rgb,res['paths']);rec['actions'][mode]=dict(paths=len(res['paths']),events=len(res['audit']),ink_mark_hypotheses=len(res['ink_mark_hypotheses']));dest=STABLE/'benchmark'/mode/r['id'];save_outputs(dest,rgb,res,dict(mode=mode,original_v31_path_file=str(source),not_real_accuracy_gold=True,trial_mode_not_deployed=mode=='ink_marks_trial'));ET.parse(dest/'lines.svg')
            records.append(rec);print('STABLE_BENCH',split,r['id'],rec['actions'],flush=True)
        keys=['point_coverage_3px','mean_best_single_path_coverage','path_count','duplicate_point_fraction','mixed_reference_paths_flagged','paper_path_density'];s={}
        for mode in ['baseline','compact','ink_marks_trial']:
            s[mode]={}
            for key in keys:
                vals=[r['metrics'][mode][key] for r in records if r['metrics'][mode].get(key) is not None];s[mode][key]=float(np.mean(vals)) if vals else None
        parts[split]=dict(summary=s,records=records);jsave(STABLE/(split+'_benchmark.json'),parts[split])
    b=parts['calibration']['summary']['baseline'];p=parts['calibration']['summary']['compact'];checks=dict(coverage_retained=p['point_coverage_3px']>=b['point_coverage_3px']-.002,continuity_retained=p['mean_best_single_path_coverage']>=b['mean_best_single_path_coverage']-.002,no_more_paths=p['path_count']<=b['path_count'],no_more_overlap=p['duplicate_point_fraction']<=b['duplicate_point_fraction']+.001,no_more_background=(p['paper_path_density'] or 0)<=(b['paper_path_density'] or 0)+.00005);chosen='compact' if all(checks.values()) else 'baseline';jsave(STABLE/'release_config.json',dict(mode=chosen,calibration_checks=checks,baseline='v31_original_machine_candidates',additive_candidates_disabled=True,ink_mark_trial_disabled=True,new_training_steps=0,complex_stroke_problem_fully_solved=False));jsave(STABLE/'frozen_benchmark_inputs.json',dict(files=input_hashes));print('STABLE_SELECTED',chosen,checks,json.dumps(parts['development']['summary']),flush=True)

def report():
    import v28_stroke_relations as old
    lockfiles=[ev.INIT,old.ROOT/'calibration.json']+list(old.ROOT.rglob('*.pt'))+[Path(__file__).parent/n for n in ['v28_stroke_relations.py','v27_separation_faint.py','v26_tone_coverage.py','v24_train.py','ink_center_v16.py']];jsave(STABLE/'baseline_lock.json',dict(files=[dict(path=str(p),sha256=digest(p)) for p in lockfiles],locks_existing_baseline_not_new_training=True))
    cfg=read(STABLE/'release_config.json');dev=read(STABLE/'development_benchmark.json');mf={r['id']:r for r in read(V31/'manifest.json')['development']};board=Image.new('RGB',(1600,1200),'#f3f0e8');draw=ImageDraw.Draw(board);draw.text((5,5),'固定基线提取：同图黑白对照；无新增补线；粗墨点转换仅是未采用试验',font=FONT,fill='black')
    from v32_experiment import paste
    for i,key in enumerate(['R04','H02','X03','N11']):
        rgb=np.asarray(Image.open(mf[key]['raw']).convert('RGB'));active=cfg['mode'];panels=[('原图',rgb),('v31原候选',np.asarray(Image.open(STABLE/'benchmark'/'baseline'/key/'lines.png'))),('当前默认 '+active,np.asarray(Image.open(STABLE/'benchmark'/active/key/'lines.png'))),('当前默认叠加',np.asarray(Image.open(STABLE/'benchmark'/active/key/'overlay.png')))]
        for j,(name,a) in enumerate(panels):draw.text((j*400+5,35+i*290),key+' '+name,font=SMALL,fill='black');paste(board,a,j*400+5,60+i*290,390,250)
    board.save(STABLE/'BASELINE_COMPARISON.jpg',quality=97)
    table='| 同一批10张旧开发图，弱参考辅助指标 | 参考点覆盖 | 单路径覆盖 | 平均路径数 | 重叠代理 |\n|---|---:|---:|---:|---:|\n'
    for mode,label in [('baseline','v31原候选'),('compact','保守精简'),('ink_marks_trial','未采用：小墨环转墨点假设')]:
        x=dev['summary'][mode];table+=f"| {label} | {x['point_coverage_3px']:.2%} | {x['mean_best_single_path_coverage']:.2%} | {x['path_count']:.1f} | {x['duplicate_point_fraction']:.2%} |\n"
    centerfile=STABLE/'development_center_benchmark.json'
    if centerfile.exists():
        x=read(centerfile)['summary']['center_guarded'];table+=f"| 有界强墨带中心校正 | {x['point_coverage_3px']:.2%} | {x['mean_best_single_path_coverage']:.2%} | {x['path_count']:.1f} | {x['duplicate_point_fraction']:.2%} |\n"
    text=f'''# 固定基线的离线线条提取入口

默认模式：{cfg['mode']}。固定使用v31时期的主线候选生成器，停用v32–v34新增候选叠加。baseline完整保留原中心线；compact只删除几乎同位置、同方向的重复描线，并以0.45px误差简化折线。center_guarded只校正双侧边界清晰的孤立强墨带，位移≤1.5px，保持端点、路径数量和分笔；淡线、近平行与交汇处保留基线。任何保留检查不合格就回到baseline。简化校准检查：{cfg['calibration_checks']}。中心校正检查：{cfg.get('center_checks','尚未运行')}。

{table}

本轮是恢复稳定基线、打包直接使用入口，并检查有界的局部中心校正；没有新神经训练，也没有证据证明已彻底解决复杂皴线。上述参考仍含非专家粗标注，数值不是整图准确率。小墨环转墨点仅作试验，普通提取不可启用；不把不确定墨块当作已知落笔动作。

## 使用

将图片拖到 F:/山河绘/山河绘/拖入图片提取线条.cmd，提取后自动打开本地预览；或运行 F:/山河绘/山河绘/scripts/run_stable_line_extract.ps1 -ImagePath 图片绝对路径。GPU运行冻结主线模型，结果统一写入 E:/Codex_Workspace_Data/Gongbi_Line_Model_20261022/stable_line_extractor_20261028/outputs/。基线权重、关系参数和主要代码哈希记录在baseline_lock.json，若被修改将拒绝静默继续使用。

每次输出原图、黑白线稿、蓝色叠加、SVG、像素路径和preview.html，并保存未经精简的baseline_paths.json与baseline_lines.png，便于逐图核对。默认最长边640，像素坐标比例写在元数据中。SVG不是机械臂坐标或压力指令；physical_z=null，ready_for_robot=false。未连接、使能或移动机械臂。

整幅青绿画、字迹和纸纹混杂处仍可能出现错误路径，主轮廓与皴线尚不能保证按语义区分。工具不声称识别出画家真实笔顺、皴法类别或完成机器人落笔。稳定入口的目的在于保留有效v31结果，防止后续失败补线继续污染输出。
''';(STABLE/'README.md').write_text(text,encoding='utf-8');snap=STABLE/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['stable_line_extract.py','run_stable_line_extract.ps1','check_stable_line_extract.py']:shutil.copy2(Path(__file__).parent/name,snap/name)
    shutil.copy2(Path(__file__).parent.parent/'拖入图片提取线条.cmd',snap/'拖入图片提取线条.cmd')
    frozen=read(STABLE/'frozen_benchmark_inputs.json')['files'];jsave(STABLE/'benchmark_verification.json',dict(inputs_unchanged=all(digest(x['file'])==x['sha256'] for x in frozen),behavior_checks_passed=read(STABLE/'behavior_checks.json')['passed'],new_training_steps=0,release_mode=cfg['mode'],mechanical_arm_used=False));print('STABLE_REPORT_READY',str(STABLE/'README.md'),flush=True)

def center_benchmark():
    checks=[];im=np.full((100,160,3),245,np.uint8);cv2.line(im,(15,50),(145,50),(40,40,40),3);source=arc([[15,51],[145,51]]);q,a=guarded_centers(im,[source]);checks.append(dict(name='known_center_recovered',passed=bool(bool(a) and np.abs(q[0][15:-15,1]-50).mean()<.6)));checks.append(dict(name='endpoints_not_moved',passed=bool(np.allclose(q[0][[0,-1]],source[[0,-1]]))));p2=source+[0,2];q,a=guarded_centers(im,[source,p2]);checks.append(dict(name='parallel_neighbour_not_shifted',passed=not a));faint=np.full_like(im,245);cv2.line(faint,(15,50),(145,50),(235,235,235),1);q,a=guarded_centers(faint,[source]);checks.append(dict(name='faint_evidence_preserves_baseline',passed=not a));step=np.full_like(im,245);step[50:]=70;q,a=guarded_centers(step,[source]);checks.append(dict(name='photo_step_not_accepted',passed=not a));jsave(STABLE/'center_behavior_checks.json',dict(checks=checks,passed=all(c['passed'] for c in checks)))
    if not all(c['passed'] for c in checks):raise RuntimeError('Center behavior test failed: '+str(checks))
    mf=read(V31/'manifest.json');summaries={}
    for split in ['calibration','development']:
        rows=[]
        for r in mf[split]:
            rgb=np.asarray(Image.open(r['raw']).convert('RGB'));base=read(V31/'initial_paths'/(r['id']+'.json'))['paths'];res=process_paths(rgb,base,'center_guarded');old=metric_case(r,rgb,base);new=metric_case(r,rgb,res['paths']);rows.append(dict(id=r['id'],baseline=old,center_guarded=new,events=res['audit']));save_outputs(STABLE/'benchmark'/'center_guarded'/r['id'],rgb,res,dict(mode='center_guarded',no_new_paths=True));print('CENTER_CHECK',split,r['id'],'adjusted',len(res['audit']),flush=True)
        keys=['point_coverage_3px','mean_best_single_path_coverage','path_count','duplicate_point_fraction','mixed_reference_paths_flagged','paper_path_density'];s={}
        for name in ['baseline','center_guarded']:
            s[name]={key:float(np.mean([r[name][key] for r in rows if r[name].get(key) is not None])) if any(r[name].get(key) is not None for r in rows) else None for key in keys}
        summaries[split]=dict(summary=s,records=rows);jsave(STABLE/(split+'_center_benchmark.json'),summaries[split])
    b=summaries['calibration']['summary']['baseline'];p=summaries['calibration']['summary']['center_guarded'];adjusted=sum(len(r['events']) for r in summaries['calibration']['records']);gate=dict(program_behavior_passed=True,coverage_retained=p['point_coverage_3px']>=b['point_coverage_3px']-.002,continuity_retained=p['mean_best_single_path_coverage']>=b['mean_best_single_path_coverage']-.002,no_more_overlap=p['duplicate_point_fraction']<=b['duplicate_point_fraction']+.001,no_more_mixed=p['mixed_reference_paths_flagged']<=b['mixed_reference_paths_flagged'],no_more_background=(p['paper_path_density'] or 0)<=(b['paper_path_density'] or 0)+.00005,real_calibration_examples_present=adjusted>=3);mode='center_guarded' if all(gate.values()) else 'baseline';jsave(STABLE/'center_release_decision.json',dict(mode=mode,checks=gate,calibration_adjusted_curves=adjusted));cfg=read(STABLE/'release_config.json');cfg['mode']=mode;cfg['center_checks']=gate;cfg['center_displacement_limit_px']=1.5;cfg['endpoints_and_path_count_preserved']=True;jsave(STABLE/'release_config.json',cfg);print('CENTER_SELECTED',mode,gate,json.dumps(summaries['development']['summary']),flush=True)

def final_verify():
    dest=STABLE/'verified_output';doc=read(dest/'paths.json');baseline=read(dest/'baseline_paths.json')['paths'];cfg=read(STABLE/'release_config.json');tree=ET.parse(dest/'lines.svg');polys=tree.findall('.//{http://www.w3.org/2000/svg}polyline');v31=read(V31/'manifest.json');cached=read(V31/'initial_paths'/'R04.json')['paths'];shape=np.asarray(Image.open(dest/'input.png')).shape[:2];checks=dict(behavior_passed=read(STABLE/'behavior_checks.json')['passed'],center_behavior_passed=read(STABLE/'center_behavior_checks.json')['passed'],baseline_files_locked=all(digest(p['path'])==p['sha256'] for p in read(STABLE/'baseline_lock.json')['files']),old_benchmark_paths_unchanged=all(digest(p['file'])==p['sha256'] for p in read(STABLE/'frozen_benchmark_inputs.json')['files']),human_review_unchanged=digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json')==v31['parent_review_sha256'],original_raster_unchanged=digest(LAB/'clean_strokes_v12_20261024'/'final.pt')==v31['old_raster_sha256'],fresh_gpu_baseline_matches_v31_raster=bool(np.array_equal(black_lines(shape,baseline),black_lines(shape,cached))),fresh_mode_matches_release=doc['meta']['mode']==cfg['mode'],fresh_inference_no_reference_reads=doc['meta']['annotation_guard_active'] and doc['meta']['blocked_annotation_opens']==0,no_new_paths=len(doc['paths'])==len(baseline) and doc['new_strokes_added']==0,endpoints_preserved=all(np.allclose(np.asarray(p)[[0,-1]],np.asarray(q)[[0,-1]],atol=1e-5) for p,q in zip(doc['paths'],baseline)),max_center_shift_bounded=all(e.get('max_shift_px',0)<=1.50001 for e in doc['audit']),svg_has_all_paths=len(polys)==len(doc['paths']),not_robot_commands=not doc['ready_for_robot'] and doc['physical_z'] is None and not doc['actually_drawn'])
    for name in ['input.png','lines.png','overlay.png','baseline_lines.png']:
        with Image.open(dest/name) as im:im.verify()
    jsave(STABLE/'FINAL_VERIFICATION.json',dict(checks=checks,passed=all(checks.values()),fresh_input=doc['meta']['input'],gpu=doc['meta']['cuda_device'],mode=cfg['mode'],new_training_steps=0,accuracy_breakthrough_claimed=False))
    if not all(checks.values()):raise RuntimeError('Final verification failed: '+str(checks))
    print('STABLE_FINAL_VERIFIED',checks,flush=True)

if __name__=='__main__':
    {'benchmark':benchmark,'center_benchmark':center_benchmark,'report':report,'final_verify':final_verify}[sys.argv[1]]()
