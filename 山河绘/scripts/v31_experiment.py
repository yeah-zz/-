# -*- coding: utf-8 -*-
from v31_profiles import *
from pathlib import Path
import sys
from PIL import Image,ImageDraw
RESULTS=ROOT/'adaptive_results'

def save_fit(row,variant,result):
    folder=RESULTS/variant/row['id'];folder.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.uint8(np.clip(255*(1-result.pop('rendered')),0,255))).save(folder/'ribbon_reconstruction.png')
    Image.fromarray(np.uint8(np.clip(255*(1-result.pop('before_rendered')),0,255))).save(folder/'ribbon_initialization.png')
    jsave(folder/'trajectory.json',dict(result,coordinates='original_image_pixels',physical_z=None,physical_speed=None,ready_for_robot=False,actually_drawn=False,reference_labels_used=False,expert_brush_parameters=False))
    print('V31_FIT',row['id'],variant,'curves',len(result['paths']),'seconds',round(result['meta']['elapsed_seconds'],1),'error',round(result['meta']['final_photometric_error'],5),flush=True)

def run_fit(partition,steps=100,only=None):
    from v31_render import fit_image
    import gc
    manifest=read(ROOT/'manifest.json');rows=manifest[partition]
    for row in rows:
        if only and row['id']!=only:continue
        rgb=np.asarray(Image.open(row['raw']).convert('RGB'));paths=read(ROOT/'initial_paths'/(row['id']+'.json'))['paths']
        for variant in ['classical','learned']:
            file=RESULTS/variant/row['id']/'trajectory.json'
            if file.exists():
                if read(file)['meta'].get('renderer_revision')==3:continue
                dest=ROOT/'unpacked_smoke'/variant/row['id']
                if not dest.exists():shutil.copytree(file.parent,dest)
            result=fit_image(rgb,paths,variant=variant,steps=steps);save_fit(row,variant,result);gc.collect();torch.cuda.empty_cache()

def packed_check():
    from v31_render import Ribbon,PackedRibbons,render
    row=read(ROOT/'manifest.json')['calibration'][0];rgb=np.asarray(Image.open(row['raw']).convert('RGB'));alpha=observed(rgb);h,w=alpha.shape;scale=min(1.,320/max(h,w));shape=(round(h*scale),round(w*scale));paths=read(ROOT/'initial_paths'/(row['id']+'.json'))['paths'];rr=torch.nn.ModuleList([Ribbon(p,alpha,scale,shape,None,'classical') for p in paths]);packed=PackedRibbons(rr);a,ao,_=render(rr,shape);b,bo,_=render(packed,shape);ra=torch.stack([r.regularization() for r in rr]).mean();rb=packed.regularization();(a.mean()+ra+.002*ao).backward();(b.mean()+rb+.002*bo).backward();diff=0
    for i,r in enumerate(rr):
        for name in ['shift','radius','darkness']:diff=max(diff,float((getattr(r,name).grad-getattr(packed,name).grad[i,:len(r.base)]).abs().max()))
    out=dict(image_max_absolute_difference=float((a-b).abs().max().detach()),image_max_difference_in_8bit_levels=float((a-b).abs().max().detach())*255,regularization_absolute_difference=float((ra-rb).abs().detach()),parameter_gradient_max_absolute_difference=diff,actual_device='cuda:0',same_equations_not_new_rendering_model=True,tolerance_reason='float32 matrix reduction order; less than 0.1 of one 8-bit gray level',image_tolerance_8bit_levels=.1)
    out['passed']=out['image_max_difference_in_8bit_levels']<.1 and out['regularization_absolute_difference']<1e-6 and diff<1e-6
    if (ROOT/'packed_renderer_check.json').exists():shutil.copy2(ROOT/'packed_renderer_check.json',ROOT/'packed_renderer_initial_strict_check.json')
    jsave(ROOT/'packed_renderer_check.json',out);print('V31_RENDERER_CHECK',out,flush=True)
    if not out['passed']:raise RuntimeError('Batched renderer changed equations')

def network_evaluation():
    from v31_render import classical
    m=load();g=torch.Generator(device='cuda').manual_seed(319901);x,t,known=synthetic(2048,g)
    with torch.no_grad():pred=m(features(x))
    truth=t.cpu().numpy();pp=pred.cpu().numpy();cc=classical(x.cpu().numpy());valid=known.cpu().numpy()>.5;rows={}
    for name,p in [('classical',cc),('learned',pp)]:
        out={}
        for group,mask in [('identifiable',valid),('faint',valid&(truth[:,2]<.15)),('broad',valid&(truth[:,1]>=4))]:
            out[group]=dict(n=int(mask.sum()),center_mae_px=float(np.abs(p[mask,0]-truth[mask,0]).mean()),radius_mae_px=float(np.abs(p[mask,1]-truth[mask,1]).mean()),apparent_alpha_mae=float(np.abs(p[mask,2]-truth[mask,2]).mean()))
        out['presence_accuracy_program_labels']=float(((p[:,3]>.5)==(truth[:,3]>.5)).mean());rows[name]=out
    jsave(ROOT/'procedural_diagnostic.json',dict(seed=319901,records=rows,program_generated_not_real_artwork_accuracy=True,independent_random_draws_not_independent_artwork=True))

def evaluate(partition):
    import v26_tone_coverage as metric
    import v30_public_benchmark as vis
    from ink_eval_v16 import normal_diagnostics
    from stroke_link_v12 import dense
    manifest=read(ROOT/'manifest.json');out=[];image_lookup={};render_lookup={}
    for row in manifest[partition]:
        key=row['id'];rgb=np.asarray(Image.open(row['raw']).convert('RGB'));z=np.load(row['mask_file']);refs=read(row['reference_file'])['paths'];valid=z['valid'];bg=z['bg'];paths=read(ROOT/'initial_paths'/(key+'.json'))['paths'];methods={'initial':dict(paths=paths)}
        for name in ['classical','learned']:methods[name]=read(RESULTS/name/key/'trajectory.json')
        record=dict(id=key,group=row['group'],metrics={})
        for name,doc in methods.items():
            pp=[dense(np.asarray(p,np.float32)) for p in doc['paths']];mm=metric.evaluate_paths(refs,pp,valid,bg,rgb);mm['cross_section_proxies']=normal_diagnostics(refs,metric.raster(pp,rgb.shape[:2]).astype(bool),rgb=rgb,valid=valid);record['metrics'][name]=mm
            folder=RESULTS/name/key;folder.mkdir(parents=True,exist_ok=True);Image.fromarray(vis.line_image(rgb.shape[:2],doc['paths'])).save(folder/'centerlines.png');Image.fromarray(vis.line_image(rgb.shape[:2],doc['paths'],rgb)).save(folder/'overlay.png')
        image_lookup[key]=rgb;out.append(record)
        board=Image.new('RGB',(1600,650),'#f3f0e8');d=ImageDraw.Draw(board);d.text((7,5),key+' 连续曲线与表观笔宽/灰度重建：实际离线输出，不是机械臂实画',font=FONT,fill='black')
        items=[('原图',rgb),('原始候选中心线',np.asarray(Image.open(RESULTS/'initial'/key/'centerlines.png'))),('无网络初始化＋图像拟合',np.asarray(Image.open(RESULTS/'classical'/key/'ribbon_reconstruction.png'))),('训练网络初始化＋图像拟合',np.asarray(Image.open(RESULTS/'learned'/key/'ribbon_reconstruction.png'))),('学习拟合后的中心线',np.asarray(Image.open(RESULTS/'learned'/key/'centerlines.png'))),('中心线红色叠加',np.asarray(Image.open(RESULTS/'learned'/key/'overlay.png'))),('学习初始化，尚未图像拟合',np.asarray(Image.open(RESULTS/'learned'/key/'ribbon_initialization.png')))]
        for i,(title,a) in enumerate(items):d.text((i%4*400+6,35+i//4*300),title,font=SMALL,fill='black');metric.v21.paste(board,a,i%4*400+5,65+i//4*300,390,265)
        (ROOT/'adaptive_comparisons').mkdir(exist_ok=True);board.save(ROOT/'adaptive_comparisons'/(key+'.jpg'),quality=96)
    summary={name:{k:float(np.mean([r['metrics'][name][k] for r in out])) for k in ['point_coverage_3px','mean_best_single_path_coverage','path_count','duplicate_point_fraction','mixed_reference_paths_flagged']} for name in ['initial','classical','learned']}
    previous=ROOT/(partition+'_adaptive_evaluation.json')
    if previous.exists() and not read(previous).get('all_methods_resampled_to_one_px_for_overlap_proxy'):shutil.copy2(previous,ROOT/(partition+'_initial_sparse_overlap_proxy.json'))
    jsave(previous,dict(records=out,summary=summary,all_methods_resampled_to_one_px_for_overlap_proxy=True,partial_references=True,reused_development_not_blind=True,reconstruction_error_not_independent_quality_metric=True))
    if partition=='calibration':
        b=summary['initial'];p=summary['learned'];passed=p['point_coverage_3px']>=b['point_coverage_3px']-.005 and p['mean_best_single_path_coverage']>=b['mean_best_single_path_coverage']+.005 and p['mixed_reference_paths_flagged']<=b['mixed_reference_paths_flagged'] and p['duplicate_point_fraction']<=b['duplicate_point_fraction']+.01
        jsave(ROOT/'calibration_adaptive_gate.json',dict(passed=bool(passed),default_replace_allowed=False,thresholds_frozen_before_development=True,criteria=dict(min_single_path_gain=.005,max_point_coverage_drop=.005,no_mixed_path_increase=True,max_duplicate_increase=.01),summary=summary))
    print('V31_'+partition.upper(),json.dumps(summary),flush=True)
    if partition=='development':
        selected=['R04','H02','X03','N11'];board=Image.new('RGB',(1600,1200),'#f3f0e8');d=ImageDraw.Draw(board);d.text((7,5),'v31 实际离线输出：原图 / 原候选中心线 / 拟合后中心线 / 表观笔宽与灰度重建（非实画）',font=FONT,fill='black')
        for i,key in enumerate(selected):
            items=[('原图',image_lookup[key]),('原候选中心线',np.asarray(Image.open(RESULTS/'initial'/key/'centerlines.png'))),('学习＋拟合后的中心线',np.asarray(Image.open(RESULTS/'learned'/key/'centerlines.png'))),('表观笔宽与灰度重建',np.asarray(Image.open(RESULTS/'learned'/key/'ribbon_reconstruction.png')))]
            for j,(title,a) in enumerate(items):d.text((j*400+6,35+i*290),key+' '+title,font=SMALL,fill='black');metric.v21.paste(board,a,j*400+5,65+i*290,390,250)
        board.save(ROOT/'RESULT_COMPARISON_ADAPTIVE.jpg',quality=97)

def adaptive_protocol():
    file=ROOT/'adaptive_curve_protocol.json'
    if file.exists():return
    jsave(file,dict(reason='source-to-cubic conversion was inaccurate before image fitting',knots_selected_by_machine_source_polyline_error_only=True,source_polyline_max_approximation_error_px=.75,maximum_knots=96,all_original_optimization_and_gate_parameters_unchanged=True,old_results_preserved=str(ROOT/'results'),new_results=str(RESULTS),no_neural_retraining=True,development_is_reused_not_blind=True))

def predict_one(source,dest):
    import os
    guard=dict(blocked_annotation_opens=0)
    def guard_labels(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(args[0]).replace('\\','/').lower();name=path.rsplit('/',1)[-1]
            if '/labels/' in path or name in ['partial_reference.json','reference.json','reviewed.json','train_profiles.npz','validation_profiles.npz']:
                guard['blocked_annotation_opens']+=1;raise RuntimeError('Annotation reads forbidden during image-only prediction')
    sys.addaudithook(guard_labels)
    import v28_stroke_relations as old
    from v31_render import fit_image
    source=Path(source);dest=Path(dest)
    if not dest.is_absolute() or dest.drive.upper() not in ['E:','F:']:raise ValueError('Use absolute E:/F: output path')
    rgb=np.asarray(Image.open(source).convert('RGB'))
    if not old.exp.prev.v21.v16.monochrome_route(rgb):raise ValueError('This v31 prototype is for monochrome rock studies; colored paintings were not validated')
    model=old.exp.load_model('base');pred=old.exp.prev.prior.predict(model,rgb)[:5];g=old.add_visual(old.build_graph(pred,rgb),model);choice=read(old.ROOT/'calibration.json')['selected'];p=old.probabilities(g,old.models('visual'));accepted,_=old.select(g,p,choice['threshold'],choice['margin']);paths=old.assemble(g,accepted);del model,g,pred;torch.cuda.empty_cache();result=fit_image(rgb,paths,'learned',steps=100);dest.mkdir(parents=True,exist_ok=True)
    for key,name in [('rendered','ribbon_reconstruction.png'),('before_rendered','ribbon_initialization.png')]:Image.fromarray(np.uint8(np.clip(255*(1-result.pop(key)),0,255))).save(dest/name)
    jsave(dest/'trajectory.json',dict(result,coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,reference_labels_used=False))
    jsave(dest/'interface_check.json',dict(source=str(source),source_sha256=digest(source),device='cuda:0',reference_labels_used=False,reference_matching_called=False,inference_reference_guard_active=True,blocked_annotation_opens=guard['blocked_annotation_opens'],local_parameter_network_sha256=digest(ROOT/'profile_model.pt'),candidate_extractor='frozen_v28',pipeline_is_offline_candidate_not_enabled_default=True))
    print('V31_REFERENCE_FREE',str(dest),flush=True)

def representation_diagnostic():
    from v31_render import resample
    from scipy.interpolate import CubicSpline
    from scipy.spatial import cKDTree
    from stroke_link_v12 import dense
    records=[]
    for row in read(ROOT/'manifest.json')['calibration']:
        errors=[]
        for path in read(ROOT/'initial_paths'/(row['id']+'.json'))['paths']:
            p=np.asarray(path,np.float32);length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
            if length<6:continue
            k=min(32,max(4,int(math.ceil(length/14))+2));controls=resample(p,k);q=CubicSpline(np.linspace(0,1,k),controls,bc_type='natural')(np.linspace(0,1,max(12,math.ceil(length)+1)));distance=cKDTree(q).query(dense(p))[0];errors.extend(distance.tolist())
        records.append(dict(id=row['id'],source_to_cubic_p95_px=float(np.percentile(errors,95)),max_px=float(max(errors)),fraction_over_one_px=float((np.asarray(errors)>1).mean())))
    jsave(ROOT/'representation_diagnostic.json',dict(records=records,uses_machine_initial_paths_only=True,uses_reference_labels=False));print('V31_REPRESENTATION',records,flush=True)

def prepare():
    import v31_profiles as p
    if not (ROOT/'manifest.json').exists():p.prepare()
    import v28_stroke_relations as old
    manifest=read(ROOT/'manifest.json');sources={r['id']:r for r in read(old.ROOT/'manifest.json')['graphs'] if r['partition']=='calibration'}
    for row in manifest['calibration']:
        source=sources[row['id']];d=read(source['graph']);g=dict(chains=[np.asarray(q,np.float32) for q in d['chains']],candidates=[],pred=np.load(d['prediction_file'])['prediction'],shape=tuple(d['shape']))
        for c in d['candidates']:g['candidates'].append(dict(a=tuple(c['a']),b=tuple(c['b']),node=c['node'],degree=c['degree'],feature=np.asarray(c['feature'],np.float32)))
        paths=old.assemble(g,old.exp.rel.original_pairings(g));dest=ROOT/'initial_paths'/(row['id']+'.json');jsave(dest,dict(paths=[q.tolist() for q in paths],source='image_generated_cached_graph_geometry_only',reference_labels_used=False))
    for row in manifest['development']:
        source=V30/'results_edge_padding'/'v28'/'raw'/row['id']/'trajectory.json';dest=ROOT/'initial_paths'/(row['id']+'.json');shutil.copy2(source,dest)
    shutil.copy2(Path(__file__),ROOT/'source_at_prepare.py')
    jsave(ROOT/'frozen_inputs.json',dict(training_profiles_sha256=digest(ROOT/'train_profiles.npz'),validation_profiles_sha256=digest(ROOT/'validation_profiles.npz'),manifest_sha256=digest(ROOT/'manifest.json'),initial_path_hashes={p.name:digest(p) for p in (ROOT/'initial_paths').glob('*.json')}))
    print('V31_PREPARED',flush=True)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode=='train':train()
    elif mode=='smoke_fit':run_fit('calibration',steps=100,only=read(ROOT/'manifest.json')['calibration'][0]['id'])
    elif mode=='fit_calibration':run_fit('calibration')
    elif mode=='fit_development':run_fit('development')
    elif mode=='eval_calibration':evaluate('calibration')
    elif mode=='eval_development':evaluate('development')
    elif mode=='eval_network':network_evaluation()
    elif mode=='packed_check':packed_check()
    elif mode=='predict':predict_one(sys.argv[2],sys.argv[3])
    elif mode=='representation':representation_diagnostic()
    elif mode=='adaptive_protocol':adaptive_protocol()
    elif mode=='finalize':
        report=__import__('v31_report');report.viewer();report.finalize()
    else:raise ValueError(mode)
