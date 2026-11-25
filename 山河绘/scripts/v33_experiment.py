# -*- coding: utf-8 -*-
from v33_replace_core import *
import sys, os, ast

METHODS=['filter_only','endpoint','classical','pooled']
RESULTS=ROOT33/'results_raw_band_verified'
CALIBRATION=ROOT33/'calibration_raw_band_verified.json'

def synthetic_checks():
    ROOT33.mkdir(parents=True,exist_ok=True);cfg=CONFIGS[1];cases=[]
    def line(a,b):return arc(np.asarray([a,b],np.float32))
    def image(ink):
        rgb=np.full((170,190,3),245,np.uint8)
        for p in ink:cv2.polylines(rgb,[np.rint(p).astype(np.int32)],False,(30,30,30),3,cv2.LINE_AA)
        return rgb
    a=line([15,80],[72,80]);b=line([82,80],[175,80]);whole=line([15,80],[175,80])
    cases.append(('continuous_ink_gap',image([whole]),[a,b],1,False))
    cases.append(('white_gap_abstain',image([a,b]),[a,b],2,False))
    pa=line([15,66],[72,66]);pb=line([82,70],[175,70]);cases.append(('offset_parallel_abstain',image([line([15,66],[175,66]),line([15,70],[175,70])]),[pa,pb],2,False))
    cross=line([77,25],[77,140]);cases.append(('third_stroke_crossing_abstain',image([whole,cross]),[a,b,cross],3,False))
    photo=np.full((170,190,3),245,np.uint8);photo[80:]=100;fa=line([15,82],[72,82]);fb=line([82,82],[175,82]);cases.append(('one_sided_photo_step_abstain',photo,[fa,fb],2,False))
    checks=[]
    for name,rgb,paths,expected,_ in cases:
        out=replace(rgb,paths,len(paths),cfg,dict(classical=[],adapted=[]),'endpoint');ok=len(out['paths'])==expected and out['preservation']['core_fraction_within_2px']>.999;checks.append(dict(name=name,expected_paths=expected,actual_paths=len(out['paths']),preservation=out['preservation'],passed=ok))
    # A curved whole candidate must cover BOTH old fragments in ordered arc intervals.
    x=np.arange(20,170,dtype=np.float32);q=np.c_[x,65+12*np.sin((x-20)/100)];parts=[q[:65],q[72:]];out=replace(image([q]),parts,2,cfg,dict(classical=[q],adapted=[]),'classical');checks.append(dict(name='ordered_curved_whole_replacement',actual_paths=len(out['paths']),whole_replacements=sum(a['event']=='whole_candidate_replacement' for a in out['audit']),passed=len(out['paths'])==1 and any(a['event']=='whole_candidate_replacement' for a in out['audit'])))
    # Filtering generated frame clutter must never erase a protected original core line.
    frame=line([20,167],[175,167]);out=replace(image([whole,frame]),[whole,frame],1,cfg,dict(classical=[],adapted=[]),'filter_only');checks.append(dict(name='generated_frame_noise_removed_core_preserved',passed=len(out['paths'])==1 and out['preservation']['core_fraction_within_2px']>.999))
    jsave(ROOT33/'synthetic_geometry_checks.json',dict(checks=checks,all_passed=all(c['passed'] for c in checks),known_program_geometry_not_real_painting_accuracy=True));print('V33_GEOMETRY_CHECKS',json.dumps(checks),flush=True)
    if not all(c['passed'] for c in checks):raise RuntimeError('Geometry behavior checks failed')

def sources():return {r['id']:r for r in read(ROOT33/'protocol.json')['sources']}

def run_one(r,cfg,method,reference_guard=False):
    source=sources()[r['id']];file=RESULTS/cfg['name']/method/r['id']/'trajectory.json'
    if file.exists():return read(file)
    rgb=np.asarray(Image.open(source['raw']).convert('RGB'));base=read(source['previous'])['paths'];ncore=len(read(source['core'])['paths']);e=Evidence(rgb);cache={}
    for name in ['classical','adapted']:
        cache[name]=candidate_cache(r['id'],name,e,read(source['proposals'][name]['path'])['proposals']) if method in ['classical','pooled'] else []
    out=replace(rgb,base,ncore,cfg,cache,method);jsave(file,out);print('V33_REPLACE',cfg['name'],method,r['id'],out['counts'],'events',len(out['audit']),flush=True);return out

def evaluate_case(r,cfg):
    rgb=np.asarray(Image.open(r['raw']).convert('RGB'));s=sources()[r['id']];base=read(s['core'])['paths'];previous=read(s['previous'])['paths'];row=dict(id=r['id'],metrics=dict(original=metric(rgb,r,base),v32=metric(rgb,r,previous)),counts={},preservation={})
    for method in METHODS:
        out=run_one(r,cfg,method);row['metrics'][method]=metric(rgb,r,out['paths']);row['counts'][method]=out['counts'];row['preservation'][method]=out['preservation']
    return row

def quality(s):
    b=s['original'];p=s['v32'];v=s['pooled'];checks=dict(input_coverage_retained=v['point_coverage_3px']>=p['point_coverage_3px']-.005,continuity_improved=v['mean_best_single_path_coverage']>=p['mean_best_single_path_coverage']+.005,path_count_reduced=v['path_count']<=p['path_count']*.9,no_mixed_increase_vs_original=v['mixed_reference_paths_flagged']<=b['mixed_reference_paths_flagged'],no_duplicate_increase=v['duplicate_point_fraction']<=p['duplicate_point_fraction'],no_paper_density_increase=(v['paper_path_density'] or 0)<=(p['paper_path_density'] or 0));return dict(passed=all(checks.values()),checks=checks)

def calibrate():
    if CALIBRATION.exists():raise FileExistsError('Freeze calibration selection')
    note=ROOT33/'raw_gray_band_revision.json'
    if not note.exists():jsave(note,dict(reason='actual zoom revealed one photo-bottom step accepted as whole brush curve; normalized alpha alone is insufficient',adds_raw_gray_two_sided_valley_check=True,minimum_raw_gray_lift=.004,normal_shoulders_px=[5,9,14],original_quality_thresholds_unchanged=True,previous_calibration=str(ROOT33/'calibration_vertex_preserved.json'),previous_results=str(ROOT33/'results_vertex_preserved'),development_already_seen=True,not_new_blind_validation=True,neural_updates=0))
    options=[]
    for cfg in read(ROOT33/'protocol.json')['configs']:
        records=[evaluate_case(r,cfg) for r in read(V31/'manifest.json')['calibration']];s=summary(records,['original','v32']+METHODS);g=quality(s);v=s['pooled'];score=v['mean_best_single_path_coverage']+.5*v['point_coverage_3px']-.02*v['mixed_reference_paths_flagged']-10*(v['paper_path_density'] or 0)-.001*v['path_count'];options.append(dict(config=cfg,summary=s,records=records,quality=g,score=score));print('V33_CALIBRATION',cfg['name'],json.dumps(s),g,flush=True)
    eligible=[o for o in options if o['quality']['passed']];selected=max(eligible or options,key=lambda o:o['score']);jsave(CALIBRATION,dict(selected=selected,options=options,development_used=False,all_images_are_reused_not_blind=True,default_replace_allowed=False));print('V33_SELECTED',selected['config']['name'],selected['quality'],flush=True)

def development():
    cfg=read(CALIBRATION)['selected']['config'];records=[evaluate_case(r,cfg) for r in read(V31/'manifest.json')['development']];s=summary(records,['original','v32']+METHODS);jsave(ROOT33/'development_raw_band_evaluation.json',dict(records=records,summary=s,config=cfg,quality=quality(s),partial_references=True,reused_not_blind=True,post_hoc_gray_band_revision=True));print('V33_DEVELOPMENT',json.dumps(s),quality(s),flush=True)

def reference_free_check():
    cfg=read(CALIBRATION)['selected']['config'];source=sources()['H01'];guard=dict(blocked=0)
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.fsdecode(args[0]).replace('\\','/').lower();name=p.rsplit('/',1)[-1]
            if '/labels/' in p or name in ['reference.json','partial_reference.json','reviewed.json','reference.npz','evaluation_masks.npz','real_train.npz','real_validation.npz']:
                guard['blocked']+=1;raise RuntimeError('Reference annotation access blocked')
    sys.addaudithook(audit)
    rgb=np.asarray(Image.open(source['raw']).convert('RGB'));base=read(source['previous'])['paths'];ncore=len(read(source['core'])['paths']);e=Evidence(rgb);cache={name:candidate_cache(source['id'],name,e,read(source['proposals'][name]['path'])['proposals']) for name in ['classical','adapted']};out=replace(rgb,base,ncore,cfg,cache,'pooled');dest=ROOT33/'reference_free_raw_band_check';jsave(dest/'trajectory.json',out);jsave(dest/'check.json',dict(guard_active=True,blocked_annotation_opens=guard['blocked'],reference_labels_used=False,tested_source_id=source['id'],source_image_sha256=digest(source['raw']),physical_z=None,ready_for_robot=False,actually_drawn=False,backend='CPU geometric matching of GPU-generated machine proposals'));print('V33_REFERENCE_FREE',out['counts'],guard,flush=True)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode=='checks':synthetic_checks()
    elif mode=='calibrate':calibrate()
    elif mode=='development':development()
    elif mode=='reference_free_check':reference_free_check()
    elif mode=='report':
        import v33_report
        v33_report.finish()
    else:raise ValueError(mode)
