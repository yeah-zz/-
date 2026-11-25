# -*- coding: utf-8 -*-
"""Reference-assisted candidate ceiling diagnosis, NEVER a deployed predictor."""
from v33_experiment import *
from stroke_link_v12 import dense

ROOT34=LAB/'candidate_ceiling_diagnosis_v34_20261028'
CATEGORY=dict(current_sufficient='当前路径已覆盖参考',selector_gap='可用候选未进入最终路径',projection_loss='原始候选可用但投影/裁剪后失去',fragmented_candidates='候选有线点但缺少完整路径',missing_pixels='候选线点覆盖不足')

def prepare():
    ROOT34.mkdir(parents=True,exist_ok=True)
    if (ROOT34/'protocol.json').exists():raise FileExistsError('Preserve v34 diagnosis')
    mf=read(V31/'manifest.json');records=[]
    for split in ['calibration','development']:
        for r in mf[split]:
            files=[r['raw'],r['reference_file'],r['mask_file'],str(RESULTS/read(CALIBRATION)['selected']['config']['name']/'pooled'/r['id']/'trajectory.json')]
            for name in ['classical','adapted']:files.extend([str(ROOT/'proposals'/r['id']/(name+'.json')),str(ROOT33/'candidate_cache'/r['id']/(name+'.json'))])
            records.append(dict(id=r['id'],split=split,files=[dict(path=f,sha256=digest(f)) for f in files]))
    jsave(ROOT34/'protocol.json',dict(records=records,distance_tolerances_px=[2,3,5],primary_tolerance_px=3,min_reference_arc_length_px=20,complete_reference_coverage=.9,min_clipped_candidate_precision=.9,min_mean_unsigned_axis_agreement=.8,max_order_backtrack_ratio=.1,oracle_reference_matching_is_diagnostic_only=True,oracle_curves_never_used_as_predictions=True,reused_partial_references_not_expert_pen_lift_gold=True,new_neural_training_updates=0,robot_connected=False,original_review_sha256=digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json'),original_raster_sha256=digest(LAB/'clean_strokes_v12_20261024'/'final.pt')))

def load_pool(key):
    cfg=read(CALIBRATION)['selected']['config'];pool=[]
    def add(source,paths):
        for i,p in enumerate(paths):
            p=dense(arc(p)).astype(np.float32)
            if len(p)<2:continue
            pool.append(dict(source=source,index=i,points=p,tree=cKDTree(p),axis=tangent(p)))
    add('original',read(V31/'initial_paths'/(key+'.json'))['paths'])
    add('v32',read(ROOT/'image_verified'/'adapted'/key/'trajectory.json')['paths'])
    for name in ['classical','adapted']:
        add('raw_'+name,[a['points'] for a in read(ROOT/'proposals'/key/(name+'.json'))['proposals']]);add('projected_'+name,read(ROOT33/'candidate_cache'/key/(name+'.json'))['paths'])
    add('final',read(RESULTS/cfg['name']/'pooled'/key/'trajectory.json')['paths'])
    return pool

def score(ref,ref_axis,ref_tree,c,tol,valid):
    p=c['points']
    if np.any(ref.max(0)<p.min(0)-tol) or np.any(ref.min(0)>p.max(0)+tol):return dict(coverage=0.,precision=0.,axis=0.,backtrack=1.,geometry_usable=False,span=[0,0])
    d,k=c['tree'].query(ref);near=d<=tol;coverage=float(near.mean())
    if near.sum()<2:return dict(coverage=coverage,precision=0.,axis=0.,backtrack=1.,geometry_usable=False,span=[0,0])
    lo,hi=int(k[near].min()),int(k[near].max());part=p[lo:hi+1];xy=np.rint(part).astype(int);xy[:,0]=np.clip(xy[:,0],0,valid.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,valid.shape[0]-1);part=part[valid[xy[:,1],xy[:,0]]>0]
    precision=float((ref_tree.query(part)[0]<=tol).mean()) if len(part) else 0.
    axis=float(np.abs((ref_axis[near]*c['axis'][k[near]]).sum(-1)).mean());order=k[near];sign=1 if order[-1]>=order[0] else -1;delta=np.diff(order)*sign;backtrack=float(np.maximum(-delta,0).sum()/max(1,hi-lo))
    usable=precision>=.9 and axis>=.8 and backtrack<=.1
    return dict(coverage=coverage,precision=precision,axis=axis,backtrack=backtrack,geometry_usable=bool(usable),span=[lo,hi])

def analyze_one(r):
    file=ROOT34/'cases'/(r['id']+'.json')
    if file.exists():return read(file)
    pool=load_pool(r['id']);z=np.load(r['mask_file']);valid=z['valid'];refs=read(r['reference_file'])['paths'];allpoints=np.concatenate([c['points'] for c in pool if c['source']!='final']);tree=cKDTree(allpoints);records=[];short=0
    for rid,raw in enumerate(refs):
        full=dense(np.asarray(raw,np.float32));length=float(np.linalg.norm(np.diff(full,axis=0),axis=1).sum());xy=np.rint(full).astype(int);xy[:,0]=np.clip(xy[:,0],0,valid.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,valid.shape[0]-1);known=valid[xy[:,1],xy[:,0]]>0;ref=full[known]
        if len(ref)<3 or length<20:short+=1;continue
        ra=tangent(full)[known];rt=cKDTree(ref);row=dict(reference_id=rid,length_px=length,valid_points=len(ref),tolerances={})
        for tol in [2,3,5]:
            scores=[score(ref,ra,rt,c,tol,valid) for c in pool];groups={}
            for name,allowed in [('current',{'final'}),('raw',{'original','v32','raw_classical','raw_adapted'}),('post',{'original','v32','projected_classical','projected_adapted'})]:
                ids=[i for i,c in enumerate(pool) if c['source'] in allowed];lib=max(ids,key=lambda i:scores[i]['coverage']);good=[i for i in ids if scores[i]['geometry_usable']];best=max(good,key=lambda i:scores[i]['coverage']) if good else None
                groups[name]=dict(liberal_coverage=scores[lib]['coverage'],usable_coverage=scores[best]['coverage'] if best is not None else 0.,best_candidate=dict(source=pool[best]['source'],index=pool[best]['index'],**scores[best]) if best is not None else None,liberal_best=dict(source=pool[lib]['source'],index=pool[lib]['index'],**scores[lib]))
            union=float((tree.query(ref)[0]<=tol).mean())
            if groups['current']['usable_coverage']>=.9:category='current_sufficient'
            elif groups['post']['usable_coverage']>=.9:category='selector_gap'
            elif groups['raw']['usable_coverage']>=.9:category='projection_loss'
            elif union>=.9:category='fragmented_candidates'
            else:category='missing_pixels'
            row['tolerances'][str(tol)]=dict(category=category,union_coverage=union,groups=groups)
        records.append(row)
    out=dict(id=r['id'],records=records,short_or_invalid_references_excluded=short,pool_counts={s:sum(c['source']==s for c in pool) for s in sorted(set(c['source'] for c in pool))},oracle_uses_reference_answers=True,not_predictor_output=True);jsave(file,out);print('V34_AUDIT',r['id'],'refs',len(records),{cat:sum(a['tolerances']['3']['category']==cat for a in records) for cat in CATEGORY},flush=True);return out

def diagnose():
    mf=read(V31/'manifest.json');results={}
    for split in ['calibration','development']:
        rows=[analyze_one(r) for r in mf[split]];summaries={}
        for tol in ['2','3','5']:
            records=[a['tolerances'][tol] for r in rows for a in r['records']];counts={key:sum(a['category']==key for a in records) for key in CATEGORY};summaries[tol]=dict(reference_chains=len(records),counts=counts,chain_weighted_means={name:float(np.mean([a['groups'][name]['usable_coverage'] for a in records])) for name in ['current','raw','post']},liberal_chain_weighted_means={name:float(np.mean([a['groups'][name]['liberal_coverage'] for a in records])) for name in ['current','raw','post']},image_balanced_usable_means={name:float(np.mean([np.mean([a['tolerances'][tol]['groups'][name]['usable_coverage'] for a in r['records']]) for r in rows if r['records']])) for name in ['current','raw','post']})
        results[split]=dict(summary=summaries,cases=[r['id'] for r in rows]);print('V34_SUMMARY',split,json.dumps(summaries['3']),flush=True)
    jsave(ROOT34/'diagnosis_summary.json',dict(results=results,eligibility_and_metrics_differ_from_formal_v33_metrics=True,oracle_is_not_real_prediction_accuracy=True))

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode=='diagnose':diagnose()
    else:raise ValueError(mode)
