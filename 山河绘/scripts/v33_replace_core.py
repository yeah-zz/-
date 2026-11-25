# -*- coding: utf-8 -*-
"""Whole-candidate replacement with conservative image and geometry checks."""
from v32_experiment import *
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d

ROOT33=LAB/'whole_candidate_replacement_v33_20261028'
CONFIGS=[dict(name='strict',gap=9.,facing=.82,axis=.90,fit=1.35,cover=.96,margin=.12,extra_support=.75),dict(name='bounded',gap=13.,facing=.74,axis=.85,fit=1.65,cover=.94,margin=.08,extra_support=.65)]

def unit(p):return p/np.maximum(np.linalg.norm(p,axis=-1,keepdims=True),1e-6)
def tangent(p):return unit(np.gradient(p,axis=0))
def runs(mask):
    d=np.diff(np.r_[False,mask,False].astype(np.int8));return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))

class Evidence:
    def __init__(self,rgb):
        self.alpha=observed(rgb);self.smooth=cv2.GaussianBlur(self.alpha,(0,0),.65);self.shape=self.alpha.shape
        self.gray=cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255,(0,0),.65)
        self.ridge,self.axis,_=ridge_field(self.alpha)
    def support(self,p):
        p=arc(p);t=tangent(p);n=np.c_[-t[:,1],t[:,0]];core=np.maximum.reduce([sample(self.smooth,p+n*d) for d in [-.75,0,.75]])
        left=sample(self.smooth,p+n*5);right=sample(self.smooth,p-n*5);peak=np.minimum(core-left,core-right)
        raw_core=np.minimum.reduce([sample(self.gray,p+n*d) for d in [-.75,0,.75]])
        raw_left=np.maximum.reduce([sample(self.gray,p+n*d) for d in [5,9,14]])
        raw_right=np.maximum.reduce([sample(self.gray,p-n*d) for d in [5,9,14]])
        # Paper normalization can turn a one-sided photo step into a false ink band.
        # Require a real gray-level valley with a brighter shoulder on BOTH sides.
        raw_valley=np.minimum(raw_left,raw_right)-raw_core
        supported=(core>=.022)&((peak>=.004)|(core>=.18))&(raw_valley>=.004)
        return supported
    def border(self,p):
        h,w=self.shape;return np.minimum.reduce([p[:,0],p[:,1],w-1-p[:,0],h-1-p[:,1]])

def prepare():
    ROOT33.mkdir(parents=True,exist_ok=True)
    if (ROOT33/'protocol.json').exists():raise FileExistsError('Preserve frozen v33 protocol')
    mf=read(V31/'manifest.json');sources=[]
    for partition in ['calibration','development']:
        for r in mf[partition]:
            f=ROOT/'image_verified'/'adapted'/r['id']/'trajectory.json';b=V31/'initial_paths'/(r['id']+'.json');pp={name:dict(path=str(ROOT/'proposals'/r['id']/(name+'.json')),sha256=digest(ROOT/'proposals'/r['id']/(name+'.json'))) for name in ['classical','adapted']}
            sources.append(dict(id=r['id'],partition=partition,raw=r['raw'],raw_sha256=digest(r['raw']),previous=str(f),previous_sha256=digest(f),core=str(b),core_sha256=digest(b),proposals=pp))
    jsave(ROOT33/'protocol.json',dict(configs=CONFIGS,sources=sources,methods=['filter_only','endpoint','classical','pooled'],all_development_images_previously_reused=True,new_independent_artworks=0,new_neural_updates=0,computation='CPU geometric matching; reuse v32 GPU-generated proposals',quality_criteria=dict(max_input_point_coverage_drop=.005,min_input_single_path_gain=.005,min_path_count_reduction_fraction=.10,no_mixed_increase_vs_original=True,no_duplicate_increase_vs_input=True,no_paper_density_increase_vs_input=True),replacement_uses_machine_paths_not_reference_labels=True,never_remove_core_path_for_paper_filter=True,physical_z=None,robot_connected=False,original_review_sha256=digest(LAB/'cross_sections_v18_20261025'/'expert_cross_sections'/'reviewed.json'),original_raster_sha256=digest(LAB/'clean_strokes_v12_20261024'/'final.pt'),v32_model_sha256=digest(ROOT/'curve_model.pt')))
    print('V33_PREPARED',len(sources),'images',flush=True)

def filter_input(e,paths,ncore,cfg):
    kept=[];audit=[]
    for i,p in enumerate(paths):
        p=np.asarray(p,np.float32);reason=None
        if i>=ncore:
            if float((e.border(p)>5).mean())<.8:reason='near_image_frame'
            elif float(e.support(p).mean())<cfg['extra_support']:reason='insufficient_normal_ink_band'
        if reason:audit.append(dict(input_id=i,event='discard_generated_extra',reason=reason))
        else:kept.append(dict(points=p,ids=[i],core=bool(i<ncore)))
    return kept,audit

def candidate_cache(key,name,e,proposals):
    file=ROOT33/'candidate_cache'/key/(name+'.json')
    if file.exists():return [np.asarray(q,np.float32) for q in read(file)['paths']]
    yy,xx=np.where((e.ridge>=.002)&(e.alpha>=.02));ridge=np.c_[xx,yy].astype(np.float32);result=[]
    if len(ridge):
        tree=cKDTree(ridge);axes=e.axis[yy,xx]
        for row in proposals:
            p=arc(row['points']);t=tangent(p);dist,idx=tree.query(p,k=min(8,len(ridge)))
            if dist.ndim==1:dist=dist[:,None];idx=idx[:,None]
            agreement=np.abs((axes[idx]*t[:,None]).sum(-1));allowed=(dist<=2.5)&(agreement>=.8);cost=np.where(allowed,dist+2*(1-agreement),1e6);choose=np.argmin(cost,axis=1);q=ridge[idx[np.arange(len(p)),choose]];good=cost[np.arange(len(p)),choose]<1e5
            good[1:]&=np.linalg.norm(np.diff(q,axis=0),axis=1)<=2.6
            good&=e.border(q)>4
            for a,b in runs(good):
                if b-a<20:continue
                z=arc(gaussian_filter1d(q[a:b],.55,axis=0))
                if len(z)>=20:result.append(z)
    jsave(file,dict(paths=[q.tolist() for q in result],reference_used=False,projection_max_distance=2.5,no_novelty_clipping=True))
    return result

def self_revisit(p):
    # Reject close returns over a long arc; this is abstention, not proof of pen identity.
    pairs=cKDTree(p).query_pairs(1.3)
    return any(abs(a-b)>16 for a,b in pairs)

def external_intersection(q,rows,covered,margin=2.):
    if len(q)<7:return False
    interior=q[3:-3]
    for i,r in enumerate(rows):
        if i in covered:continue
        p=r['points']
        if np.any(p.max(0)<interior.min(0)-margin) or np.any(p.min(0)>interior.max(0)+margin):continue
        if cKDTree(p).query(interior)[0].min()<margin:return True
    return False

def full_options(e,rows,candidates,cfg):
    result=[]
    for source,q in candidates:
        if self_revisit(q):continue
        tree=cKDTree(q);qt=tangent(q);hits=[]
        for i,row in enumerate(rows):
            p=arc(row['points'])
            if np.any(p.max(0)<q.min(0)-2) or np.any(p.min(0)>q.max(0)+2):continue
            d,k=tree.query(p);axis=np.abs((tangent(p)*qt[k]).sum(-1));good=(d<=cfg['fit'])&(axis>=cfg['axis'])
            if good.mean()<cfg['cover'] or max(d[0],d[-1])>cfg['fit'] or np.percentile(d,99)>2.:continue
            if k[-1]<k[0]:k=k[::-1]
            delta=np.diff(k);backtracking=float(np.maximum(-delta,0).sum()/max(1,k.max()-k.min()))
            if backtracking>.08:continue
            hits.append(dict(i=i,start=int(k.min()),end=int(k.max()),mean_distance=float(d.mean()),p99_distance=float(np.percentile(d,99))))
        if len(hits)<2:continue
        hits.sort(key=lambda h:h['start']);bad=False;gaps=[]
        for a,b in zip(hits,hits[1:]):
            # Overlapping arc intervals may be adjacent parallel lines, never collapse them.
            if a['end']-b['start']>2 or b['start']-a['end']>cfg['gap']:bad=True;break
            u=qt[a['end']];v=qt[b['start']];delta=q[b['start']]-q[a['end']]
            if abs(u@v)>.9 and abs(delta@np.array([-u[1],u[0]]))>max(1.5,.15*np.linalg.norm(delta)):bad=True;break
            gaps.append((a['end'],b['start']))
        if bad:continue
        z=q[hits[0]['start']:hits[-1]['end']+1]
        if len(z)<20 or e.support(z).mean()<.9:continue
        ids={h['i'] for h in hits}
        if external_intersection(z,rows,ids):continue
        # Confirm new gap pixels, not just the already-covered endpoints.
        for a,b in gaps:
            if b-a>2 and e.support(q[a:b+1]).mean()<.9:bad=True;break
        if bad:continue
        drift=float(np.mean([h['mean_distance'] for h in hits]));gain=len(ids)-1;score=10*gain-2*drift-.04*sum(max(0,b-a) for a,b in gaps)
        result.append(dict(points=z,covered=sorted(ids),source=source,score=score,mean_geometry_shift=drift,max_p99_geometry_shift=max(h['p99_distance'] for h in hits)))
    return result

def choose_whole(rows,options,cfg):
    # Distinct fragment-membership alternatives require a margin; duplicate seeds do not.
    best={}
    for opt in options:
        key=tuple(opt['covered'])
        if key not in best or opt['score']>best[key]['score']:best[key]=opt
    options=sorted(best.values(),key=lambda x:x['score'],reverse=True);taken=set();chosen=[]
    for opt in options:
        if taken.intersection(opt['covered']):continue
        ambiguous=False
        for other in options:
            if other is opt or set(other['covered'])==set(opt['covered']):continue
            if set(other['covered'])&set(opt['covered']) and abs(other['score']-opt['score'])<cfg['margin']*max(1,opt['score']):ambiguous=True;break
        if ambiguous:continue
        chosen.append(opt);taken.update(opt['covered'])
    out=[row for i,row in enumerate(rows) if i not in taken];audit=[]
    for opt in chosen:
        ids=sorted({i for j in opt['covered'] for i in rows[j]['ids']});out.append(dict(points=opt['points'],ids=ids,core=any(rows[j]['core'] for j in opt['covered'])));audit.append(dict(event='whole_candidate_replacement',source=opt['source'],input_ids=ids,replaced_path_count=len(opt['covered']),score=opt['score'],mean_geometry_shift=opt['mean_geometry_shift'],max_p99_geometry_shift=opt['max_p99_geometry_shift']))
    return out,audit

def endpoint_options(e,rows,cfg):
    ends=[]
    for i,row in enumerate(rows):
        p=arc(row['points'])
        if len(p)<6:continue
        for end in [0,1]:
            z=p if end else p[::-1];ends.append(dict(row=i,end=end,p=z[-1],out=unit(z[-1]-z[max(0,len(z)-7)])))
    result=[]
    for ia,a in enumerate(ends):
        for ib in range(ia+1,len(ends)):
            b=ends[ib]
            if a['row']==b['row']:continue
            delta=b['p']-a['p'];gap=float(np.linalg.norm(delta))
            if gap<.8 or gap>cfg['gap']:continue
            u=delta/gap;facea=float(u@a['out']);faceb=float(-u@b['out']);axis=float(-a['out']@b['out'])
            if min(facea,faceb)<cfg['facing'] or axis<cfg['axis']:continue
            if max(abs(delta@np.array([-a['out'][1],a['out'][0]])),abs(delta@np.array([-b['out'][1],b['out'][0]])))>max(1.5,.13*gap):continue
            # Cubic Hermite connector: outward directions at both old endpoints.
            c=np.stack([a['p'],a['p']+a['out']*gap/3,b['p']+b['out']*gap/3,b['p']]);t=np.linspace(0,1,max(5,int(gap*2)+1))[:,None];q=(1-t)**3*c[0]+3*(1-t)**2*t*c[1]+3*(1-t)*t*t*c[2]+t**3*c[3];q=arc(q,.5)
            if e.support(q).mean()<.9 or external_intersection(q,rows,{a['row'],b['row']},margin=2.2):continue
            pa=rows[a['row']]['points'];pb=rows[b['row']]['points'];pa=pa if a['end'] else pa[::-1];pb=pb if not b['end'] else pb[::-1];whole=arc(np.concatenate([pa,q[1:-1],pb]));
            if self_revisit(whole):continue
            score=(facea+faceb+axis)/3-.018*gap;result.append(dict(a=ia,b=ib,a_row=a['row'],b_row=b['row'],a_end=a['end'],b_end=b['end'],bridge=q,score=score,gap=gap))
    return ends,result

def merge_endpoints(e,rows,cfg):
    ends,options=endpoint_options(e,rows,cfg);by_end={i:[] for i in range(len(ends))}
    for opt in options:by_end[opt['a']].append(opt);by_end[opt['b']].append(opt)
    eligible=[]
    for opt in options:
        if all(not any(other is not opt and other['score']>=opt['score']-cfg['margin'] for other in by_end[k]) for k in [opt['a'],opt['b']]):eligible.append(opt)
    parent=list(range(len(rows)));links={};audit=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for opt in sorted(eligible,key=lambda o:o['score'],reverse=True):
        a,b=opt['a_row'],opt['b_row'];ka=(a,opt['a_end']);kb=(b,opt['b_end'])
        if ka in links or kb in links or find(a)==find(b):continue
        parent[find(a)]=find(b);links[ka]=(kb,opt['bridge']);links[kb]=(ka,opt['bridge'][::-1]);audit.append(dict(event='endpoint_candidate_join',input_ids_a=rows[a]['ids'],input_ids_b=rows[b]['ids'],gap=opt['gap'],score=opt['score']))
    visited=set();out=[]
    for i in range(len(rows)):
        if i in visited:continue
        component=[j for j in range(len(rows)) if find(j)==find(i)];start=next((j for j in component if (j,0) not in links or (j,1) not in links),i);enter=0 if (start,0) not in links else 1;parts=[];ids=[];j=start
        while j not in visited:
            visited.add(j);p=rows[j]['points'];parts.append(p if enter==0 else p[::-1]);ids.extend(rows[j]['ids']);exit_key=(j,1-enter)
            if exit_key not in links:break
            (j,enter),bridge=links[exit_key];parts.append(bridge[1:-1])
        out.append(dict(points=np.concatenate(parts),ids=sorted(set(ids)),core=any(rows[k]['core'] for k in component)))
    return out,audit

def replace(rgb,base,ncore,cfg,candidates,method):
    e=Evidence(rgb);rows,audit=filter_input(e,base,ncore,cfg);counts=dict(input_paths=len(base),after_extra_filter=len(rows));options=[]
    if method in ['classical','pooled']:
        allowed=['classical'] if method=='classical' else ['classical','adapted'];pool=[(name,q) for name in allowed for q in candidates[name]];options=full_options(e,rows,pool,cfg);rows,log=choose_whole(rows,options,cfg);audit.extend(log)
    counts['whole_candidate_options']=len(options);counts['after_whole_replacement']=len(rows)
    if method!='filter_only':rows,log=merge_endpoints(e,rows,cfg);audit.extend(log)
    counts['output_paths']=len(rows)
    # Measure preservation of every protected old-core pixel against final machine paths.
    pp=np.concatenate([arc(r['points']) for r in rows]);d=cKDTree(pp).query(np.concatenate([arc(p) for p in base[:ncore]]))[0]
    preservation=dict(core_point_distance_p99=float(np.percentile(d,99)),core_point_distance_max=float(d.max()),core_fraction_within_2px=float((d<=2).mean()))
    return dict(paths=[r['points'].tolist() for r in rows],path_input_membership=[r['ids'] for r in rows],audit=audit,counts=counts,preservation=preservation,reference_labels_used_at_inference=False,physical_z=None,ready_for_robot=False,actually_drawn=False)
