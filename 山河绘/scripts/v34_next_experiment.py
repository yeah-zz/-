# -*- coding: utf-8 -*-
from v34_candidate_audit import *
from scipy.ndimage import gaussian_filter1d
import urllib.request,heapq

TRIAL=ROOT34/'generation_trial_v2'

def freeze_trial():
    file=TRIAL/'protocol.json'
    if file.exists():return
    jsave(file,dict(normal_projection_offset_px=2.5,normal_states=11,offset_smoothness=.22,offset_prior=.01,profile_strength_normalized_per_cross_section=True,old_saturated_projection_preserved=str(ROOT34/'generation_trial'),revision_reason='known synthetic center recovery check failed under capped global ink energy',max_unbacked_gap_px=6,min_segment_points=20,beam_width=6,max_fragments_per_candidate=4,max_beam_candidates=256,generation_gap_px=24,generation_facing=.55,generation_axis=.55,acceptance_gap_options=[13,24],acceptance_support=.9,acceptance_facing=.8,acceptance_axis=.85,acceptance_margin=.12,no_reference_used_in_candidate_generation=True,oracle_diagnosis_separate=True,no_neural_training=True,reused_development_not_blind=True))

def soft_projection(rgb,proposals):
    a=observed(rgb);sm=cv2.GaussianBlur(a,(0,0),.8);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;offsets=np.linspace(-2.5,2.5,11,dtype=np.float32);trans=.22*(offsets[:,None]-offsets[None])**2;out=[]
    for row in proposals:
        p=arc(row['points']);n=np.c_[-tangent(p)[:,1],tangent(p)[:,0]]
        if len(p)<20:continue
        qq=p[:,None]+n[:,None]*offsets[None,:,None];ink=sample(sm,qq.reshape(-1,2)).reshape(len(p),11);g=sample(gray,qq.reshape(-1,2)).reshape(len(p),11);left=sample(gray,(qq+n[:,None]*7).reshape(-1,2)).reshape(len(p),11);right=sample(gray,(qq-n[:,None]*7).reshape(-1,2)).reshape(len(p),11);valley=np.maximum(np.minimum(left,right)-g,0);strength=.6*ink/np.maximum(ink.max(1,keepdims=True),.03)+.4*valley/np.maximum(valley.max(1,keepdims=True),.02);cost=-strength+.01*offsets[None]**2;dp=cost[0].copy();back=np.zeros((len(p),11),np.int16)
        for k in range(1,len(p)):
            d=dp[:,None]+trans;back[k]=np.argmin(d,axis=0);dp=cost[k]+d.min(axis=0)
        states=np.zeros(len(p),int);states[-1]=int(dp.argmin())
        for k in range(len(p)-1,0,-1):states[k-1]=back[k,states[k]]
        q=qq[np.arange(len(p)),states];supported=(ink[np.arange(len(p)),states]>.018)&(valley[np.arange(len(p)),states]>.002);keep=supported.copy()
        for start,end in runs(~supported):
            if end-start<=6 and start>0 and end<len(p):keep[start:end]=True
        for start,end in runs(keep):
            if end-start>=20:out.append(arc(q[start:end]))
    return out

def make_beam(rgb,base):
    e=Evidence(rgb);rows=[dict(points=arc(p),ids=[i],core=True) for i,p in enumerate(base)];cfg=dict(CONFIGS[1],gap=24.,facing=.55,axis=.55);ends,edges=endpoint_options_relaxed(e,rows,cfg);links={}
    for edge in edges:
        a=(edge['a_row'],edge['a_end']);b=(edge['b_row'],edge['b_end']);links.setdefault(a,[]).append((b,edge['bridge'],edge));links.setdefault(b,[]).append((a,edge['bridge'][::-1],edge))
    candidates={}
    for start,row in enumerate(rows):
        for enter in [0,1]:
            p=row['points'] if enter==0 else row['points'][::-1];states=[dict(last=start,enter=enter,points=p,ids=[start],bridges=[],score=0.)]
            for depth in range(3):
                following=[]
                for state in states:
                    for (other,end),bridge,meta in links.get((state['last'],1-state['enter']),[]):
                        if other in state['ids']:continue
                        p=rows[other]['points'];p=p if end==0 else p[::-1];q=arc(np.concatenate([state['points'],bridge[1:-1],p]))
                        if self_revisit(q):continue
                        ids=state['ids']+[other];new=dict(last=other,enter=end,points=q,ids=ids,bridges=state['bridges']+[meta],score=state['score']+meta['score']);following.append(new);key=tuple(ids) if ids<ids[::-1] else tuple(ids[::-1])
                        if key not in candidates or new['score']>candidates[key]['score']:candidates[key]=new
                states=sorted(following,key=lambda a:a['score'],reverse=True)[:6]
                if not states:break
    result=sorted(candidates.values(),key=lambda a:(a['score']/len(a['bridges'])+.08*(len(a['ids'])-1)),reverse=True)[:256]
    return result

def endpoint_options_relaxed(e,rows,cfg):
    # Generate competing hypotheses at junctions; acceptance remains separate.
    ends=[];result=[]
    for i,row in enumerate(rows):
        p=row['points']
        if len(p)<6:continue
        for end in [0,1]:
            z=p if end else p[::-1];ends.append(dict(row=i,end=end,p=z[-1],out=unit(z[-1]-z[max(0,len(z)-7)])))
    for ia,a in enumerate(ends):
        for ib in range(ia+1,len(ends)):
            b=ends[ib]
            if a['row']==b['row']:continue
            delta=b['p']-a['p'];gap=float(np.linalg.norm(delta))
            if gap<.8 or gap>24:continue
            u=delta/gap;fa=float(u@a['out']);fb=float(-u@b['out']);axis=float(-a['out']@b['out'])
            if min(fa,fb)<.55 or axis<.55:continue
            # Strongly parallel but laterally displaced strokes remain distinct.
            if axis>.95 and max(abs(delta@np.array([-a['out'][1],a['out'][0]])),abs(delta@np.array([-b['out'][1],b['out'][0]])))>2:continue
            c=np.stack([a['p'],a['p']+a['out']*gap/3,b['p']+b['out']*gap/3,b['p']]);t=np.linspace(0,1,max(5,int(gap*2)+1))[:,None];q=arc((1-t)**3*c[0]+3*(1-t)**2*t*c[1]+3*(1-t)*t*t*c[2]+t**3*c[3],.5);support=float(e.support(q).mean())
            if support<.65:continue
            crossing=external_intersection(q,rows,{a['row'],b['row']},2.2);score=(fa+fb+axis)/3-.012*gap+.25*support-.2*crossing;result.append(dict(a=ia,b=ib,a_row=a['row'],b_row=b['row'],a_end=a['end'],b_end=b['end'],bridge=q,gap=gap,facing=min(fa,fb),axis=axis,support=support,crossing=bool(crossing),score=score))
    return ends,result

def serialize_beam(c):
    return dict(points=c['points'].tolist(),ids=c['ids'],score=c['score'],bridges=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in b.items()} for b in c['bridges']])

def generate(partition):
    freeze_trial();cfg=read(CALIBRATION)['selected']['config']
    for r in read(V31/'manifest.json')[partition]:
        dest=TRIAL/'candidates'/r['id'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'pool.json').exists():continue
        rgb=np.asarray(Image.open(r['raw']).convert('RGB'));base=read(RESULTS/cfg['name']/'pooled'/r['id']/'trajectory.json')['paths'];pool=[]
        for name in ['classical','adapted']:
            pp=read(ROOT/'proposals'/r['id']/(name+'.json'))['proposals'];q=soft_projection(rgb,pp);pool.extend(dict(source='soft_'+name,points=p.tolist()) for p in q)
        beam=make_beam(rgb,base);pool.extend(dict(source='beam',points=c['points'].tolist()) for c in beam);jsave(dest/'pool.json',dict(paths=pool,beam=[serialize_beam(c) for c in beam],reference_labels_used=False,actually_drawn=False,physical_z=None));print('V34_GENERATE',r['id'],'soft',len(pool)-len(beam),'beam',len(beam),flush=True)

def diagnose_trial():
    mf=read(V31/'manifest.json');summaries={}
    for split in ['calibration','development']:
        records=[]
        for r in mf[split]:
            z=np.load(r['mask_file']);valid=z['valid'];refs=read(r['reference_file'])['paths'];pool=[]
            for i,c in enumerate(read(TRIAL/'candidates'/r['id']/'pool.json')['paths']):
                p=dense(arc(c['points']));pool.append(dict(points=p,axis=tangent(p),tree=cKDTree(p),source=c['source'],index=i))
            old=read(ROOT34/'cases'/(r['id']+'.json'));per=[]
            for row in old['records']:
                q=dense(np.asarray(refs[row['reference_id']],np.float32));axis=tangent(q);xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,valid.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,valid.shape[0]-1);known=valid[xy[:,1],xy[:,0]]>0;ref=q[known];ra=axis[known];rt=cKDTree(ref);sc=[score(ref,ra,rt,c,3,valid) for c in pool];ids=[i for i,s in enumerate(sc) if s['geometry_usable']];best=max(ids,key=lambda i:sc[i]['coverage']) if ids else None;before=row['tolerances']['3'];new=sc[best]['coverage'] if best is not None else 0;oldmax=max(before['groups'][k]['usable_coverage'] for k in ['current','raw','post']);per.append(dict(reference_id=row['reference_id'],old_category=before['category'],before_best_usable=oldmax,new_best_usable=new,new_best_source=pool[best]['source'] if best is not None else None,new_best_index=best,combined_complete=bool(max(oldmax,new)>=.9),newly_complete=bool(oldmax<.9 and new>=.9)))
            jsave(TRIAL/'oracle_diagnosis'/(r['id']+'.json'),dict(records=per,oracle_only_not_prediction=True));records.extend(per);print('V34_POOL_DIAG',r['id'],'newly_complete',sum(p['newly_complete'] for p in per),flush=True)
        summaries[split]=dict(references=len(records),complete_before=sum(p['before_best_usable']>=.9 for p in records),complete_with_new_pool=sum(p['combined_complete'] for p in records),newly_complete=sum(p['newly_complete'] for p in records),mean_best_usable_before=float(np.mean([p['before_best_usable'] for p in records])),mean_best_usable_with_new_pool=float(np.mean([max(p['before_best_usable'],p['new_best_usable']) for p in records])),records=records)
    jsave(TRIAL/'oracle_summary.json',dict(results=summaries,reference_assisted_upper_bound_only=True,not_real_extraction_accuracy=True))

def select_beam(base,beam,maxgap):
    options=[]
    for c in beam:
        edges=c['bridges']
        if not edges or any(b['gap']>maxgap or b['support']<.9 or b['facing']<.8 or b['axis']<.85 or b['crossing'] for b in edges):continue
        options.append(c)
    # Membership alternatives with similar average evidence stay uncertain.
    options=sorted(options,key=lambda c:c['score']/len(c['bridges']),reverse=True);used=set();accepted=[]
    for c in options:
        if used.intersection(c['ids']):continue
        avg=c['score']/len(c['bridges']);ambiguous=any(set(b['ids'])!=set(c['ids']) and set(b['ids'])&set(c['ids']) and abs(avg-b['score']/len(b['bridges']))<.12 for b in options)
        if ambiguous:continue
        accepted.append(c);used.update(c['ids'])
    return [p for i,p in enumerate(base) if i not in used]+[c['points'] for c in accepted],accepted

def evaluate_trial(partition):
    cfg=read(CALIBRATION)['selected']['config'];rows=[]
    if partition=='calibration':jsave(TRIAL/'selection_protocol.json',dict(methods=['v33','beam_13','beam_24','dp_whole'],dp_uses_frozen_v33_match_rules=True,eligible_min_single_gain=.005,max_coverage_drop=.005,no_mixed_increase=True,max_duplicate_increase=.005,max_background_density_increase=.0001,development_used_for_selection=False))
    for r in read(V31/'manifest.json')[partition]:
        rgb=np.asarray(Image.open(r['raw']).convert('RGB'));base=read(RESULTS/cfg['name']/'pooled'/r['id']/'trajectory.json')['paths'];doc=read(TRIAL/'candidates'/r['id']/'pool.json');rec=dict(id=r['id'],metrics=dict(v33=metric(rgb,r,base)),accepted={})
        for gap in [13,24]:
            paths,accepted=select_beam(base,doc['beam'],gap);rec['metrics']['beam_'+str(gap)]=metric(rgb,r,paths);rec['accepted'][str(gap)]=len(accepted);jsave(TRIAL/'outputs'/('beam_'+str(gap))/r['id']/'trajectory.json',dict(paths=paths,accepted_candidates=accepted,reference_labels_used=False,physical_z=None,ready_for_robot=False,actually_drawn=False))
        er=Evidence(rgb);rr=[dict(points=np.asarray(p,np.float32),ids=[i],core=True) for i,p in enumerate(base)];pool=[(c['source'],np.asarray(c['points'],np.float32)) for c in doc['paths'] if c['source'].startswith('soft_')];options=full_options(er,rr,pool,cfg);kept,audit=choose_whole(rr,options,cfg);paths=[r['points'].tolist() for r in kept];rec['metrics']['dp_whole']=metric(rgb,r,paths);rec['accepted']['dp_whole']=len(audit);jsave(TRIAL/'outputs'/'dp_whole'/r['id']/'trajectory.json',dict(paths=paths,audit=audit,reference_labels_used=False,physical_z=None,ready_for_robot=False,actually_drawn=False))
        rows.append(rec);print('V34_BEAM_SELECT',partition,r['id'],rec['accepted'],flush=True)
    s=summary(rows,['v33','beam_13','beam_24','dp_whole']);jsave(TRIAL/(partition+'_evaluation.json'),dict(summary=s,records=rows,reused_development_not_blind=True))
    if partition=='calibration':
        b=s['v33'];eligible=[k for k in ['beam_13','beam_24','dp_whole'] if s[k]['point_coverage_3px']>=b['point_coverage_3px']-.005 and s[k]['mean_best_single_path_coverage']>=b['mean_best_single_path_coverage']+.005 and s[k]['mixed_reference_paths_flagged']<=b['mixed_reference_paths_flagged'] and s[k]['duplicate_point_fraction']<=b['duplicate_point_fraction']+.005 and (s[k]['paper_path_density'] or 0)<=(b['paper_path_density'] or 0)+.0001];jsave(TRIAL/'calibration_choice.json',dict(selected=max(eligible,key=lambda k:s[k]['mean_best_single_path_coverage']) if eligible else 'v33',eligible=eligible,default_replace_allowed=False));print('V34_AUTOMATIC_CALIBRATION',json.dumps(s),'eligible',eligible,flush=True)

def checks():
    def rgb(curve,kind='line'):
        im=np.full((100,190,3),245,np.uint8)
        if kind=='line':cv2.polylines(im,[np.rint(curve).astype(np.int32)],False,(90,90,90),3,cv2.LINE_AA)
        if kind=='step':im[50:]=90
        return im
    p=arc([[15,50],[175,50]]);ink=rgb(p);shifted=p+np.array([0,2]);ss=soft_projection(ink,[dict(points=shifted)]);blank=soft_projection(rgb(p,'blank'),[dict(points=p)]);edge=soft_projection(rgb(p,'step'),[dict(points=p)]);beam=make_beam(ink,[p[:65],p[75:]]);accepted,_=select_beam([p[:65].tolist(),p[75:].tolist()],[serialize_beam(b) for b in beam],24);out=dict(center_recovers_toward_ink=bool(ss and np.abs(np.concatenate(ss)[:,1]-50).mean()<1.5),blank_does_not_invent_curve=not blank,photo_step_not_ink_band=not edge,continuous_fragments_generate_whole=bool(beam),automatic_join_reduces_two_to_one=len(accepted)==1);jsave(TRIAL/'checks.json',dict(checks=out,passed=all(out.values())));print('V34_CHECKS',out,flush=True)
    if not all(out.values()):raise RuntimeError('New candidate behavior checks failed')

def fetch_assets():
    folder=ROOT34/'new_sources';folder.mkdir(exist_ok=True);assets=[dict(id='MET_CP29',page='https://www.metmuseum.org/art/collection/search/63226',url='https://collectionapi.metmuseum.org/api/collection/v1/iiif/63226/176175/main-image'),dict(id='UMMA_ROCK',page='https://umma.umich.edu/objects/chrysanthemum-and-rock-from-the-mustard-seed-garden-manual-of-painting-1955-1-67/',url='https://umma.umich.edu/wp-content/uploads/2024/01/51839_ca_object_representations_media_37855_original.jpg'),dict(id='CMA_WANG_GAI',page='https://www.clevelandart.org/art/1975.64.2',url='https://openaccess-cdn.clevelandart.org/1975.64.2/1975.64.2_web.jpg')];records=[]
    for a in assets:
        file=folder/(a['id']+'.jpg')
        try:
            if not file.exists():
                req=urllib.request.Request(a['url'],headers={'User-Agent':'Mozilla/5.0 (offline painting research)'})
                with urllib.request.urlopen(req,timeout=35) as response:
                    data=response.read(15_000_001)
                    if len(data)>15_000_000:raise ValueError('Image exceeds budget')
                file.write_bytes(data)
            im=Image.open(file).convert('RGB');im.thumbnail((800,800));im.save(folder/(a['id']+'_preview.png'));records.append(dict(a,file=str(file),sha256=digest(file),status='downloaded'));print('V34_NEW_SOURCE',a['id'],Image.open(file).size,flush=True)
        except Exception as e:records.append(dict(a,status='failed',error=str(e)));print('V34_SOURCE_ERROR',a['id'],str(e),flush=True)
    jsave(folder/'sources.json',dict(records=records,no_expert_references=True,not_independent_accuracy_test=True,not_used_for_training_or_parameter_selection=True))

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='generate_calibration':generate('calibration')
    elif mode=='generate_development':generate('development')
    elif mode=='diagnose_trial':diagnose_trial()
    elif mode=='eval_calibration':evaluate_trial('calibration')
    elif mode=='eval_development':evaluate_trial('development')
    elif mode=='assets':fetch_assets()
    elif mode=='checks':checks()
    else:raise ValueError(mode)
