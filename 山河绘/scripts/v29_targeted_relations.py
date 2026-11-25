# -*- coding: utf-8 -*-
"""Targeted visible relationship supervision for parallel/thick/texture joins.

Assistant labels are explicit non-expert observations; ambiguity is retained.
No old development-image annotations enter optimizer or validation selection.
"""
from rock_v15_common import LAB,read,jsave,digest,FONT,SMALL
from pathlib import Path
from collections import defaultdict
import json,math,itertools,hashlib,shutil,time
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
import v28_stroke_relations as prior

ROOT=LAB/'targeted_relations_v29_20261027'
TYPES=['parallel','thick','contour_texture']
LABELS={'parallel':'近平行候选','thick':'粗墨交汇候选','contour_texture':'长轮廓/短纹理候选'}

def source_lookup():
    out={}
    for r in read(prior.exp.prev.prior.ROOT/'training_manifest.json')['records']:
        if r['split']=='train':
            z=np.load(r['data']);out[r['id']]=(z['rgb'],z['valid'])
    for r,rgb,refs,valid,bg in prior.validation_cases():out[r['id']]=(rgb,valid)
    return out

def categories(g,ends,indices):
    vectors=np.stack([prior.exp.tangent(g['chains'][e[0]],e[1]==0,span=8) for e in ends]);dot=vectors@vectors.T;np.fill_diagonal(dot,-1);near_parallel=float(dot.max())>.80;features=np.stack([g['candidates'][i]['feature'] for i in indices]);width=float(features[:,19].max()*48);ratio=float(features[:,3].min());long=float(np.expm1(features[:,2].max()*6));short=float(np.expm1(features[:,1].min()*6));return dict(parallel=near_parallel,thick=width>=6 and float(features[:,11].max())>.18,contour_texture=long>=35 and short<=35 and ratio<.55)

def safe_candidate(g,c,valid):
    points=[np.asarray(c['node'])[None]]
    for e in [c['a'],c['b']]:points.append(prior.exp.rel.endpoint_segment(g['chains'][e[0]],e[1])[1])
    q=np.rint(np.concatenate(points)).astype(int);inside=(q[:,0]>=0)&(q[:,0]<valid.shape[1])&(q[:,1]>=0)&(q[:,1]<valid.shape[0])
    return bool(inside.all() and (valid[q[:,1],q[:,0]]>.5).all())

def mine():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'node_manifest.json').exists():raise FileExistsError('Preserve selected annotation nodes')
    lookup=source_lookup();manifest=read(prior.ROOT/'manifest.json');allowed_validation=set(prior.VALIDATION_IDS)|{r['id'] for r,rgb,refs,valid,bg in prior.exp.cases('calibration')};candidates=[];ensemble=prior.models('visual')
    for r in manifest['graphs']:
        if r['view']!='base':continue
        partition='validation' if r['id'] in allowed_validation else 'train'
        if r['partition']=='calibration' and partition!='validation':raise ValueError('Split mapping')
        g=prior.load_graph(r['graph']);rgb,valid=lookup[r['id']];p=prior.probabilities(g,ensemble);nodes=defaultdict(list)
        for i,c in enumerate(g['candidates']):nodes[tuple(c['node'])].append(i)
        for xy,indices in nodes.items():
            unknown=[i for i in indices if g['candidates'][i]['label'] is None and safe_candidate(g,g['candidates'][i],valid)]
            if not unknown:continue
            ends=sorted({e for i in indices for e in [g['candidates'][i]['a'],g['candidates'][i]['b']]})
            if len(ends)>6:continue
            tags=categories(g,ends,indices)
            if not any(tags.values()):continue
            candidates.append(dict(image_id=r['id'],partition=partition,graph=r['graph'],node=list(xy),ends=[list(e) for e in ends],indices=indices,unknown_indices=unknown,tags=tags,priority=float(max(p[i] for i in unknown))+.03*len(unknown)))
    selected=[];image_counts={};taken={};rng=np.random.default_rng(292710)
    for partition,quota in [('train',10),('validation',4)]:
        pool=[r for r in candidates if r['partition']==partition];rng.shuffle(pool);pool.sort(key=lambda r:-r['priority'])
        for kind in TYPES:
            count=0
            for row in pool:
                token=(row['image_id'],tuple(row['node']))
                if not row['tags'][kind] or token in taken or image_counts.get((partition,row['image_id']),0)>=4:continue
                if any(s['image_id']==row['image_id'] and np.linalg.norm(np.asarray(s['node'])-row['node'])<24 for s in selected):continue
                out=dict(row,category=kind,key=('T' if partition=='train' else 'V')+f'{len(selected):03d}');selected.append(out);taken[token]=True;image_counts[partition,row['image_id']]=image_counts.get((partition,row['image_id']),0)+1;count+=1
                if count>=quota:break
    jsave(ROOT/'node_manifest.json',dict(records=selected,parent_manifest_sha256=digest(prior.ROOT/'manifest.json'),all_parent_graphs_sha256={r['graph']:digest(r['graph']) for r in selected},mined_from_train_and_validation_only=True,old_development_images_excluded=True,categories_are_screening_heuristics_not_verified_semantic_labels=True,old_probabilities_used_to_prioritize_not_label=True,robot_connected=False))
    folder=ROOT/'audit';folder.mkdir(exist_ok=True)
    for start in range(0,len(selected),6):
        sub=selected[start:start+6];board=Image.new('RGB',(1320,360*math.ceil(len(sub)/2)+35),'#eeeae1');d=ImageDraw.Draw(board);d.text((5,4),'v29 待判关系原图审计：左原图、右候选分支编号；不显示模型判断',font=FONT,fill='black')
        for k,r in enumerate(sub):
            g=prior.load_graph(r['graph']);rgb,valid=lookup[r['image_id']];x,y=np.rint(r['node']).astype(int);box=(int(x-48),int(y-48),int(x+48),int(y+48));a=Image.fromarray(rgb).crop(box).resize((288,288));b=a.copy();dd=ImageDraw.Draw(b)
            for j,end in enumerate(r['ends']):
                q,seg=prior.exp.rel.endpoint_segment(g['chains'][end[0]],end[1]);q=q[prior.arc(q)<=35];q=(q-[x-48,y-48])*3;color=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[j*31%180,225,190]]]),cv2.COLOR_HSV2RGB)[0,0]);dd.line([tuple(p) for p in q],fill=color,width=3)
                if len(q):
                    p=q[-1];p=np.clip(p+[(-1)**j*5,j%3*5],4,270);dd.rectangle((p[0]-2,p[1]-2,p[0]+20,p[1]+22),fill='white');dd.text(tuple(p),str(j),font=FONT,fill=color)
            xx=k%2*660;yy=35+k//2*360;d.text((xx+5,yy+2),r['key']+' '+r['image_id']+' '+r['partition'],font=FONT,fill='black');d.text((xx+5,yy+26),LABELS[r['category']]+'；其余不确定关系保留未知',font=SMALL,fill='black');board.paste(a,(xx+5,yy+57));board.paste(b,(xx+333,yy+57))
        board.save(folder/('NODES_'+str(start//6)+'.jpg'),quality=97)
    print('V29_MINED',len(selected),[(r['key'],r['image_id'],r['partition'],r['category'],len(r['ends'])) for r in selected],flush=True)

def arm_cost(g,a,h,b):
    _,u=prior.exp.rel.endpoint_segment(g['chains'][a[0]],a[1]);_,v=prior.exp.rel.endpoint_segment(h['chains'][b[0]],b[1])
    if min(len(u),len(v))<4:return 100.
    tu=prior.exp.tangent(g['chains'][a[0]],a[1]==0,span=8);tv=prior.exp.tangent(h['chains'][b[0]],b[1]==0,span=8)
    if float(tu@tv)<.90:return 100.
    from scipy.spatial import cKDTree
    return .5*(float(np.median(cKDTree(v).query(u)[0]))+float(np.median(cKDTree(u).query(v)[0])))

def prepare_labels():
    from v29_visible_labels import ANNOT
    if (ROOT/'supervision.json').exists():raise FileExistsError('Preserve frozen v29 supervision')
    nodes=read(ROOT/'node_manifest.json')['records'];parent=read(prior.ROOT/'manifest.json');lookup=source_lookup();rows=[];conflicts=[];uncertain=[];duplicates=[];reviewed=[];cache={}
    def graph(file):
        if file not in cache:cache[file]=prior.load_graph(file)
        return cache[file]
    for r in nodes:
        spec=ANNOT.get(r['key'],{});reviewed.append(dict(key=r['key'],partition=r['partition'],image=r['image_id'],note=spec.get('note','未判定'),category=r['category']))
        if spec.get('duplicate_of'):duplicates.append(dict(key=r['key'],duplicate_of=spec['duplicate_of']));continue
        g=graph(r['graph']);same={frozenset(p) for p in spec.get('same',[])};no={frozenset(p) for p in spec.get('no_connect',[])};invalid=set(spec.get('invalid',[]));ends=[tuple(e) for e in r['ends']]
        for i in r['indices']:
            c=g['candidates'][i];a=ends.index(c['a']);b=ends.index(c['b']);pair=frozenset([a,b]);label=1 if pair in same else 0 if pair in no or a in invalid or b in invalid else None
            if label is None:uncertain.append(dict(key=r['key'],pair=[a,b]));continue
            if c['label'] is not None:
                if c['label']!=label:conflicts.append(dict(key=r['key'],pair=[a,b],existing_label=c['label'],assistant_label=label,used=False))
                continue
            rgb,valid=lookup[r['image_id']]
            if not safe_candidate(g,c,valid):raise RuntimeError('Annotation entered excluded/uncertain area')
            rec=dict(image=r['image_id'],partition=r['partition'],key=r['key'],category=r['category'],feature=c['feature'].tolist(),label=label,view='base',candidate=i,provenance='assistant_visible_nonexpert',invalid_branch_rejection=bool(a in invalid or b in invalid));rows.append(rec)
            # Only training annotations propagate across prior extractor views.
            # A transferred pair must have a unique close geometric counterpart.
            if r['partition']!='train':continue
            for other in parent['graphs']:
                if other['partition']!='train' or other['id']!=r['image_id'] or other['view']=='base':continue
                h=graph(other['graph']);choices=[]
                for j,d in enumerate(h['candidates']):
                    if np.linalg.norm(np.asarray(d['node'])-c['node'])>2.5:continue
                    cost=min(arm_cost(g,c['a'],h,d['a'])+arm_cost(g,c['b'],h,d['b']),arm_cost(g,c['a'],h,d['b'])+arm_cost(g,c['b'],h,d['a']));choices.append((cost,j))
                choices.sort()
                if not choices or choices[0][0]>2.5 or (len(choices)>1 and choices[1][0]-choices[0][0]<1.0):continue
                cost,j=choices[0];d=h['candidates'][j]
                if d['label'] is not None or not safe_candidate(h,d,valid):continue
                rows.append(dict(rec,feature=d['feature'].tolist(),view=other['view'],candidate=j,provenance='unique_geometry_transfer_from_assistant_base_annotation',transfer_cost=float(cost)))
    # A candidate gets one supervised label, even if selected by two node types.
    unique={}
    for row in rows:
        token=(row['partition'],row['image'],row['view'],row['candidate'])
        if token in unique and unique[token]['label']!=row['label']:raise ValueError('Conflicting targeted labels')
        unique[token]=row
    rows=list(unique.values());counts={}
    for split in ['train','validation']:
        rr=[r for r in rows if r['partition']==split];x=np.asarray([r['feature'] for r in rr],np.float32).reshape(-1,len(prior.FEATURES));y=np.asarray([r['label'] for r in rr],np.float32);ids=np.asarray([r['image'] for r in rr]);keys=np.asarray([r['key'] for r in rr]);categories=np.asarray([r['category'] for r in rr]);np.savez_compressed(ROOT/('targeted_'+split+'.npz'),x=x,y=y,ids=ids,keys=keys,categories=categories);counts[split]=dict(positive=int((y==1).sum()),negative=int((y==0).sum()),images=int(len(np.unique(ids))),direct_base_relationships=sum(r['view']=='base' for r in rr),transferred_correlated_views=sum(r['view']!='base' for r in rr),sha256=digest(ROOT/('targeted_'+split+'.npz')))
    old=np.load(prior.ROOT/'relation_validation_pairs.npz');new=np.load(ROOT/'targeted_validation.npz');np.savez_compressed(ROOT/'combined_validation.npz',x=np.concatenate([old['x'],new['x']]),y=np.r_[old['y'],new['y']],ids=np.r_[old['ids'],new['ids']]);shutil.copy2(Path(__file__).with_name('v29_visible_labels.py'),ROOT/'visible_labels_before_training.py')
    jsave(ROOT/'supervision.json',dict(records=rows,counts=counts,reviewed_nodes=reviewed,uncertain_pairs_omitted=uncertain,duplicate_nodes_excluded=duplicates,conflicting_existing_labels_preserved=conflicts,annotation_sha256=digest(ROOT/'visible_labels_before_training.py'),node_manifest_sha256=digest(ROOT/'node_manifest.json'),combined_validation_sha256=digest(ROOT/'combined_validation.npz'),parent_split_sha256=digest(prior.ROOT/'split_policy.json'),expert_gold=False,old_development_not_labelled_or_trained=True,negative_means_no_continuation_including_spurious_branches=True,robot_connected=False));print('V29_SUPERVISION',counts,'conflicts',len(conflicts),'unknown',len(uncertain),'duplicates',len(duplicates),flush=True)

def train(variant):
    if variant not in ['control','targeted']:raise ValueError(variant)
    folder=ROOT/variant;folder.mkdir(exist_ok=True);supervision=read(ROOT/'supervision.json');old=np.load(prior.ROOT/'trusted_train_pairs.npz');fit=~np.isin(old['ids'],prior.VALIDATION_IDS);x=old['x'][fit];y=old['y'][fit];ids=old['ids'][fit];hard=np.load(ROOT/'targeted_train.npz');validation=np.load(ROOT/'combined_validation.npz');torch.set_num_threads(4);summaries=[];shutil.copy2(Path(__file__),folder/'source_at_training.py')
    if min(int((hard['y']==1).sum()),int((hard['y']==0).sum()))<2:raise RuntimeError('Too few targeted classes')
    groups={label:{str(i):np.flatnonzero((y==label)&(ids==i)) for i in np.unique(ids[y==label])} for label in [0,1]};hard_groups={label:np.flatnonzero(hard['y']==label) for label in [0,1]}
    for seed in [280271,280272,280273]:
        dest=folder/('seed_'+str(seed)+'.pt')
        if dest.exists():raise FileExistsError('Preserve finished v29 model')
        source=prior.ROOT/'visual'/('seed_'+str(seed)+'.pt');init=torch.load(source,map_location='cpu',weights_only=False);mu,sd=init['mean'],init['std'];xx=torch.from_numpy(((x-mu)/sd).astype(np.float32)).cuda();yy=torch.from_numpy(y).cuda();hx=torch.from_numpy(((hard['x']-mu)/sd).astype(np.float32)).cuda();hy=torch.from_numpy(hard['y']).cuda();vx=torch.from_numpy(((validation['x']-mu)/sd).astype(np.float32)).cuda();vy=torch.from_numpy(validation['y']).cuda();torch.manual_seed(seed+290000);rng=np.random.default_rng(seed+290000);hrng=np.random.default_rng(seed+390000);model=prior.PairNet('visual').cuda();model.load_state_dict(init['model']);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.03);schedule=hashlib.sha256();hard_schedule=hashlib.sha256();logs=[];start=time.perf_counter()
        def val_loss():
            model.eval()
            with torch.no_grad():
                loss=torch.nn.functional.binary_cross_entropy_with_logits(model(vx),vy,reduction='none');return float(.5*(loss[vy==1].mean()+loss[vy==0].mean()))
        best=val_loss();best_step=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};initial_val=best
        for step in range(1,601):
            ix=[];hi=[]
            for label in [0,1]:
                images=list(groups[label])
                for _ in range(32):
                    image=images[int(rng.integers(len(images)))];pool=groups[label][image];ix.append(int(pool[int(rng.integers(len(pool)))]))
                hi.extend(hrng.choice(hard_groups[label],16,replace=True).tolist())
            schedule.update(np.asarray(ix,np.int32).tobytes());hard_schedule.update(np.asarray(hi,np.int32).tobytes());ii=torch.tensor(ix,device='cuda');hh=torch.tensor(hi,device='cuda');model.train();base_loss=torch.nn.functional.binary_cross_entropy_with_logits(model(xx[ii]),yy[ii]);extra=base_loss.new_zeros(())
            # Both variants consume the same dropout draws. The control does
            # not backpropagate the extra relationship loss.
            extra=torch.nn.functional.binary_cross_entropy_with_logits(model(hx[hh]),hy[hh])
            loss=base_loss+.5*extra if variant=='targeted' else base_loss;opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),2)
            if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('Nonfinite targeted relation training')
            opt.step()
            if step%100==0:
                value=val_loss()
                if value<best:best=value;best_step=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                logs.append(dict(step=step,old_loss=float(base_loss.detach()),targeted_loss=float(extra.detach()),validation_loss=value))
        torch.cuda.synchronize();torch.save(dict(model=state,variant='visual',mean=mu,std=sd,seed=seed,completed_updates=600,retained_step=best_step,initial_checkpoint_sha256=digest(source),supervision_sha256=digest(ROOT/'supervision.json')),dest);summaries.append(dict(seed=seed,completed_updates=600,retained_step=best_step,initial_validation_loss=initial_val,best_validation_loss=best,elapsed_seconds=time.perf_counter()-start,actual_device=str(next(model.parameters()).device),old_sampler_sha256=schedule.hexdigest(),targeted_sampler_sha256=hard_schedule.hexdigest(),checkpoint_sha256=digest(dest),logs=logs));print('V29_TRAINED',variant,seed,'retained',best_step,'validation',round(best,4),flush=True)
    jsave(folder/'training_summary.json',dict(models=summaries,total_updates=1800,variant=variant,old_fit_positive=int((y==1).sum()),old_fit_negative=int((y==0).sum()),targeted_relationships_used=variant=='targeted',targeted_counts=supervision['counts']['train'],same_expanded_validation_for_both=True,pixel_model_updated=False,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__))

def models(variant):
    ensemble=[]
    for seed in [280271,280272,280273]:
        checkpoint=torch.load(ROOT/variant/('seed_'+str(seed)+'.pt'),map_location='cpu',weights_only=False);model=prior.PairNet('visual').cuda();model.load_state_dict(checkpoint['model']);ensemble.append((model.eval(),checkpoint))
    return ensemble

def apply_validation_labels(g,image):
    for row in read(ROOT/'supervision.json')['records']:
        if row['partition']!='validation' or row['image']!=image:continue
        c=g['candidates'][row['candidate']]
        if c['label'] is not None and c['label']!=row['label']:raise RuntimeError('Validation override conflict')
        c['label']=row['label'];c['label_reason']='assistant_visible_validation_only'
    return g

def calibrate():
    ensemble={n:models(n) for n in ['control','targeted']};cases={r['id']:(rgb,refs,valid,bg) for r,rgb,refs,valid,bg in prior.validation_cases()};graphs=[]
    for r in prior.validation_graph_records():
        g=apply_validation_labels(prior.load_graph(r['graph']),r['id']);rgb,refs,valid,bg=cases[r['id']];base=prior.exp.prev.evaluate_paths(refs,prior.assemble(g,[]),valid,bg,rgb);geometry=prior.exp.prev.evaluate_paths(refs,prior.assemble(g,prior.exp.rel.original_pairings(g)),valid,bg,rgb);pp={n:prior.probabilities(g,ensemble[n]) for n in ensemble};graphs.append((r,g,rgb,refs,valid,bg,base,geometry,pp))
    options=[];before=float(np.mean([r[6]['mean_best_single_path_coverage'] for r in graphs]));geom_mixed=float(np.mean([r[7]['mixed_reference_paths_flagged'] for r in graphs]));basepoint=float(np.mean([r[6]['point_coverage_3px'] for r in graphs]));base_dup=float(np.mean([r[6]['duplicate_point_fraction'] for r in graphs]))
    for variant in ['control','targeted']:
        for threshold in [.60,.70,.80,.85,.90,.95]:
            for margin in [.05,.10,.15]:
                rows=[];positive=0;negative=0;unknown=0
                for r,g,rgb,refs,valid,bg,base,geometry,pp in graphs:
                    indices,info=prior.select(g,pp[variant],threshold,margin);out=prior.assemble(g,indices);rows.append(prior.exp.prev.evaluate_paths(refs,out,valid,bg,rgb));positive+=sum(g['candidates'][i]['label']==1 for i in indices);negative+=sum(g['candidates'][i]['label']==0 for i in indices);unknown+=sum(g['candidates'][i]['label'] is None for i in indices)
                single=float(np.mean([r['mean_best_single_path_coverage'] for r in rows]));mixed=float(np.mean([r['mixed_reference_paths_flagged'] for r in rows]));point=float(np.mean([r['point_coverage_3px'] for r in rows]));dup=float(np.mean([r['duplicate_point_fraction'] for r in rows]));eligible=single>=before+.005 and negative==0 and mixed<=geom_mixed and point>=basepoint-.005 and dup<=base_dup+.01;score=single-.025*mixed-.10*dup;options.append(dict(variant=variant,threshold=threshold,margin=margin,single=single,baseline_single=before,gain=single-before,mixed=mixed,score=score,known_positive=int(positive),known_negative=int(negative),unknown=int(unknown),eligible=bool(eligible)))
    by_variant={}
    for variant in ['control','targeted']:
        rows=[r for r in options if r['variant']==variant];allowed=[r for r in rows if r['eligible']];by_variant[variant]=dict(selected=max(allowed or rows,key=lambda r:r['score']),enabled=bool(allowed))
    allowed=[r for r in options if r['eligible']];selected=max(allowed or options,key=lambda r:r['score']);jsave(ROOT/'calibration.json',dict(selected=selected,enabled=bool(allowed),by_variant=by_variant,options=options,expanded_validation_frozen_before_training=True,development_not_used_for_selection=True));print('V29_CALIBRATED',selected,'by_variant',by_variant,flush=True)

def classification():
    choice=read(ROOT/'calibration.json');safe=read(ROOT/'safety_selection.json')['selected'];z=np.load(ROOT/'combined_validation.npz');result={}
    for label,name in [('control','control'),('targeted','targeted'),('control_conservative','control'),('targeted_conservative','targeted')]:
        ensemble=models(name);pp=[]
        with torch.no_grad():
            for model,s in ensemble:pp.append(model(torch.from_numpy(((z['x']-s['mean'])/s['std']).astype(np.float32)).cuda()).sigmoid().cpu().numpy())
        p=np.mean(pp,0);threshold=safe['threshold'] if label.endswith('conservative') else choice['by_variant'][name]['selected']['threshold'];y=z['y'];result[label]=dict(positive=int((y==1).sum()),negative=int((y==0).sum()),true_positive=int(((p>=threshold)&(y==1)).sum()),false_positive=int(((p>=threshold)&(y==0)).sum()),false_negative=int(((p<threshold)&(y==1)).sum()),true_negative=int(((p<threshold)&(y==0)).sum()),threshold=threshold)
    jsave(ROOT/'classification.json',dict(expanded_validation=result,nonexpert_proxy_labels=True,threshold_selected_on_same_validation=True));print('V29_CLASSIFICATION',result,flush=True)

def diagnose_validation():
    choice=read(ROOT/'calibration.json')['selected'];ensemble=models(choice['variant']);manual=read(ROOT/'supervision.json')['records'];out=[]
    for r in prior.validation_graph_records():
        g=apply_validation_labels(prior.load_graph(r['graph']),r['id']);p=prior.probabilities(g,ensemble);indices,info=prior.select(g,p,choice['threshold'],choice['margin'])
        for i in indices:
            c=g['candidates'][i]
            if c['label']!=0:continue
            closed=[bool(np.linalg.norm(g['chains'][e[0]][0]-g['chains'][e[0]][-1])<1.5) for e in [c['a'],c['b']]];tags=[q['key'] for q in manual if q['partition']=='validation' and q['image']==r['id'] and q['candidate']==i];out.append(dict(image=r['id'],candidate=i,probability=float(p[i]),closed_branches=closed,manual_nodes=tags))
    jsave(ROOT/'validation_negative_diagnosis.json',dict(records=out,calibration_only=True));print('V29_VALIDATION_NEGATIVES',out,flush=True)

def freeze_safe():
    if (ROOT/'safety_selection.json').exists():raise FileExistsError('Preserve conservative selection before development evaluation')
    decision=read(ROOT/'calibration.json');old=read(prior.ROOT/'calibration.json');limit=old['selected']['geometric_mixed'];safe=[r for r in decision['options'] if r['variant']=='targeted' and r['known_negative']==0 and r['mixed']<=limit]
    if not safe:raise RuntimeError('No zero-known-negative targeted option even for conservative diagnostics')
    choice=max(safe,key=lambda r:r['score']);jsave(ROOT/'safety_selection.json',dict(selected=choice,strict_original_gain_gate_passed=decision['enabled'],original_gate_not_relaxed_or_relabelled=True,conservative_trial_only=True,not_claimed_continuity_breakthrough=True,same_threshold_control_comparison=True,validation_only_before_development=True,calibration_sha256=digest(ROOT/'calibration.json')));print('V29_CONSERVATIVE_FROZEN',choice,flush=True)

def matched_budget():
    if (ROOT/'matched_error_budget.json').exists():raise FileExistsError('Preserve error-budget selection before development')
    out={};cases={r['id']:(rgb,refs,valid,bg) for r,rgb,refs,valid,bg in prior.validation_cases()}
    for name in ['control','targeted']:
        ensemble=models(name);graphs=[];negative_scores=[]
        for r in prior.validation_graph_records():
            g=apply_validation_labels(prior.load_graph(r['graph']),r['id']);p=prior.probabilities(g,ensemble);negative_scores.extend(float(p[i]) for i,c in enumerate(g['candidates']) if c['label']==0);graphs.append((r,g,p))
        # Scores from separately trained models have different calibration. Match
        # a zero-known-negative validation budget instead of equating scores.
        threshold=min(1.,max(negative_scores)+1e-6);positive=0;negative=0;unknown=0;rows=[]
        for r,g,p in graphs:
            indices,info=prior.select(g,p,threshold,.05);positive+=sum(g['candidates'][i]['label']==1 for i in indices);negative+=sum(g['candidates'][i]['label']==0 for i in indices);unknown+=sum(g['candidates'][i]['label'] is None for i in indices);rgb,refs,valid,bg=cases[r['id']];rows.append(prior.exp.prev.evaluate_paths(refs,prior.assemble(g,indices),valid,bg,rgb))
        out[name]=dict(threshold=float(threshold),margin=.05,known_positive=int(positive),known_negative=int(negative),unknown=int(unknown),single=float(np.mean([r['mean_best_single_path_coverage'] for r in rows])),threshold_based_on_validation_negative_max=True)
    jsave(ROOT/'matched_error_budget.json',dict(variants=out,validation_only_before_development=True,not_an_independent_zero_error_guarantee=True,strict_original_gate_unchanged=True,original_strict_gate_passed=read(ROOT/'calibration.json')['enabled']));print('V29_MATCHED_ERROR_BUDGET',out,flush=True)

def evaluate():
    choice=read(ROOT/'calibration.json');safe=read(ROOT/'safety_selection.json')['selected'];budget=read(ROOT/'matched_error_budget.json')['variants'];old_choice=read(prior.ROOT/'calibration.json')['selected'];ensembles={'v28':prior.models('visual'),'control':models('control'),'targeted':models('targeted')};records=[]
    for r,rgb,refs,valid,bg in prior.exp.cases('development'):
        # These cached graphs contain only inference inputs. No new annotation
        # from these development images was used in fit or validation.
        g=prior.load_graph(prior.ROOT/'graphs'/('development_'+r['id']+'.json'));pp={n:prior.probabilities(g,e) for n,e in ensembles.items()};selected={'unpaired':([],dict(reference_used=False)),'geometric':(prior.exp.rel.original_pairings(g),dict(reference_used=False)),'v28':prior.select(g,pp['v28'],old_choice['threshold'],old_choice['margin'])}
        for name in ['control','targeted']:
            cfg=choice['by_variant'][name]['selected'];selected[name]=prior.select(g,pp[name],cfg['threshold'],cfg['margin'])
        selected['control_conservative']=prior.select(g,pp['control'],safe['threshold'],safe['margin']);selected['targeted_conservative']=prior.select(g,pp['targeted'],safe['threshold'],safe['margin']);cfg=choice['selected'];selected['calibrated_selected']=prior.select(g,pp[cfg['variant']],cfg['threshold'],cfg['margin']);outputs={name:prior.assemble(g,pairs) for name,(pairs,info) in selected.items()}
        for name in ['control','targeted']:
            cfg=budget[name];selected[name+'_zero_error']=prior.select(g,pp[name],cfg['threshold'],cfg['margin']);outputs[name+'_zero_error']=prior.assemble(g,selected[name+'_zero_error'][0])
        prior.label_graph(g,refs,valid);row=dict(id=r['id'],group=r['split'],metrics={},relationships={})
        for name,(indices,info) in selected.items():
            paths=outputs[name];row['metrics'][name]=prior.exp.prev.evaluate_paths(refs,paths,valid,bg,rgb);row['relationships'][name]=dict(accepted=len(indices),known_positive=int(sum(g['candidates'][i]['label']==1 for i in indices)),known_negative=int(sum(g['candidates'][i]['label']==0 for i in indices)),unknown=int(sum(g['candidates'][i]['label'] is None for i in indices)));events=[dict(node=g['candidates'][i]['node'],a=list(g['candidates'][i]['a']),b=list(g['candidates'][i]['b'])) for i in indices];jsave(ROOT/'trajectories'/name/(r['id']+'.json'),dict(paths=[p.round(3).tolist() for p in paths],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,reference_labels_used_at_inference=False,raw_edges_used_once=True,selected_relations=events,info=info))
        records.append(row);print('V29_EVAL',r['id'],{n:round(m['mean_best_single_path_coverage'],3) for n,m in row['metrics'].items()},'targeted',row['relationships']['targeted'],flush=True)
    variants=['unpaired','geometric','v28','control','targeted','control_conservative','targeted_conservative','control_zero_error','targeted_zero_error','calibrated_selected'];groups={}
    for g in sorted(set(r['group'] for r in records)):
        rows=[r for r in records if r['group']==g];groups[g]={}
        for name in variants:
            groups[g][name]={}
            for metric in prior.exp.prev.METRICS:
                values=[r['metrics'][name][metric] for r in rows if r['metrics'][name][metric] is not None];groups[g][name][metric]=float(np.mean(values)) if values else None
            for key in ['accepted','known_positive','known_negative','unknown']:groups[g][name][key+'_joins']=sum(r['relationships'][name][key] for r in rows)
    jsave(ROOT/'evaluation.json',dict(records=records,groups=groups,choice=choice,conservative_choice=safe,matched_error_budget=budget,partial_reference_metrics_not_full_image_accuracy=True,no_new_blind_set=True,pixel_model_unchanged=True,old_development_not_used_for_fit_or_thresholds=True));lookup={r['id']:rgb for r,rgb,refs,valid,bg in prior.exp.cases('development')};keys=['H02','R04','X03','N11'];board=Image.new('RGB',(1600,310*len(keys)+40),'#f4f2eb');d=ImageDraw.Draw(board);d.text((5,5),'v29 定向关系监督：输入／v28关系／同验证误连预算普通对照／定向监督（保守离线候选）',font=FONT,fill='black')
    for i,key in enumerate(keys):
        rgb=lookup[key];images=[('输入',rgb)]
        for name,title in [('v28','v28续接'),('control_zero_error','同误连预算普通对照'),('targeted_zero_error','定向监督保守候选')]:images.append((title,prior.exp.prev.v21.line_view(rgb.shape[:2],read(ROOT/'trajectories'/name/(key+'.json'))['paths'],True)))
        for j,(title,im) in enumerate(images):d.text((j*400+5,40+i*310),key+' '+title,font=SMALL,fill='black');prior.exp.prev.v21.paste(board,im,j*400+5,65+i*310,390,240)
    board.save(ROOT/'RESULT_COMPARISON.jpg',quality=95);print('V29_EVALUATED',flush=True)

def predict_one(input_path,output_path):
    import sys
    source=Path(input_path);out=Path(output_path)
    if not out.is_absolute() or out.drive.upper() not in ['E:','F:']:raise ValueError('Use an absolute E:/F: output path')
    rgb=np.asarray(Image.open(source).convert('RGB'))
    if not prior.exp.prev.v21.v16.monochrome_route(rgb):raise ValueError('Only monochrome rock research evaluated')
    extractor=prior.exp.load_model('base');pred=prior.exp.prev.prior.predict(extractor,rgb)[:5];g=prior.add_visual(prior.build_graph(pred,rgb),extractor);cfg=read(ROOT/'matched_error_budget.json')['variants']['targeted'];p=prior.probabilities(g,models('targeted'));indices,info=prior.select(g,p,cfg['threshold'],cfg['margin']);paths=prior.assemble(g,indices);unpaired=prior.assemble(g,[]);out.mkdir(parents=True,exist_ok=True);jsave(out/'candidate_trajectory.json',dict(paths=[q.round(3).tolist() for q in paths],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,reference_labels_used_at_inference=False,experimental_only=True,strict_original_gain_gate_passed=read(ROOT/'calibration.json')['enabled'],info=info));jsave(out/'summary.json',dict(input=str(source),input_sha256=digest(source),reference_labels_used=False,reference_label_matching_called=False,new_annotation_module_loaded='v29_visible_labels' in sys.modules,actual_device=str(next(extractor.parameters()).device),known_development_image_interface_check_not_blind_test=True,official_default_replaced=False,robot_connected=False))
    board=Image.new('RGB',(1500,410),'#f4f2eb');d=ImageDraw.Draw(board)
    for j,(title,im) in enumerate([('输入',rgb),('去重未续接',prior.exp.prev.v21.line_view(rgb.shape[:2],unpaired,True)),('定向监督保守研究候选',prior.exp.prev.v21.line_view(rgb.shape[:2],paths,True))]):d.text((500*j+5,7),title,font=FONT,fill='black');prior.exp.prev.v21.paste(board,im,500*j+5,40,490,355)
    board.save(out/'comparison.jpg',quality=95);print('V29_REFERENCE_FREE',str(out),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['mine','prepare','train','calibrate','classification','diagnose','safe','budget','evaluate','predict','finalize']);p.add_argument('--variant',choices=['control','targeted']);p.add_argument('--input');p.add_argument('--output');a=p.parse_args();{'mine':mine,'prepare':prepare_labels,'train':lambda:train(a.variant),'calibrate':calibrate,'classification':classification,'diagnose':diagnose_validation,'safe':freeze_safe,'budget':matched_budget,'evaluate':evaluate,'predict':lambda:predict_one(a.input,a.output),'finalize':lambda:__import__('v29_finalize').main()}[a.mode]()
