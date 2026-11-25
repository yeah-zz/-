# -*- coding: utf-8 -*-
"""Learn same-stroke continuation on deduplicated graph edges, offline only.

Configure the compatible E-drive GPU runtime before importing torch. Reference
chains label training relationships; inference sees only image/graph features.
"""
from rock_v15_common import LAB,read,jsave,digest,FONT,SMALL,PARENT
from pathlib import Path
from collections import defaultdict
import json,math,time,hashlib,shutil,itertools
import numpy as np,cv2,torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from PIL import Image,ImageDraw
import v27_separation_faint as exp
from topology_trace_v13 import graph_paths

ROOT=LAB/'stroke_relations_v28_20261027'
FEATURES=list(exp.rel.FEATURE_NAMES)+['visual_abs_difference_'+str(i) for i in range(16)]+['visual_product_'+str(i) for i in range(16)]+['visual_node_'+str(i) for i in range(16)]
VALIDATION_IDS=['U03','stage_035','replay_R02']

def quality_mask(x):
    # Geometry/image-only quality screen, not a prediction-derived class label.
    return (x[:,1]>=math.log1p(12)/6)&(x[:,16]>=.012)&(x[:,24]>=.008)

def arc(p):return np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]

def build_graph(pred,rgb):
    cfg=read(exp.prev.ROOT/'calibration.json')['selected']['config'];paths,trace=exp.prev.adaptive_curves(pred,cfg);sk=skeletonize(exp.prev.raster(paths,pred.shape[1:])>0);field=np.zeros((8,*sk.shape),np.float32);field[0]=np.maximum(pred[0],pred[1]*.8);field[1:3]=pred[2:4];field[3:5]=pred[2:4];chains,_=graph_paths(sk,field,False,min_length=0);nodes=defaultdict(list)
    for i,p in enumerate(chains):
        nodes[tuple(np.round(p[0],4))].append((i,0));nodes[tuple(np.round(p[-1],4))].append((i,1))
    alpha=exp.prev.v21.v16.photometry(rgb)[1];candidates=[]
    for xy,ends in nodes.items():
        for a,b in itertools.combinations(ends,2):
            if a[0]==b[0] or min(arc(chains[a[0]])[-1],arc(chains[b[0]])[-1])<8:continue
            candidates.append(dict(a=a,b=b,node=list(xy),degree=len(ends),feature=exp.rel.feature(chains,a,b,len(ends),pred,alpha),label=None))
    return dict(chains=chains,candidates=candidates,pred=pred,rgb=rgb,shape=sk.shape,trace=trace)

@torch.no_grad()
def add_visual(g,model):
    features=exp.prev.v21.v16.photometry(g['rgb'])[0];padded=np.pad(features,((0,0),(32,32),(32,32)),mode='reflect');loc=sorted({tuple(np.rint(c['node']).astype(int)) for c in g['candidates']});emb={}
    for start in range(0,len(loc),32):
        group=loc[start:start+32];batch=np.stack([padded[:,y:y+64,x:x+64] for x,y in group]);x=torch.from_numpy(batch).cuda();a=model.a(x);b=model.b(F.avg_pool2d(a,2));c=model.c(F.avg_pool2d(b,2));d=model.d(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1));z=model.e(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1)).float().cpu().numpy()
        for xy,e in zip(group,z):emb[xy]=e
    for c in g['candidates']:
        x,y=np.rint(c['node']).astype(int);z=emb[x,y];desc=[]
        for endpoint in [c['a'],c['b']]:
            p,seg=exp.rel.endpoint_segment(g['chains'][endpoint[0]],endpoint[1]);seg=seg if len(seg) else p[:6];q=seg-[x-32,y-32];desc.append(np.array([exp.rel.sample(im,q).mean() for im in z],np.float32))
        c['feature']=np.r_[c['feature'],np.abs(desc[0]-desc[1]),desc[0]*desc[1],z[:,30:35,30:35].mean((1,2))].astype(np.float32)
    return g

def label_graph(g,refs,valid):
    trees=[cKDTree(exp.dense(np.asarray(p))) for p in refs if len(p)>=2];assign={}
    for c in g['candidates']:
        for e in [c['a'],c['b']]:
            if e in assign:continue
            p,seg=exp.rel.endpoint_segment(g['chains'][e[0]],e[1]);xy=np.rint(seg).astype(int)
            if len(seg)<5 or not len(trees):assign[e]=None;continue
            inside=(xy[:,0]>=0)&(xy[:,0]<valid.shape[1])&(xy[:,1]>=0)&(xy[:,1]<valid.shape[0])
            if inside.mean()<.9 or (valid[xy[inside,1],xy[inside,0]]>.5).mean()<.7:assign[e]=None;continue
            assign[e]=exp.rel.assign_end(g['chains'][e[0]],e[1],trees)
        aa,bb=assign[c['a']],assign[c['b']];label=None;reason='unknown_or_ambiguous'
        if aa is not None and bb is not None:
            if aa['ref']!=bb['ref']:label=0;reason='distinct_reference_chains'
            elif abs(aa['node_index']-bb['node_index'])<=24:label=int(aa['direction']*bb['direction']<0);reason='same_reference_opposite_sides' if label else 'same_reference_same_side'
        if label is not None and not quality_mask(c['feature'][None])[0]:label=None;reason='uncertain_short_or_weakly_supported_branch'
        c['label']=label;c['label_reason']=reason
    return g

def save_graph(file,g,prediction_file):
    jsave(file,dict(chains=[p.round(5).tolist() for p in g['chains']],shape=list(g['shape']),prediction_file=str(prediction_file),candidates=[dict(a=list(c['a']),b=list(c['b']),node=[float(x) for x in c['node']],degree=c['degree'],feature=c['feature'].tolist(),label=c.get('label'),label_reason=c.get('label_reason','not_used_at_inference')) for c in g['candidates']]))

def load_graph(file):
    d=read(file);d['chains']=[np.asarray(p,np.float32) for p in d['chains']];d['pred']=np.load(d['prediction_file'])['prediction'];d['shape']=tuple(d['shape'])
    for c in d['candidates']:
        c['a']=tuple(c['a']);c['b']=tuple(c['b']);c['feature']=np.asarray(c['feature'],np.float32)
        if c.get('label') is not None and not quality_mask(c['feature'][None])[0]:c['raw_reference_label']=c['label'];c['label']=None;c['label_reason']='uncertain_short_or_weakly_supported_branch'
    return d

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'manifest.json').exists():raise FileExistsError('Preserve frozen v28 dataset')
    torch.set_num_threads(4);cv2.setNumThreads(2);records=read(exp.prev.prior.ROOT/'training_manifest.json')['records'];base=exp.load_model('base');arrays={'train':[],'calibration':[]};audit=[];inputs=[]
    for view in ['base','control','faint']:
        graph_model=base if view=='base' else exp.load_model(view)
        for r in records:
            if r['split']!='train':continue
            z=np.load(r['data']);rgb=z['rgb'];refs=[p['points'] for p in read(r['labels'])['strokes']];valid=z['valid'];pred_file=ROOT/'predictions'/('train_'+r['id']+'_'+view+'.npz');pred_file.parent.mkdir(exist_ok=True);pred=exp.prev.prior.predict(graph_model,rgb)[:5];np.savez_compressed(pred_file,prediction=pred);g=label_graph(add_visual(build_graph(pred,rgb),base),refs,valid);file=ROOT/'graphs'/('train_'+r['id']+'_'+view+'.json');save_graph(file,g,pred_file);known=[c for c in g['candidates'] if c['label'] is not None]
            for c in known:arrays['train'].append((c['feature'],c['label'],r['id'],view))
            audit.append(dict(id=r['id'],partition='train',view=view,graph=str(file),candidate_count=len(g['candidates']),positive=sum(c['label']==1 for c in known),negative=sum(c['label']==0 for c in known),unknown=len(g['candidates'])-len(known)))
            if view=='base':inputs.append(dict(id=r['id'],data=r['data'],data_sha256=digest(r['data']),labels=r['labels'],labels_sha256=digest(r['labels'])))
            print('V28_TRAIN_GRAPH',view,r['id'],len(g['candidates']),audit[-1]['positive'],audit[-1]['negative'],flush=True)
        if view!='base':del graph_model;torch.cuda.empty_cache()
    for r,rgb,refs,valid,bg in exp.cases('calibration'):
        pred_file=ROOT/'predictions'/('calibration_'+r['id']+'.npz');pred=exp.prev.prior.predict(base,rgb)[:5];np.savez_compressed(pred_file,prediction=pred);g=label_graph(add_visual(build_graph(pred,rgb),base),refs,valid);file=ROOT/'graphs'/('calibration_'+r['id']+'.json');save_graph(file,g,pred_file);known=[c for c in g['candidates'] if c['label'] is not None]
        for c in known:arrays['calibration'].append((c['feature'],c['label'],r['id'],'base'))
        audit.append(dict(id=r['id'],partition='calibration',view='base',graph=str(file),candidate_count=len(g['candidates']),positive=sum(c['label']==1 for c in known),negative=sum(c['label']==0 for c in known),unknown=len(g['candidates'])-len(known)));print('V28_CAL_GRAPH',r['id'],audit[-1]['positive'],audit[-1]['negative'],flush=True)
    counts={}
    for split,rows in arrays.items():
        x=np.stack([r[0] for r in rows]) if rows else np.zeros((0,len(FEATURES)),np.float32);y=np.asarray([r[1] for r in rows],np.float32);ids=np.asarray([r[2] for r in rows]);views=np.asarray([r[3] for r in rows]);file=ROOT/(split+'_pairs.npz');np.savez_compressed(file,x=x,y=y,ids=ids,views=views);counts[split]=dict(positive=int((y==1).sum()),negative=int((y==0).sum()),images=int(len(np.unique(ids))),sha256=digest(file))
    jsave(ROOT/'manifest.json',dict(inputs=inputs,graphs=audit,counts=counts,feature_names=FEATURES,base_raster_checkpoint=str(exp.prev.INIT),base_raster_sha256=digest(exp.prev.INIT),graph_view_checkpoints={n:str(exp.prev.INIT if n=='base' else exp.ROOT/n/'final.pt') for n in ['base','control','faint']},training_views_are_correlated_not_independent_images=True,reference_chain_identity_not_verified_artist_pen_lifts=True,unknown_pairs_not_negative=True,new_independent_artworks=0,robot_connected=False));shutil.copy2(Path(__file__),ROOT/'source_at_prepare.py');print('V28_COUNTS',counts,flush=True)

def validation_cases():
    result=list(exp.cases('calibration'));records=read(exp.prev.prior.ROOT/'training_manifest.json')['records']
    for r in records:
        if r['id'] not in VALIDATION_IDS:continue
        z=np.load(r['data']);refs=[p['points'] for p in read(r['labels'])['strokes']];result.append((r,z['rgb'],refs,z['valid'],z['background']>0))
    return result

def policy():
    if (ROOT/'split_policy.json').exists() and read(ROOT/'split_policy.json').get('quality_filtered'):return read(ROOT/'split_policy.json')
    if any(ROOT.glob('*/seed_*.pt')):raise RuntimeError('Freeze quality policy before fitting')
    if (ROOT/'split_policy.json').exists():shutil.copy2(ROOT/'split_policy.json',ROOT/'split_policy_initial_before_quality_filter.json')
    raw=np.load(ROOT/'train_pairs.npz');trusted=quality_mask(raw['x']);np.savez_compressed(ROOT/'trusted_train_pairs.npz',**{k:raw[k][trusted] for k in raw.files});z=np.load(ROOT/'trusted_train_pairs.npz');fit=~np.isin(z['ids'],VALIDATION_IDS);validation=np.isin(z['ids'],VALIDATION_IDS)&(z['views']=='base');old=np.load(ROOT/'calibration_pairs.npz');ok=quality_mask(old['x']);x=np.concatenate([old['x'][ok],z['x'][validation]]);y=np.r_[old['y'][ok],z['y'][validation]];ids=np.r_[old['ids'][ok],z['ids'][validation]];np.savez_compressed(ROOT/'relation_validation_pairs.npz',x=x,y=y,ids=ids)
    out=dict(manifest_sha256=digest(ROOT/'manifest.json'),fit_ids=sorted(set(z['ids'][fit].tolist())),heldout_relation_images=VALIDATION_IDS,all_views_of_heldout_images_excluded=True,fit_positive=int((z['y'][fit]==1).sum()),fit_negative=int((z['y'][fit]==0).sum()),validation_positive=int((y==1).sum()),validation_negative=int((y==0).sum()),validation_images=int(len(np.unique(ids))),validation_pair_sha256=digest(ROOT/'relation_validation_pairs.npz'),parent_raster_already_saw_some_relation_validation_images=True,validation_not_blind=True,frozen_before_relation_optimization=True)
    out.update(quality_filtered=True,quality_rule='shorter branch>=12px; branch image-alpha mean>=.012; transition image-alpha mean>=.008; otherwise omit, never relabel as negative',raw_training_relationships=len(raw['y']),trusted_training_pool_relationships=int(trusted.sum()),trusted_train_pair_sha256=digest(ROOT/'trusted_train_pairs.npz'));jsave(ROOT/'split_policy.json',out);print('V28_SPLIT',out,flush=True);return out

def audit():
    policy();manifest=read(ROOT/'manifest.json');lookup={r['id']:r for r in manifest['inputs']};cards=[];selected=[]
    for r in manifest['graphs']:
        if r['partition']!='train' or r['view']!='base' or len(cards)>=24:continue
        g=load_graph(r['graph']);rgb=np.load(lookup[r['id']]['data'])['rgb']
        for label in [1,0]:
            choices=[c for c in g['candidates'] if c['label']==label]
            if not choices:continue
            c=choices[0];x,y=np.rint(c['node']).astype(int);box=(int(x-32),int(y-32),int(x+32),int(y+32));original=Image.fromarray(rgb).crop(box).resize((180,180));overlay=original.copy();d=ImageDraw.Draw(overlay)
            for endpoint,color in [(c['a'],'#d43131'),(c['b'],'#2468d1')]:
                q,seg=exp.rel.endpoint_segment(g['chains'][endpoint[0]],endpoint[1]);q=q[arc(q)<=25];q=(q-[x-32,y-32])*180/64;d.line([tuple(p) for p in q],fill=color,width=2)
            card=Image.new('RGB',(380,224),'white');d=ImageDraw.Draw(card);d.text((4,3),r['id']+' '+('同一参考链' if label else '不同参考链'),font=SMALL,fill='black');card.paste(original,(4,30));card.paste(overlay,(194,30));cards.append(card);selected.append(dict(image=r['id'],node=c['node'],label=label,reason=c['label_reason']))
    board=Image.new('RGB',(1520,224*math.ceil(len(cards)/4)+35),'#eeeae1');d=ImageDraw.Draw(board);d.text((5,4),'v28 关系监督审计：原图局部／两条候选分支；参考链关系不是历史真实提笔金标准',font=FONT,fill='black')
    for i,card in enumerate(cards):board.paste(card,(i%4*380,35+i//4*224))
    board.save(ROOT/'RELATION_SUPERVISION_AUDIT.jpg',quality=95);jsave(ROOT/'audit_examples.json',dict(records=selected,model_predictions_not_used_for_labels=True));print('V28_AUDITED',len(cards),flush=True)

class PairNet(nn.Module):
    def __init__(self,variant):
        super().__init__();n=26 if variant=='geometry' else len(FEATURES);hidden=16 if variant=='geometry' else 32;self.net=nn.Sequential(nn.Linear(n,hidden),nn.SiLU(),nn.Dropout(.15),nn.Linear(hidden,16),nn.SiLU(),nn.Linear(16,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def train(variant):
    split=policy();z=np.load(ROOT/'trusted_train_pairs.npz');fit=~np.isin(z['ids'],VALIDATION_IDS);inner=np.isin(z['ids'],VALIDATION_IDS);n=26 if variant=='geometry' else len(FEATURES);x=z['x'][:,:n];y=z['y'];mu=x[fit].mean(0);sd=np.maximum(x[fit].std(0),.05);xx=torch.from_numpy(((x-mu)/sd).astype(np.float32)).cuda();yy=torch.from_numpy(y).cuda();validation=torch.from_numpy(np.flatnonzero(inner)).cuda();groups={}
    for label in [0,1]:groups[label]={str(k):np.flatnonzero(fit&(y==label)&(z['ids']==k)) for k in np.unique(z['ids'][fit&(y==label)])}
    if min(sum(len(v) for v in groups[0].values()),sum(len(v) for v in groups[1].values()))<10:raise RuntimeError('Not enough trusted fit relationships')
    folder=ROOT/variant;folder.mkdir(exist_ok=True);shutil.copy2(Path(__file__),folder/'source_at_training.py');summaries=[];torch.set_num_threads(4)
    for seed in [280271,280272,280273]:
        dest=folder/('seed_'+str(seed)+'.pt')
        if dest.exists():raise FileExistsError('Preserve completed relation model')
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=PairNet(variant).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.03);best=float('inf');best_state=None;best_step=0;logs=[];schedule=hashlib.sha256();start=time.perf_counter()
        for step in range(1,801):
            indices=[]
            for label in [0,1]:
                keys=list(groups[label])
                for _ in range(32):
                    key=keys[int(rng.integers(len(keys)))];arr=groups[label][key];indices.append(int(arr[int(rng.integers(len(arr)))]))
            schedule.update(np.asarray(indices,np.int32).tobytes());ix=torch.tensor(indices,device='cuda');model.train();loss=F.binary_cross_entropy_with_logits(model(xx[ix]),yy[ix]);opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),2)
            if not torch.isfinite(loss) or not torch.isfinite(norm):raise RuntimeError('Nonfinite relation training')
            opt.step()
            if step%100==0:
                model.eval()
                with torch.no_grad():
                    vv=F.binary_cross_entropy_with_logits(model(xx[validation]),yy[validation],reduction='none');labels=yy[validation];val=float(.5*(vv[labels==1].mean()+vv[labels==0].mean()))
                if val<best:best=val;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                log=dict(step=step,fit_loss=float(loss.detach()),inner_validation_loss=val);logs.append(log)
        torch.cuda.synchronize();torch.save(dict(model=best_state,variant=variant,mean=mu,std=sd,completed_updates=800,retained_step=best_step,seed=seed,feature_names=FEATURES[:n],split_policy_sha256=digest(ROOT/'split_policy.json')),dest);summary=dict(seed=seed,completed_updates=800,retained_step=best_step,inner_validation_loss=best,elapsed_seconds=time.perf_counter()-start,actual_device=str(next(model.parameters()).device),schedule_sha256=schedule.hexdigest(),checkpoint_sha256=digest(dest),logs=logs);summaries.append(summary);print('V28_TRAINED',variant,seed,'best_step',best_step,'val',round(best,4),flush=True)
    jsave(folder/'training_summary.json',dict(models=summaries,total_updates=2400,original_raster_updated=False,fit_images=len(split['fit_ids']),fit_positive=split['fit_positive'],fit_negative=split['fit_negative'],heldout_relation_images=VALIDATION_IDS,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,robot_connected=False))

def models(variant):
    result=[]
    for seed in [280271,280272,280273]:
        state=torch.load(ROOT/variant/('seed_'+str(seed)+'.pt'),map_location='cpu',weights_only=False);model=PairNet(variant).cuda();model.load_state_dict(state['model']);result.append((model.eval(),state))
    return result

@torch.no_grad()
def probabilities(g,ensemble):
    if not g['candidates']:return np.zeros(0,np.float32)
    x=np.stack([c['feature'] for c in g['candidates']]);out=[]
    for model,s in ensemble:
        n=len(s['mean']);xx=torch.from_numpy(((x[:,:n]-s['mean'])/s['std']).astype(np.float32)).cuda();out.append(model(xx).sigmoid().cpu().numpy())
    return np.mean(out,0)

def assemble(g,accepted):
    paired={}
    for i in accepted:
        c=g['candidates'][i];a,b=c['a'],c['b']
        if a in paired or b in paired:raise ValueError('Endpoint paired twice')
        paired[a]=b;paired[b]=a
    used=set();paths=[];starts=[(i,s) for i in range(len(g['chains'])) for s in [0,1] if (i,s) not in paired]+[(i,0) for i in range(len(g['chains']))]
    for i,s in starts:
        if i in used:continue
        pieces=[]
        while i not in used:
            used.add(i);pieces.append(g['chains'][i] if s==0 else g['chains'][i][::-1]);nxt=paired.get((i,1-s))
            if nxt is None:break
            i,s=nxt
        q=np.concatenate(pieces)
        if arc(q)[-1]>=6:paths.append(q)
    return paths

def select(g,prob,threshold,margin):
    by_end=defaultdict(list);eligible=[]
    for i,c in enumerate(g['candidates']):
        if prob[i]<threshold:continue
        va=exp.tangent(g['chains'][c['a'][0]],c['a'][1]==0,span=6);vb=exp.tangent(g['chains'][c['b'][0]],c['b'][1]==0,span=6)
        if float(va@(-vb))<-.10:continue
        eligible.append(i);by_end[c['a']].append(i);by_end[c['b']].append(i)
    accepted=[];used=set();parent=list(range(len(g['chains'])));cycles=0
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in sorted(eligible,key=lambda k:-prob[k]):
        c=g['candidates'][i];a,b=c['a'],c['b']
        if a in used or b in used:continue
        alternatives=[j for end in [a,b] for j in by_end[end] if j!=i]
        if alternatives and prob[i]-max(prob[j] for j in alternatives)<margin:continue
        ra,rb=root(a[0]),root(b[0])
        if ra==rb:cycles+=1;continue
        accepted.append(i);used.update([a,b]);parent[ra]=rb
    return accepted,dict(candidate_count=len(g['candidates']),accepted_relations=len(accepted),new_cycles_rejected=cycles,endpoint_used_at_most_once=True,unique_original_graph_edges=True,reference_used=False)

def validation_graph_records():
    manifest=read(ROOT/'manifest.json');out=[]
    for r in manifest['graphs']:
        if r['partition']=='calibration' or (r['partition']=='train' and r['view']=='base' and r['id'] in VALIDATION_IDS):out.append(r)
    return out

def calibrate():
    policy();cases={r['id']:(rgb,refs,valid,bg) for r,rgb,refs,valid,bg in validation_cases()};graphs=[];ensemble={name:models(name) for name in ['geometry','visual']}
    for r in validation_graph_records():
        g=load_graph(r['graph']);rgb,refs,valid,bg=cases[r['id']];base=exp.prev.evaluate_paths(refs,assemble(g,[]),valid,bg,rgb);geometry_pairs=exp.rel.original_pairings(g);geom=exp.prev.evaluate_paths(refs,assemble(g,geometry_pairs),valid,bg,rgb);pp={name:probabilities(g,ensemble[name]) for name in ensemble};graphs.append((r,g,rgb,refs,valid,bg,base,geom,pp))
    options=[]
    for name in ['geometry','visual']:
        for threshold in [.60,.70,.80,.90,.95]:
            for margin in [.05,.10,.15]:
                rows=[];negative=0;positive=0;unknown=0
                for r,g,rgb,refs,valid,bg,base,geom,pp in graphs:
                    selected,info=select(g,pp[name],threshold,margin);met=exp.prev.evaluate_paths(refs,assemble(g,selected),valid,bg,rgb);rows.append(dict(id=r['id'],metrics=met));negative+=sum(g['candidates'][i]['label']==0 for i in selected);positive+=sum(g['candidates'][i]['label']==1 for i in selected);unknown+=sum(g['candidates'][i]['label'] is None for i in selected)
                single=float(np.mean([r['metrics']['mean_best_single_path_coverage'] for r in rows]));before=float(np.mean([r[6]['mean_best_single_path_coverage'] for r in graphs]));mixed=float(np.mean([r['metrics']['mixed_reference_paths_flagged'] for r in rows]));geom_mixed=float(np.mean([r[7]['mixed_reference_paths_flagged'] for r in graphs]));point=float(np.mean([r['metrics']['point_coverage_3px'] for r in rows]));basepoint=float(np.mean([r[6]['point_coverage_3px'] for r in graphs]));dup=float(np.mean([r['metrics']['duplicate_point_fraction'] for r in rows]));base_dup=float(np.mean([r[6]['duplicate_point_fraction'] for r in graphs]));eligible=single>=before+.005 and negative==0 and mixed<=geom_mixed and point>=basepoint-.005 and dup<=base_dup+.01;score=single-.025*mixed-.10*dup;options.append(dict(variant=name,threshold=threshold,margin=margin,score=score,single=single,baseline_single=before,gain=single-before,mixed=mixed,geometric_mixed=geom_mixed,known_positive=int(positive),known_negative=int(negative),unknown=int(unknown),eligible=bool(eligible)))
    allowed=[r for r in options if r['eligible']];selected=max(allowed or options,key=lambda r:r['score']);by_variant={}
    for name in ['geometry','visual']:
        sub=[r for r in options if r['variant']==name];good=[r for r in sub if r['eligible']];by_variant[name]=dict(selected=max(good or sub,key=lambda r:r['score']),enabled=bool(good))
    jsave(ROOT/'calibration.json',dict(selected=selected,enabled=bool(allowed),by_variant=by_variant,options=options,optimizer_never_used_validation_images=True,heldout_development_not_used_for_choice=True,selection_before_development=True));print('V28_CALIBRATED',bool(allowed),selected,flush=True)

def classification():
    choice=read(ROOT/'calibration.json');policy();datasets={};z=np.load(ROOT/'trusted_train_pairs.npz');datasets['fit']=(z['x'][~np.isin(z['ids'],VALIDATION_IDS)],z['y'][~np.isin(z['ids'],VALIDATION_IDS)]);datasets['inner_validation']=(z['x'][np.isin(z['ids'],VALIDATION_IDS)],z['y'][np.isin(z['ids'],VALIDATION_IDS)]);v=np.load(ROOT/'relation_validation_pairs.npz');datasets['relation_validation']=(v['x'],v['y']);out={}
    for name in ['geometry','visual']:
        ensemble=models(name);threshold=choice['by_variant'][name]['selected']['threshold'];out[name]={}
        for split,(x,y) in datasets.items():
            probs=[]
            with torch.no_grad():
                for model,s in ensemble:
                    n=len(s['mean']);probs.append(model(torch.from_numpy(((x[:,:n]-s['mean'])/s['std']).astype(np.float32)).cuda()).sigmoid().cpu().numpy())
            p=np.mean(probs,0);out[name][split]=dict(positive=int((y==1).sum()),negative=int((y==0).sum()),true_positive=int(((p>=threshold)&(y==1)).sum()),false_positive=int(((p>=threshold)&(y==0)).sum()),false_negative=int(((p<threshold)&(y==1)).sum()),true_negative=int(((p<threshold)&(y==0)).sum()),threshold=threshold)
    jsave(ROOT/'classification.json',dict(reports=out,reference_chain_proxy_not_artist_stroke_accuracy=True,calibration_used_for_threshold_selection=True));print('V28_CLASSIFICATION',out,flush=True)

def evaluate():
    torch.set_num_threads(4);cv2.setNumThreads(2);choice=read(ROOT/'calibration.json');ensemble={name:models(name) for name in ['geometry','visual']};extractor=exp.load_model('base');rows=[];keys=['H02','R04','X04','N10','N11'];views={}
    for r,rgb,refs,valid,bg in exp.cases('development'):
        pred_file=ROOT/'predictions'/('development_'+r['id']+'.npz');pred=exp.prev.prior.predict(extractor,rgb)[:5];np.savez_compressed(pred_file,prediction=pred);g=add_visual(build_graph(pred,rgb),extractor);graph_file=ROOT/'graphs'/('development_'+r['id']+'.json');save_graph(graph_file,g,pred_file);pp={name:probabilities(g,ensemble[name]) for name in ensemble};selections={'unpaired':([],dict(reference_used=False,accepted_relations=0)),'geometric':(exp.rel.original_pairings(g),dict(reference_used=False,rule='previous_tangent_axis_mutual_margin'))}
        for name in ['geometry','visual']:
            selected=choice['by_variant'][name]['selected'];selections['learned_'+name]=select(g,pp[name],selected['threshold'],selected['margin'])
        selected=choice['selected'];selections['learned_selected']=select(g,pp[selected['variant']],selected['threshold'],selected['margin']);row=dict(id=r['id'],group=r['split'],metrics={},relationships={},candidate_count=len(g['candidates']))
        paths_by_variant={}
        # All model/geometry choices and assembly finish BEFORE reference matching.
        for variant,(indices,info) in selections.items():paths_by_variant[variant]=assemble(g,indices)
        label_graph(g,refs,valid)
        for variant,(indices,info) in selections.items():
            paths=paths_by_variant[variant];row['metrics'][variant]=exp.prev.evaluate_paths(refs,paths,valid,bg,rgb);row['relationships'][variant]=dict(accepted=len(indices),known_positive=int(sum(g['candidates'][i]['label']==1 for i in indices)),known_negative=int(sum(g['candidates'][i]['label']==0 for i in indices)),unknown=int(sum(g['candidates'][i]['label'] is None for i in indices)))
            events=[dict(node=g['candidates'][i]['node'],a=list(g['candidates'][i]['a']),b=list(g['candidates'][i]['b'])) for i in indices];jsave(ROOT/'trajectories'/variant/(r['id']+'.json'),dict(paths=[p.round(3).tolist() for p in paths],physical_z=None,coordinates='pixels',ready_for_robot=False,actually_drawn=False,reference_labels_used_at_inference=False,raw_edges_used_once=True,selected_relations=events,relation_info=info))
            if r['id'] in keys:views[r['id'],variant]=paths
        rows.append(row);print('V28_EVAL',r['id'],{k:round(v['mean_best_single_path_coverage'],3) for k,v in row['metrics'].items()},'learned_relations',row['relationships']['learned_selected'],flush=True)
    groups={};variants=['unpaired','geometric','learned_geometry','learned_visual','learned_selected']
    for group in sorted(set(r['group'] for r in rows)):
        rr=[r for r in rows if r['group']==group];groups[group]={}
        for name in variants:
            groups[group][name]={}
            for metric in exp.prev.METRICS:
                vals=[r['metrics'][name][metric] for r in rr if r['metrics'][name][metric] is not None];groups[group][name][metric]=float(np.mean(vals)) if vals else None
            groups[group][name]['known_positive_joins']=sum(r['relationships'][name]['known_positive'] for r in rr);groups[group][name]['known_negative_joins']=sum(r['relationships'][name]['known_negative'] for r in rr);groups[group][name]['unknown_joins']=sum(r['relationships'][name]['unknown'] for r in rr)
    jsave(ROOT/'evaluation.json',dict(records=rows,groups=groups,choice=choice,partial_references_not_full_image_accuracy=True,no_new_blind_test=True,no_reference_labels_at_inference=True,pixel_model_unchanged=True));lookup={r['id']:rgb for r,rgb,refs,valid,bg in exp.cases('development')};board=Image.new('RGB',(1600,len(keys)*310+40),'#f4f2eb');d=ImageDraw.Draw(board);d.text((5,5),'v28 同笔连接：输入／去重后未续接／几何续接／关系模型候选（全为离线轨迹）',font=FONT,fill='black')
    for i,key in enumerate(keys):
        rgb=lookup[key];images=[('输入',rgb)]+[(title,exp.prev.v21.line_view(rgb.shape[:2],views[key,name],True)) for name,title in [('unpaired','去重后未续接'),('geometric','几何规则续接'),('learned_selected','学习关系续接候选')]]
        for j,(title,im) in enumerate(images):d.text((j*400+5,40+i*310),key+' '+title,font=SMALL,fill='black');exp.prev.v21.paste(board,im,j*400+5,65+i*310,390,240)
    board.save(ROOT/'RESULT_COMPARISON.jpg',quality=95);print('V28_EVALUATED',flush=True)

def predict_one(input_path,output_path):
    source=Path(input_path);out=Path(output_path)
    if not out.is_absolute() or out.drive.upper() not in ['E:','F:']:raise ValueError('Use an absolute E:/F: output path')
    rgb=np.asarray(Image.open(source).convert('RGB'))
    if not exp.prev.v21.v16.monochrome_route(rgb):raise ValueError('Only monochrome rock research was evaluated')
    choice=read(ROOT/'calibration.json');model=exp.load_model('base');pred=exp.prev.prior.predict(model,rgb)[:5];g=add_visual(build_graph(pred,rgb),model);selected=choice['selected'];prob=probabilities(g,models(selected['variant']));indices,info=select(g,prob,selected['threshold'],selected['margin']);trial=assemble(g,indices);unpaired=assemble(g,[]);actual=trial if choice['enabled'] else unpaired
    out.mkdir(parents=True,exist_ok=True);jsave(out/'trajectory.json',dict(paths=[p.round(3).tolist() for p in actual],physical_z=None,coordinates='pixels',ready_for_robot=False,actually_drawn=False,reference_labels_used_at_inference=False,calibration_gate_passed=choice['enabled'],relations=info));np.savez_compressed(out/'prediction.npz',prediction=pred)
    board=Image.new('RGB',(1500,410),'#f4f2eb');d=ImageDraw.Draw(board)
    for j,(title,im) in enumerate([('输入',rgb),('去重后未续接',exp.prev.v21.line_view(rgb.shape[:2],unpaired,True)),('关系续接候选',exp.prev.v21.line_view(rgb.shape[:2],trial,True))]):d.text((500*j+5,7),title,font=FONT,fill='black');exp.prev.v21.paste(board,im,500*j+5,40,490,355)
    board.save(out/'comparison.jpg',quality=95);jsave(out/'summary.json',dict(input=str(source),input_sha256=digest(source),output_curves=len(actual),reference_labels_used=False,reference_label_matching_called=False,development_image_interface_check_not_blind_evaluation=True,device=str(next(model.parameters()).device),official_default_replaced=False,robot_connected=False));print('V28_REFERENCE_FREE',str(out),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','audit','train','calibrate','classification','evaluate','predict','finalize']);p.add_argument('--variant',choices=['geometry','visual']);p.add_argument('--input');p.add_argument('--output');a=p.parse_args();{'prepare':prepare,'audit':audit,'train':lambda:train(a.variant),'calibrate':calibrate,'classification':classification,'evaluate':evaluate,'predict':lambda:predict_one(a.input,a.output),'finalize':lambda:__import__('v28_finalize').main()}[a.mode]()
