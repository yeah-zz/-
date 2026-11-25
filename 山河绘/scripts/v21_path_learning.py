# -*- coding: utf-8 -*-
"""Learn conservative endpoint decisions from visible-chain cuts and nearby chains.
Targets describe assistant chain identities, not verified physical brush lifts.
"""
from v21_stroke_learning import *

class PairNet(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(11,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def feature(a,b,ta,tb,alpha):
    delta=b-a;length=float(np.linalg.norm(delta))
    if length<.1:return None
    direction=delta/length;normal=np.array([-direction[1],direction[0]],np.float32);p=a[None]+np.linspace(0,1,21)[:,None]*delta[None]
    def sample(off):
        q=p+normal[None]*off;return cv2.remap(alpha,q[:,0].astype(np.float32)[None],q[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]
    v=sample(0);side=(sample(4)+sample(-4))*.5
    return np.array([length/20,float(ta@direction),float(tb@(-direction)),float(ta@(-tb)),v.mean(),v.std(),np.quantile(v,.2),(v>.012).mean(),(v-side).mean(),(v>.04).mean(),(v[0]+v[-1])*.5],np.float32)

def unit(a):return a/max(1e-6,np.linalg.norm(a))

def pair_examples(r,seed):
    rng=np.random.default_rng(seed);rgb=np.asarray(Image.open(r['image']).convert('RGB'));alpha=v16.photometry(rgb)[1];paths=[dense(s['points']).astype(np.float32) for s in read(r['labels'])['strokes']];rows=[]
    for p in paths:
        if len(p)<18:continue
        for _ in range(22):
            gap=int(rng.integers(3,13));i=int(rng.integers(5,max(6,len(p)-gap-5)));j=min(len(p)-5,i+gap)
            if j<=i:continue
            f=feature(p[i],p[j],unit(p[i]-p[i-4]),unit(p[j]-p[j+4]),alpha)
            if f is not None and f[7]>.30:rows.append((f,1))
    for k,p in enumerate(paths):
        for l,q in enumerate(paths):
            if l<=k or len(p)<10 or len(q)<10:continue
            tree=cKDTree(q);pick=np.unique(np.linspace(4,len(p)-5,min(24,len(p)-8)).astype(int));dd,ii=tree.query(p[pick]);valid=(dd>=3)&(dd<=16)&(ii>=4)&(ii<len(q)-4)
            for a,b in zip(pick[valid],ii[valid]):
                for sa in [-1,1]:
                    for sb in [-1,1]:
                        ta=unit(p[a]-p[a+sa*4]);tb=unit(q[b]-q[b+sb*4]);f=feature(p[a],q[b],ta,tb,alpha)
                        if f is not None:rows.append((f,0))
    return rows

def train_pairs():
    torch.set_num_threads(3);torch.manual_seed(211026);rng=np.random.default_rng(211026);records=read(ROOT/'manifest.json')['records'];parts={}
    for split in ['train','calibration','development_holdout']:
        data=[]
        for i,r in enumerate(records):
            if r['split']==split:data.extend(pair_examples(r,211026+i))
        x=np.stack([a for a,b in data]);y=np.array([b for a,b in data],np.float32);parts[split]=(x,y)
    folder=ROOT/'endpoint_model';folder.mkdir(exist_ok=True)
    if (folder/'final.pt').exists():raise FileExistsError('Preserve trained endpoint model')
    x,y=parts['train'];mu=x.mean(0);sd=np.maximum(x.std(0),.08);xx=torch.from_numpy((x-mu)/sd);yy=torch.from_numpy(y);pos=np.flatnonzero(y>0);neg=np.flatnonzero(y==0);model=PairNet();opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    for step in range(600):
        ids=np.r_[rng.choice(pos,128),rng.choice(neg,128)];out=model(xx[ids]);loss=F.binary_cross_entropy_with_logits(out,yy[ids]);opt.zero_grad();loss.backward();opt.step()
    def scores(split):
        x,y=parts[split]
        with torch.no_grad():p=model(torch.from_numpy((x-mu)/sd)).sigmoid().numpy()
        return p,y
    p,y=scores('calibration');choices=[]
    for threshold in [.90,.95,.97,.98,.99,.995,.999,1.001]:
        fp=int(((p>=threshold)&(y==0)).sum());tp=int(((p>=threshold)&(y>0)).sum());n=int((y==0).sum());choices.append(dict(threshold=threshold,tp=tp,fp=fp,positive=int((y>0).sum()),negative=n,false_positive_rate=fp/max(1,n)))
    allowed=[a for a in choices if a['false_positive_rate']<=.005];chosen=max(allowed,key=lambda a:(a['tp'],-a['fp'])) if allowed else choices[-1]
    report={}
    for split in parts:
        p,y=scores(split);report[split]=dict(positive=int((y>0).sum()),negative=int((y==0).sum()),accepted_positive=int(((p>=chosen['threshold'])&(y>0)).sum()),accepted_negative=int(((p>=chosen['threshold'])&(y==0)).sum()))
    torch.save(dict(model=model.state_dict(),mean=mu,std=sd,threshold=chosen['threshold']),folder/'final.pt');jsave(folder/'summary.json',dict(successful_updates=600,selected_calibration=chosen,threshold_candidates=choices,records=report,assistant_chain_targets_not_expert_pen_lifts=True,artificial_cuts_not_real_gap_accuracy=True,external_X_images_not_used=True));print('V21_ENDPOINT_TRAINED',chosen,report,flush=True)

def load_pairs():
    folder=ROOT/'endpoint_real_fragments' if (ROOT/'endpoint_real_fragments'/'final.pt').exists() else ROOT/'endpoint_model';z=torch.load(folder/'final.pt',map_location='cpu',weights_only=False);z['model_kind']=folder.name;m=PairNet();m.load_state_dict(z['model']);return m.eval(),z

def real_fragments():
    folder=ROOT/'endpoint_real_fragments';folder.mkdir(exist_ok=True);base=load_model('improved');policy=read(ROOT/'improved'/'calibration.json')['selected'];parts={};detail=[]
    for split in ['train','calibration']:
        xx=[];yy=[]
        for r in read(ROOT/'manifest.json')['records']:
            if r['split']!=split:continue
            rgb=np.asarray(Image.open(r['image']).convert('RGB'));alpha=v16.photometry(rgb)[1];refs=[dense(s['points']).astype(np.float32) for s in read(r['labels'])['strokes']];trees=[cKDTree(p) for p in refs];pred=v16.predict(base,rgb);paths,_,_=prior_eval.decode(pred,policy['kind'],policy['high'],policy['low']);ends=[]
            for k,p in enumerate(paths):
                if len(p)<5:continue
                for end in [0,1]:
                    a=p[-1] if end else p[0];inner=p[-min(5,len(p))] if end else p[min(4,len(p)-1)];dist=np.array([t.query(a)[0] for t in trees]);order=np.argsort(dist)
                    if dist[order[0]]>3.5 or (len(order)>1 and dist[order[1]]-dist[order[0]]<1.5):continue
                    refid=int(order[0]);ends.append(dict(path=k,end=end,p=a,t=unit(a-inner),ref=refid,index=int(trees[refid].query(a)[1]),inner_index=int(trees[refid].query(inner)[1])))
            counters=dict(near_pairs=0,geometry_supported=0,positive=0,negative=0)
            for i,a in enumerate(ends):
                for b in ends[i+1:]:
                    if a['path']==b['path']:continue
                    gap=float(np.linalg.norm(a['p']-b['p']))
                    if not 2.5<=gap<=14:continue
                    counters['near_pairs']+=1;f=feature(a['p'],b['p'],a['t'],b['t'],alpha)
                    if f[1]<.45 or f[2]<.45 or f[7]<.35 or f[6]<.006:continue
                    counters['geometry_supported']+=1
                    if a['ref']==b['ref']:
                        if abs(a['index']-b['index'])>26:continue
                        ispositive=(b['index']-a['index'])*(a['index']-a['inner_index'])>0 and (a['index']-b['index'])*(b['index']-b['inner_index'])>0
                    else:ispositive=False
                    xx.append(f);yy.append(float(ispositive));counters['positive' if ispositive else 'negative']+=1
            detail.append(dict(id=r['id'],split=split,**counters))
        parts[split]=(np.stack(xx) if xx else np.zeros((0,11),np.float32),np.array(yy,np.float32))
    jsave(folder/'data_audit.json',dict(records=detail,annotation_and_decoding_derived_not_expert_true_joins=True,reference_model='v21_improved_not_updated_by_gap_training'))
    for split,(x,y) in parts.items():np.savez_compressed(folder/(split+'_pairs.npz'),x=x,y=y)
    counts={s:dict(positive=int((y>0).sum()),negative=int((y==0).sum())) for s,(x,y) in parts.items()};print('V21_REAL_FRAGMENT_COUNTS',counts,flush=True)
    if min(counts['train'].values())<10 or min(counts['calibration'].values())<5:
        jsave(folder/'summary.json',dict(trained=False,counts=counts,reason='Insufficient unambiguous real-decoder positive/negative endpoint pairs; keep abstention',official_model_promoted=False));return
    # A second checkpoint can only be fitted if real endpoint evidence is sufficient.
    torch.set_num_threads(3);torch.manual_seed(211027);rng=np.random.default_rng(211027);x,y=parts['train'];mu=x.mean(0);sd=np.maximum(x.std(0),.08);xx=torch.from_numpy((x-mu)/sd);yp=torch.from_numpy(y);pos=np.flatnonzero(y>0);neg=np.flatnonzero(y==0);m=PairNet();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    for _ in range(400):
        ids=np.r_[rng.choice(pos,64),rng.choice(neg,64)];loss=F.binary_cross_entropy_with_logits(m(xx[ids]),yp[ids]);opt.zero_grad();loss.backward();opt.step()
    x,y=parts['calibration']
    with torch.no_grad():p=m(torch.from_numpy((x-mu)/sd)).sigmoid().numpy()
    options=[]
    for t in [.9,.95,.97,.99,.995,.999,1.001]:options.append(dict(threshold=t,tp=int(((p>=t)&(y>0)).sum()),fp=int(((p>=t)&(y==0)).sum())))
    choice=max([a for a in options if a['fp']==0],key=lambda a:a['tp']);torch.save(dict(model=m.state_dict(),mean=mu,std=sd,threshold=choice['threshold']),folder/'final.pt');jsave(folder/'summary.json',dict(trained=True,successful_updates=400,counts=counts,selected=choice,calibration_options=options,small_development_sample_not_reliable_accuracy=True));print('V21_REAL_ENDPOINT_FIT',choice,flush=True)

def touching_junctions(paths,alpha):
    paths=[np.asarray(p,np.float32) for p in paths];events=[];ambiguous=0
    for iteration in range(4):
        candidates=[]
        for i,p in enumerate(paths):
            if len(p)<5:continue
            for j in range(i+1,len(paths)):
                q=paths[j]
                if len(q)<5:continue
                for ai,bi in [(0,0),(0,1),(1,0),(1,1)]:
                    a=p[-1] if ai else p[0];b=q[-1] if bi else q[0];gap=float(np.linalg.norm(a-b))
                    if gap>2.75:continue
                    ta=unit(a-p[-min(7,len(p))] if ai else a-p[min(6,len(p)-1)]);tb=unit(b-q[-min(7,len(q))] if bi else b-q[min(6,len(q)-1)]);align=float(ta@(-tb))
                    node=(a+b)*.5;others=[v for k,v in enumerate(paths) if k not in [i,j]];third_near=bool(others and cKDTree(np.concatenate(others)).query(node)[0]<6)
                    # A confirmed degree-two contact may be a legitimate sharp rock corner.
                    # Three-way junctions still require almost straight continuation.
                    if align<(.9 if third_near else -.25):continue
                    xy=np.rint((a+b)*.5).astype(int)
                    if not (0<=xy[0]<alpha.shape[1] and 0<=xy[1]<alpha.shape[0]) or alpha[xy[1],xy[0]]<.01:continue
                    candidates.append(dict(i=i,j=j,ai=ai,bi=bi,score=align,gap=gap,degree_two_contact=not third_near))
        used=set();merged=[]
        for c in sorted(candidates,key=lambda c:-c['score']):
            i,j=c['i'],c['j']
            if i in used or j in used:continue
            alternatives=[x['score'] for x in candidates if x is not c and ((x['i']==i and x['ai']==c['ai']) or (x['j']==i and x['bi']==c['ai']) or (x['i']==j and x['ai']==c['bi']) or (x['j']==j and x['bi']==c['bi']))]
            if alternatives and c['score']-max(alternatives)<.12:ambiguous+=1;continue
            p=paths[i] if c['ai'] else paths[i][::-1];q=paths[j][::-1] if c['bi'] else paths[j];merged.append(np.concatenate([p,q]));used|={i,j};events.append(dict(iteration=iteration,alignment=c['score'],gap=c['gap'],degree_two_contact=c['degree_two_contact']))
        if not merged:break
        paths=[p for k,p in enumerate(paths) if k not in used]+merged
    return paths,events,ambiguous

def joins(paths,rgb,model,params):
    paths=[np.asarray(p,np.float32) for p in paths];alpha=v16.photometry(rgb)[1];paths,node_events,node_abstained=touching_junctions(paths,alpha);events=[];rejected_ambiguity=0;threshold=params['threshold']
    for pass_id in range(3):
        candidates=[]
        for i,p in enumerate(paths):
            if len(p)<3:continue
            for j in range(i+1,len(paths)):
                q=paths[j]
                if len(q)<3:continue
                for ai,bi in [(0,0),(0,1),(1,0),(1,1)]:
                    a=p[-1] if ai else p[0];b=q[-1] if bi else q[0];gap=float(np.linalg.norm(b-a))
                    if not 2.5<=gap<=14:continue
                    ta=unit(a-p[-min(5,len(p))] if ai else a-p[min(4,len(p)-1)]);tb=unit(b-q[-min(5,len(q))] if bi else b-q[min(4,len(q)-1)]);f=feature(a,b,ta,tb,alpha)
                    if f[1]<.45 or f[2]<.45 or f[7]<.35 or f[6]<.006:continue
                    with torch.no_grad():score=float(model(torch.from_numpy((f-params['mean'])/params['std'])[None]).sigmoid()[0])
                    if score>=threshold:candidates.append(dict(i=i,j=j,ai=ai,bi=bi,score=score,gap=gap))
        if not candidates:break
        used=set();merges=[]
        for c in sorted(candidates,key=lambda x:-x['score']):
            i,j=c['i'],c['j']
            if i in used or j in used:continue
            competitors=[x['score'] for x in candidates if x is not c and ((x['i']==i and x['ai']==c['ai']) or (x['j']==i and x['bi']==c['ai']) or (x['i']==j and x['ai']==c['bi']) or (x['j']==j and x['bi']==c['bi']))]
            if competitors and c['score']-max(competitors)<.08:rejected_ambiguity+=1;continue
            p=paths[i] if c['ai'] else paths[i][::-1];q=paths[j][::-1] if c['bi'] else paths[j];mid=np.linspace(p[-1],q[0],max(3,round(c['gap'])))[1:-1];other=[x for k,x in enumerate(paths) if k not in [i,j]]
            if other and len(mid) and (cKDTree(np.concatenate(other)).query(mid)[0]<2).any():continue
            merges.append(np.concatenate([p,mid,q]));used|={i,j};events.append(dict(pass_id=pass_id,score=c['score'],gap=c['gap']))
        if not merges:break
        paths=[p for k,p in enumerate(paths) if k not in used]+merges
    return paths,dict(accepted_joins=events,touching_junction_splices=node_events,touching_candidates_abstained=node_abstained,competing_candidates_abstained=rejected_ambiguity,self_loop_joins_allowed=False,reference_labels_used_at_inference=False)

def calibrate_joins(variant):
    base=load_model(variant);pm,params=load_pairs();policy=read(ROOT/variant/'calibration.json')['selected'];rows=[]
    for r in read(ROOT/'manifest.json')['records']:
        if r['split']!='calibration':continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));refs=[s['points'] for s in read(r['labels'])['strokes']];pred=v16.predict(base,rgb);paths,_,_=prior_eval.decode(pred,policy['kind'],policy['high'],policy['low']);joined,info=joins(paths,rgb,pm,params);valid=np.ones(rgb.shape[:2]);rows.append(dict(id=r['id'],before=path_metrics(refs,paths,valid),after=path_metrics(refs,joined,valid),accepted_gap_joins=len(info['accepted_joins']),touching_node_splices=len(info['touching_junction_splices'])))
    old=float(np.mean([r['before']['mean_best_single_path_coverage'] for r in rows]));new=float(np.mean([r['after']['mean_best_single_path_coverage'] for r in rows]));oldmixed=sum(r['before']['mixed_reference_paths_flagged'] for r in rows);newmixed=sum(r['after']['mixed_reference_paths_flagged'] for r in rows);enabled=new>old+.005 and newmixed<=oldmixed;file=ROOT/variant/'endpoint_calibration.json'
    if file.exists() and not (file.parent/'initial_artificial_endpoint_calibration.json').exists():shutil.copy2(file,file.parent/'initial_artificial_endpoint_calibration.json')
    jsave(file,dict(enabled=bool(enabled),before=old,after=new,mixed_before=oldmixed,mixed_after=newmixed,records=rows,endpoint_model_kind=params['model_kind'],not_expert_false_join_ground_truth=True));print('V21_JOIN_CALIBRATION',variant,enabled,old,new,flush=True)

def evaluate_joins(variant):
    model=load_model(variant);pm,params=load_pairs();policy=read(ROOT/variant/'calibration.json')['selected'];enabled=read(ROOT/variant/'endpoint_calibration.json')['enabled'];folder=ROOT/variant/'joined_evaluation';folder.mkdir(exist_ok=True);rows=[]
    for r,rgb,refs,valid,bg,_ in evaluation_cases():
        pred=v16.predict(model,rgb);paths,_,_=prior_eval.decode(pred,policy['kind'],policy['high'],policy['low']);candidate,info=joins(paths,rgb,pm,params);actual=candidate if enabled else paths;before=diagnostic(refs,pred,paths,valid,rgb,bg);after=diagnostic(refs,pred,actual,valid,rgb,bg);trial=path_metrics(refs,candidate,valid);out=folder/r['id'];out.mkdir(exist_ok=True);jsave(out/'trajectory.json',dict(paths=[p.round(3).tolist() for p in actual],coordinates='pixels',ready_for_robot=False,actually_drawn=False,endpoint_policy_enabled=enabled,info=info));rows.append(dict(id=r['id'],split=r['split'],before=before,after=after,trial_if_enabled=trial,accepted_candidates=len(info['accepted_joins'])))
        page=Image.new('RGB',(1400,350),'#f5f3ec');d=ImageDraw.Draw(page)
        for j,(title,a) in enumerate([('原图',rgb),('原解码路径',line_view(rgb.shape[:2],paths,True)),('连接试验候选',line_view(rgb.shape[:2],candidate,True)),('辅助参考（非预测）',line_view(rgb.shape[:2],refs,True))]):d.text((350*j+4,4),r['id']+' '+title,font=SMALL,fill='black');paste(page,a,350*j+4,32,340,310)
        page.save(out/'comparison.jpg',quality=95);print('V21_JOIN_EVAL',variant,r['id'],len(info['accepted_joins']),round(trial['mean_best_single_path_coverage'],3),flush=True)
    jsave(folder/'summary.json',dict(records=rows,endpoint_policy_enabled=enabled,calibration_decision_frozen=True,reference_used_at_inference=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','real','calibrate','evaluate']);p.add_argument('variant',nargs='?',default='improved');a=p.parse_args()
    if a.mode=='train':train_pairs()
    elif a.mode=='real':real_fragments()
    elif a.mode=='calibrate':calibrate_joins(a.variant)
    else:evaluate_joins(a.variant)
