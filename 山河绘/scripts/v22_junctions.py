# -*- coding: utf-8 -*-
"""Offline branch association at real predicted junctions; frozen v21 pixel model."""
from v21_stroke_learning import *
import v21_stroke_learning as v21
from stroke_link_v12 import tangent
BASE21=v21.ROOT
ROOT=LAB/'junction_strokes_v22_20261025'

def graph_parts(sk):
    h,w=sk.shape;pixels=set(map(tuple,np.argwhere(sk)));neighbors={}
    for y,x in pixels:
        near=[]
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                q=(y+dy,x+dx)
                if (dx==0 and dy==0) or q not in pixels:continue
                if dx and dy and ((y,x+dx) in pixels or (y+dy,x) in pixels):continue
                near.append(q)
        neighbors[y,x]=near
    high=np.zeros((h,w),np.uint8)
    for p,near in neighbors.items():
        if len(near)>2:high[p]=1
    expanded=cv2.dilate(high,np.ones((5,5),np.uint8))*sk.astype(np.uint8);n,cc=cv2.connectedComponents(expanded,8);node_of={};centers={};sizes={};nn=0
    for i in range(1,n):
        yy,xx=np.nonzero(cc==i);centers[nn]=np.array([xx.mean(),yy.mean()]);sizes[nn]=len(yy);node_of.update({(y,x):nn for y,x in zip(yy.tolist(),xx.tolist())});nn+=1
    for p,near in neighbors.items():
        if len(near)!=2 and p not in node_of:node_of[p]=nn;centers[nn]=np.array([p[1],p[0]],float);sizes[nn]=1;nn+=1
    visited=set();chains=[];nodes=[]
    def edge(a,b):return tuple(sorted((a,b)))
    for pixel,node in sorted(node_of.items()):
        for q in neighbors[pixel]:
            if node_of.get(q)==node or edge(pixel,q) in visited:continue
            visited.add(edge(pixel,q));points=[centers[node],np.array([pixel[1],pixel[0]],float)];previous=pixel;position=q;end=None
            for _ in range(len(pixels)+1):
                if position in node_of:end=node_of[position];points.append(centers[end]);break
                points.append(np.array([position[1],position[0]],float));options=[z for z in neighbors[position] if z!=previous]
                if not options:break
                following=options[0];visited.add(edge(position,following));previous,position=position,following
            if len(points)>2:chains.append(np.asarray(points));nodes.append((node,end))
    for a in sorted(pixels):
        for b in neighbors[a]:
            if edge(a,b) in visited or (a in node_of and b in node_of):continue
            p=[np.array([a[1],a[0]],float)];previous=a;position=b;visited.add(edge(a,b))
            for _ in range(len(pixels)+1):
                p.append(np.array([position[1],position[0]],float))
                if position==a:break
                options=[z for z in neighbors[position] if z!=previous and edge(position,z) not in visited]
                if not options:break
                following=options[0];visited.add(edge(position,following));previous,position=position,following
            if len(p)>2:chains.append(np.asarray(p));nodes.append((None,None))
    incident={}
    for k,ends in enumerate(nodes):
        for side,node in enumerate(ends):
            if node is not None:incident.setdefault(node,[]).append((k,side))
    return dict(chains=chains,nodes=nodes,centers=centers,sizes=sizes,incident=incident,shape=(h,w))

def chain_length(p):return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())

def usable_incident(g,prune=True):
    active=set(range(len(g['chains'])));removed=[]
    if prune:
        for k,p in enumerate(g['chains']):
            ends=g['nodes'][k];leaf=any(n is not None and len(g['incident'].get(n,[]))==1 for n in ends)
            if leaf and chain_length(p)<8:active.remove(k);removed.append(k)
    incident={node:[e for e in ends if e[0] in active] for node,ends in g['incident'].items()}
    return active,incident,removed

def arm(p,side,span=24):
    q=p if side==0 else p[::-1];q=dense(q);distance=np.linalg.norm(q-q[0],axis=1);return q[(distance>=4)&(distance<=span)]

def mean_at(a,q):
    if not len(q):return 0.
    return float(cv2.remap(a,q[:,0].astype(np.float32)[None],q[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).mean())

def pair_features(g,node,a,b,pred,alpha,ends):
    ia,sa=a;ib,sb=b;pa=g['chains'][ia];pb=g['chains'][ib];va=[tangent(pa,sa==0,span) for span in [4,8,14]];vb=[tangent(pb,sb==0,span) for span in [4,8,14]];align=[float(v@(-u)) for v,u in zip(va,vb)];la=chain_length(pa);lb=chain_length(pb);qa=arm(pa,sa);qb=arm(pb,sb);x,y=np.rint(g['centers'][node]).astype(int);x=int(np.clip(x,0,alpha.shape[1]-1));y=int(np.clip(y,0,alpha.shape[0]-1));theta=.5*math.atan2(pred[3,y,x],pred[2,y,x]);axis=np.array([math.cos(theta),math.sin(theta)]);field=abs(float(va[1]@axis))*abs(float(vb[1]@axis));compa=[];compb=[]
    for e in ends:
        if e in [a,b]:continue
        t=tangent(g['chains'][e[0]],e[1]==0,8);compa.append(float(va[1]@(-t)));compb.append(float(vb[1]@(-t)))
    wa=mean_at(pred[4],qa);wb=mean_at(pred[4],qb)
    return np.array(align+[float(va[0]@va[2]),float(vb[0]@vb[2]),math.log1p(min(la,lb))/6,math.log1p(max(la,lb))/6,len(ends)/8,math.log1p(g['sizes'][node])/4,field,float(pred[0,y,x]),float(pred[1,y,x]),float(alpha[y,x]),mean_at(pred[0],qa),mean_at(pred[0],qb),mean_at(alpha,qa),mean_at(alpha,qb),abs(wa-wb),align[1]-max(compa or [-1]),align[1]-max(compb or [-1])],np.float32)

def match_arm(p,side,refs,trees,node_xy):
    q=arm(p,side,28)
    if len(q)<7:return None
    matches=[]
    for k,tree in enumerate(trees):
        d,ii=tree.query(q);frac=float((d<=3.5).mean());med=float(np.median(d));matches.append((frac,-med,k,ii))
    matches.sort(key=lambda a:(a[0],a[1]),reverse=True);best=matches[0]
    if best[0]<.72 or -best[1]>3 or (len(matches)>1 and best[0]-matches[1][0]<.22 and best[1]-matches[1][1]<1.5):return None
    refid=best[2];node_dist,node_index=trees[refid].query(node_xy)
    if node_dist>7:return None
    index=float(np.median(best[3]));return dict(ref=refid,reference_index=index,node_reference_index=int(node_index),purity=best[0])

def sources():
    rows=read(BASE21/'manifest.json')['records'];out=[]
    for r in rows:
        if r['split'] in ['train','calibration']:out.append(dict(r,kind='assistant_chain'))
    for r in read(V19/'manifest_v2.json')['records']:
        if r['id'] in ['R01','R02']:out.append(dict(r,split='train',work_group='user_'+r['id'],kind='user_partial_chain'))
    return out

def prepare():
    torch.set_num_threads(4);cv2.setNumThreads(2);ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'cache').mkdir(exist_ok=True);model=v21.load_model('ribbon_improved');policy=read(BASE21/'ribbon_improved'/'calibration.json')['selected'];data=[];manifest=[];audit=[]
    for r in sources():
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));refs=[dense(s['points']) for s in read(r['labels'])['strokes']];trees=[cKDTree(q) for q in refs];alpha=v16.photometry(rgb)[1];pred=v16.predict(model,rgb);paths,sk,info=prior_eval.decode(pred,policy['kind'],policy['high'],policy['low']);g=graph_parts(sk);active,incident,removed=usable_incident(g,True);counts=dict(positive=0,negative=0,unknown=0,degree2=0,junctions=0);samples=[];overlay=rgb.copy()
        for node,ends in incident.items():
            if len(ends)<2:continue
            counts['junctions']+=1
            if len(ends)==2:counts['degree2']+=1
            matches={e:match_arm(g['chains'][e[0]],e[1],refs,trees,g['centers'][node]) for e in ends}
            for i,a in enumerate(ends):
                for b in ends[i+1:]:
                    if a[0]==b[0]:continue
                    ma,mb=matches[a],matches[b];label=None
                    if ma is not None and mb is not None:
                        if ma['ref']==mb['ref']:
                            da=ma['reference_index']-ma['node_reference_index'];db=mb['reference_index']-mb['node_reference_index']
                            if da*db<0 and abs(da)>3 and abs(db)>3:label=1
                        else:label=0
                    counts['unknown' if label is None else 'positive' if label else 'negative']+=1
                    if label is None:continue
                    f=pair_features(g,node,a,b,pred,alpha,ends);sample=dict(id=r['id'],group=r['work_group'],split=r['split'],node=node,degree=len(ends),a=list(a),b=list(b),target=label,features=f.tolist(),provenance=r['kind']);samples.append(sample);data.append(sample)
                    xy=tuple(np.rint(g['centers'][node]).astype(int));cv2.circle(overlay,xy,3,(40,170,50) if label else (220,70,40),-1)
        np.savez_compressed(ROOT/'cache'/(r['id']+'.npz'),rgb=rgb,prediction=pred,skeleton=sk);manifest.append(dict(id=r['id'],image=r['image'],labels=r['labels'],image_sha256=digest(r['image']),label_sha256=digest(r['labels']),split=r['split'],work_group=r['work_group'],raw_chains=len(g['chains']),pruned_short_leaves=len(removed),**counts));print('V22_NODE_LABELS',r['id'],counts,flush=True)
        if r['id'] in ['R01','R02','stage_035','stage_055','sina_6586334590_26']:
            page=Image.new('RGB',(1200,350),'white');d=ImageDraw.Draw(page)
            for j,(title,a) in enumerate([('原图',rgb),('旧路径',line_view(rgb.shape[:2],paths,True)),('派生节点监督，绿正／红负',overlay)]):d.text((400*j+5,5),r['id']+' '+title,font=SMALL,fill='black');paste(page,a,400*j+5,35,390,305)
            audit.append(page)
    jsave(ROOT/'node_samples.json',dict(records=data,targets_are_reference_chain_relations_not_expert_pen_lifts=True));jsave(ROOT/'manifest.json',dict(records=manifest,frozen_pixel_weights=str(BASE21/'ribbon_improved'/'final.pt'),frozen_pixel_sha256=digest(BASE21/'ribbon_improved'/'final.pt'),v21_reference_manifest_sha256=digest(BASE21/'manifest.json'),old_skeleton_decode_policy=policy,unknown_relations_not_negative=True,robot_connected=False));shutil.copy2(Path(__file__),ROOT/'source_before_training.py')
    board=Image.new('RGB',(1200,350*len(audit)),'white')
    for i,a in enumerate(audit):board.paste(a,(0,i*350))
    board.save(ROOT/'NODE_SUPERVISION_AUDIT.jpg',quality=95);print('V22_DATA',len(data),flush=True)

def transform_image(rgb,refs,rot,scale):
    h,w=rgb.shape[:2];im=np.rot90(rgb,rot).copy();rr=[]
    for p in refs:
        p=np.asarray(p).copy();x=p[:,0].copy();y=p[:,1].copy()
        if rot==1:p=np.c_[y,w-1-x]
        elif rot==2:p=np.c_[w-1-x,h-1-y]
        elif rot==3:p=np.c_[h-1-y,x]
        rr.append(p)
    hh,ww=im.shape[:2];nh,nw=round(hh*scale),round(ww*scale);im=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR);rr=[(p+.5)*np.array([nw/ww,nh/hh])-.5 for p in rr];return im,rr

def label_graph(g,pred,rgb,refs,r,tag):
    trees=[cKDTree(q) for q in refs];alpha=v16.photometry(rgb)[1];active,incident,removed=usable_incident(g,True);rows=[]
    for node,ends in incident.items():
        if len(ends)<2:continue
        matches={e:match_arm(g['chains'][e[0]],e[1],refs,trees,g['centers'][node]) for e in ends}
        for i,a in enumerate(ends):
            for b in ends[i+1:]:
                if a[0]==b[0]:continue
                ma,mb=matches[a],matches[b];label=None
                if ma is not None and mb is not None:
                    if ma['ref']==mb['ref']:
                        da=ma['reference_index']-ma['node_reference_index'];db=mb['reference_index']-mb['node_reference_index']
                        if da*db<0 and abs(da)>3 and abs(db)>3:label=1
                    else:label=0
                if label is not None:rows.append(dict(id=r['id'],group=r['work_group'],split=r['split'],augmentation=tag,node=node,degree=len(ends),target=label,features=pair_features(g,node,a,b,pred,alpha,ends).tolist(),provenance=r['kind']))
    return rows

def augment():
    torch.set_num_threads(4);cv2.setNumThreads(2);model=v21.load_model('ribbon_improved');policy=read(ROOT/'manifest.json')['old_skeleton_decode_policy'];data=read(ROOT/'node_samples.json')['records'][:];counts=[]
    for r in sources():
        if r['split']!='train':continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));refs=[dense(s['points']) for s in read(r['labels'])['strokes']];total=[]
        for rot in range(4):
            for scale in [.8,1.2]:
                im,rr=transform_image(rgb,refs,rot,scale);pred=v16.predict(model,im);_,sk,_=prior_eval.decode(pred,policy['kind'],policy['high'],policy['low']);rows=label_graph(graph_parts(sk),pred,im,rr,r,str(rot)+'_'+str(scale));data.extend(rows);total.extend(rows)
        counts.append(dict(id=r['id'],positive=sum(x['target']==1 for x in total),negative=sum(x['target']==0 for x in total)));print('V22_AUGMENT',r['id'],counts[-1],flush=True)
    jsave(ROOT/'augmented_node_samples.json',dict(records=data,augmentation_counts=counts,underlying_training_images=len(counts),augmented_views_are_not_independent_artworks=True,calibration_views_not_augmented=True));print('V22_AUGMENT_TOTAL',len(data),flush=True)

class JunctionNet(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(20,48),nn.SiLU(),nn.Dropout(.1),nn.Linear(48,24),nn.SiLU(),nn.Linear(24,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def train():
    torch.set_num_threads(3);torch.manual_seed(221025);rng=np.random.default_rng(221025);doc=read(ROOT/'augmented_node_samples.json');rows=[r for r in doc['records'] if r['split']=='train'];x=np.array([r['features'] for r in rows],np.float32);y=np.array([r['target'] for r in rows],np.float32);pos=np.flatnonzero(y>0);neg=np.flatnonzero(y==0)
    if len(pos)<20 or len(neg)<20:raise RuntimeError('Insufficient explicit branch targets; do not force unknown into negatives')
    mean=x.mean(0);std=np.maximum(x.std(0),.08);xx=torch.from_numpy((x-mean)/std);yy=torch.from_numpy(y);model=JunctionNet();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02);folder=ROOT/'junction_model';folder.mkdir(exist_ok=True)
    if (folder/'final.pt').exists():raise FileExistsError('Preserve completed checkpoint')
    def weights(ids):
        groups=[rows[i]['group'] for i in ids];return np.array([1/groups.count(g) for g in groups])/len(set(groups))
    wp,wn=weights(pos),weights(neg);start=time.perf_counter()
    for step in range(600):
        ids=np.r_[rng.choice(pos,64,p=wp),rng.choice(neg,64,p=wn)];logits=model(xx[ids]);loss=F.binary_cross_entropy_with_logits(logits,yy[ids]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        if step%100==99:print('V22_JUNCTION_TRAIN',step+1,float(loss.detach()),flush=True)
    model.eval();torch.save(dict(model=model.state_dict(),mean=mean,std=std),folder/'final.pt');cal=[r for r in doc['records'] if r['split']=='calibration'];calx=torch.from_numpy((np.array([r['features'] for r in cal],np.float32)-mean)/std)
    with torch.no_grad():cp=model(calx).sigmoid().numpy()
    jsave(folder/'training_summary.json',dict(successful_updates=600,elapsed_seconds=time.perf_counter()-start,training_pairs=len(rows),positive=int(len(pos)),negative=int(len(neg)),underlying_training_work_groups=sorted(set(r['group'] for r in rows)),calibration_pairs=[dict(id=r['id'],target=r['target'],score=float(p)) for r,p in zip(cal,cp)],tiny_partial_development_labels_not_expert_accuracy=True,unknown_relations_never_trained_as_negative=True,frozen_pixel_weights_sha256=digest(BASE21/'ribbon_improved'/'final.pt')));print('V22_JUNCTION_TRAINED',len(pos),len(neg),flush=True)

def load_junction():
    z=torch.load(ROOT/'junction_model'/'final.pt',map_location='cpu',weights_only=False);m=JunctionNet();m.load_state_dict(z['model']);return m.eval(),z

def assemble(g,active,paired,min_length=8):
    used=set();paths=[];starts=[(i,s) for i in sorted(active) for s in [0,1] if (i,s) not in paired]+[(i,0) for i in sorted(active)]
    for index,side in starts:
        if index in used:continue
        segments=[];first=(index,side)
        while index not in used:
            used.add(index);segments.append(g['chains'][index] if side==0 else g['chains'][index][::-1]);nxt=paired.get((index,1-side))
            if nxt is None:break
            index,side=nxt
            if (index,side)==first:break
        p=np.concatenate(segments)
        if chain_length(p)>=min_length:paths.append(p)
    return paths

def legacy_pairs(g,pred,incident):
    paired={};events=[]
    for node,ends in incident.items():
        if len(ends)<2:continue
        x,y=np.rint(g['centers'][node]).astype(int);theta=.5*math.atan2(pred[3,y,x],pred[2,y,x]);axis=np.array([math.cos(theta),math.sin(theta)]);candidates=[];by_end={}
        for i,a in enumerate(ends):
            for b in ends[i+1:]:
                if a[0]==b[0]:continue
                va=tangent(g['chains'][a[0]],a[1]==0);vb=tangent(g['chains'][b[0]],b[1]==0);score=.8*float(va@(-vb))+.2*abs(float(va@axis))*abs(float(vb@axis))
                if score<.72:continue
                k=len(candidates);candidates.append((score,a,b));by_end.setdefault(a,[]).append((score,k));by_end.setdefault(b,[]).append((score,k))
        for k,(score,a,b) in enumerate(candidates):
            aa=sorted(by_end[a],reverse=True);bb=sorted(by_end[b],reverse=True)
            if aa[0][1]!=k or bb[0][1]!=k:continue
            if (len(aa)>1 and aa[0][0]-aa[1][0]<.08) or (len(bb)>1 and bb[0][0]-bb[1][0]<.08):continue
            if a not in paired and b not in paired:paired[a]=b;paired[b]=a;events.append(dict(node=node,kind='legacy',score=score))
    return paired,events

def decode_graph(g,pred,rgb,config,model=None,params=None):
    mode=config['mode'];active,incident,removed=usable_incident(g,mode!='legacy');paired,events=legacy_pairs(g,pred,incident);alpha=v16.photometry(rgb)[1]
    if mode in ['degree2','learned']:
        for node,ends in incident.items():
            if len(ends)!=2 or ends[0][0]==ends[1][0] or any(e in paired for e in ends):continue
            a,b=ends;x,y=np.rint(g['centers'][node]).astype(int);va=tangent(g['chains'][a[0]],a[1]==0,8);vb=tangent(g['chains'][b[0]],b[1]==0,8);alignment=float(va@(-vb))
            if min(chain_length(g['chains'][a[0]]),chain_length(g['chains'][b[0]]))<12 or g['sizes'][node]>24 or alignment<-.35 or alpha[y,x]<.01 or pred[0,y,x]<.35:continue
            paired[a]=b;paired[b]=a;events.append(dict(node=node,kind='unique_degree2_corner',alignment=alignment))
    if mode=='learned':
        for node,ends in incident.items():
            free=[e for e in ends if e not in paired]
            if len(free)<2 or len(free)>8:continue
            candidates=[]
            for i,a in enumerate(free):
                for b in free[i+1:]:
                    if a[0]==b[0]:continue
                    f=pair_features(g,node,a,b,pred,alpha,ends)
                    with torch.no_grad():p=float(model(torch.from_numpy((f-params['mean'])/params['std'])[None]).sigmoid()[0])
                    if p>=config['threshold']:candidates.append((p,a,b))
            # Global node matching includes abstaining; reject close alternative matchings.
            options=[]
            def visit(available,selected,score):
                if not available:options.append((score,selected));return
                a=available[0];visit(available[1:],selected,score)
                for p,aa,bb in candidates:
                    if a not in [aa,bb]:continue
                    b=bb if a==aa else aa
                    if b not in available[1:]:continue
                    visit([e for e in available[1:] if e!=b],selected+[(p,a,b)],score+(p-config['threshold'])/max(1e-5,1-config['threshold']))
            visit(free,[],0.);options.sort(key=lambda x:-x[0])
            if not options or not options[0][1] or (len(options)>1 and options[0][0]-options[1][0]<config['margin']):continue
            for p,a,b in options[0][1]:paired[a]=b;paired[b]=a;events.append(dict(node=node,kind='learned_branch_relation',score=p))
    paths=assemble(g,active,paired);return paths,dict(raw_chains=len(g['chains']),short_leaf_chains_pruned=len(removed),pairings=events,reference_at_inference=False,center_pixels_added=False)

def calibrate():
    model,params=load_junction();configs=[dict(mode='legacy'),dict(mode='prune'),dict(mode='degree2')]+[dict(mode='learned',threshold=t,margin=m) for t in [.8,.9,.97] for m in [.04,.1]];cases=[]
    for r in read(ROOT/'manifest.json')['records']:
        if r['split']!='calibration':continue
        z=np.load(ROOT/'cache'/(r['id']+'.npz'));refs=[s['points'] for s in read(r['labels'])['strokes']];cases.append((r,z['rgb'],z['prediction'],graph_parts(z['skeleton']),refs))
    rows=[]
    for cfg in configs:
        results=[]
        for r,rgb,pred,g,refs in cases:
            paths,info=decode_graph(g,pred,rgb,cfg,model,params);metrics=path_metrics(refs,paths,np.ones(rgb.shape[:2]));results.append(dict(id=r['id'],metrics=metrics,new_pairings=sum(e['kind']!='legacy' for e in info['pairings'])))
        score=float(np.mean([.55*x['metrics']['mean_best_single_path_coverage']+.45*x['metrics']['point_coverage_3px']-.04*x['metrics']['mixed_reference_paths_flagged'] for x in results]));rows.append(dict(config=cfg,score=score,records=results))
    baseline=rows[0];basepoint=float(np.mean([r['metrics']['point_coverage_3px'] for r in baseline['records']]));basemixed=sum(r['metrics']['mixed_reference_paths_flagged'] for r in baseline['records']);allowed=[r for r in rows if np.mean([x['metrics']['point_coverage_3px'] for x in r['records']])>=basepoint-.003 and sum(x['metrics']['mixed_reference_paths_flagged'] for x in r['records'])<=basemixed];selected=max(allowed,key=lambda r:r['score']);jsave(ROOT/'calibration.json',dict(selected=selected,candidates=rows,selection_ids=[r['id'] for r,_,_,_,_ in cases],partial_development_not_blind=True,external_X_and_R03_R04_not_used=True));print('V22_CALIBRATED',selected['config'],selected['score'],flush=True)

def loop_count(paths):return sum(bool(np.linalg.norm(p[0]-p[-1])<2 and chain_length(p)<60) for p in paths if len(p)>2)

def evaluate():
    model,params=load_junction();selected=read(ROOT/'calibration.json')['selected']['config'];configs={'legacy':dict(mode='legacy'),'degree2':dict(mode='degree2'),'selected':selected};rows=[];comparisons=[];reproduction=[];physical=[]
    for r,rgb,refs,valid,bg,_ in v21.evaluation_cases():
        key=r['id'];cached=BASE21/'ribbon_improved'/'evaluation'/key;pred=np.load(cached/'prediction.npz')['prediction'];policy=read(ROOT/'manifest.json')['old_skeleton_decode_policy'];v21_paths,sk,_=prior_eval.decode(pred,policy['kind'],policy['high'],policy['low']);g=graph_parts(sk);rawmask=v21.raster(g['chains'],rgb.shape[:2]);out=ROOT/'evaluation'/key;out.mkdir(parents=True,exist_ok=True);case=dict(id=key,split='legacy_related_probe' if key in ['stage_018','stage_029'] else r['split'],models={});pathsets={};saved=read(cached/'trajectory.json')['paths'];savedmask=v21.raster(saved,rgb.shape[:2])
        for name,cfg in configs.items():
            paths,info=decode_graph(g,pred,rgb,cfg,model,params);pathsets[name]=paths;metric=v21.diagnostic(refs,pred,paths,valid,rgb,bg);metric['small_closed_path_count']=int(loop_count(paths));metric['new_pairings']=sum(e['kind']!='legacy' for e in info['pairings']);metric['learned_pairings']=sum(e['kind']=='learned_branch_relation' for e in info['pairings']);metric['removed_short_leaves']=info['short_leaf_chains_pruned'];raster=v21.raster(paths,rgb.shape[:2]);metric['drawn_pixels_outside_raw_graph']=int(((raster>0)&(rawmask==0)).sum());case['models'][name]=metric;jsave(out/(name+'_trajectory.json'),dict(paths=[p.round(3).tolist() for p in paths],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,reference_at_inference=False,frozen_pixel_weights=True,config=cfg,trace_info=info));physical.append(bool(all(np.isfinite(p).all() and (p>=0).all() and (p[:,0]<rgb.shape[1]).all() and (p[:,1]<rgb.shape[0]).all() for p in paths)))
            if name=='legacy':reproduction.append(dict(id=key,same_path_count=len(paths)==len(saved),same_raster=np.array_equal(raster,savedmask),coverage_difference=metric['mean_best_single_path_coverage']-path_metrics(refs,saved,valid)['mean_best_single_path_coverage']))
        jsave(out/'metrics.json',case);rows.append(case);page=Image.new('RGB',(1800,365),'#f5f3ec');d=ImageDraw.Draw(page)
        for j,(title,a) in enumerate([('原图',rgb),('v21 旧解码',line_view(rgb.shape[:2],pathsets['legacy'],True)),('短支线／拐角规则试验',line_view(rgb.shape[:2],pathsets['degree2'],True)),('校准选定的 v22 候选',line_view(rgb.shape[:2],pathsets['selected'],True))]):d.text((450*j+5,5),key+' '+title,font=SMALL,fill='black');paste(page,a,450*j+5,35,440,320)
        page.save(out/'comparison.jpg',quality=95);comparisons.append((key,page));print('V22_EVAL',key,{k:round(v['mean_best_single_path_coverage'],3) for k,v in case['models'].items()},flush=True)
    aggregate={}
    for split in ['development_holdout','fixed_external_development','user_reviewed_old_development','legacy_related_probe']:
        rr=[r for r in rows if r['split']==split];aggregate[split]={}
        for name in configs:
            m=[r['models'][name] for r in rr];a={k:float(np.mean([x[k] for x in m])) for k in ['point_coverage_3px','mean_best_single_path_coverage','mean_reference_fragments','path_count']};a.update(images=len(m),mixed_flags=sum(x['mixed_reference_paths_flagged'] for x in m),small_closed_path_count=sum(x['small_closed_path_count'] for x in m),new_pairings=sum(x['new_pairings'] for x in m),learned_pairings=sum(x['learned_pairings'] for x in m));aggregate[split][name]=a
    jsave(ROOT/'evaluation'/'summary.json',dict(records=rows,aggregate=aggregate,selected_config=selected,pixel_model_unchanged=True,probability_false_positive_rates_unchanged=True,source_labels_partial_not_expert=True));jsave(ROOT/'baseline_reproduction.json',dict(records=reproduction,all_same_raster=all(r['same_raster'] for r in reproduction),all_same_path_count=all(r['same_path_count'] for r in reproduction)));jsave(ROOT/'trajectory_validation.json',dict(all_finite_and_in_bounds=all(physical),all_offline=True,raw_graph_geometry_only=all(x['drawn_pixels_outside_raw_graph']==0 for r in rows for x in r['models'].values())))
    selected_ids=['stage_065','stage_049','sina_6586334590_31','X01','X04','R04'];pictures=dict(comparisons);board=Image.new('RGB',(1800,365*len(selected_ids)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((5,4),'v22 真实离线路径对照；冻结像素模型，颜色区分路径，未在机械臂执行',font=FONT,fill='black')
    for i,key in enumerate(selected_ids):board.paste(pictures[key],(0,35+i*365))
    board.save(ROOT/'RESULT_COMPARISON.jpg',quality=95);print('V22_AGGREGATE',json.dumps(aggregate),flush=True)

def report():
    summary=read(ROOT/'evaluation'/'summary.json');data=read(ROOT/'augmented_node_samples.json');training=read(ROOT/'junction_model'/'training_summary.json');manifest=read(ROOT/'manifest.json');cal=read(ROOT/'calibration.json');checks=dict(frozen_pixel_weights_unchanged=digest(manifest['frozen_pixel_weights'])==manifest['frozen_pixel_sha256'],source_images_and_labels_unchanged=all(digest(r['image'])==r['image_sha256'] and digest(r['labels'])==r['label_sha256'] for r in manifest['records']),v21_reference_manifest_unchanged=digest(BASE21/'manifest.json')==manifest['v21_reference_manifest_sha256'],legacy_raster_reproduced=read(ROOT/'baseline_reproduction.json')['all_same_raster'],legacy_path_counts_reproduced=read(ROOT/'baseline_reproduction.json')['all_same_path_count'],trajectories_finite=read(ROOT/'trajectory_validation.json')['all_finite_and_in_bounds'],trajectories_use_existing_raw_graph_only=read(ROOT/'trajectory_validation.json')['raw_graph_geometry_only'])
    state=torch.load(ROOT/'junction_model'/'final.pt',map_location='cpu',weights_only=False)['model'];checks['junction_weights_finite']=all(bool(torch.isfinite(v).all()) for v in state.values());jsave(ROOT/'verification.json',dict(checks=checks,passed=all(checks.values()),robot_connected=False))
    ag=summary['aggregate'];gain={s:ag[s]['selected']['mean_best_single_path_coverage']-ag[s]['legacy']['mean_best_single_path_coverage'] for s in ag};nonregress=all(ag[s]['selected']['point_coverage_3px']>=ag[s]['legacy']['point_coverage_3px']-.003 and ag[s]['selected']['mixed_flags']<=ag[s]['legacy']['mixed_flags'] and ag[s]['selected']['small_closed_path_count']<=ag[s]['legacy']['small_closed_path_count'] for s in ['development_holdout','fixed_external_development','user_reviewed_old_development']);gate=dict(research_candidate=cal['selected']['config'],offline_default_replaced=False,original_v16_baseline_untouched=True,development_regression_gate_passed=bool(nonregress and min(gain[s] for s in ['development_holdout','fixed_external_development','user_reviewed_old_development'])>=0 and max(gain.values())>.005),not_independent_blind_test=True,robot_connected=False);jsave(ROOT/'deployment_gate.json',gate)
    table='\n'.join('| '+label+' | '+' | '.join(f"{ag[s][name]['mean_best_single_path_coverage']:.3f}" for name in ['legacy','degree2','selected'])+' |' for s,label in [('development_holdout','6 张开发图'),('fixed_external_development','X01–X04 固定开发图'),('user_reviewed_old_development','R03/R04 用户局部参考')]);detail='\n'.join('| '+label+' | '+str(ag[s]['legacy']['mixed_flags'])+' → '+str(ag[s]['selected']['mixed_flags'])+' | '+f"{ag[s]['legacy']['mean_reference_fragments']:.2f} → {ag[s]['selected']['mean_reference_fragments']:.2f}"+' |' for s,label in [('development_holdout','6 张开发图'),('fixed_external_development','X01–X04'),('user_reviewed_old_development','R03/R04')]);learned=sum(r['models']['selected']['learned_pairings'] for r in summary['records'])
    text=f'''# v22 分叉与拐角关系优化

日期：2026-10-25。数据、缓存与权重均在 E 盘；未安装新依赖，未连接机械臂。

本轮冻结 v21 修正墨带监督的像素模型，优化从中心骨架到逐条路径的关系。原始墨迹/中心概率和它们的空白误检率没有变化，因此不能把本轮描述为像素分割网络更准确。

## 实际工作

从 11 张 v21 训练图和 R01/R02 用户局部参考图的真实预测骨架生成节点分支关系。训练图另作 8 个旋转/尺度视图，重新运行冻结像素模型和骨架提取；这些不是新作品。只用能可靠对应参考线链的分支关系：同参考线且沿线位于节点两侧为正，不同明确参考链为负，歧义关系保持未知。

最终训练关系 {training['training_pairs']} 个（正 {training['positive']}、负 {training['negative']}），小型节点网络实际训练 600 次更新。其标签是用户局部/助手辅助线链的派生关系，不是画家实际笔顺、起收笔或完整专家金标。校准原生视图只有 {len(training['calibration_pairs'])} 个可明确对应的节点关系，十分有限。原始归属筛选中距离差符号曾过于保守，修正前文件保存在 initial_conservative_label_trial/，没有用那批文件冒充额外训练。

同时对照三个步骤：旧解码；过滤小于 8px 的短叶支线后重新判断明确两路拐角；在剩余分叉加入节点关系网络和有拒判的节点匹配。三路及以上分叉不会只按距离强接。所有新路径由原骨架链段组成。

先在固定 5 张校准图选择配置，未用 X01–X04 或 R03/R04 选参。选定配置为：{cal['selected']['config']}。选定方案在 14 张开发/相关探针中实际使用了 {learned} 次学习配对；其余变化可能来自短支线与明确拐角规则，不能都归功于学习网络。

## 同一像素预测下的路径指标

| 数据组 | 旧解码单路径覆盖 | 拐角规则试验 | 校准选定方案 |
|---|---:|---:|---:|
{table}

| 数据组 | 混合参考路径提示数 | 平均碎段数 |
|---|---:|---:|
{detail}

均为局部参考线附近 3px 容差的开发指标，不是整幅提取准确率。混合参考路径只是潜在误连提示，不能据此声称识别了所有错误分笔；闭合路径也可能是合法轮廓。来源已反复查看，存在历史训练关联，不是新盲测。stage_018/stage_029 单列为旧资料相关探针。

## 当前判断

晋级检查：{gate['development_regression_gate_passed']}。完整性检查：{all(checks.values())}。本轮保留真实成功与退化样本，未替换 v16 操作基线，候选仅用于离线研究；轨迹均为像素坐标，physical_z=null、ready_for_robot=false。

分叉的语义仍受参考完整性限制：同一墨块中两条交叉线是否一笔或两笔，单靠图像未必能确定。本轮没有训练皴法类别，也没有把粗细换算成真实毛笔下压量。

## 文件

- RESULT_COMPARISON.jpg：原图／旧解码／拐角规则／校准选定候选，均为真实路径。
- NODE_SUPERVISION_AUDIT.jpg：节点监督审计，非预测效果图。
- evaluation/summary.json：各图完整度、混线提示、短闭合路径、学习配对数。
- calibration.json：固定五张校准图上的选择依据与全部候选。
- junction_model/final.pt、training_summary.json：学习网络实际权重与训练记录。
- baseline_reproduction.json、verification.json、deployment_gate.json：复现、完整性和晋级决定。
'''
    (ROOT/'REPORT.md').write_text(text,encoding='utf-8');snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True);shutil.copy2(Path(__file__),snap/Path(__file__).name);print('V22_VERIFIED',checks,'GATE',gate,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode');a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='augment':augment()
    elif a.mode=='train':train()
    elif a.mode=='calibrate':calibrate()
    elif a.mode=='evaluate':evaluate()
    elif a.mode=='report':report()
