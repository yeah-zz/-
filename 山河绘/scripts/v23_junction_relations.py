# -*- coding: utf-8 -*-
"""Learn incident-branch relationships on real decoder nodes, offline only.

The reference chain identities are assistant annotations, not artist pen-lift
truth. Unknown branches are omitted from supervision. No labels at inference.
"""
import argparse
import itertools
import json
import math
import time
import shutil
from collections import defaultdict
from pathlib import Path

import v21_stroke_learning as v21
from stroke_link_v12 import tangent, dense
from topology_trace_v13 import graph_paths
import numpy as np
import cv2
import torch
from torch import nn
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw

ROOT = v21.LAB / 'junction_relations_v23_20261026'
BASE = v21.ROOT
WEIGHT = BASE/'ribbon_improved'/'final.pt'
POLICY = ('center', .65, .30)
FEATURE_NAMES = ['degree','short_length','long_length','length_ratio',
    'alignment_4','alignment_8','alignment_16','bend_min','bend_max',
    'node_center','node_ink','node_alpha','branch_center_min','branch_center_max',
    'branch_ink_min','branch_ink_max','branch_alpha_min','branch_alpha_max',
    'width_min','width_max','width_ratio','axis_pair_support',
    'transition_center_mean','transition_center_min','transition_alpha_mean',
    'transition_ink_mean']

def sample(im,q):
    q=np.asarray(q,np.float32)
    return cv2.remap(im.astype(np.float32),q[:,0][None],q[:,1][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]

def unit(x):
    return x/max(float(np.linalg.norm(x)),1e-6)

def arc(q):
    return np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]

def endpoint_segment(q,side):
    q=dense(q if side==0 else q[::-1]).astype(np.float32)
    a=arc(q)
    segment=q[(a>=4)&(a<=22)]
    return q,segment

def feature(chains,a,b,degree,pred,alpha):
    pa,sa=endpoint_segment(chains[a[0]],a[1]);pb,sb=endpoint_segment(chains[b[0]],b[1])
    la=float(arc(pa)[-1]);lb=float(arc(pb)[-1])
    va=[tangent(pa,True,s) for s in [4,8,16]]
    vb=[tangent(pb,True,s) for s in [4,8,16]]
    node=(pa[0]+pb[0])*.5
    x,y=np.rint(node).astype(int);h,w=alpha.shape;x=int(np.clip(x,0,w-1));y=int(np.clip(y,0,h-1))
    region=(slice(max(0,y-2),min(h,y+3)),slice(max(0,x-2),min(w,x+3)))
    if len(sa)==0:sa=pa[:min(6,len(pa))]
    if len(sb)==0:sb=pb[:min(6,len(pb))]
    values=[degree/6,math.log1p(min(la,lb))/6,math.log1p(max(la,lb))/6,min(la,lb)/max(la,lb,1)]
    values += [float(u@(-v)) for u,v in zip(va,vb)]
    bend=[float(va[0]@va[2]),float(vb[0]@vb[2])];values+=sorted(bend)
    values += [float(pred[0][region].mean()),float(pred[1][region].mean()),float(alpha[region].mean())]
    for im in [pred[0],pred[1],alpha]:
        values+=sorted([float(sample(im,sa).mean()),float(sample(im,sb).mean())])
    widths=sorted([float(sample(pred[4],sa).mean()),float(sample(pred[4],sb).mean())])
    values+=widths+[widths[0]/max(widths[1],.001)]
    th=.5*math.atan2(float(pred[3,y,x]),float(pred[2,y,x]));ax=np.array([math.cos(th),math.sin(th)])
    values.append(abs(float(va[1]@ax))*abs(float(vb[1]@ax)))
    # A short quadratic connector uses only the observed branch geometry.
    p=pa[min(len(pa)-1,8)];q=pb[min(len(pb)-1,8)];t=np.linspace(0,1,17)[:,None]
    transition=(1-t)**2*p+2*(1-t)*t*node+t*t*q
    cc=sample(pred[0],transition)
    values += [float(cc.mean()),float(cc.min()),float(sample(alpha,transition).mean()),float(sample(pred[1],transition).mean())]
    return np.asarray(values,np.float32)

def graph(pred,rgb,policy=POLICY):
    baseline,sk,oldinfo=v21.prior_eval.decode(pred,*policy)
    field=np.zeros((8,*sk.shape),np.float32);field[0]=np.maximum(pred[0],pred[1]*.8);field[1:3]=pred[2:4];field[3:5]=pred[2:4]
    chains,_=graph_paths(sk,field,False,min_length=0)
    nodes=defaultdict(list)
    for i,p in enumerate(chains):
        for side,xy in [(0,p[0]),(1,p[-1])]:
            nodes[tuple(np.round(xy,4))].append((i,side))
    alpha=v21.v16.photometry(rgb)[1]
    candidates=[]
    for xy,ends in nodes.items():
        if len(ends)<2:continue
        for a,b in itertools.combinations(ends,2):
            if a[0]==b[0]:continue
            f=feature(chains,a,b,len(ends),pred,alpha)
            candidates.append(dict(node=list(xy),a=a,b=b,feature=f,degree=len(ends)))
    return dict(chains=chains,candidates=candidates,baseline=baseline,oldinfo=oldinfo,shape=sk.shape,pred=pred,rgb=rgb)

def original_pairings(g):
    candidates=[];byend=defaultdict(list);pred=g['pred'];h,w=g['shape']
    for i,c in enumerate(g['candidates']):
        a,b=c['a'],c['b'];pa=g['chains'][a[0]];pb=g['chains'][b[0]]
        va=tangent(pa,a[1]==0);vb=tangent(pb,b[1]==0)
        x,y=np.rint(c['node']).astype(int);x=int(np.clip(x,0,w-1));y=int(np.clip(y,0,h-1))
        th=.5*math.atan2(float(pred[3,y,x]),float(pred[2,y,x]));ax=np.array([math.cos(th),math.sin(th)])
        score=.8*float(va@(-vb))+.2*abs(float(va@ax))*abs(float(vb@ax))
        if score<.72:continue
        candidates.append((score,i));byend[a].append((score,i));byend[b].append((score,i))
    accepted=[]
    for score,i in sorted(candidates,reverse=True):
        c=g['candidates'][i];aa=sorted(byend[c['a']],reverse=True);bb=sorted(byend[c['b']],reverse=True)
        if aa[0][1]!=i or bb[0][1]!=i:continue
        if (len(aa)>1 and aa[0][0]-aa[1][0]<.08) or (len(bb)>1 and bb[0][0]-bb[1][0]<.08):continue
        accepted.append(i)
    return accepted

def assemble(g,accepted):
    paired={}
    for i in accepted:
        c=g['candidates'][i];a,b=c['a'],c['b']
        if a in paired or b in paired:raise ValueError('Duplicate endpoint pairing')
        paired[a]=b;paired[b]=a
    chains=g['chains'];used=set();paths=[]
    starts=[(i,s) for i in range(len(chains)) for s in [0,1] if (i,s) not in paired]+[(i,0) for i in range(len(chains))]
    for index,side in starts:
        if index in used:continue
        segments=[]
        while index not in used:
            used.add(index);segments.append(chains[index] if side==0 else chains[index][::-1]);nxt=paired.get((index,1-side))
            if nxt is None:break
            index,side=nxt
        q=np.concatenate(segments)
        if arc(q)[-1]>=8:paths.append(q)
    return paths

def assign_end(p,side,trees):
    q,seg=endpoint_segment(p,side)
    if len(seg)<5:return None
    ds=[t.query(seg) for t in trees]
    med=np.array([np.median(z[0]) for z in ds]);order=np.argsort(med);k=int(order[0])
    if med[k]>3.5 or np.mean(ds[k][0]<=4)<.7:return None
    if len(order)>1 and med[order[1]]-med[k]<1.5:return None
    ni=int(trees[k].query(q[0])[1]);inner=float(np.median(ds[k][1]))
    if abs(inner-ni)<2:return None
    return dict(ref=k,direction=float(np.sign(inner-ni)),node_index=ni,median_distance=float(med[k]))

def labels_for_graph(g,refs,key=None):
    trees=[cKDTree(dense(p)) for p in refs];assigned={}
    for c in g['candidates']:
        for end in [c['a'],c['b']]:
            if end not in assigned:assigned[end]=assign_end(g['chains'][end[0]],end[1],trees)
        aa,bb=assigned[c['a']],assigned[c['b']]
        label=None;reason='unknown_or_ambiguous_branch'
        if aa is not None and bb is not None:
            if aa['ref']!=bb['ref']:label=0;reason='distinct_reference_chains'
            elif abs(aa['node_index']-bb['node_index'])<=24:
                label=int(aa['direction']*bb['direction']<0);reason='same_reference_opposite_sides' if label else 'same_reference_same_side'
        c['label']=label;c['label_reason']=reason;c['assignment_a']=aa;c['assignment_b']=bb
    if key is not None and (ROOT/'node_audit'/'nodes.json').exists():
        from v23_assistant_nodes import SPECS
        records=v21.read(ROOT/'node_audit'/'nodes.json')['records']
        for r in records:
            if r['image_id']!=key or r['key'] not in SPECS:continue
            spec=SPECS[r['key']]
            for kind,y in [('positive',1),('negative',0)]:
                for a,b in spec.get(kind,[]):
                    ends={tuple(r['ends'][a]),tuple(r['ends'][b])}
                    matches=[c for c in g['candidates'] if np.linalg.norm(np.asarray(c['node'])-r['node'])<.001 and {c['a'],c['b']}==ends]
                    if len(matches)!=1:raise ValueError(('Manual node mapping changed',r['key'],a,b))
                    c=matches[0];c['initial_auto_label']=c['label'];c['label']=y;c['label_reason']='assistant_visual_node_'+kind
    return g

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'predictions').mkdir(exist_ok=True)
    if any(ROOT.glob('seed_*/final.pt')):raise FileExistsError('Preserve frozen training data after fitting')
    if (ROOT/'data_audit.json').exists() and not (ROOT/'initial_auto_data_audit.json').exists():
        shutil.copy2(ROOT/'data_audit.json',ROOT/'initial_auto_data_audit.json')
        for s in ['train','calibration']:shutil.copy2(ROOT/(s+'_pairs.npz'),ROOT/('initial_auto_'+s+'_pairs.npz'))
    torch.set_num_threads(3);cv2.setNumThreads(2)
    model=v21.load_model('ribbon_improved');rows=[];records=v21.read(BASE/'manifest.json')['records']
    datasets={s:[] for s in ['train','calibration']}
    before=dict(weights=v21.digest(WEIGHT),manifest=v21.digest(BASE/'manifest.json'))
    for r in records:
        if r['split'] not in datasets:continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));refs=[s['points'] for s in v21.read(r['labels'])['strokes']]
        cache=ROOT/'predictions'/(r['id']+'.npz')
        if cache.exists():pred=np.load(cache)['prediction']
        else:pred=v21.v16.predict(model,rgb);np.savez_compressed(cache,prediction=pred)
        # Multiple actual decoder views are augmentation, not independent works.
        policies=[POLICY,('center',.80,.45),('center',.55,.25)] if r['split']=='train' else [POLICY]
        for policy in policies:
            g=labels_for_graph(graph(pred,rgb,policy),refs,r['id'] if policy==POLICY else None);known=[c for c in g['candidates'] if c['label'] is not None]
            reconstructed=assemble(g,original_pairings(g));valid=np.ones(rgb.shape[:2],np.float32)
            old=v21.path_metrics(refs,g['baseline'],valid);new=v21.path_metrics(refs,reconstructed,valid)
            if old!=new:raise ValueError(('Baseline reconstruction mismatch',r['id'],old,new))
            for c in known:datasets[r['split']].append((c['feature'],c['label'],r['id']))
            rows.append(dict(id=r['id'],split=r['split'],policy=list(policy),candidates=len(g['candidates']),positive=sum(c['label']==1 for c in known),negative=sum(c['label']==0 for c in known),unknown=len(g['candidates'])-len(known),known_pairs=[dict(node=[float(v) for v in c['node']],a=list(c['a']),b=list(c['b']),label=int(c['label']),reason=c['label_reason'],initial_auto_label=c.get('initial_auto_label',c['label'])) for c in known],label_sha256=r['label_sha256'],image_sha256=r['image_sha256']))
        print('V23_DATA',r['id'],r['split'],rows[-1]['positive'],rows[-1]['negative'],rows[-1]['unknown'],flush=True)
    counts={}
    for split,data in datasets.items():
        x=np.stack([a for a,b,k in data]) if data else np.zeros((0,len(FEATURE_NAMES)),np.float32);y=np.array([b for a,b,k in data],np.float32);ids=np.array([k for a,b,k in data])
        np.savez_compressed(ROOT/(split+'_pairs.npz'),x=x,y=y,ids=ids)
        counts[split]=dict(positive=int((y>0).sum()),negative=int((y==0).sum()),images=int(len(np.unique(ids))))
    v21.jsave(ROOT/'data_audit.json',dict(records=rows,counts=counts,feature_names=FEATURE_NAMES,input_hashes=before,training_graph_views_are_correlated_augmentation=True,reference_chain_labels_not_expert_stroke_truth=True,unknowns_omitted=True,baseline_reconstruction_matches=True,robot_connected=False))
    print('V23_COUNTS',counts,flush=True)

def audit_nodes():
    folder=ROOT/'node_audit';folder.mkdir(exist_ok=True);cards=[];records=[]
    for r in v21.read(BASE/'manifest.json')['records']:
        if r['split'] not in ['train','calibration']:continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));refs=[s['points'] for s in v21.read(r['labels'])['strokes']];pred=np.load(ROOT/'predictions'/(r['id']+'.npz'))['prediction'];g=labels_for_graph(graph(pred,rgb),refs)
        groups=defaultdict(list)
        for i,c in enumerate(g['candidates']):groups[tuple(c['node'])].append(i)
        for ni,(node,indices) in enumerate(sorted(groups.items())):
            if all(g['candidates'][i]['label'] is not None for i in indices):continue
            ends=sorted({e for i in indices for e in [g['candidates'][i]['a'],g['candidates'][i]['b']]})
            x,y=map(float,node);radius=38;box=[round(x-radius),round(y-radius),round(x+radius),round(y+radius)];source=Image.fromarray(rgb).crop(box).resize((240,240));overlay=source.copy();dd=ImageDraw.Draw(overlay)
            for j,e in enumerate(ends):
                q=dense(g['chains'][e[0]] if e[1]==0 else g['chains'][e[0]][::-1]);q=q[arc(q)<=34];q=(q-np.array(box[:2]))*240/(2*radius);color=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[j*29%180,225,190]]]),cv2.COLOR_HSV2RGB)[0,0]);dd.line([tuple(v) for v in q],fill=color,width=3)
                if len(q):
                    pos=q[min(len(q)-1,max(1,len(q)-3))];dd.rectangle((pos[0]-1,pos[1]-1,pos[0]+21,pos[1]+21),fill='white');dd.text(tuple(pos),str(j),font=v21.FONT,fill=color)
            key=r['id']+'__'+str(ni);card=Image.new('RGB',(540,300),'white');d=ImageDraw.Draw(card);d.text((4,3),key+' '+r['split'],font=v21.SMALL,fill='black');card.paste(source,(4,30));card.paste(overlay,(280,30));d.text((4,273),'原图局部 / 真实解码分支；编号是局部分支 ID',font=v21.SMALL,fill='black');cards.append(card)
            records.append(dict(key=key,image_id=r['id'],split=r['split'],node=[x,y],ends=[list(e) for e in ends],pairs=[dict(local_a=ends.index(g['candidates'][i]['a']),local_b=ends.index(g['candidates'][i]['b']),initial_label=g['candidates'][i]['label']) for i in indices]))
    for start in range(0,len(cards),8):
        board=Image.new('RGB',(1080,300*math.ceil(min(8,len(cards)-start)/2)),'#dddddd')
        for k,card in enumerate(cards[start:start+8]):board.paste(card,(k%2*540,k//2*300))
        board.save(folder/('NODES_'+str(start//8)+'.jpg'),quality=96)
    v21.jsave(folder/'nodes.json',dict(records=records,not_model_results=True,source_image_patches_for_assistant_annotation=True))
    print('V23_NODE_AUDIT',len(cards),'sheets',math.ceil(len(cards)/8),flush=True)

class RelationNet(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(len(FEATURE_NAMES),24),nn.SiLU(),nn.Linear(24,12),nn.SiLU(),nn.Linear(12,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def train():
    torch.set_num_threads(3);audit=v21.read(ROOT/'data_audit.json')
    z=np.load(ROOT/'train_pairs.npz');x=z['x'];y=z['y'];mu=x.mean(0);sd=np.maximum(x.std(0),.05);xx=torch.from_numpy((x-mu)/sd);yy=torch.from_numpy(y)
    positive=np.flatnonzero(y>0);negative=np.flatnonzero(y==0)
    if min(len(positive),len(negative))<8:raise RuntimeError('Too few confidently assigned real-node pairs to fit the relation model')
    logs=[]
    for seed in [231026,231027,231028]:
        folder=ROOT/('seed_'+str(seed));folder.mkdir(exist_ok=True)
        if (folder/'final.pt').exists():raise FileExistsError('Preserve previous relation checkpoint')
        torch.manual_seed(seed);rng=np.random.default_rng(seed);m=RelationNet();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.02);start=time.time();hist=[]
        for step in range(800):
            ids=np.r_[rng.choice(positive,32),rng.choice(negative,32)];loss=nn.functional.binary_cross_entropy_with_logits(m(xx[ids]),yy[ids]);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step()
            if step%100==0 or step==799:hist.append(dict(step=step+1,loss=float(loss.detach())))
        torch.save(dict(model=m.state_dict(),mean=mu,std=sd,features=FEATURE_NAMES,seed=seed,steps=800),folder/'final.pt')
        v21.jsave(folder/'training.json',dict(records=hist,updates=800,elapsed_seconds=time.time()-start,known_pairs=len(y)))
        logs.append(dict(seed=seed,updates=800,final_loss=hist[-1]['loss'],sha256=v21.digest(folder/'final.pt')))
        print('V23_FIT',logs[-1],flush=True)
    v21.jsave(ROOT/'training_summary.json',dict(models=logs,total_updates=2400,extractor_backbone_updated=False,real_decoder_nodes_used=True,synthetic_cut_pairs_used=False,counts=audit['counts'],calibration_not_in_optimizer=True))

def models():
    out=[]
    for seed in [231026,231027,231028]:
        z=torch.load(ROOT/('seed_'+str(seed))/'final.pt',map_location='cpu',weights_only=False);m=RelationNet();m.load_state_dict(z['model']);out.append((m.eval(),z))
    return out

def probabilities(g,ensemble):
    if not g['candidates']:return np.zeros(0)
    x=np.stack([c['feature'] for c in g['candidates']]);probs=[]
    with torch.no_grad():
        for m,z in ensemble:probs.append(m(torch.from_numpy((x-z['mean'])/z['std'])).sigmoid().numpy())
    return np.mean(probs,axis=0)

def select_pairs(g,prob,threshold,mode):
    accepted=original_pairings(g) if mode=='add' else []
    used={end for i in accepted for end in [g['candidates'][i]['a'],g['candidates'][i]['b']]}
    eligible=[];byend=defaultdict(list)
    for i,c in enumerate(g['candidates']):
        if prob[i]<threshold or c['a'] in used or c['b'] in used:continue
        # Tiny skeleton spurs cannot drive a new path decision.
        if min(arc(g['chains'][c['a'][0]])[-1],arc(g['chains'][c['b'][0]])[-1])<8:continue
        eligible.append(i);byend[c['a']].append(i);byend[c['b']].append(i)
    for i in sorted(eligible,key=lambda i:-prob[i]):
        c=g['candidates'][i]
        if c['a'] in used or c['b'] in used:continue
        alternatives=[j for e in [c['a'],c['b']] for j in byend[e] if j!=i]
        if alternatives and prob[i]-max(prob[j] for j in alternatives)<.10:continue
        # Do not form a closed route by combining already joined branches.
        edges=defaultdict(set)
        for j in accepted:
            u=g['candidates'][j]['a'][0];v=g['candidates'][j]['b'][0];edges[u].add(v);edges[v].add(u)
        todo=[c['a'][0]];seen=set(todo)
        while todo:
            for v in edges[todo.pop()]:
                if v not in seen:seen.add(v);todo.append(v)
        if c['b'][0] in seen:continue
        accepted.append(i);used.update([c['a'],c['b']])
    return accepted

def calibrate():
    torch.set_num_threads(3);ensemble=models();cases=[]
    for r in v21.read(BASE/'manifest.json')['records']:
        if r['split']!='calibration':continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));refs=[s['points'] for s in v21.read(r['labels'])['strokes']];pred=np.load(ROOT/'predictions'/(r['id']+'.npz'))['prediction'];g=labels_for_graph(graph(pred,rgb),refs,r['id']);p=probabilities(g,ensemble);valid=np.ones(rgb.shape[:2],np.float32)
        cases.append((r['id'],refs,g,p,valid))
    options=[]
    for mode in ['add','replace']:
        for threshold in [.70,.85,.95,.99,.999,1.001]:
            rows=[];fp=tp=pos=neg=0
            for key,refs,g,p,valid in cases:
                pairs=select_pairs(g,p,threshold,mode);before=v21.path_metrics(refs,g['baseline'],valid);after=v21.path_metrics(refs,assemble(g,pairs),valid)
                added=set(pairs)-set(original_pairings(g))
                for i,c in enumerate(g['candidates']):
                    if c['label'] is None:continue
                    pos+=int(c['label']==1);neg+=int(c['label']==0)
                    if i in pairs:tp+=int(c['label']==1);fp+=int(c['label']==0)
                rows.append(dict(id=key,before=before,after=after,new_pairs=len(added),known_new_negative=sum(g['candidates'][i]['label']==0 for i in added)))
            old=float(np.mean([r['before']['mean_best_single_path_coverage'] for r in rows]));new=float(np.mean([r['after']['mean_best_single_path_coverage'] for r in rows]));oldmix=sum(r['before']['mixed_reference_paths_flagged'] for r in rows);newmix=sum(r['after']['mixed_reference_paths_flagged'] for r in rows);newnegative=sum(r['known_new_negative'] for r in rows)
            option=dict(mode=mode,threshold=threshold,before=old,after=new,gain=new-old,mixed_before=oldmix,mixed_after=newmix,known_selected_positive=tp,known_selected_negative=fp,known_available_positive=pos,known_available_negative=neg,new_known_negative=newnegative,records=rows)
            option['eligible']=bool(new>=old+.005 and newmix<=oldmix and newnegative==0 and min(pos,neg)>=3)
            options.append(option)
    eligible=[o for o in options if o['eligible']]
    selected=max(eligible or options,key=lambda o:(o['gain'],-o['mixed_after'],-o['new_known_negative']))
    v21.jsave(ROOT/'calibration.json',dict(selected=selected,enabled=bool(eligible),options=options,minimum_single_path_gain=.005,no_added_known_negative=True,external_development_not_used_for_selection=True,small_correlated_calibration=True))
    print('V23_CALIBRATION',bool(eligible),{k:selected[k] for k in ['mode','threshold','before','after','gain','new_known_negative']},flush=True)

def evaluate():
    torch.set_num_threads(3);cv2.setNumThreads(2);ensemble=models();decision=v21.read(ROOT/'calibration.json');sel=decision['selected'];out=ROOT/'evaluation';out.mkdir(exist_ok=True);rows=[];images={}
    for r,rgb,refs,valid,bg,_ in v21.evaluation_cases():
        key=r['id'];pred=np.load(BASE/'ribbon_improved'/'evaluation'/key/'prediction.npz')['prediction'];g=graph(pred,rgb);p=probabilities(g,ensemble)
        # Inference and selection happen before reference correspondence is made.
        pairs=select_pairs(g,p,sel['threshold'],sel['mode']);trial=assemble(g,pairs);actual=trial if decision['enabled'] else g['baseline']
        before=v21.path_metrics(refs,g['baseline'],valid);after=v21.path_metrics(refs,trial,valid);labels_for_graph(g,refs)
        added=set(pairs)-set(original_pairings(g));removed=set(original_pairings(g))-set(pairs)
        newneg=sum(g['candidates'][i]['label']==0 for i in added);newpos=sum(g['candidates'][i]['label']==1 for i in added);unknown=sum(g['candidates'][i]['label'] is None for i in added)
        split=r['split']
        if key in ['stage_018','stage_029']:split='legacy_related_probe'
        row=dict(id=key,split=split,before=before,trial=after,new_pairs=len(added),removed_pairs=len(removed),new_known_positive=int(newpos),new_known_negative=int(newneg),new_unknown=int(unknown),candidate_count=len(g['candidates']))
        folder=out/key;folder.mkdir(exist_ok=True)
        v21.jsave(folder/'trajectory.json',dict(paths=[q.round(3).tolist() for q in actual],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,relation_enabled=decision['enabled'],references_used_at_inference=False))
        v21.jsave(folder/'trial_trajectory.json',dict(paths=[q.round(3).tolist() for q in trial],experimental_only=True,events=[dict(node=[float(v) for v in g['candidates'][i]['node']],probability=float(p[i]),posthoc_reference_label=g['candidates'][i]['label']) for i in added]))
        rows.append(row);images[key]=(rgb,g['baseline'],trial)
        print('V23_EVAL',key,round(before['mean_best_single_path_coverage'],3),round(after['mean_best_single_path_coverage'],3),'add',len(added),'wrong',newneg,flush=True)
    groups={}
    for split in sorted(set(r['split'] for r in rows)):
        rr=[r for r in rows if r['split']==split];groups[split]={tag:{metric:float(np.mean([r[tag][metric] for r in rr])) for metric in ['point_coverage_3px','mean_best_single_path_coverage','mean_reference_fragments','path_count','mixed_reference_paths_flagged']} for tag in ['before','trial']}
        groups[split]['known_new_positive']=sum(r['new_known_positive'] for r in rr);groups[split]['known_new_negative']=sum(r['new_known_negative'] for r in rr);groups[split]['new_unknown']=sum(r['new_unknown'] for r in rr)
    keys=['stage_065','sina_6586334590_31','X01','X04','R04','stage_049'];board=Image.new('RGB',(1500,320*len(keys)+45),'#f5f3ec');d=ImageDraw.Draw(board)
    d.text((8,6),'v23 真实节点关系训练：原图 / v21 原路径 / v23 关系模型候选（离线、非实画）',font=v21.FONT,fill='black')
    for i,key in enumerate(keys):
        rgb,before,trial=images[key]
        for j,(title,a) in enumerate([('原图',rgb),('原路径',v21.line_view(rgb.shape[:2],before,True)),('局部关系模型试验候选',v21.line_view(rgb.shape[:2],trial,True))]):
            d.text((j*500+6,45+320*i),key+' '+title,font=v21.SMALL,fill='black');v21.paste(board,a,j*500+6,70+320*i,485,285)
    board.save(ROOT/'RESULT_COMPARISON.jpg',quality=95)
    v21.jsave(ROOT/'evaluation.json',dict(records=rows,groups=groups,calibration_enabled=decision['enabled'],not_blind=True,reference_labels_not_used_for_inference=True,model_is_local_junction_relation_network=True))
    print('V23_GROUPS',groups,flush=True)

def diagnostics():
    ensemble=models();reports={}
    for split in ['train','calibration']:
        z=np.load(ROOT/(split+'_pairs.npz'));x=z['x'];y=z['y'];pp=[]
        with torch.no_grad():
            for m,params in ensemble:pp.append(m(torch.from_numpy((x-params['mean'])/params['std'])).sigmoid().numpy())
        p=np.mean(pp,axis=0);threshold=v21.read(ROOT/'calibration.json')['selected']['threshold']
        reports[split]=dict(positive=int((y==1).sum()),negative=int((y==0).sum()),true_positive=int(((p>=threshold)&(y==1)).sum()),false_positive=int(((p>=threshold)&(y==0)).sum()),false_negative=int(((p<threshold)&(y==1)).sum()),true_negative=int(((p<threshold)&(y==0)).sum()),positive_scores=[float(v) for v in p[y==1]],negative_scores=[float(v) for v in p[y==0]],threshold=threshold)
    v21.jsave(ROOT/'pair_classification.json',dict(reports=reports,training_views_not_independent=True,calibration_small_and_source_correlated=True,not_expert_accuracy=True))
    print('V23_PAIR_COUNTS',{s:{k:v for k,v in r.items() if not k.endswith('scores')} for s,r in reports.items()},flush=True)

def predict_one(input_path,output_path,grayscale=False):
    source=Path(input_path);out=Path(output_path)
    if not out.is_absolute() or out.drive.upper() not in ['E:','F:']:raise ValueError('Use an absolute E:/F: output path')
    rgb=np.asarray(Image.open(source).convert('RGB'))
    if grayscale:rgb=np.repeat(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)[:,:,None],3,2)
    if not v21.v16.monochrome_route(rgb):raise ValueError('Only monochrome rock research has been evaluated')
    model=v21.load_model('ribbon_improved');pred=v21.v16.predict(model,rgb);g=graph(pred,rgb);p=probabilities(g,models());decision=v21.read(ROOT/'calibration.json');sel=decision['selected'];pairs=select_pairs(g,p,sel['threshold'],sel['mode']);trial=assemble(g,pairs);actual=trial if decision['enabled'] else g['baseline'];out.mkdir(parents=True,exist_ok=True)
    v21.jsave(out/'trajectory.json',dict(paths=[q.round(3).tolist() for q in actual],physical_z=None,ready_for_robot=False,actually_drawn=False,relation_enabled=decision['enabled'],references_used_at_inference=False))
    v21.jsave(out/'trial_trajectory.json',dict(paths=[q.round(3).tolist() for q in trial],experimental_candidate=True,references_used_at_inference=False,ready_for_robot=False))
    board=Image.new('RGB',(1500,430),'#f5f3ec');d=ImageDraw.Draw(board)
    for j,(title,a) in enumerate([('无标注输入原图',rgb),('按校准门槛输出的路径',v21.line_view(rgb.shape[:2],actual,True)),('未启用的关系模型候选',v21.line_view(rgb.shape[:2],trial,True))]):
        d.text((500*j+6,8),title,font=v21.FONT,fill='black');v21.paste(board,a,500*j+6,40,485,380)
    board.save(out/'comparison.jpg',quality=95)
    v21.jsave(out/'summary.json',dict(input=str(source),input_sha256=v21.digest(source),extractor_sha256=v21.digest(WEIGHT),input_shape=list(rgb.shape[:2]),reference_labels_used=False,relation_enabled=decision['enabled'],official_model_replaced=False,robot_connected=False,path_count=len(actual)))
    print('V23_REFERENCE_FREE',str(out),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','audit','train','calibrate','evaluate','diagnostics','predict']);p.add_argument('--input');p.add_argument('--output');p.add_argument('--grayscale',action='store_true');a=p.parse_args()
    if a.mode=='predict':
        if not a.input or not a.output:p.error('predict requires --input and --output')
        predict_one(a.input,a.output,a.grayscale)
    else:{'prepare':prepare,'audit':audit_nodes,'train':train,'calibrate':calibrate,'evaluate':evaluate,'diagnostics':diagnostics}[a.mode]()
