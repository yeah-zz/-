# -*- coding: utf-8 -*-
"""Image context for fragment/duplicate/separate pairs, not semantic cun labels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,time,hashlib
import cv2,numpy as np,torch
from torch.nn import functional as F
from scipy.spatial import cKDTree
from context_curve_model_20261105 import ContextCurveNet,pack,predict
from context_curve_data_20261105 import ROOT,source_records,stroke,write
from landscape_stroke_refine_20261104 import samples,length,curve_tangents
from source_stroke_width_20261104 import fields,sample

def synthetic(n,seed,raw=False):
    rng=np.random.default_rng(seed);x=[];y=[];examples=[]
    for i in range(n):
        label=i%3;angle=rng.uniform(-np.pi,np.pi);u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-u[1],u[0]])
        center=np.array([48.,48.]);arc=rng.uniform(12,46);bend=rng.uniform(-8,8);width=int(rng.integers(1,8))
        full=stroke(center-u*arc/2,center+u*arc/2,bend)
        if label==0:
            cut=int(rng.integers(10,25));p=full[:cut+1];q=full[cut:];ink=[full]
        elif label==1:
            q=full;p=full.copy()+v*rng.uniform(-width*.35,width*.35)
            if rng.random()<.6:p=p[5:28]
            ink=[full]
        else:
            p=full
            if rng.random()<.7:q=full+v*rng.uniform(width+1,width+5)
            else:q=stroke(center+rng.normal(0,1,2),center+v*rng.uniform(14,30),rng.uniform(-3,3))
            ink=[p,q]
        paper=rng.uniform(.72,.98);rgb=np.full((96,96,3),paper,np.float32)
        rgb+=rng.normal(0,.006,(96,96,1)).astype(np.float32)
        rgb+=np.array([-.04,0,.02],np.float32)*rng.uniform(0,1)
        for path in ink:
            layer=np.zeros((96,96),np.uint8);cv2.polylines(layer,[np.rint(path).astype(np.int32)],False,255,width,cv2.LINE_AA)
            alpha=layer.astype(np.float32)/255*rng.uniform(.04,.85)
            if label==1 and rng.random()<.5:
                # Known dry-brush holes stay inside ONE rendered stroke; they
                # are not labelled as two independent artist lines.
                yy,xx=np.mgrid[:96,:96];cx,cy=full[len(full)//2]
                hole=((xx-cx)**2+(yy-cy)**2)<rng.uniform(.3,1.5)**2
                alpha[hole]*=.1
            rgb-=alpha[:,:,None]
        rgb=np.clip(rgb*255,0,255).astype(np.uint8)
        x.append(pack(rgb,p,[q]));y.append(label)
        if raw:examples.append((rgb,p,q,label))
    return (x,y,examples) if raw else (x,y)

def real_pairs(r,n,seed,raw=False):
    rng=np.random.default_rng(seed);x=[];y=[];examples=[]
    for i in range(n):
        label=i%3;p=samples(r['refs'][rng.integers(len(r['refs']))],.6)
        if len(p)<6:continue
        if label==0:
            cut=int(rng.integers(2,len(p)-2));a=p[:cut+1];b=p[cut:]
        elif label==1:
            a=p.copy();b=p+rng.normal(0,.05,p.shape)
        else:
            b=r['refs'][rng.integers(len(r['refs']))]
            if np.mean(cKDTree(b).query(p)[0])<2:continue
            a=p
        x.append(pack(r['rgb'],a,[b]));y.append(label)
        if raw:examples.append((r['rgb'],a,b,label))
    return (x,y,examples) if raw else (x,y)

def train():
    dest=ROOT/'relations';dest.mkdir(exist_ok=True)
    if (dest/'model.pt').exists():raise FileExistsError('Preserve completed relation experiment')
    records=source_records();sets={};protocol=[]
    for split,n,seed in [('train',2100,110511),('calibration',450,110512),('test',450,110513)]:
        x,y=synthetic(n,seed)
        for j,r in enumerate(records):
            if r['split']!=split:continue
            xx,yy=real_pairs(r,150 if split=='train' else 90,seed+100+j);x+=xx;y+=yy
            protocol.append(dict(id=r['id'],split=split,pairs=len(yy),source_family=r['source_family']))
        sets[split]=(np.stack(x),np.array(y,np.int64));print('REL_DATA',split,len(y),flush=True)
    np.savez_compressed(dest/'train_data.npz',x=sets['train'][0],y=sets['train'][1])
    tx=torch.from_numpy(sets['train'][0]).cuda();ty=torch.from_numpy(sets['train'][1]).cuda()
    torch.manual_seed(110511);model=ContextCurveNet(3).cuda()
    initial=torch.load(ROOT/'model.pt',map_location='cpu',weights_only=False)['model']
    model.encoder.load_state_dict({k[len('encoder.'):]:v for k,v in initial.items() if k.startswith('encoder.')})
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0005)
    best=1e9;state=None;step_saved=0;history=[];start=time.monotonic();cx,cy=sets['calibration']
    for step in range(1,1001):
        model.train();ix=torch.randint(len(tx),(32,),device='cuda');optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(model(tx[ix].float()/255),ty[ix])
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if step%100==0:
            p=predict(model,cx);ce=float(-np.log(np.maximum(p[np.arange(len(cy)),cy],1e-8)).mean())
            history.append(dict(step=step,calibration_ce=ce,loss=float(loss.detach())))
            if ce<best:best=ce;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};step_saved=step
            print('REL_TRAIN',step,round(ce,4),flush=True)
    model.load_state_dict(state);torch.save(dict(model=state,completed_updates=1000,retained_step=step_saved),dest/'model.pt')
    cp=predict(model,cx);options=[]
    for t in [.9,.95,.97,.98,.99,.995,.999,.9999]:
        pred=cp[:,1]>=t;fp=int((pred&(cy!=1)).sum());tp=int((pred&(cy==1)).sum())
        if fp==0 and tp>=10:options.append(dict(threshold=t,tp=tp,fp=fp))
    threshold=max(options,key=lambda r:r['tp'])['threshold'] if options else 1.01
    p=predict(model,sets['test'][0]);y=sets['test'][1];pred=p[:,1]>=threshold
    result=dict(completed_updates=1000,retained_step=step_saved,gpu=torch.cuda.get_device_name(),
        seconds=time.monotonic()-start,threshold=threshold,calibration_choices=options,history=history,records=protocol,
        test=dict(samples=len(y),true_duplicate=int((y==1).sum()),duplicate_accepted=int((pred&(y==1)).sum()),
            separate_or_continuous_wrongly_accepted=int((pred&(y!=1)).sum()),argmax_accuracy=float((p.argmax(1)==y).mean())),
        synthetic_and_user_curve_augmentation_only=True,independent_real_painting_accuracy=False,
        model_sha256=hashlib.sha256((dest/'model.pt').read_bytes()).hexdigest(),robot_connected=False)
    write(dest/'training_summary.json',result);print('REL_DONE',json.dumps(result['test']),threshold,flush=True)

def load_relation():
    d=json.loads((ROOT/'relations/training_summary.json').read_text('utf8'));model=ContextCurveNet(3).cuda()
    model.load_state_dict(torch.load(ROOT/'relations/model.pt',map_location='cpu',weights_only=False)['model']);model.eval()
    return model,d['threshold']

def eligible(rgb,p,q):
    s=max(1.,max(rgb.shape[:2])/496);p=np.asarray(p,float);q=np.asarray(q,float)
    if np.linalg.norm(p[0]-p[-1])<.1 or length(q)+1e-6<length(p):return None
    a=samples(p,.5*s);b=samples(q,.5*s);d,ix=cKDTree(b).query(a)
    if (d<2.2*s).mean()<.96 or d.max()>3*s or max(d[0],d[-1])>2.2*s:return None
    # An adjacent fragment touching only the representative's endpoint is a
    # continuation, not a contained duplicate. Require progress along its body.
    projected_length=(int(ix.max())-int(ix.min()))*length(b)/max(1,len(b)-1)
    if projected_length<length(a)*.5:return None
    if np.mean(abs((curve_tangents(a)*curve_tangents(b)[ix]).sum(1)))<.78:return None
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;paired=b[ix]
    values=np.stack([sample(gray,a*(1-t)+paired*t) for t in np.linspace(0,1,13)],1)
    gap=values[:,1:-1].max(1)-np.maximum(values[:,0],values[:,-1])>.015
    longest=cur=0
    for g in gap:
        cur=cur+1 if g else 0;longest=max(longest,cur)
    if gap.mean()>.20 or longest*.5*s>1.5*s:return None
    return dict(mean_distance_px=float(d.mean()),max_distance_px=float(d.max()),gap_fraction=float(gap.mean()),projected_length_px=projected_length)

def fresh_pipeline_test():
    model,t=load_relation();x,y,examples=synthetic(600,110514,raw=True)
    full=LAB/'user_review_v11_20261024';doc=json.loads((full/'labels/A06.json').read_text('utf8'))
    assert doc['review_state']=='checked_partial'
    rgb=cv2.cvtColor(cv2.imread(str(full/'samples/A06.png')),cv2.COLOR_BGR2RGB)
    r=dict(rgb=rgb,refs=[np.array(s['points'],float) for s in doc['strokes'] if s['id'].startswith('s_') and len(s['points'])>=2])
    xx,yy,ee=real_pairs(r,180,11051406,raw=True);x+=xx;y+=yy;examples+=ee
    p=predict(model,np.stack(x));eligible_flags=np.array([eligible(rgb,a,b) is not None for rgb,a,b,_ in examples])
    y=np.array(y);use=(p[:,1]>=t)&eligible_flags
    result=dict(samples=len(y),synthetic_samples=600,reserved_A06_pairs=len(yy),
        raw_network_false_duplicate=int(((p[:,1]>=t)&(y!=1)).sum()),
        joint_false_duplicate=int((use&(y!=1)).sum()),joint_duplicate_accepted=int((use&(y==1)).sum()),
        true_duplicate_pairs=int((y==1).sum()),threshold=t,
        unseen_synthetic_seed=True,reserved_image_not_used_in_this_module_training=True,
        reserved_image_same_painting_family=True,independent_painting_accuracy=False)
    write(ROOT/'relations/fresh_pipeline_test.json',result);print('FRESH_REL',result,flush=True);return result

def suppress(model,threshold,rgb,rows):
    paths=[np.asarray(r['points'],float) for r in rows];s=max(1.,max(rgb.shape[:2])/496)
    q=[samples(p,.5*s) for p in paths];tan=[curve_tangents(a) for a in q];trees=[cKDTree(a) for a in q]
    lens=[length(p) for p in paths];gray,_=fields(rgb);pool=[];views=[]
    # One representative must cover an entire shorter curve. Artist-closed
    # motifs remain protected; a partial match cannot delete a unique tail.
    order=sorted(range(len(rows)),key=lambda i:-lens[i])
    for rank,i in enumerate(order):
        if np.linalg.norm(paths[i][0]-paths[i][-1])<.1:continue
        choices=[]
        for j in order[:rank]:
            if rows[i].get('module',0)!=rows[j].get('module',0):continue
            if (q[i].min(0)>q[j].max(0)+3*s).any() or (q[i].max(0)<q[j].min(0)-3*s).any():continue
            d,ix=trees[j].query(q[i])
            if (d<2.2*s).mean()<.96 or d.max()>3*s:continue
            info=eligible(rgb,paths[i],paths[j])
            if info is None:continue
            choices.append((info['mean_distance_px'],j))
        for dist,j in sorted(choices)[:3]:
            pool.append(dict(curve=i,representative=j,mean_distance_px=dist));views.append(pack(rgb,paths[i],[paths[j]]))
    if not pool:return dict(selected=list(range(len(rows))),removed=[],events=[],candidate_pairs=0)
    probs=predict(model,np.stack(views));removed=set();events=[];pinned=set()
    for r,p in sorted(zip(pool,probs),key=lambda z:(-lens[z[0]['representative']],-float(z[1][1]))):
        i,j=r['curve'],r['representative']
        if i in removed or i in pinned or j in removed or p[1]<threshold:continue
        removed.add(i);pinned.add(j);events.append(dict(**r,duplicate_probability=float(p[1]),source_identity_verified=False))
    return dict(selected=[i for i in range(len(rows)) if i not in removed],removed=sorted(removed),events=events,
        candidate_pairs=len(pool),threshold=threshold,whole_curve_coverage_required=True,closed_motifs_protected=True,
        extended_bright_gap_protected=True,raw_geometry_archived=True)

def actual_audit(context_model,context_threshold,relation_model,relation_threshold):
    from context_curve_apply_20261105 import select,metrics
    records=[]
    for r in source_records():
        if not r['prediction_file']:continue
        rows=[dict(points=p,module=0,source_curve_ids=[i]) for i,p in enumerate(r['paths'])]
        context=select(context_model,context_threshold,r['rgb'],rows)
        part=[rows[i] for i in context['selected']]
        relation=suppress(relation_model,relation_threshold,r['rgb'],part)
        ids=[context['selected'][i] for i in relation['selected']]
        before=metrics(r['paths'],r['refs'],r['background'])
        after=metrics([r['paths'][i] for i in ids],r['refs'],r['background'])
        drop=max(a['covered_fraction']-b['covered_fraction'] for a,b in zip(before['reference_details'],after['reference_details']))
        records.append(dict(id=r['id'],split=r['split'],before=before,after=after,selected=ids,
            context_selection=context,relation_selection=relation,
            max_per_reference_coverage_drop=drop,passes_coverage_guard=drop<=.01,
            source_image=r['image'],prediction_file=r['prediction_file'],labels=r['labels']))
    result=dict(records=records,passes_all_actual_output_coverage_guards=all(r['passes_coverage_guard'] for r in records),
        scope='actual archived predictions, partial references, same source families',independent_painting_accuracy=False)
    write(ROOT/'actual_prediction_relation_audit.json',result)
    print('REL_ACTUAL',[(r['id'],r['relation_selection']['candidate_pairs'],len(r['relation_selection']['removed']),r['max_per_reference_coverage_drop']) for r in records],flush=True)
    return result

if __name__=='__main__':
    torch.set_num_threads(4);cv2.setNumThreads(2);train()
