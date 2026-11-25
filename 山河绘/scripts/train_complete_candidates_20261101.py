# -*- coding: utf-8 -*-
"""Curve/profile classifier with explicitly scoped non-expert supervision."""
import sys
sys.dont_write_bytecode=True
from mountain_complete_evidence_20261101 import *
import torch,time,json
from torch import nn
NEW={
 'G01_01':dict(keep=['v4_1643','ridge_77','ridge_326','ridge_1031','ridge_717','ridge_871','ridge_962'],reject=['ridge_386','ridge_488','ridge_546','ridge_342','ridge_534']),
 'G05_01':dict(keep=['v4_1075','v4_511','v4_482','v4_613','v4_554','v4_359','ridge_880','ridge_353','ridge_374','ridge_254','ridge_700','ridge_243','ridge_256','ridge_493','ridge_378'],reject=['ridge_847','ridge_659','ridge_283','ridge_398']),
 'G03_02':dict(keep=['v4_1169','v4_1478','v4_869','v4_504','ridge_927','ridge_1380','ridge_219','ridge_173','ridge_560'],reject=['v4_577','v4_759','ridge_827','ridge_398','ridge_635','ridge_703','ridge_619']),
 'G02_01':dict(keep=['v4_481','v4_16','v4_373','ridge_411','ridge_89','ridge_556','ridge_86','ridge_393','ridge_164'],reject=['v4_110','v4_416','ridge_680','ridge_709','ridge_330','ridge_859','ridge_290','ridge_496']),
 'G04_01':dict(keep=['v4_774','ridge_267','ridge_286','ridge_95','ridge_346','ridge_353','ridge_238','ridge_670'],reject=['v4_61','ridge_621','ridge_397','ridge_685','ridge_606','ridge_497','ridge_464','ridge_262','ridge_146']),
}
SPLIT={'G01_01':'train','G03_02':'train','G05_01':'train','G02_01':'calibration','G04_01':'regression'}
FIT=ROOT/'fit_r2'
PROTOCOL=dict(version='curve_profile_mlp_r2',features=FEATURES,architecture=[len(FEATURES),24,1],seeds=[143,277,433],updates_per_seed=400,lr=.006,weight_decay=.03,photometric_fade_alphas=[1.,.8,.6],augmentation_scope='Two RGB-to-paper blends per training source; same reference geometry, not additional artworks',train_parent_groups=[1,3,5],calibration_parent_groups=[2],regression_parent_groups=[4],unknown_loss=False,expert_ground_truth=False,images_reused=True,accept_threshold_candidates=[.8,.85,.9,.95,.98,.99,.995,.999,.9995,.9999,1.01])

def network():return nn.Sequential(nn.Linear(len(FEATURES),24),nn.SiLU(),nn.Linear(24,1))

def vector(row):return np.array([row['features'][n] for n in FEATURES],np.float32)

def prediction(artifact,rows,device='cpu'):
    if not rows:return np.zeros(0),np.zeros(0)
    x=(np.stack([vector(r) for r in rows])-np.asarray(artifact['mean'],np.float32))/np.asarray(artifact['std'],np.float32);x=torch.from_numpy(x).to(device);ps=[]
    with torch.no_grad():
        for state in artifact['states']:
            m=network().to(device);m.load_state_dict(state);m.eval();ps.append(torch.sigmoid(m(x)).flatten().cpu().numpy())
    ps=np.stack(ps);return ps.mean(0),ps.std(0)

def geometry_gate(row):
    f=row['features'];new=row.get('key','').startswith('ridge_');minimum=CFG['min_new_ridge_length'] if new else CFG['min_candidate_length'];return (math.expm1(f['log_length'])>=minimum and math.expm1(f['log_extent'])>=CFG['min_extent'] and f['support_adaptive']>=CFG['minimum_supported_fraction'] and f['rgb_support']>=CFG['rgb_supported_fraction'] and f['bend_q90']<=CFG['max_bend_q90'] and not f['compact_loop'] and f['hairpin']<.25 and f.get('narrow_return',0)<.15)

def references():
    # Retain all previous scoped labels including unknowns. Features are
    # recomputed from RGB along those candidates, not from previous decisions.
    refs=[];seen=set()
    for r in read(PREVIOUS/'partial_references.json')['references']:
        ident=r['image'];key='v4_'+str(r['candidate']);refs.append(dict(image=ident,key=key,label=r['label'],points=r['points'],split=SPLIT[ident],basis=r['basis'],expert_verified=False,annotation_author='assistant_nonexpert_local_rgb_review',origin='previous_partial_reference'));seen.add((ident,key))
    for ident,label in NEW.items():
        data=read(ROOT/'prepared'/ident/'candidates.json');bykey={r['key']:r for r in data['rows']};keys=read(ROOT/'review_cards'/f'{ident}_manifest.json')
        assert not set(label['keep'])&set(label['reject'])
        for key in keys:
            if (ident,key) in seen:continue
            r=bykey[key];value=1 if key in label['keep'] else 0 if key in label['reject'] else None
            refs.append(dict(image=ident,key=key,label=value,points=r['points'],split=SPLIT[ident],basis='Visible ink/contour supports the curved or faint candidate' if value==1 else 'RGB shows wash fluctuation, blot perimeter, or a hairpin joining parallel sides instead of one centerline' if value==0 else 'Ambiguous image evidence or stroke identity; excluded from supervision',expert_verified=False,annotation_author='assistant_nonexpert_local_rgb_review',origin='new_curve_faint_review'));seen.add((ident,key))
        assert set(label['keep']+label['reject'])<=set(keys)
    return refs

def prepare_training(refs):
    rows=[]
    for ident in SPLIT:
        rgb=np.asarray(Image.open(STRUCTURE/'outputs'/ident/'input.png').convert('RGB'));subset=[r for r in refs if r['image']==ident and r['label'] is not None];paper=np.quantile(rgb.reshape(-1,3),.96,axis=0)
        for alpha in PROTOCOL['photometric_fade_alphas'] if SPLIT[ident]=='train' else [1.]:
            modified=np.rint(rgb*alpha+paper*(1-alpha)).clip(0,255).astype(np.uint8);field=evidence_field(modified)
            for ref in subset:
                f=arc_profile(modified,np.asarray(ref['points']),field);rows.append(dict(image=ident,key=ref['key'],label=ref['label'],split=ref['split'],fade_alpha=alpha,features=f))
    jsave(ROOT/'training_rows.json',dict(rows=rows,unlabeled_excluded=True));return rows

def main():
    if (FIT/'model.pt').exists():raise RuntimeError('Completed fit is immutable; use another experiment version')
    refs=references();jsave(ROOT/'partial_references.json',dict(references=refs,expert_verified=False,unlabeled_are_negative=False));protocol=dict(PROTOCOL,reference_sha256=digest(ROOT/'partial_references.json'),training_code_sha256=digest(__file__),evidence_code_sha256=digest(Path(__file__).with_name('mountain_complete_evidence_20261101.py')),config=CFG);jsave(ROOT/'training_protocol.json',protocol)
    rows=prepare_training(refs);train=[r for r in rows if r['split']=='train'];x=np.stack([vector(r) for r in train]);mean=x.mean(0);std=np.maximum(x.std(0),.03);xx=torch.from_numpy((x-mean)/std);y=np.array([r['label'] for r in train],np.float32);yy=torch.from_numpy(y)
    assert torch.cuda.is_available();device='cuda';torch.set_num_threads(2);torch.use_deterministic_algorithms(True);xx=xx.to(device);yy=yy.to(device);weights=torch.where(yy>0,.5/max(1,int(y.sum())),.5/max(1,int((1-y).sum())));states=[];logs=[];started=time.time()
    for seed in PROTOCOL['seeds']:
        torch.manual_seed(seed);m=network().to(device);opt=torch.optim.AdamW(m.parameters(),lr=PROTOCOL['lr'],weight_decay=PROTOCOL['weight_decay'])
        for step in range(1,PROTOCOL['updates_per_seed']+1):
            pred=m(xx).flatten();loss=(nn.functional.binary_cross_entropy_with_logits(pred,yy,reduction='none')*weights).sum();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step in [1,100,200,400]:logs.append(dict(seed=seed,step=step,loss=float(loss.detach().cpu())))
        states.append({k:v.detach().cpu() for k,v in m.state_dict().items()});print('COMPLETE_GPU_FIT',seed,logs[-1],flush=True)
    artifact=dict(states=states,mean=mean.tolist(),std=std.tolist(),features=FEATURES,threshold=1.01,protocol=protocol,updates_executed=1200,device=torch.cuda.get_device_name(0));cal=[r for r in rows if r['split']=='calibration'];cp,cs=prediction(artifact,cal,device);gate=np.array([geometry_gate(r) for r in cal]);threshold=1.01
    for t in PROTOCOL['accept_threshold_candidates']:
        yes=(cp>=t)&(cs<=CFG['model_uncertainty'])&gate
        if sum(yes[i] and r['label']==0 for i,r in enumerate(cal))==0:threshold=t;break
    artifact['threshold']=threshold;FIT.mkdir(exist_ok=True);torch.save(artifact,FIT/'model.pt');jsave(FIT/'training_log.json',logs);evaluated=[]
    for ident in SPLIT:
        subset=[r for r in rows if r['image']==ident and r['fade_alpha']==1.];p,u=prediction(artifact,subset,device);predictions=[]
        for r,score,unc in zip(subset,p,u):predictions.append(dict(key=r['key'],label=r['label'],score=float(score),ensemble_std=float(unc),geometry_gate=geometry_gate(r),accepted=bool(geometry_gate(r) and score>=threshold and unc<=CFG['model_uncertainty'])))
        evaluated.append(dict(image=ident,split=SPLIT[ident],positive=sum(r['label']==1 for r in predictions),negative=sum(r['label']==0 for r in predictions),true_accept=sum(r['label']==1 and r['accepted'] for r in predictions),false_accept=sum(r['label']==0 and r['accepted'] for r in predictions),raw_false_accept=sum(r['label']==0 and r['score']>=threshold and r['ensemble_std']<=CFG['model_uncertainty'] for r in predictions),predictions=predictions))
    summary=dict(records=evaluated,updates_executed=1200,training_rows=len(train),original_training_labels=sum(r['split']=='train' and r['label'] is not None for r in refs),reviewed_candidates=len(refs),unknown_excluded=sum(r['label'] is None for r in refs),threshold=threshold,seconds=time.time()-started,device=artifact['device'],full_image_accuracy=None,scope='Small candidate/profile MLP; reused sources, nonexpert partial labels; fading variants not new artworks')
    jsave(FIT/'evaluation.json',summary);print('COMPLETE_FIT_SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
