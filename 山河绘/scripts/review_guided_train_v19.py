# -*- coding: utf-8 -*-
"""V19 reviewed sparse supervision + automatic conservative QC, no robot SDK."""
from autonomous_review_v19 import *
import autonomous_review_v19 as reviewer
import train_aligned_v17 as train17
from scipy.ndimage import distance_transform_edt


def features_with_gray(gray,p,n):
    u=np.array([n[1],-n[0]],np.float32);coords=np.asarray(p)[None,None]+OFFSETS[None,:,None]*n+np.arange(-4,5,dtype=np.float32)[:,None,None]*u
    raw=cv2.remap(gray,coords[:,:,0].astype(np.float32),coords[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);av=np.sort(raw,axis=0)[1:-1].mean(0);l=float(np.quantile(av[OFFSETS<-15],.75));r=float(np.quantile(av[OFFSETS>15],.75));paper=np.interp(OFFSETS,[-20,20],[l,r]).clip(.15,1);contrast=np.maximum((paper-av)/paper,0);valid=((coords[:,:,0]>=0)&(coords[:,:,0]<gray.shape[1])&(coords[:,:,1]>=0)&(coords[:,:,1]<gray.shape[0])).mean(0);x=np.stack((np.clip(av/paper,0,1.2),contrast,raw.std(0)/paper,OFFSETS/24,valid)).astype(np.float32);pr,band=reviewer.geometry_prior(x);return np.concatenate((x,pr[None],band[None]))


@torch.no_grad()
def automatic_qc(model,rgb,t,m,bg):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;ys,xs=np.nonzero((t[0]>.65)&(m[2]>.2));points=np.column_stack((xs,ys))[::3];pass_map=np.zeros(gray.shape,np.uint8);out=[]
    for start in range(0,len(points),128):
        group=points[start:start+128];rows=[]
        for x,y in group:
            theta=.5*math.atan2(float(t[3,y,x]),float(t[2,y,x]));n=np.array([-math.sin(theta),math.cos(theta)],np.float32);f=features_with_gray(gray,np.array([x,y]),n);rows.append(dict(x=f,target=None,present=1,identity='auto',center=0))
        result=reviewer.assess(model,rows)
        for p,r in zip(group,result):
            passed=r['auto_accept'] and abs(r['estimated_offset'])<=2
            if passed:cv2.circle(pass_map,tuple(p.astype(int)),5,1,-1)
            out.append(dict(point=p.tolist(),automatic_pass=bool(passed),**r))
    revised=m.copy();ink=t[1]>.5;factor=.20+.80*pass_map;known_non_bg=bg==0
    revised[0][known_non_bg]*=factor[known_non_bg];revised[1][known_non_bg]*=.5+.5*pass_map[known_non_bg];revised[2:]*=pass_map[None]
    return revised,out


def inject_reviews(rgb,t,m,valid,rows,candidates):
    h,w=rgb.shape[:2];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;yy,xx=np.mgrid[:h,:w];log=[];focus=[]
    for a in rows:
        r=candidates[a['id']];p=np.asarray(r['p']);n=np.asarray(r['normal']);u=np.array([n[1],-n[0]]);along=(xx-p[0])*u[0]+(yy-p[1])*u[1];normal=(xx-p[0])*n[0]+(yy-p[1])*n[1];strip=np.abs(along)<=3.5
        if a['label']=='uncertain':
            zone=strip&(np.abs(normal)<=10)&(valid>0);m[:,zone]=0;log.append(dict(id=a['id'],decision='uncertain_excluded_not_negative'));continue
        if a['label']=='background':
            zone=(np.abs(along)<6)&(np.abs(normal)<6)&(valid>0);t[0:2,zone]=0;m[0:2,zone]=1;m[2:,zone]=0;focus.append(p.tolist());log.append(dict(id=a['id'],decision='user_confirmed_background'));continue
        center=float(a['center']);features=features_with_gray(gray,p,n);index=int(np.argmin(np.abs(OFFSETS-center)));evidence=float(features[1,max(0,index-2):min(97,index+3)].max());coords=p+center*n;focus.append(coords.tolist());broad=(a['right']-a['left'])>=24;clipped=a['left']<=-23 or a['right']>=23
        # User class remains recorded; no hard center geometry on contradictory/censored scans.
        if evidence<.012 or features[4,index]<.9:
            zone=strip&(np.abs(normal-center)<=5)&(valid>0);m[2:,zone]=0;log.append(dict(id=a['id'],decision='class_retained_geometry_abstained',evidence=evidence));continue
        seed=float(features[1,max(0,index-2):min(97,index+3)].max());on=gaussian_filter1d(features[1],.8)>max(.008,.27*seed);left=index;right=index+1
        while left>0 and on[left-1]:left-=1
        while right<len(on) and on[right]:right+=1
        lo=float(OFFSETS[left]);hi=float(OFFSETS[min(96,right-1)]);single=lo<center<hi and hi-lo<=28
        if not single:lo=center-2;hi=center+2
        # Geometry edits are used locally. Broad unchanged drafts do not become precise edge truth.
        user_edges_modified=any(abs(a[k]-r[k])>.02 for k in ['left','right'])
        trust_edges=user_edges_modified and not clipped and not broad
        if trust_edges:lo=float(a['left']);hi=float(a['right'])
        half=max(3,abs(center-lo)+2,abs(hi-center)+2);zone=strip&(np.abs(normal-center)<=half)&(valid>0);inkzone=strip&(normal>=lo)&(normal<=hi)&(valid>0);weight=1. if a.get('geometry_changed',False) else .65
        if broad and not a.get('geometry_changed',False):weight=.35
        t[0,zone]=np.exp(-(normal[zone]-center)**2/(2*1.1**2));m[0,zone]=weight;t[1,inkzone]=1;m[1,inkzone]=max(weight,.5);theta=math.atan2(u[1],u[0]);axiszone=strip&(np.abs(normal-center)<1.8)&(valid>0);t[2,axiszone]=math.cos(2*theta);t[3,axiszone]=math.sin(2*theta);m[2,axiszone]=weight if not broad else 0
        if trust_edges:t[4,axiszone]=(hi-lo)/48;m[3,axiszone]=.8
        else:m[3,zone]=0
        log.append(dict(id=a['id'],decision='user_center_and_presence',center_weight=weight,precise_edges_from_user=trust_edges,user_broad_boundary_uncertainty_respected=broad and not trust_edges,edge_source='user_modified' if trust_edges else 'automatic_photometric_not_expert',evidence=evidence))
    return t,m,log,focus


def prepare():
    torch.set_num_threads(4);cv2.setNumThreads(2);folder=ROOT/'data';folder.mkdir(parents=True,exist_ok=True);old=read(OLD/'manifest.json');reviews=read(ROOT/'review_audit.json')['rows'];candidate={r['id']:r for r in read(ROOT/'draft_candidates_snapshot.json')};qc=reviewer.load_reviewer();teacher=v16.RibbonNet().cuda();teacher.load_state_dict(torch.load(OLD/'aligned'/'final.pt',map_location='cuda',weights_only=False)['model']);teacher.eval();out=[]
    for r in old['records']:
        if r['split']!='train':continue
        z=np.load(r['data']);rgb=z['rgb'];t=z['target'].copy();m=z['mask'].copy();bg=z['background'];valid=z['valid'];filtered,decisions=automatic_qc(qc,rgb,t,m,bg);mine=[a for a in reviews if a['sample_id']==r['id']];t,filtered,logs,focus=inject_reviews(rgb,t,filtered,valid,mine,candidate);prediction=v16.predict(teacher,rgb);file=folder/(r['id']+'.npz');np.savez_compressed(file,rgb=rgb,target=t,mask=filtered,control_target=z['target'],control_mask=z['mask'],teacher_center=prediction[0],background=bg,valid=valid);jsave(folder/(r['id']+'_qc.json'),dict(automatic_decisions=decisions,user_injection=logs,automatic_results_not_expert=True));out.append(dict(r,data=str(file),focus=focus,manual_local_annotations=len(mine),automatic_pass=sum(a['automatic_pass'] for a in decisions),automatic_total=len(decisions),image_sha256=digest(r['image']),label_sha256=digest(r['labels'])));print('V19_DATA',r['id'],'review',len(mine),'auto_pass',out[-1]['automatic_pass'],'/',len(decisions),flush=True)
    jsave(ROOT/'manifest.json',dict(records=out,training_ids=[r['id'] for r in out],calibration_ids=['N07','N08'],development_ids=['R03','R04','N01','N10','N11','T01','T02'],init_weights=str(OLD/'aligned'/'final.pt'),init_sha256=digest(OLD/'aligned'/'final.pt'),original_parent_sha256=digest(PARENT),review_snapshot_sha256=digest(ROOT/'user_review_snapshot.json'),reviewer_sha256=digest(ROOT/'reviewer'/'final.pt'),uncertain_not_negative=True,unchanged_broad_drafts_not_precise_edge_gold=True,user_faint_labels_not_physical_ink_concentration=True,robot_connected=False))


def train(variant):
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(192510);rng=np.random.default_rng(192510);manifest=read(ROOT/('manifest_v2.json' if variant.endswith('_v2') else 'manifest.json'));folder=ROOT/variant;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'final.pt').exists():raise FileExistsError('Completed model immutable')
    data=[]
    for r in manifest['records']:
        z=np.load(r['data']);prefix='control_' if variant.startswith('control') else '';data.append(dict(id=r['id'],rgb=z['rgb'].copy(),t=np.concatenate((z[prefix+'target'],z['teacher_center'][None],z['uncertain_exclusion'][None] if 'uncertain_exclusion' in z.files else np.zeros_like(z['teacher_center'][None]))),m=z[prefix+'mask'].copy(),focus=r['focus']))
    model=v16.RibbonNet().cuda();model.load_state_dict(torch.load(manifest['init_weights'],map_location='cuda',weights_only=False)['model']);opt=torch.optim.AdamW([dict(params=list(model.a.parameters())+list(model.b.parameters())+list(model.c.parameters()),lr=.000015),dict(params=list(model.d.parameters())+list(model.e.parameters())+list(model.head.parameters()),lr=.00008)],weight_decay=.0001);steps=800;usage=dict(user=0,assistant=0,review_focus=0);start=time.perf_counter();config=dict(variant=variant,steps=steps,seed=192510,init_sha256=manifest['init_sha256'],training_ids=manifest['training_ids'],calibration_ids=manifest['calibration_ids'],unknown_distillation_weight=.2,unchanged_5_head_architecture=True,uncertain_excluded_from_all_losses=variant.endswith('_v2'),robot_connected=False);jsave(folder/'config.json',config);shutil.copy2(Path(__file__),folder/'source_at_start.py')
    for step in range(1,steps+1):
        batch=[]
        for _ in range(6):
            group=[a for a in data if a['id'].startswith('R')] if rng.random()<.65 else [a for a in data if not a['id'].startswith('R')];a=group[int(rng.integers(len(group)))];usage['user' if a['id'].startswith('R') else 'assistant']+=1;rgb=a['rgb'];t=a['t'];m=a['m']
            if a['focus'] and rng.random()<.55:
                point=a['focus'][int(rng.integers(len(a['focus'])))];h,w=rgb.shape[:2];x=int(np.clip(point[0]-128,0,max(0,w-256)));y=int(np.clip(point[1]-128,0,max(0,h-256)));rgb=rgb[y:y+256,x:x+256];t=t[:,y:y+256,x:x+256];m=m[:,y:y+256,x:x+256];usage['review_focus']+=1
            batch.append(v16.crop(rgb,t,m,rng,fade=False))
        x,t,m=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(3)];factor=.4+.6*.5*(1+math.cos(math.pi*step/steps))
        for g,lr in zip(opt.param_groups,[.000015,.00008]):g['lr']=lr*factor
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            o=model(x);loss=train17.loss_fn(o,t[:,:5],m);old=t[:,5];unknown=(m[:,0]<.1)&((old>.9)|(old<.1))&(t[:,6]<.5);distill=(F.binary_cross_entropy_with_logits(o[:,0].float(),old,reduction='none')*unknown).sum()/unknown.sum().clamp_min(1);loss=loss+.2*distill
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();g=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(g):raise RuntimeError('Nonfinite gradient')
        opt.step()
        if step%100==0:
            row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start)
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('V19_TRAIN',variant,json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),config=config),folder/'final.pt');jsave(folder/'training_summary.json',dict(successful_updates=steps,usage=usage,elapsed_seconds=time.perf_counter()-start,robot_connected=False));print('V19_TRAINED',variant,flush=True)


def load_model(variant):
    m=v16.RibbonNet().cuda();m.load_state_dict(torch.load(ROOT/variant/'final.pt',map_location='cuda',weights_only=False)['model']);return m.eval()


def evaluate(variant,mode):
    previous_root=prior_eval.ROOT;previous_load=prior_eval.load;prior_eval.ROOT=ROOT;prior_eval.load=load_model
    try:prior_eval.calibrate(variant) if mode=='calibrate' else prior_eval.evaluate(variant)
    finally:prior_eval.ROOT=previous_root;prior_eval.load=previous_load


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','calibrate','evaluate']);p.add_argument('variant',nargs='?',choices=['control','reviewed','control_v2','reviewed_v2']);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='train':train(a.variant)
    else:evaluate(a.variant,a.mode)

