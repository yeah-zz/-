# -*- coding: utf-8 -*-
"""Real user-reviewed fitting experiment; unmarked pixels remain unknown."""
import sys,json,time,math,shutil,os,copy
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw
from stroke_topology_v13 import model_new,targets,heat_loss,infer as infer_tiled
from human_mountain_v11 import fields,digest,jsave,FONT,SMALL,paste
from gongbi_training.review_server_v11 import validate_document
BASE=LAB/'reviewed_rocks_v14_20261024'
CORRECTED=os.environ.get('SHANHE_REVIEW_VARIANT')=='corrected'
ROOT=BASE/'corrected' if CORRECTED else BASE
REVIEW=LAB/'stroke_topology_v13_20261024'/'user_review'
PARENT=LAB/'clean_strokes_v12_20261024'/'final.pt'

def curate(doc):
    d=copy.deepcopy(doc)
    if d['sample_id']!='R01':return d
    from stroke_link_v12 import dense
    from scipy.spatial import cKDTree
    by={s['id']:s for s in d['strokes']};mapping={'draft_1':'s_mvmallc6_k5k5','draft_2':'s_mvmalaj0_pdcy'};fragments=[];changes=[]
    for old,new in mapping.items():
        if old not in by or new not in by:raise ValueError('Confirmed replacement IDs changed; preserve source and re-audit')
        a=dense(by[old]['points']);b=dense(by[new]['points']);dist=cKDTree(b).query(a)[0];coverage=float(np.mean(dist<=5));changes.append(dict(old_id=old,new_id=new,old_fraction_near_new=coverage,user_confirmation='new line replaces corresponding overlapping old draft'))
        if coverage<.7:
            # Unmatched old geometry is kept, but artificial cut points never
            # become user-confirmed endpoints or a new brush-lift annotation.
            keep=dist>5;starts=np.flatnonzero(keep&~np.r_[False,keep[:-1]]);ends=np.flatnonzero(keep&~np.r_[keep[1:],False])+1
            for k,(start,end) in enumerate(zip(starts,ends)):
                p=a[start:end]
                if len(p)>2 and np.linalg.norm(np.diff(p,axis=0),axis=1).sum()>=8:fragments.append(dict(id=old+'_unmatched_geometry_'+str(k),kind='stroke',role='contour',points=p.round(2).tolist(),geometry_only=True))
    d['strokes']=[s for s in d['strokes'] if s['id'] not in mapping];d['geometry_only_fragments']=fragments;d['replacement_log']=changes;d['label_provenance']='user_reviewed_with_chat_confirmed_replacement';return d

def build_targets(doc):
    t,m=targets(doc,False)
    if doc.get('geometry_only_fragments'):
        ext=copy.deepcopy(doc);ext['strokes']+=doc['geometry_only_fragments'];et,em=targets(ext,False)
        # Geometry contributes to center/directions only. Endpoint and junction
        # targets/confidence remain those of the confirmed complete strokes.
        et[5:7]=t[5:7];em[4:6]=m[4:6]*em[7][None];return et,em
    return t,m

def prepare():
    ROOT.mkdir(exist_ok=True);folder=ROOT/'data';folder.mkdir(exist_ok=True);records=[];package=json.loads((REVIEW/'package_manifest.json').read_text(encoding='utf-8'));board=Image.new('RGB',(1500,440*4),'#f5f3ec');d=ImageDraw.Draw(board)
    for i,r in enumerate(package['samples']):
        path=REVIEW/'labels'/(r['id']+'.json');doc=json.loads(path.read_text(encoding='utf-8'));validate_document(doc,r,package['package_id'])
        if doc['review_state']!='checked_partial':raise ValueError('User check missing: '+r['id'])
        if digest(REVIEW/r['image'])!=r['image_sha256']:raise ValueError('image changed')
        im=folder/(r['id']+'.png');label=folder/(r['id']+'.json');shutil.copy2(REVIEW/r['image'],im);shutil.copy2(path,folder/(r['id']+'_original_user.json'));doc=curate(doc) if CORRECTED else doc;jsave(label,doc);center,axis,mask,dist=fields(doc);t,m=build_targets(doc)
        # Native content only; artificial batch padding is never a negative.
        p=np.asarray(Image.open(im).convert('RGB'));h,w=p.shape[:2];box=r['source_box'];cw,ch=box[2]-box[0],box[3]-box[1];left=max(0,(w-cw)//2);top=max(0,(h-ch)//2);content=np.zeros((h,w),np.uint8);content[top:top+ch,left:left+cw]=1;m*=content[None]
        np.savez_compressed(folder/(r['id']+'_targets.npz'),target=t,mask=m);overlay=p.copy();overlay[cv2.dilate(center,np.ones((2,2),np.uint8))>0]=[235,65,20]
        for reg in doc['regions']:cv2.polylines(overlay,[np.rint(reg['points']).astype(np.int32)],True,(20,170,65) if reg['kind']=='background' else (150,50,190),2)
        ref=np.full_like(p,255);ref[center>0]=[235,65,20];ref[mask[1]>0]=[180,230,185]
        for j,(name,a) in enumerate([('原图',p),('已复核描线与背景',ref),('标注叠加',overlay)]):d.text((j*500+8,i*440+6),r['id']+' '+name,font=FONT,fill='black');paste(board,a,j*500+8,i*440+42,484,390)
        records.append(dict(id=r['id'],source_id=r['source_id'],image=str(im),labels=str(label),target=str(folder/(r['id']+'_targets.npz')),image_sha256=digest(im),label_sha256=digest(label),original_user_label_sha256=digest(path),source_group=r['source_group'],source_page=r['source_page'],strokes=len(doc['strokes']),geometry_only_fragments=len(doc.get('geometry_only_fragments',[])),replacement_log=doc.get('replacement_log',[]),background_regions=sum(a['kind']=='background' for a in doc['regions']),uncertain_regions=sum(a['kind']=='uncertain' for a in doc['regions']),known_positive_pixels=int(m[0].sum()),known_background_pixels=int(m[1].sum()),ignored_pixels=int((m[7]==0).sum()),native_content=[left,top,cw,ch],provenance='user_reviewed_target_robot_strokes',complete=False))
    jsave(ROOT/'manifest.json',dict(records=records,unmarked='unknown',reviewed_strokes=sum(r['strokes'] for r in records),small_fit_ids=['R01','R02','R03'],small_fit_development_id='R04',all_images_seen_by_parent=True,same_motif_step_images=True,independent_test=False,source_work_independence_not_verified=True));board.save(ROOT/'annotation_audit.jpg',quality=95);return records

def loss_fn(o,t,m):
    o=o.float();b=F.binary_cross_entropy_with_logits(o[:,0],t[:,0],reduction='none');center=(b*m[:,0]).sum()/m[:,0].sum().clamp_min(1)+(F.softplus(o[:,0])*m[:,1]).sum()/m[:,1].sum().clamp_min(1)
    a=F.normalize(o[:,1:3],dim=1);b=F.normalize(o[:,3:5],dim=1);aa=1-(a*t[:,1:3]).sum(1);ab=1-(a*t[:,3:5]).sum(1);ba=1-(b*t[:,1:3]).sum(1);bb=1-(b*t[:,3:5]).sum(1)
    # Canonical primary direction on unambiguous strokes. The secondary
    # direction is used only at predicted crossings, avoiding a free second
    # axis creating arbitrary extra ridges on ordinary real brush marks.
    direction=(aa*(1-m[:,3])+.5*torch.minimum(aa+bb,ab+ba)*m[:,3])*m[:,2];axis=direction.sum()/m[:,2].sum().clamp_min(1)
    ep=heat_loss(o[:,5],t[:,5],m[:,4]);joint=heat_loss(o[:,6],t[:,6],m[:,5]);b=F.binary_cross_entropy_with_logits(o[:,7],t[:,7],reduction='none');multiple=(b*m[:,6]).sum()/m[:,6].sum().clamp_min(1)
    return center+.15*axis+.03*ep+.03*joint+.08*multiple,dict(center=float(center.detach()),direction=float(axis.detach()),endpoints=float(ep.detach()),junctions=float(joint.detach()))

@torch.no_grad()
def direct(model,rgb):
    p=model(torch.from_numpy(rgb.transpose(2,0,1).copy())[None].cuda().float()/255).float();p[:,[0,5,6,7]]=p[:,[0,5,6,7]].sigmoid();p[:,1:3]=F.normalize(p[:,1:3],dim=1);p[:,3:5]=F.normalize(p[:,3:5],dim=1);return p[0].cpu().numpy()

def train(mode):
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(142410 if mode=='small' else 142411);folder=ROOT/mode
    if (folder/'training_summary.json').exists():raise FileExistsError('Preserve completed run')
    folder.mkdir(parents=True,exist_ok=True)
    records=prepare() if not (ROOT/'manifest.json').exists() else json.loads((ROOT/'manifest.json').read_text())['records'];chosen=[r for r in records if mode!='small' or r['id']!='R04']
    arrays=[]
    for r in chosen:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));z=np.load(r['target']);arrays.append((rgb,z['target'].copy(),z['mask'].copy()))
    model=model_new().cuda();lr=1.5e-4 if mode=='small' else 6e-5;steps=600 if mode=='small' else 800;rng=np.random.default_rng(142410 if mode=='small' else 142411);opt=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=lr*.2),dict(params=[p for n,p in model.named_parameters() if not n.startswith('encoder.')],lr=lr)],weight_decay=0 if mode=='small' else 1e-4);start=time.perf_counter()
    replay=[]
    if mode=='replay':
        old_records=json.loads((LAB/'clean_strokes_v12_20261024'/'manifest.json').read_text(encoding='utf-8'))['records'];ids=['C01','C06']+['U_A0'+str(i) for i in range(1,6)];teacher=model_new().cuda().eval()
        for identity in ids:
            r=next(r for r in old_records if r['id']==identity);assert r['split']=='train'
            rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=json.loads(Path(r['labels']).read_text(encoding='utf-8'));_,_,mm,_=fields(doc);valid=mm[3].copy();left,top=r.get('padding',[0,0]);box=r.get('source_box',[0,0,rgb.shape[1],rgb.shape[0]]);native=np.zeros(valid.shape,np.float32);native[top:top+box[3]-box[1],left:left+box[2]-box[0]]=1;valid*=native
            replay.append((rgb,direct(teacher,rgb),valid,r))
        del teacher;torch.cuda.empty_cache()
    config=dict(mode=mode,initialization=str(PARENT),initialization_sha256=digest(PARENT),steps=steps,ids=[r['id'] for r in chosen],user_strokes=sum(r['strokes'] for r in chosen),unmarked_real_regions_unknown=True,synthetic_samples=0,initialization_not_failed_v13_joint_model=True,canonical_primary_direction=True,weights_not_selected_on_holdout=True,robot_connected=False)
    if replay:config.update(replay_ids=[r[3]['id'] for r in replay],replay_teacher='v12 soft probabilities and primary tangent, not new expert labels',replay_uses_test_images=False,replay_teacher_weight=1.,replay_one_image_per_update=True)
    jsave(folder/'config.json',config)
    for step in range(1,steps+1):
        model.train()
        for layer in model.encoder.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
        factor=.3+.7*.5*(1+math.cos(math.pi*step/steps))
        for g,mult in zip(opt.param_groups,[.2,1]):g['lr']=lr*mult*factor
        opt.zero_grad(set_to_none=True);parts_list=[];values=[]
        for rgb,target,mask in arrays:
            # Accumulate native-size pictures independently. No batching pad,
            # geometric augmentation or raster cleanup in the small fit.
            x=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].cuda().float()/255;t=torch.from_numpy(target[None]).cuda();m=torch.from_numpy(mask[None]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=loss_fn(model(x),t,m)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            (loss/len(arrays)).backward();values.append(float(loss.detach()));parts_list.append(parts)
        if replay:
            rgb,tp,valid,_=replay[(step-1)%len(replay)];x=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].cuda().float()/255;tp=torch.from_numpy(tp[None]).cuda();valid=torch.from_numpy(valid[None]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                ro=model(x).float();bce=F.binary_cross_entropy_with_logits(ro[:,0],tp[:,0],reduction='none');pos=valid*tp[:,0];neg=valid*(1-tp[:,0]);center_replay=.5*((bce*pos).sum()/pos.sum().clamp_min(1)+(bce*neg).sum()/neg.sum().clamp_min(1));dw=valid*tp[:,0].square();direction_replay=((1-(F.normalize(ro[:,1:3],dim=1)*tp[:,1:3]).sum(1))*dw).sum()/dw.sum().clamp_min(1);replay_loss=center_replay+.15*direction_replay
            if not torch.isfinite(replay_loss):raise RuntimeError('nonfinite replay loss')
            replay_loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(grad):raise RuntimeError('nonfinite gradient')
        opt.step()
        if step%50==0:
            row=dict(step=step,loss=float(np.mean(values)),elapsed_seconds=time.perf_counter()-start,**{k:float(np.mean([p[k] for p in parts_list])) for k in parts_list[0]})
            if replay:row['replay_soft_teacher_loss']=float(replay_loss.detach())
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('REAL_FIT_'+mode.upper(),json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),config=config),folder/'final.pt');jsave(folder/'training_summary.json',dict(successful_updates=steps,elapsed_seconds=time.perf_counter()-start,real_user_reviewed_images=len(chosen),reviewed_strokes=sum(r['strokes'] for r in chosen),parent_unchanged=digest(PARENT)==config['initialization_sha256'],small_model_is_fit_diagnostic_only=mode=='small',robot_connected=False));shutil.copy2(Path(__file__),folder/'source_snapshot.py');print('REAL_REVIEW_TRAINED',mode,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','small','main','replay']);a=p.parse_args();prepare() if a.mode=='prepare' else train(a.mode)
