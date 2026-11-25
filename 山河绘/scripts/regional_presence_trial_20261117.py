# -*- coding: utf-8 -*-
"""Frozen paired trial of positive regional object presence. No robot control."""
import os,sys,json,hashlib,random,time,math
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,cv2
from PIL import Image
from gongbi_training.regional_presence import regional_presence_loss
from gongbi_training.scene_objects import SceneObjects,scene_metrics
from gongbi_training.scene_object_reference import sparse_scene_reference
from gongbi_training.sparse_object_supervision import masked_teacher_kl
from gongbi_training.cunfa_supervision import partial_role_loss
from guohua_objects_train_20261112 import pack,predict
ROOT=LAB/'regional_presence_trial_20261117';BASE=LAB/'guohua_scene4_mapping_fix_20261116';SOURCE=LAB/'clear_rock_sources_20261117'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,v):(ROOT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')

def projected_support(box,geometry):
    left,top,nw,nh,w,h=geometry;x0,y0,x1,y1=box
    mask=np.zeros((256,256),bool)
    mask[top+int(math.floor(y0*nh/h)):top+int(math.ceil(y1*nh/h)),
         left+int(math.floor(x0*nw/w)):left+int(math.ceil(x1*nw/w))]=True
    return mask

def prepare():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Frozen trial already prepared'
    incoming=read(SOURCE/'local_references.json');records=incoming['records']
    oldp=read(LAB/'pima_static_discovery_20261109/protocol.json')
    assert 'new_fuchun' in oldp['external_work_ids'] and not oldp['external_used_for_training']
    admission=[]
    for r in records:
        assert sha(r['frame'])==r['frame_sha256'] and r['style_training_target'] is None
        admitted=r['object_checked'] and r['object_type'] in {'rock','tree'}
        admission.append(dict(id=r['id'],admitted_for='weak_object_presence_only' if admitted else None,
            source_reference_training_admitted=r['training_admitted'],
            original_reference_unchanged=True,reviewer='assistant_native_crop_review_20261117',
            expert_verified=False,dense_pixel_target=False,ink_target=False,cunfa_target=None,
            known_development_material=True,reason=r['visual_evidence']))
    write('admission.json',dict(records=admission,review_scope='new task-specific weak admission; does not promote earlier full-stroke prohibition',
        source_manifest=str(SOURCE/'local_references.json'),source_manifest_sha256=sha(SOURCE/'local_references.json')))
    write('split_retirement.json',dict(work='ganjing:1gga38p7dgh3iKX2kYUhFzF7k1ns1c',
        historical_protocol=str(LAB/'pima_static_discovery_20261109/protocol.json'),historical_protocol_sha256=sha(LAB/'pima_static_discovery_20261109/protocol.json'),
        former_use='exploratory external tracer check; already inspected',new_use='development object-presence training for this trial',
        all_frames_and_resolutions_one_work=True,may_not_be_called_independent_test_after_training=True,
        historical_artifacts_unchanged=True,new_reserved_works_opened=False))
    p=dict(seed=171127,steps=128,arms=['continuation_control','regional_presence'],
        initialization=str(BASE/'mapping_fixed.pt'),initial_sha256=sha(BASE/'mapping_fixed.pt'),
        dataset_sha256=sha(BASE/'dataset_manifest.json'),old_reference_sha256=sha(BASE/'references.json'),
        region_reference_sha256=sha(SOURCE/'local_references.json'),admission_sha256=sha(ROOT/'admission.json'),
        script_sha256=sha(__file__),loss_module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/regional_presence.py'),
        class_ids={'0':'open_scene','1':'rock','2':'tree','3':'other','255':'unknown'},
        old_replay='2 coarse train images sampled by work +1 previous photo/video partial reference',
        weak_input='one of three full1280x720 frames, packed256 with exact box projection; same inputs both arms',
        weak_positives=8,uncertain_excluded=2,weak_source_works=1,
        common_loss='old partial CE +0.2 frozen four-class teacher KL on labelled old pixels',
        new_loss='0.3 mean per-image/class -log(mean of top20percent target probabilities in reviewed region); unknown boxes removed',
        optimizer='AdamW lr0.0001 wd0.0001; frozen image encoder, trainable existing decoder',
        selection='fixed last128 checkpoint; no checkpoint/threshold/top_fraction selection on evaluation',
        evaluation='known regressions only; historical new_test is now known14; full-source outputs, no reference crop at inference',
        independent_validation=False,cunfa_trained=False,tree_skill_trained=False,robot_connected=False)
    write('protocol.json',p)
    torch.set_num_threads(4)
    logits=torch.randn(1,4,8,8,requires_grad=True);mask=torch.zeros(8,8,dtype=torch.bool);mask[2:6,2:6]=True;mask[3,3]=False
    regional_presence_loss(logits,[(0,2,mask)]).backward()
    assert logits.grad[0,:,~mask].count_nonzero()==0 and logits.grad[0,:,mask].count_nonzero()>0
    z=torch.randn(1,4,8,8,requires_grad=True);loss=regional_presence_loss(z,[]);loss.backward();assert z.grad.count_nonzero()==0
    try:regional_presence_loss(logits,[(0,0,mask)])
    except ValueError:pass
    else:raise AssertionError('Unreviewed paper presence accepted')
    write('pretrain_checks.json',dict(unknown_outside_bag_gradients_zero=True,empty_bag_safe=True,
        object_presence_not_dense_target=True,paper_cunfa_and_pressure_not_supervised=True,
        historical_external_retired_from_independent_use=True,original_reference_files_unchanged=True))
    print('PREPARED',json.dumps({k:p[k] for k in ['steps','weak_positives','uncertain_excluded','weak_source_works']},ensure_ascii=False),flush=True)

def setup_data():
    ds=read(BASE/'dataset_manifest.json');refs=read(BASE/'references.json');data={}
    for r in ds['records']:
        data[r['id']]=(np.array(Image.open(r['image']).convert('RGB')),np.array(Image.open(r['target'])))
    for r in refs['photo_cases']+refs['video_cases']:
        scene,ink=sparse_scene_reference(r['size'],r['regions']);assert (ink==255).all()
        data[r['id']]=(np.array(Image.open(r['image']).convert('RGB')),scene)
    return ds,refs,data

def train():
    assert not (ROOT/'results.json').exists(),'Never retrain after seeing results'
    p=read(ROOT/'protocol.json');assert sha(__file__)==p['script_sha256']
    assert sha(p['initialization'])==p['initial_sha256'] and sha(SOURCE/'local_references.json')==p['region_reference_sha256']
    ds,refs,data=setup_data();allregions=read(SOURCE/'local_references.json')['records']
    positive=[r for r in allregions if r['object_checked'] and r['object_type'] in {'rock','tree'}]
    weakframes={}
    for path in sorted({r['frame'] for r in positive}):
        im=np.array(Image.open(path).convert('RGB'));x,_,geometry=pack(im)
        unknown=np.zeros((256,256),bool)
        for r in allregions:
            if r['frame']==path and r['object_type']=='unknown':unknown|=projected_support(r['box'],geometry)
        bags=[]
        for r in positive:
            if r['frame']!=path:continue
            support=projected_support(r['box'],geometry)&~unknown
            assert support.any();bags.append((1 if r['object_type']=='rock' else 2,support,r['id']))
        weakframes[path]=(x,bags)
    old=[r for r in ds['records'] if r['split']=='train'];oldgroups=[[r for r in old if r['work_group']==g] for g in sorted({r['work_group'] for r in old})]
    previous=[r for r in refs['photo_cases'] if r['split']=='train_photo']+refs['video_cases']
    prevgroups=[[r for r in previous if r['work_group']==g] for g in sorted({r['work_group'] for r in previous})]
    rng=random.Random(p['seed']);schedule=[]
    for _ in range(p['steps']):
        schedule.append(dict(replay=[rng.choice(rng.choice(oldgroups))['id'],rng.choice(rng.choice(oldgroups))['id'],rng.choice(rng.choice(prevgroups))['id']],weak=rng.choice(sorted(weakframes))))
    write('schedule.json',schedule)
    packed={ident:pack(*data[ident])[:2] for ident in {i for step in schedule for i in step['replay']}}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available(),'Use the existing isolated CUDA runtime'
    state=torch.load(p['initialization'],map_location='cpu',weights_only=False)['model']
    teacher=SceneObjects().cuda();teacher.load_state_dict(state);teacher.eval();teacher.requires_grad_(False)
    start=time.time();losses={};checkpoint_hashes={}
    for arm in p['arms']:
        random.seed(p['seed']);np.random.seed(p['seed']);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
        net=SceneObjects().cuda();net.load_state_dict(state);net.train()
        opt=torch.optim.AdamW([v for v in net.parameters() if v.requires_grad],lr=.0001,weight_decay=.0001)
        curve=[]
        for step,item in enumerate(schedule):
            xs=[packed[i][0] for i in item['replay']];ts=[packed[i][1] for i in item['replay']]
            wx,bags=weakframes[item['weak']];x=torch.from_numpy(np.stack(xs+[wx])).cuda();target=torch.from_numpy(np.stack(ts).astype('int64')).cuda()
            z=net(x)
            with torch.no_grad():tz=teacher(x[:3])
            common=partial_role_loss(z[:3],target)+.2*masked_teacher_kl(z[:3],tz,target.ne(255))
            extra=regional_presence_loss(z,[(3,c,torch.from_numpy(s).cuda()) for c,s,_ in bags])
            loss=common+(.3*extra if arm=='regional_presence' else extra*0)
            opt.zero_grad();loss.backward();opt.step();curve.append(dict(common=float(common.detach()),presence=float(extra.detach())))
            if (step+1)%32==0:print(arm,step+1,round(curve[-1]['common'],4),round(curve[-1]['presence'],4),flush=True)
        assert all(torch.equal(net.state_dict()[k].cpu(),v) for k,v in state.items() if k.startswith('encoder.'))
        torch.save(dict(model=net.state_dict(),protocol=p,arm=arm),ROOT/(arm+'.pt'))
        checkpoint_hashes[arm]=sha(ROOT/(arm+'.pt'));losses[arm]=curve
        del net,opt
    write('checkpoint_freeze.json',checkpoint_hashes);write('losses.json',losses)
    (ROOT/'predictions').mkdir(exist_ok=True);models={'frozen_mapping':teacher}
    for arm in p['arms']:
        net=SceneObjects().cuda();net.load_state_dict(torch.load(ROOT/(arm+'.pt'),map_location='cuda',weights_only=False)['model']);net.eval();models[arm]=net
    # Metrics use the same old source targets for all arms. Every previously viewed
    # case is explicitly known, regardless of its historical split string.
    cases=read(BASE/'results.json')['records'];rows=[];hmask=np.load(LAB/'cunfa_object_correction_20261112/H_14_object_reference.npz')['object_verified'];foliage={}
    for case in cases:
        ident=case['id'];image=np.array(Image.open(case['image']).convert('RGB'));scores={}
        for arm,net in models.items():
            probability,pred=predict(net,image);np.savez_compressed(ROOT/f'predictions/{ident}_{arm}.npz',prediction=pred)
            if ident in data:scores[arm]=scene_metrics(data[ident][1],pred)
            if ident=='H_14':foliage[arm]=float((pred[hmask]==2).mean())
        rows.append(dict(id=ident,image=case['image'],historical_split=case['split'],evaluation_scope='known_regression',scores=scores))
    aggregates={}
    for split in sorted({r['historical_split'] for r in rows if r['scores']}):
        aggregates[split]={}
        for arm in models:
            confusion=sum(np.array(r['scores'][arm]['confusion']) for r in rows if r['historical_split']==split)
            union=confusion.sum(0)+confusion.sum(1)-confusion.diagonal();ious=[float(confusion[k,k]/union[k]) if union[k] else None for k in range(4)]
            aggregates[split][arm]=dict(confusion=confusion.tolist(),iou=ious,mean_iou=float(np.mean([v for v in ious if v is not None])))
    weakresults=[]
    for path in sorted(weakframes):
        im=np.array(Image.open(path).convert('RGB'));item=dict(frame=path,regions=[],evaluation_scope='training_work_only_not_acceptance')
        predictions={}
        for arm,net in models.items():
            probability,pred=predict(net,im);predictions[arm]=(probability,pred)
            np.savez_compressed(ROOT/f'predictions/F{Path(path).stem[-4:]}_{arm}.npz',prediction=pred)
        for r in positive:
            if r['frame']!=path:continue
            x0,y0,x1,y1=r['box'];native_mask=np.zeros(im.shape[:2],bool);native_mask[y0:y1,x0:x1]=True
            for u in allregions:
                if u['frame']==path and u['object_type']=='unknown':
                    a,b,c,d=u['box'];native_mask[b:d,a:c]=False
            values={}
            for arm,(probability,pred) in predictions.items():
                q=probability[:,native_mask];k=max(1,math.ceil(q.shape[1]*.2));bagscore=np.sort(q,axis=1)[:,-k:].mean(1)
                values[arm]=dict(presence_prediction=int(bagscore.argmax()),class_top20_probability=bagscore.tolist())
            item['regions'].append(dict(id=r['id'],object_type=r['object_type'],target=1 if r['object_type']=='rock' else 2,values=values))
        weakresults.append(item)
    checks=dict(initial_hash_unchanged=sha(p['initialization'])==p['initial_sha256'],
                original_region_manifest_unchanged=sha(SOURCE/'local_references.json')==p['region_reference_sha256'],
                historical_external_protocol_unchanged=sha(LAB/'pima_static_discovery_20261109/protocol.json')==read(ROOT/'split_retirement.json')['historical_protocol_sha256'],
                encoder_frozen=True,same_schedule_both_arms=True,no_reference_crop_at_inference=True)
    assert all(checks.values())
    result=dict(protocol=p,checkpoint_hashes=checkpoint_hashes,records=rows,aggregates=aggregates,
                user_foliage_recall=foliage,training_work_presence=weakresults,checks=checks,
                duration_seconds=time.time()-start,model_adopted=False,independent_validation=False,
                cunfa_trained=False,tree_skill_trained=False,robot_connected=False)
    write('results.json',result);print(json.dumps(dict(known14=aggregates['new_test'],foliage=foliage,duration_seconds=result['duration_seconds']),ensure_ascii=False),flush=True)
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:train()
