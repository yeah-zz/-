# -*- coding: utf-8 -*-
import os,sys,json,hashlib,random,time
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import torch,numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.scene_objects import SceneObjects
from gongbi_training.scene_object_reference import sparse_scene_reference
from gongbi_training.cunfa_supervision import partial_role_loss
from gongbi_training.global_detail_gate import GlobalDetailGate
from guohua_objects_train_20261112 import pack
ROOT=LAB/'global_detail_gate_20261118';BASE=LAB/'guohua_scene4_mapping_fix_20261116';DIAG=LAB/'object_scale_pairs_20261118'
ARMS=['global256','detail512','fixed_mean','learned_gate'];COLORS=np.array([[241,237,222],[196,140,62],[70,143,105],[110,112,163]],'uint8')
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,x):(ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def prepare():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Already frozen'
    ds=read(BASE/'dataset_manifest.json')['records'];by={r['id']:r for r in ds}
    evaluation=read(DIAG/'results.json')['records'];excluded={by[r['id']]['work_group'] for r in evaluation if r['id'] in by}
    train=[dict(r,reference_kind='original_coarse_scene') for r in ds if r['split']=='train' and r['work_group'] not in excluded]
    refs=read(BASE/'references.json')
    for r in refs['photo_cases']+refs['video_cases']:
        if r['split']=='photo_holdout' or r['work_group'] in excluded:continue
        train.append(dict(r,reference_kind='prior_partial_scene'))
    assert not ({r['work_group'] for r in train}&excluded)
    write('train_manifest.json',dict(records=train,excluded_known_work_groups=sorted(excluded),
        evaluation_core_polygons_never_training_targets=True,new_sources=0,source_labels_not_expert_ink_truth=True))
    write('protocol.json',dict(seed=181128,steps=128,lr=.0003,batch=2,scales=[256,512],arms=ARMS,
        train_manifest_sha256=sha(ROOT/'train_manifest.json'),script_sha256=sha(__file__),
        module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/global_detail_gate.py'),
        frozen_checkpoint=str(BASE/'mapping_fixed.pt'),frozen_checkpoint_sha256=sha(BASE/'mapping_fixed.pt'),
        input='Full image only: normalized RGB plus frozen global/detail probability maps on aligned512 grid',
        trainable='Small11->16->16->1 gate only, initialized alpha0.5; one coefficient shared by4classes',
        loss='partial scene CE +0.01 spatial total variation of gate; unknown pixels have no direct CE',
        selection='Fixed final128 steps; no thresholds or checkpoint search; report all4arms',
        evaluation='17 previously inspected cases; no independent acceptance; whole known work groups excluded from gate training',
        reference_boxes_used_at_inference=False,cunfa_trained=False,tree_skill_trained=False,robot_connected=False))
    print('PREPARED',len(train),'training frames',flush=True)
def make_input(net,image):
    rgb,_,geom=pack(image,size=512);left,top,nw,nh,w,h=geom;grids=[];native=[]
    for size in [256,512]:
        x,_,(lx,ty,ww,hh,_,_)=pack(image,size=size)
        with torch.inference_mode():p=net(torch.from_numpy(x)[None].cuda()).softmax(1)[0].cpu().numpy()[:,ty:ty+hh,lx:lx+ww]
        native.append(np.stack([cv2.resize(a,(w,h),interpolation=cv2.INTER_LINEAR) for a in p]))
        grid=np.full((4,512,512),.25,'float32');grid[:,top:top+nh,left:left+nw]=np.stack([cv2.resize(a,(nw,nh),interpolation=cv2.INTER_LINEAR) for a in p]);grids.append(grid)
    return np.concatenate([rgb]+grids),geom,native
def restore(prob,geom):
    left,top,nw,nh,w,h=geom
    return np.stack([cv2.resize(a[top:top+nh,left:left+nw],(w,h),interpolation=cv2.INTER_LINEAR) for a in prob])
def run():
    p=read(ROOT/'protocol.json');assert not (ROOT/'results.json').exists(),'Already evaluated'
    assert sha(__file__)==p['script_sha256'] and sha(p['frozen_checkpoint'])==p['frozen_checkpoint_sha256']
    assert sha(ROOT/'train_manifest.json')==p['train_manifest_sha256']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    random.seed(p['seed']);np.random.seed(p['seed']);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    net=SceneObjects().cuda();net.load_state_dict(torch.load(p['frozen_checkpoint'],map_location='cuda',weights_only=False)['model']);net.eval();net.requires_grad_(False)
    gate=GlobalDetailGate().cuda();opt=torch.optim.AdamW(gate.parameters(),lr=p['lr'],weight_decay=.0001)
    # Exact neutral initialization and unknown-gradient contract.
    z=torch.randn(1,4,4,4,requires_grad=True);t=torch.full((1,4,4),255,dtype=torch.long);t[:,1,1]=1;partial_role_loss(z,t).backward();assert z.grad[0,:,t[0]==255].count_nonzero()==0
    probe=torch.cat([torch.zeros(1,3,4,4),torch.randn(1,4,4,4).softmax(1),torch.randn(1,4,4,4).softmax(1)],1).cuda()
    with torch.no_grad():out,alpha=gate(probe);assert torch.all(alpha==.5);assert torch.allclose(out.exp(),(probe[:,3:7]+probe[:,7:11])*.5)
    rows=read(ROOT/'train_manifest.json')['records'];cache={};(ROOT/'cache').mkdir(exist_ok=True)
    for i,r in enumerate(rows):
        image=np.array(Image.open(r['image']).convert('RGB'));features,geom,_=make_input(net,image)
        if r['reference_kind']=='prior_partial_scene':target,ink=sparse_scene_reference(r['size'],r['regions']);assert (ink==255).all()
        else:target=np.array(Image.open(r['target']))
        _,packed,_=pack(image,target,size=512)
        path=ROOT/f'cache/{r["id"]}.npz';np.savez_compressed(path,features=features.astype('float16'),target=packed)
        cache[r['id']]=path
        if (i+1)%16==0:print('CACHE',i+1,'/',len(rows),flush=True)
    groups=[[r['id'] for r in rows if r['work_group']==g] for g in sorted({r['work_group'] for r in rows})]
    losses=[];schedule=[];start=time.time()
    for step in range(p['steps']):
        ids=[random.choice(random.choice(groups)) for _ in range(2)];schedule.append(ids);items=[np.load(cache[i]) for i in ids]
        x=torch.from_numpy(np.stack([a['features'].astype('float32') for a in items])).cuda();target=torch.from_numpy(np.stack([a['target'].astype('int64') for a in items])).cuda()
        logits,alpha=gate(x);tv=(alpha[:,:,1:]-alpha[:,:,:-1]).abs().mean()+(alpha[:,:,:,1:]-alpha[:,:,:,:-1]).abs().mean()
        loss=partial_role_loss(logits,target)+.01*tv;opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        if (step+1)%32==0:print('TRAIN',step+1,round(losses[-1],4),flush=True)
    torch.save(dict(model=gate.state_dict(),protocol=p),ROOT/'gate.pt');write('checkpoint_freeze.json',dict(gate_sha256=sha(ROOT/'gate.pt')));write('schedule.json',schedule)
    gate.eval();(ROOT/'predictions').mkdir(exist_ok=True);cases=read(DIAG/'results.json')['records'];ds={r['id']:r for r in read(BASE/'dataset_manifest.json')['records']};results=[]
    known14={r['id'] for r in read(BASE/'results.json')['records'] if r['split']=='new_test'}
    for r in cases:
        ident=r['id'];image=np.array(Image.open(r['image']).convert('RGB'));features,geom,native=make_input(net,image)
        with torch.inference_mode():logits,alpha=gate(torch.from_numpy(features)[None].cuda());learned=restore(logits.exp()[0].cpu().numpy(),geom)
        fixed=restore((features[3:7]+features[7:11])*.5,geom);probs=[native[0],native[1],fixed,learned];scores={}
        for arm,prob in zip(ARMS,probs):
            pred=prob.argmax(0).astype('uint8');np.savez_compressed(ROOT/f'predictions/{ident}_{arm}.npz',prediction=pred)
            if ident in known14:
                target=np.array(Image.open(ds[ident]['target']));valid=target!=255
                scores[arm]=np.bincount((target[valid].astype('int64')*4+pred[valid]).ravel(),minlength=16).reshape(4,4).tolist()
        # Both single-resolution controls must reproduce the previous run exactly.
        for arm,size in [('global256',256),('detail512',512)]:assert np.array_equal(np.load(ROOT/f'predictions/{ident}_{arm}.npz')['prediction'],np.load(DIAG/f'predictions/{ident}_{size}.npz')['prediction'])
        results.append(dict(id=ident,image=r['image'],confusion=scores,scope='known_case_not_independent'));print('EVAL',ident,flush=True)
    aggregates={}
    for arm in ARMS:
        c=sum(np.array(r['confusion'][arm]) for r in results if arm in r['confusion']);union=c.sum(0)+c.sum(1)-c.diagonal();iou=[float(c[k,k]/union[k]) if union[k] else None for k in range(4)]
        aggregates[arm]=dict(confusion=c.tolist(),iou=iou,mean_iou=float(np.mean([v for v in iou if v is not None])))
    corecases=read(DIAG/'pairs.json')['cases'];hcase=next(c for c in read(LAB/'guohua_boundary_reference_20261116/references.json')['cases'] if c['id']=='H_14')
    corecases=[hcase]+[c for c in corecases if c['id']!='H_14'];regional=[]
    for c in corecases:
        for reg in c['regions']:
            if reg['kind']!='core_polygon':continue
            mask=Image.new('L',tuple(c['size']),0);ImageDraw.Draw(mask).polygon([tuple(q) for q in reg['points']],fill=1);mask=np.array(mask)>0;cls=1 if reg['object']=='rock' else 2
            regional.append(dict(id=c['id'],region=reg['name'],object=reg['object'],correct_fraction={a:float((np.load(ROOT/f'predictions/{c["id"]}_{a}.npz')['prediction'][mask]==cls).mean()) for a in ARMS}))
    assert sha(p['frozen_checkpoint'])==p['frozen_checkpoint_sha256']
    write('results.json',dict(protocol=p,records=results,known14=aggregates,regional=regional,loss_curve=losses,duration_training_and_evaluation_seconds=time.time()-start,
        adopted=False,new_gate_trained=True,original_model_changed=False,independent_validation=False,robot_connected=False))
    print(json.dumps(dict(known14=aggregates,regional=regional),ensure_ascii=False),flush=True)
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:run()
