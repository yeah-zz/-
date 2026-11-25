# -*- coding: utf-8 -*-
import sys,json,hashlib,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,torch
from PIL import Image
from guohua_objects_train_20261112 import ContextObjects,pack,predict,metrics,confusion
from guohua_objects_v2_train_20261113 import normalize
from gongbi_training.cunfa_supervision import partial_role_loss
from gongbi_training.sparse_object_supervision import reference_masks,masked_teacher_kl,sparse_class_loss
ROOT=LAB/'guohua_objects_v4_sparse_20261116';OLD=LAB/'guohua_objects_v2_20261113';REF=LAB/'guohua_boundary_reference_20261116';ARMS=['matched_control','sparse_weak'];ALL=['previous']+ARMS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def checks():
    t=torch.full((1,8,8),255,dtype=torch.long);t[:,2:4,2:4]=2;z=torch.randn(1,3,8,8,requires_grad=True);teacher=torch.randn_like(z);valid=t.ne(255);loss=sparse_class_loss(z,t)+masked_teacher_kl(z,teacher,valid);loss.backward();assert z.grad[0,:,~valid[0]].count_nonzero()==0;assert teacher.grad is None
    empty=torch.full_like(t,255);assert sparse_class_loss(z,empty).item()==0;assert masked_teacher_kl(z,teacher,empty.ne(255)).item()==0
    return dict(unknown_has_no_sparse_or_teacher_gradient=True,teacher_detached=True,empty_sparse_safe=True)
def main():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Never overwrite frozen run';write('contract_checks.json',checks());ds=read(OLD/'dataset_manifest.json');refs=read(REF/'reviewed_references.json');train=[r for r in ds['records'] if r['split']=='train'];byid={r['id']:r for r in train};weak={c['id']:c for c in refs['cases'] if c['prior_use']=='old_model_training_work'};assert len(weak)==3 and 'H_14' not in weak
    for ident,c in weak.items():assert ident in byid and c['image_sha256']==sha(byid[ident]['image'])
    count=sum(r['kind']=='core_polygon' and r['object']!='unknown' for c in weak.values() for r in c['regions']);assert count==13
    p=dict(seed=161116,steps=192,batch=4,arms=ARMS,initial_sha256=sha(OLD/'expanded_control.pt'),old_result_sha256=sha(OLD/'results.json'),reference_sha256=sha(REF/'reviewed_references.json'),dataset_sha256=sha(OLD/'dataset_manifest.json'),script_sha256=sha(__file__),loss_module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/sparse_object_supervision.py'),reference_cases=sorted(weak),weak_core_regions=count,unknown_and_corridors_not_weak_labels=True,H_14_training=False,encoder='Frozen same ImageNet MobileNetV3 Small',budget='Same192 steps, batch4, AdamW lr0.0001 wd0.0001',common_loss='Source weighted CE plus0.2 teacher KL; assistant unknown and sparse core excluded from both common losses',difference='Only sparse_weak adds0.25 equal-class sparse core CE on13 assistant-reviewed core regions from3 old training works',sampling='One weak-reference training image plus3 work-uniform training images per batch, identical RNG for both arms',selection='Fixed last192 step; no checkpoint,threshold or loss-weight adjustment after results',test='Known14 and old12 regression only; H_14 known user case only; three weak-reference works are in-sample diagnostics',expert_truth=False,new_independent_test=False,cunfa_trained=False,tree_skill_trained=False,robot_connected=False);write('protocol.json',p);write('dataset_manifest.json',ds)
    data={r['id']:(np.array(Image.open(r['image']).convert('RGB')),np.array(Image.open(r['target']))) for r in ds['records']};sparse={i:reference_masks(c) for i,c in weak.items()};write('weak_reference_manifest.json',dict(cases=list(weak.values()),regions=count,used_as='assistant object weak supervision; no ink/stroke truth',excluded_case='H_14'))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;initial=torch.load(OLD/'expanded_control.pt',map_location='cuda',weights_only=False)['model'];teacher=ContextObjects().cuda();teacher.load_state_dict(initial);teacher.eval();teacher.requires_grad_(False);groups=[[r for r in train if r['work_group']==g] for g in sorted({r['work_group'] for r in train})];weakrows=[byid[i] for i in sorted(weak)];counts=sum(np.bincount(data[r['id']][1][data[r['id']][1]!=255],minlength=3) for r in train);weights=torch.tensor(1/np.sqrt(counts),dtype=torch.float32,device='cuda');weights/=weights.mean();models={};curves={};start=time.time()
    for a in ARMS:
        random.seed(p['seed']);np.random.seed(p['seed']);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed']);net=ContextObjects().cuda();net.load_state_dict(initial);opt=torch.optim.AdamW([x for x in net.parameters() if x.requires_grad],lr=.0001,weight_decay=.0001);curve=[]
        for step in range(p['steps']):
            xs=[];ts=[];ws=[]
            chosen=[random.choice(weakrows)]+[random.choice(random.choice(groups)) for _ in range(3)]
            for r in chosen:
                im,t=data[r['id']];sp,u=sparse.get(r['id'],(np.full(t.shape,255,'uint8'),np.zeros(t.shape,bool)));base=t.copy();base[u|(sp!=255)]=255;x,base,_=pack(im,base);_,sp,_=pack(im,sp);im=np.clip((x.transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype('uint8');m=cv2.getRotationMatrix2D((128,128),random.uniform(-7,7),random.uniform(.9,1.1));im=cv2.warpAffine(im,m,(256,256),borderValue=(210,210,210));base=cv2.warpAffine(base,m,(256,256),flags=cv2.INTER_NEAREST,borderValue=255);sp=cv2.warpAffine(sp,m,(256,256),flags=cv2.INTER_NEAREST,borderValue=255)
                if random.random()<.5:im=im[:,::-1].copy();base=base[:,::-1].copy();sp=sp[:,::-1].copy()
                if random.random()<.3:im=np.repeat(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)[...,None],3,2)
                im=np.clip(im.astype('float32')*random.uniform(.9,1.1)+random.uniform(-8,8),0,255).astype('uint8');xs.append(normalize(im));ts.append(base.astype('int64'));ws.append(sp.astype('int64'))
            x=torch.from_numpy(np.stack(xs)).cuda();t=torch.from_numpy(np.stack(ts)).cuda();w=torch.from_numpy(np.stack(ws)).cuda();z=net(x)
            with torch.no_grad():tz=teacher(x)
            loss=partial_role_loss(z,t,weights[t.clamp(0,2)]*t.ne(255))+.2*masked_teacher_kl(z,tz,t.ne(255))
            if a=='sparse_weak':loss=loss+.25*sparse_class_loss(z,w)
            opt.zero_grad();loss.backward();opt.step();curve.append(float(loss.detach()))
            if (step+1)%48==0:print(a,step+1,round(curve[-1],4),flush=True)
        net.eval();models[a]=net;curves[a]=curve;torch.save(dict(model=net.state_dict(),protocol=p),ROOT/(a+'.pt'))
    hashes={a:sha(ROOT/(a+'.pt')) for a in ARMS};write('checkpoint_freeze.json',hashes);(ROOT/'predictions').mkdir(exist_ok=True);out=[]
    for r in ds['records']:
        if r['split']=='train' and r['id'] not in weak:continue
        im,t=data[r['id']];scores={};old=np.load(OLD/'predictions'/(r['id']+'_expanded_control.npz'));np.savez_compressed(ROOT/'predictions'/(r['id']+'_previous.npz'),prediction=old['prediction'],probability=old['probability']);scores['previous']=metrics(confusion(t,old['prediction']))
        for a,net in models.items():
            prob,pred=predict(net,im);np.savez_compressed(ROOT/'predictions'/(r['id']+'_'+a+'.npz'),prediction=pred,probability=prob.astype('float16'));scores[a]=metrics(confusion(t,pred))
        out.append(dict(id=r['id'],split=r['split'],work_group=r['work_group'],scores=scores))
    aggregates={s:{a:metrics(sum(np.array(r['scores'][a]['confusion']) for r in out if r['split']==s)) for a in ALL} for s in sorted({r['split'] for r in out if r['split']!='train'})}
    im=np.array(Image.open(LAB/'cunfa_roles_v2_20261112/references/H_14_input.png').convert('RGB'));region=np.load(LAB/'cunfa_object_correction_20261112/H_14_object_reference.npz')['object_verified'];hm={};old=np.load(OLD/'predictions/H_14_expanded_control.npz');np.savez_compressed(ROOT/'predictions/H_14_previous.npz',prediction=old['prediction'],probability=old['probability']);hm['previous']=float((old['prediction'][region]==2).mean())
    for a,net in models.items():
        prob,pred=predict(net,im);np.savez_compressed(ROOT/'predictions'/('H_14_'+a+'.npz'),prediction=pred,probability=prob.astype('float16'));hm[a]=float((pred[region]==2).mean())
    assert sha(OLD/'results.json')==p['old_result_sha256'];assert sha(OLD/'expanded_control.pt')==p['initial_sha256'];assert all(torch.equal(v,models[a].state_dict()[k]) for a in ARMS for k,v in initial.items() if k.startswith('encoder.'))
    write('results.json',dict(protocol=p,checkpoint_hashes=hashes,records=out,aggregates=aggregates,user_foliage_recall=hm,loss_curves=curves,duration_seconds=time.time()-start,model_adopted=False,new_independent_validation=False,expert_accuracy=False,tree_skill_enabled=False,robot_connected=False));print(json.dumps(dict(aggregates=aggregates,user=hm),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
