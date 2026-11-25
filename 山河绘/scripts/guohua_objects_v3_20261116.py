# -*- coding: utf-8 -*-
"""Fixed budget experiment: ordinary continuation vs coarse-interface/core loss."""
import sys,json,hashlib,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from guohua_objects_train_20261112 import ContextObjects,pack,predict,confusion,metrics
from guohua_objects_v2_train_20261113 import normalize
from gongbi_training.cunfa_supervision import partial_role_loss
from gongbi_training.object_core_supervision import core_reference,core_balanced_loss
ROOT=LAB/'guohua_objects_v3_20261116';OLD=LAB/'guohua_objects_v2_20261113'
ARMS=['continuation_control','core_balanced'];COLORS=np.array([[145,157,170],[183,134,75],[65,149,91]],'uint8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def check_contract():
    t=np.full((32,32),255,'uint8');t[4:28,4:16]=1;t[4:28,16:28]=2
    revised,cores,band=core_reference(t);assert band.any();assert np.all(revised[t==255]==255);assert np.all(cores[t==255]==255);assert set(np.unique(cores))=={1,2,255}
    z=torch.randn(1,3,32,32,requires_grad=True);tr=torch.from_numpy(revised.astype('int64'))[None];cr=torch.from_numpy(cores.astype('int64'))[None];loss=core_balanced_loss(z,tr,cr,tr.ne(255).float());loss.backward();assert z.grad[0,:,t==255].count_nonzero()==0;assert z.grad[0,:,band].count_nonzero()==0
    x=np.full((32,32,3),200,'uint8');assert np.array_equal(pack(x,t)[0],pack(x,np.zeros_like(t))[0])
    empty=torch.full((1,32,32),255);assert core_balanced_loss(z,empty,empty,empty.ne(255)).item()==0
    return dict(void_preserved=True,unknown_and_interface_gradients_zero=True,input_independent_of_target=True,empty_reference_safe=True)
def prepare():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Protocol already frozen; do not overwrite.'
    ds=read(OLD/'dataset_manifest.json');write('dataset_manifest.json',ds)
    for r in ds['records']:assert sha(r['image'])==r['image_sha256'] and sha(r['target'])==r['target_sha256']
    protocol=dict(seed=161106,steps=384,batch=4,arms=ARMS,initial_sha256=sha(OLD/'expanded_control.pt'),old_result_sha256=sha(OLD/'results.json'),dataset_sha256=sha(ROOT/'dataset_manifest.json'),script_sha256=sha(__file__),loss_module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/object_core_supervision.py'),optimizer='AdamW lr0.0003 weight_decay0.0001',initial='Same previous expanded_control, frozen encoder, train decoder only',difference='Candidate ignores 2px coarse rock/tree interface and adds0.5 times equal rock/tree core CE; control unchanged weighted CE',sampling='Same uniform work-group/image sampling, same seed, original RGB with affine/flip/grayscale/color only; no synthetic line view',test_status='All previous14 test images are now known regression only; no fresh independent test this round',reference_status='Coarse source regions, eroded cores are not expert ink boundaries; no new manual truth',selection='Fixed final384 steps, no checkpoint/threshold selection',new_data=False,robot_connected=False,cunfa_trained=False,tree_skill_trained=False)
    write('protocol.json',protocol);write('contract_checks.json',check_contract());(ROOT/'views').mkdir(exist_ok=True)
    # Pretraining audit of training data only, one example per work, selected by interface size.
    candidates=[]
    for r in ds['records']:
        if r['split']!='train':continue
        im=np.array(Image.open(r['image']).convert('RGB'));target=np.array(Image.open(r['target']));x,t,_=pack(im,target);revised,core,band=core_reference(t)
        candidates.append((int(band.sum()),r,x,t,core,band))
    chosen=[];seen=set()
    for q in sorted(candidates,key=lambda q:q[0],reverse=True):
        if q[1]['work_group'] in seen:continue
        chosen.append(q);seen.add(q[1]['work_group'])
        if len(chosen)==6:break
    audit=Image.new('RGB',(1024,6*290),'#f4f1e8');d=ImageDraw.Draw(audit);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);auditrows=[]
    for i,(n,r,x,t,core,band) in enumerate(chosen):
        rgb=np.clip((x.transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype('uint8');ref=np.full((256,256,3),[229,223,210],dtype='uint8');ref[t!=255]=COLORS[t[t!=255]];c=np.full_like(ref,[229,223,210]);c[core!=255]=COLORS[core[core!=255]];b=rgb.copy();b[band]=[210,60,140]
        for j,img in enumerate([rgb,ref,c,b]):audit.paste(Image.fromarray(img),(j*256,i*290+30))
        d.text((4,i*290+5),r['id'][:90],font=font,fill='#294735');auditrows.append(dict(id=r['id'],split='train',interface_pixels=n,source_labels_not_expert=True))
    audit.save(ROOT/'training_reference_audit.jpg',quality=93);write('training_reference_audit.json',dict(columns=['RGB','source_coarse_objects','eroded_cores','interface_to_ignore'],records=auditrows));print(json.dumps(protocol,ensure_ascii=False),flush=True)
def train():
    assert not (ROOT/'results.json').exists(),'Do not retrain on viewed results.';p=read(ROOT/'protocol.json');assert sha(__file__)==p['script_sha256'];ds=read(ROOT/'dataset_manifest.json');rows=ds['records'];data={r['id']:(np.array(Image.open(r['image']).convert('RGB')),np.array(Image.open(r['target']))) for r in rows};training=[r for r in rows if r['split']=='train'];groups=[[r for r in training if r['work_group']==g] for g in sorted({r['work_group'] for r in training})]
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    initial=torch.load(OLD/'expanded_control.pt',map_location='cuda',weights_only=False)['model'];counts=sum(np.bincount(t[t!=255],minlength=3) for r in training for t in [data[r['id']][1]]);weights=torch.tensor(1/np.sqrt(counts),dtype=torch.float32,device='cuda');weights/=weights.mean();models={};curves={};started=time.time()
    for a in ARMS:
        random.seed(p['seed']);np.random.seed(p['seed']);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed']);net=ContextObjects().cuda();net.load_state_dict(initial);opt=torch.optim.AdamW([v for v in net.parameters() if v.requires_grad],lr=.0003,weight_decay=.0001);curve=[]
        for step in range(p['steps']):
            xs=[];ts=[];cs=[]
            for _ in range(p['batch']):
                r=random.choice(random.choice(groups));im,t=data[r['id']];x,t,_=pack(im,t);im=np.clip((x.transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype('uint8');m=cv2.getRotationMatrix2D((128,128),random.uniform(-7,7),random.uniform(.9,1.1));im=cv2.warpAffine(im,m,(256,256),borderValue=(210,210,210));t=cv2.warpAffine(t,m,(256,256),flags=cv2.INTER_NEAREST,borderValue=255)
                if random.random()<.5:im=im[:,::-1].copy();t=t[:,::-1].copy()
                if random.random()<.3:im=np.repeat(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)[...,None],3,2)
                im=np.clip(im.astype('float32')*random.uniform(.9,1.1)+random.uniform(-8,8),0,255).astype('uint8');revised,core,_=core_reference(t);xs.append(normalize(im));ts.append((revised if a=='core_balanced' else t).astype('int64'));cs.append(core.astype('int64'))
            z=net(torch.from_numpy(np.stack(xs)).cuda());target=torch.from_numpy(np.stack(ts)).cuda();w=weights[target.clamp(0,2)]*target.ne(255);loss=core_balanced_loss(z,target,torch.from_numpy(np.stack(cs)).cuda(),w) if a=='core_balanced' else partial_role_loss(z,target,w);opt.zero_grad();loss.backward();opt.step();curve.append(float(loss.detach()))
            if (step+1)%96==0:print(a,step+1,round(curve[-1],4),flush=True)
        net.eval();models[a]=net;curves[a]=curve;torch.save(dict(model=net.state_dict(),protocol=p),ROOT/(a+'.pt'))
    hashes={a:sha(ROOT/(a+'.pt')) for a in ARMS};write('checkpoint_freeze.json',hashes);(ROOT/'predictions').mkdir(exist_ok=True);results=[]
    for r in rows:
        if r['split']=='train':continue
        im,t=data[r['id']];scores={}
        # Frozen baseline predictions are reused byte-for-byte, never manually masked.
        z=np.load(OLD/'predictions'/(r['id']+'_expanded_control.npz'));np.savez_compressed(ROOT/'predictions'/(r['id']+'_previous.npz'),prediction=z['prediction'],probability=z['probability']);scores['previous']=metrics(confusion(t,z['prediction']))
        for a,net in models.items():
            prob,pred=predict(net,im);np.savez_compressed(ROOT/'predictions'/(r['id']+'_'+a+'.npz'),prediction=pred,probability=prob.astype('float16'));scores[a]=metrics(confusion(t,pred))
        results.append(dict(id=r['id'],split=r['split'],work_group=r['work_group'],scores=scores))
    aggregates={s:{a:metrics(sum(np.array(r['scores'][a]['confusion']) for r in results if r['split']==s)) for a in ['previous']+ARMS} for s in sorted({r['split'] for r in results})}
    im=np.array(Image.open(LAB/'cunfa_roles_v2_20261112/references/H_14_input.png').convert('RGB'));region=np.load(LAB/'cunfa_object_correction_20261112/H_14_object_reference.npz')['object_verified'];user={}
    old=np.load(OLD/'predictions/H_14_expanded_control.npz');np.savez_compressed(ROOT/'predictions/H_14_previous.npz',prediction=old['prediction'],probability=old['probability']);user['previous']=float((old['prediction'][region]==2).mean())
    for a,net in models.items():
        prob,pred=predict(net,im);np.savez_compressed(ROOT/'predictions'/('H_14_'+a+'.npz'),prediction=pred,probability=prob.astype('float16'));user[a]=float((pred[region]==2).mean())
    assert sha(OLD/'results.json')==p['old_result_sha256'];assert sha(OLD/'expanded_control.pt')==p['initial_sha256']
    for a in ARMS:assert all(torch.equal(v,models[a].state_dict()[k]) for k,v in initial.items() if k.startswith('encoder.'))
    result=dict(protocol=p,checkpoint_hashes=hashes,records=results,aggregates=aggregates,user_foliage_recall=user,loss_curves=curves,duration_seconds=time.time()-started,adopted=False,object_accepted=False,cunfa_accepted=False,tree_skill_enabled=False,robot_connected=False);write('results.json',result);print(json.dumps(dict(aggregates=aggregates,user=user),ensure_ascii=False),flush=True)
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:train()
