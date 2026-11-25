# -*- coding: utf-8 -*-
"""Matched expanded-data control versus image-only line/dilution augmentation."""
import sys,json,hashlib,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,torch
from PIL import Image
from gongbi_training.cunfa_supervision import partial_role_loss
from guohua_objects_train_20261112 import ContextObjects,pack,predict,metrics,confusion
ROOT=LAB/'guohua_objects_v2_20261113';OLD=LAB/'guohua_objects_pilot_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def line_view(image):
 """Training-only approximation from source RGB, never from object masks."""
 g=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);f=g.astype('float32');local=np.maximum(cv2.GaussianBlur(f,(0,0),5)-f,0);a=np.clip(255-3.2*local,0,255).astype('uint8');edges=cv2.Canny(cv2.GaussianBlur(g,(3,3),.7),40,110);a[edges>0]=np.minimum(a[edges>0],80);return np.repeat(a[...,None],3,2)
def normalize(image):
 a=image.astype('float32')/255.;return ((a-np.array([.485,.456,.406],'float32'))/np.array([.229,.224,.225],'float32')).transpose(2,0,1)
def main():
 assert not (ROOT/'results.json').exists(),'Never rerun against frozen new test.';ds=json.loads((ROOT/'dataset_manifest.json').read_text('utf8'));records=ds['records'];assert all(sha(r['image'])==r['image_sha256'] and sha(r['target'])==r['target_sha256'] for r in records);train=[r for r in records if r['split']=='train'];data={r['id']:(np.array(Image.open(r['image']).convert('RGB')),np.array(Image.open(r['target']))) for r in records};seed=131101;steps=600;batch=4
 protocol=dict(seed=seed,steps_per_arm=steps,batch_size=batch,arms=['expanded_control','line_aware_aug'],initial_model_sha256=sha(OLD/'context_object_head.pt'),old_result_sha256=sha(OLD/'results.json'),dataset_sha256=sha(ROOT/'dataset_manifest.json'),selection_sha256=sha(ROOT/'selection_before_download.json'),script_sha256=sha(__file__),architecture='Same frozen MobileNetV3 Small encoder and multiscale object decoder',optimizer='AdamW lr0.001 weight_decay0.0001',common_input='Aspect-preserving256 static full context, affine +/-7deg,0.9..1.1scale,flip,grayscale,color perturbation',augmentation_difference='Only line-aware arm:35% deterministic source-RGB line view,20% diluted local contrast; no target mask edges',sampling='Uniform filename work group then image; identical RNG schedule between arms',new_test_selection='14 images in4 filename artist/work candidate groups, selection and files frozen before training; no image preview or new test predictions before checkpoint freeze',previous_eight_collection_test_now_explicit_development=True,true_author_or_work_identity_authenticated=False,new_test_is_exploratory_not_expert_acceptance=True,model_selection='Fixed final step; no checkpoint/threshold selection from validation/test',source_void_ignored=True,cunfa_or_tree_painting_trained=False,robot_connected=False);write('protocol.json',protocol)
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);initial=torch.load(OLD/'context_object_head.pt',map_location='cuda',weights_only=False)['model'];counts=np.zeros(3)
 for r in train:
  t=data[r['id']][1];counts+=np.bincount(t[t!=255],minlength=3)
 weights=torch.tensor(1/np.sqrt(counts),dtype=torch.float32,device='cuda');weights/=weights.mean();groups=[[r for r in train if r['work_group']==g] for g in sorted({r['work_group'] for r in train})];models={};curves={};started=time.time()
 for arm in protocol['arms']:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);net=ContextObjects().cuda();net.load_state_dict(initial);opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.001,weight_decay=.0001);losses=[]
  for step in range(steps):
   xs=[];ts=[]
   for _ in range(batch):
    r=random.choice(random.choice(groups));image,target=data[r['id']];x,t,_=pack(image,target);image=np.clip((x.transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype('uint8');m=cv2.getRotationMatrix2D((128,128),random.uniform(-7,7),random.uniform(.9,1.1));image=cv2.warpAffine(image,m,(256,256),borderValue=(210,210,210));t=cv2.warpAffine(t,m,(256,256),flags=cv2.INTER_NEAREST,borderValue=255)
    flip=random.random();gray=random.random();mode=random.random();dilution=random.uniform(.55,.9);gain=random.uniform(.9,1.1);bias=random.uniform(-8,8)
    if flip<.5:image=image[:,::-1].copy();t=t[:,::-1].copy()
    if gray<.3:image=np.repeat(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)[...,None],3,2)
    if arm=='line_aware_aug' and mode<.35:image=line_view(image)
    elif arm=='line_aware_aug' and mode<.55:
     f=image.astype('float32');background=cv2.GaussianBlur(f,(0,0),12);image=np.clip(background+dilution*(f-background),0,255).astype('uint8')
    image=np.clip(image.astype('float32')*gain+bias,0,255).astype('uint8');xs.append(normalize(image));ts.append(t.astype('int64'))
   logits=net(torch.from_numpy(np.stack(xs)).cuda());target=torch.from_numpy(np.stack(ts)).cuda();loss=partial_role_loss(logits,target,weights[target.clamp(0,2)]*target.ne(255));opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
   if (step+1)%150==0:print(arm,step+1,round(losses[-1],4),flush=True)
  net.eval();models[arm]=net;curves[arm]=losses;torch.save(dict(model=net.state_dict(),protocol=protocol,arm=arm),ROOT/(arm+'.pt'))
 hashes={a:sha(ROOT/(a+'.pt')) for a in models};write('checkpoint_freeze.json',dict(checkpoints=hashes,new_test_predictions_generated=False));old=ContextObjects().cuda();old.load_state_dict(initial);old.eval();models['previous_frozen']=old;(ROOT/'predictions').mkdir(exist_ok=True);results=[]
 for r in records:
  image,t=data[r['id']];score={}
  for arm,net in models.items():
   prob,p=predict(net,image);np.savez_compressed(ROOT/'predictions'/(r['id']+'_'+arm+'.npz'),prediction=p,probability=prob.astype('float16'));score[arm]=metrics(confusion(t,p))
  results.append(dict(id=r['id'],split=r['split'],work_group=r['work_group'],scores=score))
 aggregates={}
 for split in sorted({r['split'] for r in results}):
  aggregates[split]={}
  for arm in models:
   relevant=[r for r in results if r['split']==split];value=metrics(sum(np.array(r['scores'][arm]['confusion']) for r in relevant));works=sorted({r['work_group'] for r in relevant});perwork=[metrics(sum(np.array(r['scores'][arm]['confusion']) for r in relevant if r['work_group']==work))['mean_iou'] for work in works];value.update(filename_work_groups=len(works),work_macro_iou=float(np.mean(perwork)));aggregates[split][arm]=value
 image=np.array(Image.open(LAB/'cunfa_roles_v2_20261112/references/H_14_input.png').convert('RGB'));region=np.load(LAB/'cunfa_object_correction_20261112/H_14_object_reference.npz')['object_verified'];hmetrics={}
 for arm,net in models.items():
  prob,p=predict(net,image);np.savez_compressed(ROOT/'predictions'/('H_14_'+arm+'.npz'),prediction=p,probability=prob.astype('float16'));hmetrics[arm]=dict(tree_core_reference_pixels=int(region.sum()),tree_region_recall=float((p[region]==2).mean()),not_new_blind=True,no_complete_faint_rock_reference=True)
 result=dict(protocol=protocol,checkpoint_hashes=hashes,records=results,aggregates=aggregates,user_case=hmetrics,loss_curves=curves,duration_seconds=time.time()-started,model_adopted=False,object_formal_acceptance=False,cunfa_formal_acceptance=False,tree_skill_training_enabled=False,robot_connected=False);write('results.json',result);print(json.dumps(dict(new_test=aggregates['new_test'],old_regression=aggregates['old_named_regression'],user_case=hmetrics),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
