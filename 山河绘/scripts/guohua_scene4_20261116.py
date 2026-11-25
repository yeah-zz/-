# -*- coding: utf-8 -*-
import sys,json,hashlib,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from gongbi_training.scene_objects import SceneObjects,scene_metrics
from gongbi_training.scene_object_reference import sparse_scene_reference,legacy_object_to_scene
from gongbi_training.cunfa_supervision import partial_role_loss
from gongbi_training.sparse_object_supervision import masked_teacher_kl
from guohua_objects_train_20261112 import ContextObjects,pack,predict
from guohua_objects_v2_train_20261113 import normalize
ROOT=LAB/'guohua_scene4_20261116';OLD=LAB/'guohua_objects_v2_20261113';NEW=LAB/'guohua_faint_sources_20261116';ARMS=['scene_base','scene_plus_line']
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def region(name,obj,points):return dict(name=name,object=obj,points=points,kind='core_polygon')
def prepare():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Run frozen'
 data=read(OLD/'dataset_manifest.json');photo=read(NEW/'references.json');prior=read(LAB/'cunfa_roles_v2_20261112/dataset_manifest.json');pby={r['id']:r for r in prior['records']};videos=[]
 for ident in ['L_149_3','L_170','M_1','M_2']:
  image=LAB/'cunfa_roles_v2_20261112/references'/(ident+'_input.png');size=list(Image.open(image).size)
  if ident.startswith('L_'):
   regs=[region('white_inner_mountain','rock',[[90,169],[105,153],[123,158],[141,183],[138,208],[119,219],[92,211]]),region('back_peak_inner','rock',[[172,58],[181,49],[193,64],[205,91],[214,120],[199,125],[185,104]]),region('front_hatched_face','rock',[[168,259],[189,233],[203,239],[198,270],[182,300],[164,314],[151,304]]),region('outside_left','open_scene',[[2,116],[16,113],[22,136],[15,174],[3,187]])]
  else:
   regs=[region('rock_interior_left','rock',[[62,90],[78,78],[96,85],[109,101],[99,118],[79,121],[62,108]]),region('rock_interior_right','rock',[[223,58],[237,48],[256,54],[255,70],[245,84],[224,81]])]
  for r in regs:assert all(0<=x<size[0] and 0<=y<size[1] for x,y in r['points'])
  videos.append(dict(id=ident,image=str(image),image_sha256=sha(image),size=size,work_group='tutorial:'+pby[ident]['source_family'],source_url=pby[ident]['source_url'],source_status='Reused real tutorial frames; previously used for separate role tasks, new object-region assistant references; not new footage',regions=regs,split='train_video'))
 photo_cases=[]
 for c in photo['cases']:
  q=dict(c);q['split']='photo_holdout' if c['id'].startswith('zhao-ji_') else 'train_photo';photo_cases.append(q)
 assert len([c for c in photo_cases if c['split']=='train_photo'])==4;assert len({c['work_group'] for c in videos})==2
 refs=dict(photo_cases=photo_cases,video_cases=videos,expert_reviewed=False,H_14_training=False,video_frames_grouped_by_work=True,ink_truth=False,roi_selected_before_new_predictions=True);write('references.json',refs);write('dataset_manifest.json',data)
 protocol=dict(seed=161126,steps=256,batch_roles=['old_work_uniform','old_work_uniform','photo_work_uniform','video_work_uniform'],arms=ARMS,initial_sha256=sha(OLD/'expanded_control.pt'),old_results_sha256=sha(OLD/'results.json'),dataset_sha256=sha(ROOT/'dataset_manifest.json'),reference_sha256=sha(ROOT/'references.json'),new_photo_reference_sha256=sha(NEW/'references.json'),script_sha256=sha(__file__),scene_module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/scene_objects.py'),class_ids={'0':'open_scene','1':'rock','2':'tree','3':'other_object','255':'unknown'},initialization='legacy0->new3; legacy1->new1; legacy2->new2; new open weights0 and bias-2; frozen ImageNet encoder',optimizer='AdamW lr0.0003 wd0.0001; fixed last256 steps',common_loss='Old source CE +0.2 conditional legacy teacher KL;0.7 photo sparse CE',difference='Only scene_plus_line adds0.7 video sparse CE. Same images/augmentations/seed/budget for both arms; base sees frames but no labels',new_reference_count=dict(photo_train=4,video_train=4,video_works=2,photo_holdout=2),heldout_work='filename:zhao-ji_autumn-colors',holdout_status='Already viewed new-development work, held out of all object training but not blind/expert acceptance; no tuning on results',H_14_status='Repeated known user case, no training',reserved_status='47 paths in2 other work candidate groups remain metadata-only, no download/preview/inference',new_independent_acceptance=False,expert_truth=False,cunfa_trained=False,tree_skill_trained=False,robot_connected=False);write('protocol.json',protocol)
 print(json.dumps(protocol,ensure_ascii=False),flush=True)
def train():
 assert not (ROOT/'results.json').exists(),'Never rerun against results';p=read(ROOT/'protocol.json');assert sha(__file__)==p['script_sha256'];refs=read(ROOT/'references.json');assert sha(ROOT/'references.json')==p['reference_sha256'];ds=read(ROOT/'dataset_manifest.json');training=[r for r in ds['records'] if r['split']=='train'];photos=[c for c in refs['photo_cases'] if c['split']=='train_photo'];videos=refs['video_cases'];assert 'H_14' not in [c['id'] for c in photos+videos];assert not ({c['work_group'] for c in photos}&{c['work_group'] for c in refs['photo_cases'] if c['split']=='photo_holdout'})
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;oldstate=torch.load(OLD/'expanded_control.pt',map_location='cuda',weights_only=False)['model'];teacher=ContextObjects().cuda();teacher.load_state_dict(oldstate);teacher.eval();teacher.requires_grad_(False)
 data={r['id']:(np.array(Image.open(r['image']).convert('RGB')),legacy_object_to_scene(np.array(Image.open(r['target'])))) for r in ds['records']}
 for c in refs['photo_cases']+videos:
  assert sha(c['image'])==c['image_sha256'];scene,ink=sparse_scene_reference(c['size'],c['regions']);assert (ink==255).all();data[c['id']]=(np.array(Image.open(c['image']).convert('RGB')),scene)
 oldgroups=[[r for r in training if r['work_group']==g] for g in sorted({r['work_group'] for r in training})];photogroups=[[c for c in photos if c['work_group']==g] for g in sorted({c['work_group'] for c in photos})];videogroups=[[c for c in videos if c['work_group']==g] for g in sorted({c['work_group'] for c in videos})]
 models={};curves={};start=time.time()
 for a in ARMS:
  random.seed(p['seed']);np.random.seed(p['seed']);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed']);net=SceneObjects(oldstate).cuda();opt=torch.optim.AdamW([v for v in net.parameters() if v.requires_grad],lr=.0003,weight_decay=.0001);curve=[]
  # The transferred old conditional logits exactly preserve their original order.
  assert torch.equal(net.head[-1].weight[3],teacher.head[-1].weight[0])
  for step in range(p['steps']):
   chosen=[random.choice(random.choice(oldgroups)),random.choice(random.choice(oldgroups)),random.choice(random.choice(photogroups)),random.choice(random.choice(videogroups))];xs=[];ts=[]
   for r in chosen:
    im,t=data[r['id']];x,t,_=pack(im,t);im=np.clip((x.transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype('uint8');m=cv2.getRotationMatrix2D((128,128),random.uniform(-7,7),random.uniform(.9,1.1));im=cv2.warpAffine(im,m,(256,256),borderValue=(210,210,210));t=cv2.warpAffine(t,m,(256,256),flags=cv2.INTER_NEAREST,borderValue=255)
    if random.random()<.5:im=im[:,::-1].copy();t=t[:,::-1].copy()
    if random.random()<.3:im=np.repeat(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)[...,None],3,2)
    im=np.clip(im.astype('float32')*random.uniform(.9,1.1)+random.uniform(-8,8),0,255).astype('uint8');xs.append(normalize(im));ts.append(t.astype('int64'))
   x=torch.from_numpy(np.stack(xs)).cuda();t=torch.from_numpy(np.stack(ts)).cuda();z=net(x)
   with torch.no_grad():tz=teacher(x[:2])
   loss=partial_role_loss(z[:2],t[:2])+.2*masked_teacher_kl(z[:2][:,[3,1,2]],tz,t[:2].ne(255))+.7*partial_role_loss(z[2:3],t[2:3])
   if a=='scene_plus_line':loss=loss+.7*partial_role_loss(z[3:4],t[3:4])
   opt.zero_grad();loss.backward();opt.step();curve.append(float(loss.detach()))
   if (step+1)%64==0:print(a,step+1,round(curve[-1],4),flush=True)
  net.eval();models[a]=net;curves[a]=curve;torch.save(dict(model=net.state_dict(),protocol=p),ROOT/(a+'.pt'))
 hashes={a:sha(ROOT/(a+'.pt')) for a in ARMS};write('checkpoint_freeze.json',hashes);(ROOT/'predictions').mkdir(exist_ok=True);out=[]
 cases=[dict(r) for r in ds['records'] if r['split']!='train']+refs['photo_cases']+videos
 him=np.array(Image.open(LAB/'cunfa_roles_v2_20261112/references/H_14_input.png').convert('RGB'));data['H_14']=(him,None);cases.insert(0,dict(id='H_14',image=str(LAB/'cunfa_roles_v2_20261112/references/H_14_input.png'),split='known_user_regression',work_group='known_H_tutorial'))
 for r in cases:
  im,t=data[r['id']];scores={}
  for a,net in dict(legacy=teacher,**models).items():
   prob,pred=predict(net,im)
   if a=='legacy':pred=legacy_object_to_scene(pred)
   np.savez_compressed(ROOT/'predictions'/(r['id']+'_'+a+'.npz'),prediction=pred)
   if t is not None:scores[a]=scene_metrics(t,pred)
  out.append(dict(id=r['id'],image=r['image'],split=r['split'],work_group=r['work_group'],scores=scores))
 aggregates={}
 for split in sorted({r['split'] for r in out if r['scores']}):
  aggregates[split]={}
  for a in ['legacy']+ARMS:
   c=sum(np.array(r['scores'][a]['confusion']) for r in out if r['split']==split);support=c.sum(1);union=c.sum(0)+support-c.diagonal();iou=[float(c[k,k]/union[k]) if union[k] else None for k in range(4)];aggregates[split][a]=dict(confusion=c.tolist(),support=support.tolist(),iou=iou,known_foreground_mean_iou=float(np.mean([iou[k] for k in [1,2,3] if iou[k] is not None])),recall=[float(c[k,k]/support[k]) if support[k] else None for k in range(4)])
 region=np.load(LAB/'cunfa_object_correction_20261112/H_14_object_reference.npz')['object_verified'];h={a:float((np.load(ROOT/'predictions'/('H_14_'+a+'.npz'))['prediction'][region]==2).mean()) for a in ['legacy']+ARMS}
 assert sha(OLD/'results.json')==p['old_results_sha256'];assert sha(OLD/'expanded_control.pt')==p['initial_sha256'];assert all(torch.equal(v,models[a].state_dict()[k]) for a in ARMS for k,v in oldstate.items() if k.startswith('encoder.'))
 write('results.json',dict(protocol=p,checkpoint_hashes=hashes,records=out,aggregates=aggregates,user_foliage_recall=h,loss_curves=curves,duration_seconds=time.time()-start,model_adopted=False,expert_accuracy=False,independent_validation=False,tree_skill_enabled=False,robot_connected=False));print(json.dumps(dict(aggregates=aggregates,user=h),ensure_ascii=False),flush=True)
if __name__=='__main__':
 if '--prepare' in sys.argv:prepare()
 else:train()
