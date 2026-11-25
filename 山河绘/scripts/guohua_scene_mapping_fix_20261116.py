# -*- coding: utf-8 -*-
import sys,json,hashlib,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import torch,cv2,numpy as np
from PIL import Image,ImageDraw
from gongbi_training.source_scene_mapping import source_to_scene,OPEN_SCENE_IDS
from gongbi_training.scene_objects import SceneObjects,scene_metrics
from gongbi_training.scene_object_reference import sparse_scene_reference
from gongbi_training.sparse_object_supervision import masked_teacher_kl
from gongbi_training.cunfa_supervision import partial_role_loss
from guohua_objects_train_20261112 import ContextObjects,pack,predict
from guohua_objects_v2_train_20261113 import normalize
ROOT=LAB/'guohua_scene4_mapping_fix_20261116';PREV=LAB/'guohua_scene4_20261116';OLD=LAB/'guohua_objects_v2_20261113';NEW=LAB/'guohua_faint_sources_20261116'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def prepare():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Frozen experiment';(ROOT/'targets').mkdir(exist_ok=True)
 ds=read(OLD/'dataset_manifest.json');refs=read(PREV/'references.json');oldp=read(PREV/'protocol.json');count={};rows=[]
 for r in ds['records']:
  assert sha(r['source_mask'])==r['source_mask_sha256'];raw=np.array(Image.open(r['source_mask']));target=source_to_scene(raw);path=ROOT/'targets'/(r['id']+'_scene.png');Image.fromarray(target).save(path);q=dict(r);q['legacy_target']=r['target'];q['target']=str(path);q['target_sha256']=sha(path);rows.append(q)
  if r['split']=='train':
   for k in OPEN_SCENE_IDS:count[str(k)]=count.get(str(k),0)+int((raw==k).sum())
 write('dataset_manifest.json',dict(records=rows,source_manifest_sha256=sha(OLD/'dataset_manifest.json'),mapping='original1/4/10->open0;2/13->rock1;3->tree2;5/6/7/8/9/11/12/14->other3;0/255->unknown255',coarse_source_not_expert_ink_truth=True));write('references.json',refs)
 # Cross-check explicit open-scene reference cores against ORIGINAL source masks.
 newds={r['id']:r for r in read(NEW/'dataset_manifest.json')['records']};evidence=[]
 for c in read(NEW/'references.json')['cases']:
  for r in c['regions']:
   if r['object']!='open_scene':continue
   mask=Image.new('L',tuple(c['size']),0);ImageDraw.Draw(mask).polygon([tuple(p) for p in r['points']],fill=255);raw=np.array(Image.open(newds[c['id']]['source_mask']));vals,ns=np.unique(raw[np.array(mask)>0],return_counts=True);assert set(vals)<=set(OPEN_SCENE_IDS);evidence.append(dict(id=c['id'],region=r['name'],source_code_pixels=dict(zip(map(str,vals),map(int,ns))),old_scene_target=3,correct_scene_target=0,assistant_region_not_expert=True))
 # Pair existing within-image scene references without generating new stroke truth.
 pairs=[]
 for c in refs['photo_cases']+refs['video_cases']:
  opens=[r['name'] for r in c['regions'] if r['object']=='open_scene'];rocks=[r['name'] for r in c['regions'] if r['object']=='rock'];trees=[r['name'] for r in c['regions'] if r['object']=='tree']
  if opens and (rocks or trees):pairs.append(dict(id=c['id'],work_group=c['work_group'],split=c['split'],open_regions=opens,object_interior_regions=rocks+trees,ink_not_derived_from_scene=True))
 write('supervision_conflict_audit.json',dict(previous_run=str(PREV),conflicting_train_pixels_by_source_id=count,total_conflicting_train_pixels=sum(count.values()),evidence=evidence,original_coco_taxonomy=str(LAB/'guohua_hierarchy_20261112/sources/instances_validation.json'),public_taxonomy_url='https://github.com/shengbei4980/ShanShuiSeg#分割类别--classes',status='Project merge correction supported by public category table and sampled originals/masks; not official full CLPD expert certification',river_code_review_image='source_code_audit.jpg',historical_other_included_open_scene=True,old_artifacts_unchanged=True))
 write('paired_reference_manifest.json',dict(pairs=pairs,new_independent_works=0,reuses_existing_assistant_regions=True,unknown_preserved=True))
 p=dict(oldp);p.update(arms=['mapping_fixed'],script_sha256=sha(__file__),mapping_module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/source_scene_mapping.py'),dataset_sha256=sha(ROOT/'dataset_manifest.json'),reference_sha256=sha(ROOT/'references.json'),conflicted_control_sha256=sha(PREV/'scene_plus_line.pt'),conflicted_control_results_sha256=sha(PREV/'results.json'),control='Reuse frozen previous scene_plus_line, same seed161126,256steps,initial,images,transforms,optimizer,loss,video references; only original source coarse target mapping changes',mapping_fix='source1/4/10 now open0 instead of inherited legacy-other3; raw0 and255 stay unknown',selection='Fixed last256 steps; no checkpoint/threshold/augmentation retuning',evaluation='Both control and correction evaluated against SAME corrected raw-source targets; historical three-foreground means have a different taxonomy and cannot be compared directly',photometric_checks='Post-freeze original/warm/dim/bright perturbations on the same known photo_holdout work, no new independent samples',robot_connected=False);write('protocol.json',p)
 raw=np.arange(15,dtype='uint8')[None];expected=np.array([[255,0,1,2,0,3,3,3,3,3,0,3,3,1,3]],'uint8');assert np.array_equal(source_to_scene(raw),expected);assert source_to_scene(np.array([[255]],'uint8'))[0,0]==255
 z=torch.randn(1,4,3,3,requires_grad=True);t=torch.full((1,3,3),255,dtype=torch.long);t[:,1,1]=0;partial_role_loss(z,t).backward();assert z.grad[0,:,t[0]==255].count_nonzero()==0
 write('pretrain_checks.json',dict(original_source_remap_all_ids_checked=True,source_void_never_becomes_background=True,open_scene_no_longer_supervised_as_other=True,all_five_open_reference_cores_match_open_source_codes=True,unknown_loss_gradients_zero=True,old_files_preserved=True,control_frozen=True,no_new_expert_truth=True,robot_connected=False));print(json.dumps(dict(conflicting_pixels=sum(count.values()),paired_cases=len(pairs)),ensure_ascii=False),flush=True)
def train():
 assert not (ROOT/'results.json').exists(),'Never rerun after results';p=read(ROOT/'protocol.json');assert sha(__file__)==p['script_sha256'];assert sha(PREV/'scene_plus_line.pt')==p['conflicted_control_sha256'];ds=read(ROOT/'dataset_manifest.json');refs=read(ROOT/'references.json');oldrows=[r for r in ds['records'] if r['split']=='train'];photos=[r for r in refs['photo_cases'] if r['split']=='train_photo'];videos=refs['video_cases'];assert not ({c['work_group'] for c in photos}&{c['work_group'] for c in refs['photo_cases'] if c['split']=='photo_holdout'})
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;state=torch.load(OLD/'expanded_control.pt',map_location='cuda',weights_only=False)['model'];teacher=ContextObjects().cuda();teacher.load_state_dict(state);teacher.eval();teacher.requires_grad_(False);data={r['id']:(np.array(Image.open(r['image']).convert('RGB')),np.array(Image.open(r['target']))) for r in ds['records']}
 for c in refs['photo_cases']+videos:
  scene,ink=sparse_scene_reference(c['size'],c['regions']);assert (ink==255).all();data[c['id']]=(np.array(Image.open(c['image']).convert('RGB')),scene)
 groups=lambda rows:[[r for r in rows if r['work_group']==g] for g in sorted({r['work_group'] for r in rows})]
 og,pg,vg=groups(oldrows),groups(photos),groups(videos);random.seed(p['seed']);np.random.seed(p['seed']);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed']);net=SceneObjects(state).cuda();opt=torch.optim.AdamW([v for v in net.parameters() if v.requires_grad],lr=.0003,weight_decay=.0001);curve=[];start=time.time()
 for step in range(p['steps']):
  chosen=[random.choice(random.choice(og)),random.choice(random.choice(og)),random.choice(random.choice(pg)),random.choice(random.choice(vg))];xs=[];ts=[]
  for r in chosen:
   im,t=data[r['id']];x,t,_=pack(im,t);im=np.clip((x.transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype('uint8');m=cv2.getRotationMatrix2D((128,128),random.uniform(-7,7),random.uniform(.9,1.1));im=cv2.warpAffine(im,m,(256,256),borderValue=(210,210,210));t=cv2.warpAffine(t,m,(256,256),flags=cv2.INTER_NEAREST,borderValue=255)
   if random.random()<.5:im=im[:,::-1].copy();t=t[:,::-1].copy()
   if random.random()<.3:im=np.repeat(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)[...,None],3,2)
   im=np.clip(im.astype('float32')*random.uniform(.9,1.1)+random.uniform(-8,8),0,255).astype('uint8');xs.append(normalize(im));ts.append(t.astype('int64'))
  x=torch.from_numpy(np.stack(xs)).cuda();t=torch.from_numpy(np.stack(ts)).cuda();z=net(x)
  with torch.no_grad():tz=teacher(x[:2])
  loss=partial_role_loss(z[:2],t[:2])+.2*masked_teacher_kl(z[:2][:,[3,1,2]],tz,t[:2].ne(255))+.7*partial_role_loss(z[2:3],t[2:3])+.7*partial_role_loss(z[3:4],t[3:4]);opt.zero_grad();loss.backward();opt.step();curve.append(float(loss.detach()))
  if (step+1)%64==0:print('mapping_fixed',step+1,round(curve[-1],4),flush=True)
 net.eval();torch.save(dict(model=net.state_dict(),protocol=p),ROOT/'mapping_fixed.pt');write('checkpoint_freeze.json',dict(mapping_fixed=sha(ROOT/'mapping_fixed.pt')));control=SceneObjects().cuda();control.load_state_dict(torch.load(PREV/'scene_plus_line.pt',map_location='cuda',weights_only=False)['model']);control.eval();models={'conflicted_control':control,'mapping_fixed':net};(ROOT/'predictions').mkdir(exist_ok=True)
 previous=read(PREV/'results.json');out=[]
 for r in previous['records']:
  ident=r['id'];im,t=data.get(ident,(np.array(Image.open(r['image']).convert('RGB')),None));scores={}
  for a,model in models.items():
   prob,pred=predict(model,im);np.savez_compressed(ROOT/'predictions'/(ident+'_'+a+'.npz'),prediction=pred)
   if t is not None:scores[a]=scene_metrics(t,pred)
  out.append(dict(id=ident,image=r['image'],split=r['split'],work_group=r['work_group'],scores=scores))
 aggregates={}
 for split in sorted({r['split'] for r in out if r['scores']}):
  aggregates[split]={}
  for a in models:
   c=sum(np.array(r['scores'][a]['confusion']) for r in out if r['split']==split);support=c.sum(1);union=c.sum(0)+support-c.diagonal();iou=[float(c[k,k]/union[k]) if union[k] else None for k in range(4)];aggregates[split][a]=dict(confusion=c.tolist(),support=support.tolist(),iou=iou,mean_iou=float(np.mean([x for x in iou if x is not None])),recall=[float(c[k,k]/support[k]) if support[k] else None for k in range(4)])
 # Fixed photometric stress views do not add works, labels or independent tests.
 stresses=[]
 for c in refs['photo_cases']:
  if c['split']!='photo_holdout':continue
  im,t=data[c['id']]
  for name,gain,bias,color in [('original',1.,0.,[1,1,1]),('warm',.95,-6,[1.06,1.,.9]),('dim',.78,4,[1,1,1]),('bright',1.12,8,[1,1,1])]:
   view=np.clip(im.astype('float32')*gain*np.array(color)+bias,0,255).astype('uint8');row=dict(id=c['id'],variant=name,scores={})
   for a,model in models.items():_,pred=predict(model,view);row['scores'][a]=scene_metrics(t,pred)
   stresses.append(row)
 region=np.load(LAB/'cunfa_object_correction_20261112/H_14_object_reference.npz')['object_verified'];h={a:float((np.load(ROOT/'predictions'/('H_14_'+a+'.npz'))['prediction'][region]==2).mean()) for a in models};assert sha(PREV/'results.json')==p['conflicted_control_results_sha256'];assert sha(OLD/'results.json')==p['old_results_sha256'];assert all(torch.equal(v,net.state_dict()[k]) for k,v in state.items() if k.startswith('encoder.'))
 write('results.json',dict(protocol=p,records=out,aggregates=aggregates,photometric_stress=stresses,user_foliage_recall=h,checkpoint_sha256=sha(ROOT/'mapping_fixed.pt'),loss_curve=curve,duration_seconds=time.time()-start,model_adopted=False,expert_accuracy=False,independent_acceptance=False,robot_connected=False));print(json.dumps(dict(aggregates=aggregates,user=h),ensure_ascii=False),flush=True)
if __name__=='__main__':
 if '--prepare' in sys.argv:prepare()
 else:train()
