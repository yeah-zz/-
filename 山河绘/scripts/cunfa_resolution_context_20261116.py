# -*- coding: utf-8 -*-
"""Frozen-head diagnostic of resolution and larger rock-face context, not training."""
import sys,json,hashlib,datetime,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,joblib
from torchvision.models import mobilenet_v3_small
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_resolution_context_20261116';OLD=LAB/'cunfa_fresh_validation_20261111';ARMS=['original_views','half_resolution','rock_context'];CLASSES=['披麻皴','斧劈皴']
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def module_box(r):
 x0,y0,x1,y1=r['source_box']
 if r['work_id']=='231531':return [18,168,340,448] if y0<500 else [10,505,220,745]
 if r['work_id']=='234350':return [42,350,280,925]
 if r['work_id']=='230757':
  if y0<500:return [586,150,1014,510]
  return [12,815,596,967] if y0>=815 else [8,610,262,851]
 raise ValueError('Unreviewed source work')
def prepare():
 assert not (ROOT/'protocol.json').exists(),'Frozen diagnostic';ds=read(OLD/'test_manifest.json');rows=[dict(r) for r in ds['records'] if r['admission']=='source_label_test' and r['publisher']=='DPM'];assert len(rows)==9;freeze=read(OLD/'freeze_manifest.json')
 for f in freeze['files']:assert sha(f['path'])==f['sha256']
 for r in rows:
  im=Image.open(r['source_image']);box=module_box(r);assert 0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height;x0,y0,x1,y1=r['source_box'];assert box[0]<=x0<x1<=box[2] and box[1]<=y0<y1<=box[3];r.update(rock_context_box=box,rock_context_selected_by_assistant=True,rock_context_is_not_instance_mask=True,local_cunfa_expert_verified=False,training_eligible=False,original_crop_native_size=list(Image.open(r['image']).size))
 manifest=dict(records=rows,excluded_comparison='4 NPM crops remain old known cases; no public higher-detail source acquired here, not silently counted into9-case comparison',source_labels='Museum work-level descriptions plus old provisional local boxes, not new expert labels',source_work_count=3,new_independent_test=False);write('comparison_manifest.json',manifest)
 p=dict(prepared_before_new_predictions=True,prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(ROOT/'comparison_manifest.json'),old_prediction_sha256=sha(OLD/'predictions.json'),old_features_sha256=sha(OLD/'features.npz'),freeze_manifest_sha256=sha(OLD/'freeze_manifest.json'),script_sha256=sha(__file__),arms=ARMS,input_change='Same frozen encoder, grayscale equalization/letterbox224 and linear SVM; original local+2.5x context vs same views downsampled half before preprocessing vs original local+assistant larger rock-face context',no_new_training=True,selection='All9 previously admitted DPM cases from3 works, all reported; no model/threshold selection',larger_context_scope='Assistant-selected visible rock-face windows, not model recovered whole rocks or per-stroke truth',resolution_status='Public native URLs returned smaller539-pixel versions;2048 variants404; no higher-resolution genuine source acquired. Existing1024 previews remain inputs, no upscaling claimed as new detail',causal_scope='Exploratory same-case input ablation; not independent expert accuracy or learned autonomous recognition',robot_connected=False);write('protocol.json',p)
 (ROOT/'views').mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);sources={r['work_id']:r['source_image'] for r in rows};board=Image.new('RGB',(1200,1050),'#f4f1e8');d=ImageDraw.Draw(board)
 for j,(work,path) in enumerate(sources.items()):
  im=Image.open(path).convert('RGB');draw=ImageDraw.Draw(im)
  for r in rows:
   if r['work_id']!=work:continue
   draw.rectangle(r['rock_context_box'],outline='#367b60',width=3);draw.rectangle(r['source_box'],outline='#c77d35',width=2)
  im.thumbnail((385,950));board.paste(im,(j*400+5,45));d.text((j*400+5,8),work+'：橙局部／绿较完整岩面窗口',font=font,fill='#294735')
 board.save(ROOT/'object_context_review.jpg',quality=95);print('FROZEN',len(rows),'cases',len(sources),'works',flush=True)
def evaluate():
 assert not (ROOT/'results.json').exists(),'Never tune on displayed output';p=read(ROOT/'protocol.json');assert sha(__file__)==p['script_sha256'];assert sha(ROOT/'comparison_manifest.json')==p['manifest_sha256'];assert sha(OLD/'predictions.json')==p['old_prediction_sha256'];assert sha(OLD/'features.npz')==p['old_features_sha256'];freeze=read(OLD/'freeze_manifest.json')
 for f in freeze['files']:assert sha(f['path'])==f['sha256']
 spec=importlib.util.spec_from_file_location('frozen_cunfa_input',OLD/'frozen/encoder_and_preprocessing.py');frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen);rows=read(ROOT/'comparison_manifest.json')['records'];torch.set_num_threads(4);weight=next(f for f in freeze['files'] if f['path'].endswith('.pth'));net=mobilenet_v3_small(weights=None);net.load_state_dict(torch.load(weight['path'],map_location='cpu',weights_only=True));net=net.features.cuda().eval();head=joblib.load(OLD/'frozen/context_head_exploratory.joblib')['model'];arrays=[];views=[]
 for r in rows:
  assert sha(r['image'])==r['crop_sha256'] and sha(r['source_image'])==r['source_sha256'];a,b,_=frozen.input_views(r);source=Image.open(r['source_image']).convert('RGB');c=source.crop(tuple(r['rock_context_box']));parts=dict(original_views=(a,b),half_resolution=(a.resize((max(1,a.width//2),max(1,a.height//2)),Image.Resampling.LANCZOS),b.resize((max(1,b.width//2),max(1,b.height//2)),Image.Resampling.LANCZOS)),rock_context=(a,c))
  for arm,(local,context) in parts.items():
   lp=ROOT/'views'/(r['id']+'_'+arm+'_local.png');cp=ROOT/'views'/(r['id']+'_'+arm+'_context.png');local.save(lp);context.save(cp);arrays.extend([frozen.preprocess(local),frozen.preprocess(context)]);views.append(dict(id=r['id'],arm=arm,local='views/'+lp.name,context='views/'+cp.name,local_native_size=list(local.size),context_native_size=list(context.size)))
 features=[]
 with torch.inference_mode():
  for i in range(0,len(arrays),12):
   z=net(torch.from_numpy(np.stack(arrays[i:i+12])).cuda()).mean((2,3));features.append(torch.nn.functional.normalize(z,dim=1).cpu().numpy())
 z=np.concatenate(features);x=np.concatenate([z[::2],z[1::2]],axis=1)/np.sqrt(2);pred=head.predict(x);margins=head.decision_function(x);old={r['id']:r for r in read(OLD/'predictions.json')['records']};oldcases=[r for r in read(OLD/'test_manifest.json')['records'] if r['admission']=='source_label_test'];oldx=np.load(OLD/'features.npz')['local_plus_context'];out=[]
 for i,v in enumerate(views):
  r=next(r for r in rows if r['id']==v['id']);result=dict(**v,source_label=r['label'],source_url=r['source_url'],work_id=r['work_id'],prediction=CLASSES[int(pred[i])],signed_margin=float(margins[i]),source_label_agreement=CLASSES[int(pred[i])]==r['label'])
  if v['arm']=='original_views':
   assert result['prediction']==old[r['id']]['predictions']['local_plus_context']['predicted'];j=next(j for j,q in enumerate(oldcases) if q['id']==r['id']);assert np.allclose(x[i],oldx[j],atol=1e-5,rtol=1e-5)
  out.append(result)
 summary={arm:dict(regions=9,agreement_count=sum(r['source_label_agreement'] for r in out if r['arm']==arm),source_label_agreement=sum(r['source_label_agreement'] for r in out if r['arm']==arm)/9,per_work={work:dict(regions=sum(r['work_id']==work for r in rows),agreement_count=sum(r['source_label_agreement'] for r in out if r['arm']==arm and r['work_id']==work)) for work in sorted({r['work_id'] for r in rows})}) for arm in ARMS}
 np.savez_compressed(ROOT/'features.npz',features=x);write('results.json',dict(protocol=p,records=out,summary=summary,old_baseline_features_reproduced=True,weights_unchanged=True,no_new_training=True,original_source_crop_pixels_preserved=True,source_labels_not_local_expert_truth=True,new_independent_test=False,model_adopted=False,robot_connected=False));print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
 if '--prepare' in sys.argv:prepare()
 else:evaluate()
