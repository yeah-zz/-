# -*- coding: utf-8 -*-
"""Fair frozen-base before/after, complete procedural masks and sparse real data."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_ink_ownership_20261110 import *
from train_rock_context_joint_20261109 import infer
from evaluate_rock_context_joint_20261109 import external_examples,match,group_plan
from PIL import Image,ImageDraw,ImageFont
from scipy.optimize import linear_sum_assignment
CALFILE=OUT/('mask_calibration_hr.json' if (OUT/'ink_decoder_hr.pt').exists() else 'mask_calibration.json')
MASK_THRESHOLD=json.loads(CALFILE.read_text('utf8'))['selected'] if CALFILE.exists() else .5

def iou(a,b):return float((a&b).sum()/max(1,(a|b).sum()))
def decode(net,old,image,aa):
 features,dist,curves=condition_features(old,image,aa);masks=[];quality=[]
 with torch.no_grad():
  for start in range(0,len(aa),8):
   if getattr(net,'high_resolution',False):m,q=net(features[start:start+8],image[aa[start:start+8,0].long()])
   else:m,q=net(features[start:start+8])
   masks.extend(m.sigmoid().cpu().numpy());quality.extend(q.sigmoid().cpu().numpy().tolist())
 return masks,quality,dist.cpu().numpy(),curves.cpu().numpy()*SIZE

def automatic(base,net,image):
 old_records,_=infer(base,image,.35);tensor=torch.tensor(image.transpose(2,0,1)[None],dtype=torch.float32,device='cuda')
 with torch.no_grad():old=base(tensor)
 if not old_records:return old_records,[],[]
 aa=torch.tensor([[0,*r['anchor']] for r in old_records],device='cuda');masks,qualities,_,_=decode(net,old,tensor,aa);raw=[]
 for i,(r,m,q) in enumerate(zip(old_records,masks,qualities)):raw.append(dict(r,mask=m,mask_threshold=MASK_THRESHOLD,nonempty_probability=float(q),source_candidate=i))
 # Disclosed output filtering only; not a learned reduction of stroke count.
 # Similar masks alone do not authorize joining or smoothing two curves.
 keep=[];rejected=[]
 for i in sorted(range(len(raw)),key=lambda i:raw[i]['confidence']*raw[i]['nonempty_probability'],reverse=True):
  r=raw[i];binary=r['mask']>MASK_THRESHOLD;reason=None
  if r['nonempty_probability']<.5:reason='predicted_empty_query'
  elif binary.sum()<3:reason='fewer_than_3_mask_pixels'
  else:
   for j in keep:
    if iou(binary,raw[j]['mask']>MASK_THRESHOLD)>.8:
     a=np.array(r['curve']);b=np.array(raw[j]['curve']);d=np.linalg.norm(a[:,None]-b[None],axis=2)
     if (d.min(0).mean()+d.min(1).mean())/2<6:reason='duplicate_mask_and_curve';break
  if reason:rejected.append(dict(candidate=i,reason=reason))
  else:keep.append(i)
 return old_records,[raw[i] for i in keep],dict(raw=raw,rejected=rejected,keep=keep)

def mask_f1(records,truth):
 if not records:return dict(predicted=0,true=len(truth),matches_iou50=0,precision=0.,recall=0.,f1=0.)
 costs=np.array([[1-iou(cv2.resize(p['mask'],(t.shape[1],t.shape[0]),interpolation=cv2.INTER_LINEAR)>p.get('mask_threshold',.5),t>.5) for p in records] for t in truth]);pairs=list(zip(*linear_sum_assignment(costs)));n=sum(costs[i,j]<=.5 for i,j in pairs)
 return dict(predicted=len(records),true=len(truth),matches_iou50=int(n),precision=n/len(records),recall=n/len(truth),f1=2*n/(len(records)+len(truth)))

def panel(image,records,path,curves=False):
 rgb=np.uint8(np.clip(image,0,1)*255);out=cv2.resize(rgb,(512,512)).astype(float);palette=np.array([[28,128,172],[184,121,25],[101,146,58],[142,83,162],[176,74,93]],float)
 if not curves:
  # One blend per pixel avoids opacity increasing merely with duplicate count.
  rgb_sum=np.zeros_like(out);count=np.zeros(out.shape[:2])
  for i,r in enumerate(records):
   m=cv2.resize(r['mask'],(512,512))>r.get('mask_threshold',.5);rgb_sum[m]+=palette[i%len(palette)];count[m]+=1
  active=count>0;out[active]=out[active]*.65+(rgb_sum[active]/count[active,None])*.35
 im=Image.fromarray(np.uint8(out));d=ImageDraw.Draw(im)
 if curves:
  for r in records:
   d.line([tuple(p) for p in np.array(r['curve'])*2],fill='#b47e25' if r['role']==0 else '#1d81a7',width=2)
 im.save(path)

def main():
 torch.set_num_threads(4);base=old_model();net=load_decoder();holdout=[]
 # Untouched new seeds. All stroke instances are counted, including coincident
 # anchors; correct-anchor results below are diagnostics, not production scores.
 for seed in range(990401,990417):
  r=synthetic(seed);tensor=torch.tensor(r['image'].transpose(2,0,1)[None],device='cuda');truth=[s['mask'] for s in r['strokes']];aa=[]
  for s in r['strokes']:
   x,y=np.clip(np.floor(s['curve'][POINTS//2]/SIZE*GRID).astype(int),0,GRID-1);aa.append([0,int(x),int(y)])
  aa=torch.tensor(aa,device='cuda')
  with torch.no_grad():old=base(tensor)
  new,q,dist,_=decode(net,old,tensor,aa);old_masks=[]
  for _,x,y in aa:
   logit=(old['instance'][0,:,y,x,None,None]*old['pixel'][0]).sum(0)/(8**.5)+old['mask_bias'][0,0,y,x];old_masks.append(cv2.resize(logit.sigmoid().cpu().numpy(),(SIZE,SIZE),interpolation=cv2.INTER_LINEAR))
  gray=r['image'].mean(2);ink=(np.quantile(gray,.95)-gray)>.10;geo=[(cv2.resize(d,(SIZE,SIZE))<8)&ink for d in dist]
  old_auto,new_auto,detail=automatic(base,net,r['image']);row=dict(seed=seed,truth_count=len(truth),distinct_anchor_count=len(set(map(tuple,aa.cpu().numpy().tolist()))),
   correct_anchor_old_iou=float(np.mean([iou(a>.5,t>.5) for a,t in zip(old_masks,truth)])),correct_anchor_new_iou=float(np.mean([iou(a>MASK_THRESHOLD,t>.5) for a,t in zip(new,truth)])),
   correct_anchor_geometry_only_iou=float(np.mean([iou(a,t>.5) for a,t in zip(geo,truth)])),old_automatic=mask_f1(old_auto,truth),new_automatic=mask_f1(new_auto,truth))
  holdout.append(row)
  if seed==990401:
   font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);board=Image.new('RGB',(1280,370),'#f2efe6');d=ImageDraw.Draw(board);j=0
   imgs=[Image.fromarray(np.uint8(r['image']*255)),Image.fromarray(np.uint8(255*(1-truth[j]))).convert('RGB'),Image.fromarray(np.uint8(255*(1-(old_masks[j]>.5)))).convert('RGB'),Image.fromarray(np.uint8(255*(1-(new[j]>MASK_THRESHOLD)))).convert('RGB')]
   for i,(im,label) in enumerate(zip(imgs,['完整程序样本·非真实皴法','同一笔的正确墨迹','旧单笔墨迹预测','修改后单笔墨迹预测'])):
    board.paste(im.resize((310,310)),(i*320,35));d.text((i*320+4,5),label,font=font,fill='#243e37')
   board.save(OUT/'single_stroke_before_after.jpg',quality=95)
  print('HOLDOUT',row,flush=True)
 real=real_example();scenes=[dict(real,id='real_training_module',scope='sparse real training scene; not independent test')]+external_examples();results=[]
 for scene in scenes:
  before,after,detail=automatic(base,net,scene['image']);name=scene['id'];Image.fromarray(np.uint8(scene['image']*255)).resize((512,512)).save(OUT/(name+'_input.png'));panel(scene['image'],before,OUT/(name+'_old_masks.png'));panel(scene['image'],detail['raw'] if detail else [],OUT/(name+'_new_raw_masks.png'));panel(scene['image'],after,OUT/(name+'_new_kept_masks.png'));panel(scene['image'],after,OUT/(name+'_retained_curves.png'),curves=True)
  np.savez_compressed(OUT/(name+'_raw_masks.npz'),old=np.array([r['mask'] for r in before]),new_raw=np.array([r['mask'] for r in detail['raw']]) if detail else np.empty((0,GRID,GRID)),new_kept=np.array([r['mask'] for r in after]))
  group_plan(after);serial=lambda records:[{k:v for k,v in r.items() if k!='mask'} for r in records];row=dict(id=name,scope=scene['scope'],old_candidates=len(before),new_raw_candidates=len(detail['raw']) if detail else 0,new_kept_candidates=len(after),
    before_partial_matches=match(before,scene['strokes']),after_partial_matches=match(after,scene['strokes']),predictions=serial(after),rejections=detail['rejected'] if detail else [],
    reference_seed_at_inference=False,manual_module_box=True,curves_and_roles_unchanged=True,instance_masks_complete_truth=False)
  results.append(row);print('REAL',name,len(before),len(after),flush=True)
  if name=='real_training_module':
   # Saved new decoder must reproduce the same masks with frozen base.
   check=load_decoder();_,replayed,_=automatic(base,check,scene['image']);assert len(replayed)==len(after);assert all(np.allclose(a['mask'],b['mask'],atol=1e-5) for a,b in zip(after,replayed))
 avg={key:float(np.mean([r[key] for r in holdout])) for key in ['correct_anchor_old_iou','correct_anchor_new_iou','correct_anchor_geometry_only_iou']};avg['old_auto_mask_f1']=float(np.mean([r['old_automatic']['f1'] for r in holdout]));avg['new_auto_mask_f1']=float(np.mean([r['new_automatic']['f1'] for r in holdout]))
 write(OUT/'evaluation.json',dict(mean=avg,holdout=holdout,real=results,evaluation_mask_resolution=SIZE,old_probabilities_bilinearly_resized_to_same_grid=True,heldout_seeds_never_trained=True,oracle_metrics_are_diagnostic_only=True,real_blind_test=False,
   reference_curves_not_input=True,old_curve_predictor_frozen=True,mask_nms_is_explicit_postprocess=True,mask_threshold=MASK_THRESHOLD,threshold_selected_on_procedural_development_only=True,nonempty_threshold=.5,mask_nms_iou=.8,geometry_nms_symmetric_distance_px=6,coincident_anchor_crossing_supported=False,robot_connected=False))
 previous=json.loads((ROOT/'verification.json').read_text('utf8'));assert all(hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()==p['sha256'] for p in previous['approved_plans_fingerprints']);write(OUT/'verification.json',dict(passed=True,new_decoder_replayed=True,approved_plans_unchanged=True,base_model_file_unchanged=True,oracle_not_claimed_as_automatic=True,robot_connected=False));print('SUMMARY',avg,flush=True)
if __name__=='__main__':main()
