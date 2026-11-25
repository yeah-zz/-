# -*- coding: utf-8 -*-
"""Isolate anchor/curve/instance errors using fully known procedural strokes.

Correct anchors here are deliberately supplied ONLY for diagnosis, not as a
production result, real-stroke accuracy claim or training example.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.rock_context_data import LAB,ROOT,synthetic,targets,GRID,POINTS,SIZE,write
from gongbi_training.rock_context_model import RockContextNet
import torch,numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
OUT=LAB/'rock_context_diagnostics_20261110'
def iou(a,b):return float((a&b).sum()/max(1,(a|b).sum()))
def pair_iou(masks):return float(np.mean([iou(masks[i],masks[j]) for i in range(len(masks)) for j in range(i+1,len(masks))])) if len(masks)>1 else 0.
def main():
 OUT.mkdir(exist_ok=True);torch.set_num_threads(4);ck=torch.load(ROOT/'whole_context_step1250.pt',map_location='cpu',weights_only=False);net=RockContextNet(project_curve=True).cuda();net.load_state_dict(ck['model']);net.eval();rows=[]
 for seed in [990101,990102,990103,990104,990105,990106]:
  r=synthetic(seed);t=targets(r)
  with torch.no_grad():p=net(torch.tensor(r['image'].transpose(2,0,1)[None],device='cuda'))
  gt=[];pm=[];curves=[];roles=[];query=[]
  for a in t['anchors']:
   y,x=a['y'],a['x'];logit=(p['instance'][0,:,y,x,None,None]*p['pixel'][0]).sum(0)/(8**.5)+p['mask_bias'][0,0,y,x];pred=logit.sigmoid().cpu().numpy()>.5;truth=a['mask']>.5;gt.append(truth);pm.append(pred)
   q=(p['curves'][0,:,y,x].reshape(POINTS,2).cpu().numpy()+np.array([(x+.5)/GRID,(y+.5)/GRID]))*SIZE;b=a['curve']*SIZE;curves.append(float(min(np.linalg.norm(q-b,axis=1).mean(),np.linalg.norm(q-b[::-1],axis=1).mean())));roles.append(int(p['roles'][0,:,y,x].argmax())==a['role']);query.append(p['instance'][0,:,y,x].cpu().numpy())
  actual=[iou(a,b) for a,b in zip(pm,gt)];union_true=np.any(gt,axis=0);union_pred=np.any(pm,axis=0);q=np.array(query);q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8);similar=q@q.T;pair=similar[np.triu_indices(len(q),1)]
  rows.append(dict(seed=seed,true_anchor_instances=len(gt),oracle_anchor_mean_instance_iou=float(np.mean(actual)),oracle_anchor_union_ink_iou=iou(union_true,union_pred),
   predicted_mask_pairwise_iou=pair_iou(pm),true_mask_pairwise_iou=pair_iou(gt),instance_query_pairwise_cosine=float(pair.mean()) if len(pair) else None,
   oracle_anchor_mean_curve_error_px=float(np.mean(curves)),oracle_anchor_role_accuracy=float(np.mean(roles))))
  if seed==990101:
   font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);board=Image.new('RGB',(1280,366),'#f2efe6');d=ImageDraw.Draw(board);panels=[Image.fromarray(np.uint8(r['image']*255)),Image.fromarray(np.uint8(255*(1-union_true))).convert('RGB'),Image.fromarray(np.uint8(255*(1-gt[0]))).convert('RGB'),Image.fromarray(np.uint8(255*(1-pm[0]))).convert('RGB')]
   for i,(im,label) in enumerate(zip(panels,['已知程序路径·非真实皴法','全部笔迹参考','指定单笔的正确墨迹','给正确位置后·模型墨迹'])):
    board.paste(im.resize((310,310)),(i*320,33));d.text((i*320+4,4),label,font=font,fill='#243e37')
   board.save(OUT/'instance_diagnosis.jpg',quality=95)
 avg={k:float(np.mean([r[k] for r in rows])) for k in ['oracle_anchor_mean_instance_iou','oracle_anchor_union_ink_iou','predicted_mask_pairwise_iou','true_mask_pairwise_iou','instance_query_pairwise_cosine','oracle_anchor_mean_curve_error_px','oracle_anchor_role_accuracy']}
 report=dict(rows=rows,mean=avg,correct_anchors_given_for_diagnosis_only=True,production_inference_claim=False,real_painting_accuracy_claim=False,checkpoint_changed=False,robot_connected=False)
 write(OUT/'oracle_diagnosis.json',report)
 gate=dict(passed=avg['oracle_anchor_mean_instance_iou']>=.5,minimum_procedural_instance_iou=.5,observed_oracle_instance_iou=avg['oracle_anchor_mean_instance_iou'],
  gate_scope='minimum engineering sanity on fully labelled procedural strokes; not an art-quality standard',
  block_full_painting_trial=True,block_claim_of_whole_rock_success=True,local_reference_coverage_not_acceptance_metric=True,
  required_next_work='Repair single-stroke ownership decoder and instance objective; pass full-label per-stroke tests before more real-work expansion',robot_connected=False)
 write(OUT/'quality_gate.json',gate);print(json.dumps(dict(mean=avg,gate=gate),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
