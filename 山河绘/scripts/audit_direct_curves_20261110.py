# -*- coding: utf-8 -*-
"""Audit unsupported direct curves and preprocessing; never beautify predictions.

Image support is a diagnostic, not annotation precision or artist stroke recall.
The failing coordinate model cannot emit an approved drawing plan.
"""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image
from scipy.ndimage import binary_propagation
OUT=LAB/'rock_skill_offline_20261110'

def resample(q):
 q=np.asarray(q,float);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
 keep=np.r_[True,np.diff(arc)>1e-6];arc=arc[keep];q=q[keep]
 t=np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]))+1))
 return np.column_stack([np.interp(t,arc,q[:,i]) for i in (0,1)])

def audit(rows,rgb):
 # Frozen image evidence thresholds from source_ink_baseline_20261028.py.
 g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype('float32')/255
 g=cv2.GaussianBlur(g,(0,0),.45);local=cv2.GaussianBlur(g,(0,0),3)-g
 valley=cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
 ink=binary_propagation((local>.045)&(valley>.09),mask=(local>.018)&(valley>.045))
 distance=cv2.distanceTransform((~ink).astype('uint8'),cv2.DIST_L2,5)
 h,w=g.shape;stats=[]
 for i,r in enumerate(rows):
  q=resample(r['curve']);inside=(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
  xy=np.round(q).astype(int);xy[:,0]=xy[:,0].clip(0,w-1);xy[:,1]=xy[:,1].clip(0,h-1)
  supported=inside&(distance[xy[:,1],xy[:,0]]<=2.)
  stats.append(dict(id=r.get('id',str(i)),confidence=r['confidence'],samples=len(q),out_of_image_fraction=float((~inside).mean()),
   source_supported_fraction=float(supported.mean()),unsupported_fraction=float((~supported).mean())))
 return dict(candidates=len(rows),curves_leaving_image=sum(r['out_of_image_fraction']>0 for r in stats),
  curves_less_than_half_supported=sum(r['source_supported_fraction']<.5 for r in stats),
  mean_source_supported_fraction=float(np.mean([r['source_supported_fraction'] for r in stats])) if stats else None,
  rows=stats,evidence_kind='fixed pixel diagnostic, not annotated precision; faint true lines may be missed')

def main():
 original=json.loads((OUT/'model_direct.json').read_text('utf-8'))
 source=json.loads((OUT/'visible_reference.json').read_text('utf-8'));x0,y0,x1,y1=source['box']
 full=np.array(Image.open(source['source']).convert('RGB'));rgb=full[y0:y1,x0:x1]
 raw=audit(original['curves'],rgb)
 # Control the aspect/spacing mismatch in the prior viewer inference.
 import torch
 sys.path.insert(0,str(Path(__file__).resolve().parent))
 from gongbi_training.rock_context_model import RockContextNet
 from train_rock_context_joint_20261109 import infer
 torch.set_num_threads(4);net=RockContextNet(project_curve=True).cuda()
 ck=torch.load(original['checkpoint'],map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);net.eval()
 h,w=rgb.shape[:2];s=min(232/w,232/h);origin=np.array([(256-w*s)/2,(256-h*s)/2])
 mat=np.float32([[s,0,origin[0]],[0,s,origin[1]]]);letter=cv2.warpAffine(rgb,mat,(256,256),borderMode=cv2.BORDER_REFLECT_101)
 pred,_=infer(net,letter.astype('float32')/255,threshold=.35,max_predictions=32)
 for i,p in enumerate(pred):p['curve']=((np.array(p['curve'])-origin)/s).tolist();p.pop('mask',None);p['id']=f'letterbox_{i:02d}'
 corrected=audit(pred,rgb)
 (OUT/'letterbox_direct.json').write_text(json.dumps(dict(curves=pred,preprocessing='same 12px padded aspect-preserving convention as training',reference_input=False,approved=False),ensure_ascii=False,indent=2),encoding='utf-8')
 decision=dict(route='whole_context_step1250_direct_coordinates',status='rejected',drawing_enabled=False,approved_trajectories=[],
  reason='Raw output visibly fails. Pixel support and resize ablation are diagnostics only; no complete independent stroke validation exists.',
  confidence_is_not_stroke_quality=True,post_filtering_would_not_establish_reconstruction=True,raw_preserved=True,new_weights_trained=False,
  authoritative_baseline='approved_execution_refine_20261107/G03_01_500/plan.json',robot_connected=False)
 (OUT/'direct_model_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
 result=dict(raw=raw,aspect_preserving=corrected,weights_unchanged=True,reference_labels_used=False,decision=decision)
 (OUT/'direct_curve_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:{a:b for a,b in v.items() if a!='rows'} for k,v in [('raw',raw),('aspect_preserving',corrected)]},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
