# -*- coding: utf-8 -*-
"""Repair reference coordinates before a new immutable refinement experiment."""
import sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import cunfa_band_dataset_20261112 as build
ROOT=LAB/'cunfa_band_checked_20261112';OLD=LAB/'cunfa_roles_v2_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'results.json').exists();guides=copy.deepcopy(build.GUIDES);guides['M_1']=[g for g in guides['M_1'] if g['id']!='M_contour_b'];guides['L_149_3']=[g for g in guides['L_149_3'] if g['id']!='L_cun_b']
 guides['H_14'][2]['points']=[[94,225],[98,218],[100,210],[102,205]]
 # This crowded second hatch failed the pre-training support check; do not
 # relax the gate or force its dry/ambiguous pixels into either class.
 guides['H_14']=[g for g in guides['H_14'] if g['id']!='H_cun_b']
 # Inspection discovered coordinate errors. Existing runs and metric files are
 # retained verbatim, but explicitly invalidated for acceptance.
 invalid=dict(reason='Several assistant-drawn guides were misplaced on paper, including M_contour_b, L_cun_b and H_cun_a. Their labels can contaminate training and width/recall evaluation.',valid_for_acceptance=False,old_metrics_not_revised=True,automatic_predictions_used_to_relabel=False,action='Exclude two ambiguous training guides, correct clear H ink path from native pixels, constrain guide refinement to image evidence and independently review overlays. No blind-test claim.')
 for folder in ['cunfa_band_v3_20261112','cunfa_band_refiner_20261112']:(LAB/folder/'reference_invalidation.json').write_text(json.dumps(invalid,ensure_ascii=False,indent=2),encoding='utf8')
 # Reference construction assistance only. Inference receives none of these
 # curves. A fixed small normal search finds native image ink support, not a
 # model prediction. This creates weak reference, not independent expert gold.
 audit=[]
 for ident,gs in guides.items():
  z=np.load(OLD/'references'/(ident+'.npz'));image=z['image'];gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype('float32');lc=cv2.GaussianBlur(gray,(0,0),4)-gray
  for g in gs:
   original=np.array(g['points'],'float32');s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(original,axis=0),axis=1))];ts=np.arange(0,s[-1]+.01,1.);q=np.column_stack([np.interp(ts,s,original[:,k]) for k in range(2)]);tangent=np.gradient(q,axis=0);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True).clip(.01);normal=np.column_stack([-tangent[:,1],tangent[:,0]]);offset=np.arange(-2.5,2.51,.5,dtype='float32');candidates=q[:,None,:]+normal[:,None,:]*offset[None,:,None];signal=cv2.remap(lc,candidates[:,:,0].astype('float32'),candidates[:,:,1].astype('float32'),cv2.INTER_LINEAR);cost=-signal+.4*offset[None,:]**2;current=cost[0];backs=[]
   for i in range(1,len(q)):
    transition=.8*(offset[:,None]-offset[None,:])**2;v=transition+current[None,:];parent=v.argmin(1);current=cost[i]+v[np.arange(len(offset)),parent];backs.append(parent)
   k=int(current.argmin());ids=[k]
   for parent in backs[::-1]:k=int(parent[k]);ids.append(k)
   ids=np.array(ids[::-1]);refined=candidates[np.arange(len(q)),ids];g['manual_original_points']=g['points'];g['points']=refined.tolist();g['inner']=.45 if ident!='M_1' else .8
   if ident=='H_14':g['outer']=4.5;g['paper_margin']=2.5
   dist,along,total=build.field(gray.shape,g['points']);inside=(dist<=g['inner'])&(along>3)&(along<total-3);values=lc[inside];fraction=float((values<0).mean()) if len(values) else 1.;audit.append(dict(id=ident,guide=g['id'],ink_core_pixels=len(values),fraction_negative_local_contrast=fraction,median_local_contrast=float(np.median(values)),max_reference_adjustment_px=float(np.abs(offset[ids]).max()),reference_only_not_model_output=True,not_expert_gold=True))
   assert len(values)>=5 and fraction<.08,(ident,g['id'],fraction)
 build.ROOT=ROOT;build.GUIDES=guides;build.main()
 # Reject paper candidates which still contain visible dark ink (e.g. a close
 # neighbouring hatch). This is weak image-supported reference construction,
 # not a complete independent ground-truth ink mask.
 dataset=json.loads((ROOT/'dataset_manifest.json').read_text('utf8'))
 for r in dataset['records']:
  z=dict(np.load(r['path']));g=cv2.cvtColor(z['image'],cv2.COLOR_RGB2GRAY).astype('float32');lc=cv2.GaussianBlur(g,(0,0),4)-g;reject=(z['band_targets']==0)&(lc>=2.5);z['band_targets'][reject]=255;z['near_paper'][reject]=False;r['paper_candidates_rejected_for_visible_ink']=int(reject.sum());r['band_counts']={str(k):int((z['band_targets']==k).sum()) for k in range(3)};np.savez_compressed(r['path'],**z);r['sha256']=sha(r['path']);overlay=np.array(Image.open(ROOT/'references'/(r['id']+'_overlay.png')));overlay[reject]=z['image'][reject];Image.fromarray(overlay).save(ROOT/'references'/(r['id']+'_overlay.png'));target=np.array(Image.open(ROOT/'references'/(r['id']+'_target.png')));target[reject]=[218,212,199];Image.fromarray(target).save(ROOT/'references'/(r['id']+'_target.png'))
 (ROOT/'dataset_manifest.json').write_text(json.dumps(dataset,ensure_ascii=False,indent=2),encoding='utf8');(ROOT/'reference_coordinate_audit.json').write_text(json.dumps(dict(records=audit,invalidated_prior_runs=invalid,construction='Manual native-pixel anchors plus fixed +/-2.5px image-evidence normal adjustment; visible-ink paper candidates rejected; not automatic model prediction',image_evidence_not_independent_gold=True,reference_curves_never_model_input=True),ensure_ascii=False,indent=2),encoding='utf8')
 # Native crop overlays for a second visual check before training.
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
 for ident,gs in guides.items():
  z=np.load(ROOT/'references'/(ident+'.npz'));im=Image.fromarray(z['image']);overlay=Image.open(ROOT/'references'/(ident+'_overlay.png'));target=Image.open(ROOT/'references'/(ident+'_target.png'))
  for g in gs:
   q=np.array(g['points']);x0=max(0,int(q[:,0].min())-11);x1=min(im.width,int(q[:,0].max())+12);y0=max(0,int(q[:,1].min())-11);y1=min(im.height,int(q[:,1].max())+12);box=(x0,y0,x1,y1);board=Image.new('RGB',(1200,470),'#f4f1e8');d=ImageDraw.Draw(board)
   for j,(p,title) in enumerate([(im,'原生像素局部'),(overlay,'核查墨迹 / 两侧纸面'),(target,'边缘未知')]):
    a=p.crop(box);a=a.resize((a.width*4,a.height*4),Image.Resampling.NEAREST);a.thumbnail((380,390));board.paste(a,(j*400+5,35));d.text((j*400+5,5),title,font=font,fill='#243e32')
   d.text((8,438),g['id']+'：放大仅供检查；没有新增真实图像信息。',font=font,fill='#815238');board.save(ROOT/'references'/(g['id']+'_zoom.jpg'),quality=95)
 print('CHECKED_REFERENCE_AUDIT',audit,flush=True)
if __name__=='__main__':main()
