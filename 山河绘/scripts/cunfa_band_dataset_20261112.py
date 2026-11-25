# -*- coding: utf-8 -*-
"""Reviewed ink/paper intervals, not automatically thresholded ink truth."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_band_v3_20261112';OLD=LAB/'cunfa_roles_v2_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
# These are assistant traces reviewed on enlarged native pixel crops. Radii
# describe a conservative interval, not measured physical brush width.
GUIDES={
 'M_1':[
  dict(id='M_contour_a',role=1,points=[[55,62],[59,58],[64,54],[68,51]],inner=1.4,outer=6.0,paper_margin=3.,endpoint_status='partial segment; true endpoints outside review'),
  dict(id='M_contour_b',role=1,points=[[239,120],[238,130],[238,140],[239,146]],inner=1.5,outer=6.0,paper_margin=3.,endpoint_status='partial segment; curve bend excluded')],
 'L_149_3':[
  dict(id='L_outer_left',role=1,points=[[59,150],[57,160],[52,170],[49,180],[47,190],[44,200],[43,210],[41,220],[39,230],[38,240],[36,250]],inner=.8,outer=3.4,paper_margin=3.,endpoint_status='middle of contour; neither end is a brush endpoint'),
  dict(id='L_outer_right',role=1,points=[[124,76],[131,83],[140,90],[147,101],[155,113],[161,124],[166,138]],inner=.8,outer=3.4,paper_margin=3.,endpoint_status='middle of contour'),
  dict(id='L_cun_a',role=2,points=[[103,156],[99,161],[93,170],[88,180],[83,190],[78,200],[72,210],[68,218]],inner=.65,outer=2.8,paper_margin=2.8,endpoint_status='visible upper ink tip candidate; lower endpoint occluded',ink_tip_candidate=[103,156],ink_tip_uncertainty_px=3),
  dict(id='L_cun_b',role=2,points=[[85,157],[79,165],[73,173],[69,183],[64,193],[60,203],[56,213],[53,222]],inner=.65,outer=2.8,paper_margin=2.8,endpoint_status='partial second hatch; contact/lift not inferred')],
 'H_14':[
  dict(id='H_faint_contour',role=1,points=[[115,122],[126,115],[135,110],[144,106],[153,99],[161,91]],inner=.5,outer=4.0,paper_margin=3.,endpoint_status='middle of faint contour'),
  dict(id='H_curved_contour',role=1,points=[[196,52],[203,44],[209,36],[218,31],[227,28],[239,27],[248,25],[257,21]],inner=.6,outer=4.0,paper_margin=3.,endpoint_status='large turn preserved; true endpoints not reviewed'),
  dict(id='H_cun_a',role=2,points=[[101,229],[104,220],[107,213],[110,207]],inner=.6,outer=3.1,paper_margin=2.5,endpoint_status='partial hatch; crowded ends excluded'),
  dict(id='H_cun_b',role=2,points=[[80,215],[86,208],[92,202],[97,196]],inner=.6,outer=3.1,paper_margin=2.5,endpoint_status='partial hatch; neighboring ink crossing excluded')]
}
def field(shape,points):
 yy,xx=np.indices(shape);p=np.stack([xx,yy],-1).astype('float32');dist=np.full(shape,1e6,'float32');along=np.zeros(shape,'float32');total=0.
 for a,b in zip(points[:-1],points[1:]):
  a=np.array(a,'float32');b=np.array(b,'float32');v=b-a;length=float(np.linalg.norm(v));t=np.clip(((p-a)*v).sum(-1)/(length*length),0,1);dd=np.sqrt(((p-a-t[...,None]*v)**2).sum(-1));better=dd<dist;along[better]=total+t[better]*length;dist=np.minimum(dist,dd);total+=length
 return dist,along,total
def main():
 assert not (ROOT/'results.json').exists(),'References frozen after evaluation.'
 (ROOT/'references').mkdir(exist_ok=True);source=json.loads((OLD/'dataset_manifest.json').read_text('utf8'));records=[];font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 for r in source['records']:
  ident=r['id'];z=np.load(r['path']);image=z['image'];oldtarget=z['targets'];shape=oldtarget.shape;target=np.full(shape,255,'uint8');inner=np.zeros(shape,bool);outer=np.zeros(shape,bool);paper=np.zeros(shape,bool);guides=GUIDES.get(ident,[])
  for g in guides:
   dist,along,length=field(shape,g['points']);# Discard end caps: a reviewed segment is not a true whole stroke.
   longitudinal=(along>3)&(along<length-3);ink=(dist<=g['inner'])&longitudinal;tolerance=dist<=g['outer'];inner|=ink;outer|=tolerance;paper|=(dist>g['outer'])&(dist<=g['outer']+g['paper_margin'])&longitudinal;target[ink]=g['role']
  paper&=~outer
  # Explicit mask for occluding brush; never learn its shaft as mountain ink.
  occlusion=np.zeros(shape,'uint8')
  if ident.startswith('L_'):cv2.fillPoly(occlusion,[np.array([[64,220],[94,233],[239,298],[239,340],[86,249]],'int32')],1)
  # Conflict is excluded for review, not resolved by overwriting older positives.
  conflict=paper&np.isin(oldtarget,[1,2]);paper[conflict]=False;target[paper]=0;target[occlusion>0]=255;paper[occlusion>0]=False;inner[occlusion>0]=False
  refined=oldtarget.copy();take=(target!=255)&(oldtarget==255);refined[take]=target[take]
  p=ROOT/'references'/(ident+'.npz');np.savez_compressed(p,image=image,core_targets=oldtarget,band_targets=target,refined_targets=refined,ink_core=inner,outer_tolerance=outer,near_paper=paper,occlusion=occlusion)
  record=dict(id=ident,path=str(p),sha256=sha(p),split=r['split'],source_family=r['source_family'],source_url=r['source_url'],source_reference_sha256=r['sha256'],guides=guides,band_counts={str(k):int((target==k).sum()) for k in range(3)},new_training_pixel_counts={str(k):int((take&(target==k)).sum()) for k in range(3)},paper_conflicts_ignored=int(conflict.sum()),assistant_reviewed=True,expert_verified=False,automatic_prediction_as_label=False,role_reference_checked=True,role_mask_path=str(p),reference_kind='manual native-pixel guide and conservative ink/paper interval; edge transition and end caps ignored',whole_rock_mask_complete=False,complete_stroke_identity=False,endpoint_direction_or_pressure_verified=False);records.append(record)
  colours={0:np.array([90,150,90]),1:np.array([192,128,40]),2:np.array([40,135,180])};overlay=image.copy();mask=np.full_like(image,[218,212,199])
  for k,c in colours.items():overlay[target==k]=(.35*image[target==k]+.65*c).astype('uint8');mask[target==k]=c
  for name,im in [('input',image),('overlay',overlay),('target',mask)]:Image.fromarray(im).save(ROOT/'references'/(ident+'_'+name+'.png'))
  board=Image.new('RGB',(1200,460),'#f4f1e8');d=ImageDraw.Draw(board)
  for j,(im,title) in enumerate([(image,'原始静态上下文'),(overlay,'确定墨迹与两侧纸面'),(mask,'边缘 / 端点 / 遮挡留未知')]):
   pi=Image.fromarray(im);pi.thumbnail((380,355));board.paste(pi,(j*400+8,40));d.text((j*400+8,8),title,font=font(21),fill='#243e32')
  d.text((8,405),ident+'：本图为助手复核的区间参考，不是完整墨带真值或模型输出。',font=font(20),fill='#815238');board.save(ROOT/'references'/(ident+'_review.jpg'),quality=95)
 dataset=dict(records=records,previous_dataset_sha256=sha(OLD/'dataset_manifest.json'),train_works=2,reviewed_training_band_works=2,diagnostic_source_works=1,diagnostic_not_blind=True,diagnostic_artist_identity_verified=False,unknown_value=255,manual_intervals_not_exact_ink_mask=True,automatic_model_predictions_used_as_labels=False,no_original_image_threshold_as_target=True,complete_crossing_or_endpoint_gold=False,robot_connected=False)
 write('dataset_manifest.json',dataset);print('BAND_REFERENCES',[(r['id'],r['band_counts'],r['paper_conflicts_ignored']) for r in records],flush=True)
if __name__=='__main__':main()
