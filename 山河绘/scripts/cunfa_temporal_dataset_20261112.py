# -*- coding: utf-8 -*-
"""Conservative temporal positive cores, not full stroke ownership or expert gold."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_temporal_roles_20261112';SINGLE=LAB/'pima_single_stroke_20261109'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def register(before,after,mask=None):
 a=cv2.cvtColor(before,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(after,cv2.COLOR_RGB2GRAY);sift=cv2.SIFT_create(nfeatures=8000,contrastThreshold=.006)
 ka,da=sift.detectAndCompute(a,mask);kb,db=sift.detectAndCompute(b,mask);pairs=cv2.BFMatcher().knnMatch(da,db,k=2);good=[m for m,n in pairs if m.distance<.68*n.distance]
 if len(good)<15:raise RuntimeError('Insufficient same-camera reference matches')
 p=np.float32([ka[m.queryIdx].pt for m in good]);q=np.float32([kb[m.trainIdx].pt for m in good]);matrix,valid=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=1.2,maxIters=5000)
 if matrix is None:raise RuntimeError('Registration failed')
 inside=valid[:,0].astype(bool);errors=np.linalg.norm(cv2.transform(p[:,None,:],matrix)[:,0]-q,axis=1);rms=float(np.sqrt((errors[inside]**2).mean()));assert inside.sum()>=15 and rms<1.0,(inside.sum(),rms)
 warped=cv2.warpAffine(before,matrix,(after.shape[1],after.shape[0]),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
 return warped,dict(matrix=matrix.tolist(),inliers=int(inside.sum()),rms_px=rms,matched_pairs=len(good))
def main():
 assert not (ROOT/'role_probe_results.json').exists(),'Do not change targets after role model evaluation.'
 (ROOT/'references').mkdir(exist_ok=True);(ROOT/'previews').mkdir(exist_ok=True);frames=json.loads((ROOT/'frame_manifest.json').read_text('utf8'));records=[];font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 def frame(t):return np.array(Image.open(next(r['path'] for r in frames['records'] if r['t']==t)).convert('RGB'))
 def keep(ident,before,after,box,role,reference,roi_positive=None,background_boxes=(),split='train',family='mixiaolu_short_pima',metadata=None,structure_reference=None):
  a=cv2.cvtColor(before,cv2.COLOR_RGB2GRAY).astype('float32');b=cv2.cvtColor(after,cv2.COLOR_RGB2GRAY).astype('float32');delta=a-b
  # Robust global brightness offset. It is not a camera-cut ink difference.
  offset=float(np.median(delta));delta-=offset
  local=cv2.GaussianBlur(b,(0,0),4)-b
  positive=(delta>12)&(local>3)
  if roi_positive is not None:positive&=roi_positive
  # Isolated video compression changes are ignored, never converted to class0.
  n,lab,stats,_=cv2.connectedComponentsWithStats(positive.astype('uint8'),8);valid=np.zeros_like(positive)
  for i in range(1,n):
   if stats[i,cv2.CC_STAT_AREA]>=3:valid|=lab==i
  target=np.full(a.shape,255,'uint8');target[valid]=role
  near_old_ink=cv2.dilate((cv2.GaussianBlur(a,(0,0),4)-a>8).astype('uint8'),np.ones((3,3),'uint8'))>0
  if role==2:target[near_old_ink]=255
  if structure_reference is not None:
   structure_supported=structure_reference&(local>3)
   near_new=cv2.dilate((target==2).astype('uint8'),np.ones((3,3),'uint8'))>0
   target[structure_supported&~near_new]=1
   target[structure_reference&near_new]=255
  near_visible_ink=cv2.dilate((local>5).astype('uint8'),np.ones((5,5),'uint8'))>0
  for x0,y0,x1,y1 in background_boxes:
   region=np.zeros_like(valid);region[y0:y1,x0:x1]=True;target[region&(abs(delta)<3)&(abs(local)<5)&~near_visible_ink]=0
  x0,y0,x1,y1=box;target=target[y0:y1,x0:x1];image=after[y0:y1,x0:x1];orig_before=before[y0:y1,x0:x1];maskpos=target==role
  assert maskpos.sum()>=20,(ident,int(maskpos.sum()))
  path=ROOT/'references'/(ident+'.npz');np.savez_compressed(path,image=image,targets=target,before=orig_before,delta=delta[y0:y1,x0:x1])
  Image.fromarray(image).save(ROOT/'references'/(ident+'_after.png'));Image.fromarray(orig_before).save(ROOT/'references'/(ident+'_before.png'))
  overlay=image.copy();colours={0:np.array([90,150,90]),1:np.array([192,128,40]),2:np.array([40,135,180])}
  for k,c in colours.items():overlay[target==k]=(.5*overlay[target==k]+.5*c).astype('uint8')
  Image.fromarray(overlay).save(ROOT/'references'/(ident+'_cores.png'));mask=np.full_like(image,245);mask[target==255]=[218,212,199]
  for k,c in colours.items():mask[target==k]=c
  Image.fromarray(mask).save(ROOT/'references'/(ident+'_targets.png'))
  record=dict(id=ident,path=str(path),role_mask_path=str(path),sha256=sha(path),split=split,source_family=family,role=role,reference_kind=reference,role_reference_checked=True,
   confirmed_partial_pixel_counts={str(k):int((target==k).sum()) for k in [0,1,2]},unknown_pixels=int((target==255).sum()),brightness_offset_gray=offset,
   source_based_partial_reference=True,assistant_reviewed=True,expert_verified=False,whole_box_is_not_labelled=True,complete_instance_ownership=False,style_target=None,wash_target=None,pressure=None,stroke_order=None,**(metadata or {}))
  records.append(record)
  board=Image.new('RGB',(1200,370),'#f4f1e8');d=ImageDraw.Draw(board)
  for j,(im,title) in enumerate([(orig_before,'前帧/对齐参考'),(image,'模型静态输入'),(overlay,'新增墨迹核心参考'),(mask,'只监督有证据像素')]):
   p=Image.fromarray(im);p.thumbnail((285,285));board.paste(p,(j*300+5,40));d.text((j*300+5,5),title,font=font(19),fill='#243e32')
  d.text((8,326),ident+'：灰色为待定；彩色核心是视频辅助参考，并非完整笔画真值。',font=font(18),fill='#8f4933');board.save(ROOT/'previews'/(ident+'.jpg'),quality=94)
 # Structure evidence stays within the first camera view, from4s to49s.
 before=frame(4);after=frame(49);mask=np.zeros(before.shape[:2],'uint8');mask[55:440,90:710]=255;mask[180:440,90:350]=0;mask[245:440,325:710]=0
 aligned,reg=register(before,after,mask);roi=np.zeros(before.shape[:2],bool);roi[245:440,325:675]=True
 keep('T_structure',aligned,after,[310,225,680,450],1,'same_camera_initial_structure_phase_new_ink_core',roi,[(315,228,390,245),(650,229,675,250)],metadata=dict(before_seconds=4,after_seconds=49,registration=reg,individual_contacts_verified=False,phase_role_basis='Visible initial drawing consists of rock contours and plane turns; not a named local cun label.'))
 # The second camera view begins around50s. Never subtract across that cut.
 # Geometry transfer is separately checked against unchanged early structure.
 first=np.load(ROOT/'references/T_structure.npz');full_mask=np.zeros(frame(49).shape[:2],'uint8');full_mask[225:450,310:680]=(first['targets']==1)
 _,crossreg=register(frame(49),frame(51.6));matrix=np.asarray(crossreg['matrix']);zoom=frame(51.6)
 transferred=cv2.warpAffine(full_mask,matrix,(zoom.shape[1],zoom.shape[0]),flags=cv2.INTER_NEAREST)>0
 gz=cv2.cvtColor(zoom,cv2.COLOR_RGB2GRAY).astype('float32');supported=cv2.dilate((cv2.GaussianBlur(gz,(0,0),4)-gz>3).astype('uint8'),np.ones((3,3),'uint8'))>0
 support_fraction=float(supported[transferred].mean());assert support_fraction>.70,support_fraction
 transferred &= supported
 save('structure_transfer.json',dict(**crossreg,visible_ink_support=support_fraction,geometry_transfer_only=True,cross_cut_pixel_difference=False,original_structure_scene='T_structure',model_receives_no_registration_or_reference=True))
 for ident,a,b,positivebox in [('T_cun_group1',51.6,54.5,[290,285,465,400]),('T_cun_group2',54.5,60,[290,250,510,410])]:
  before=frame(a);after=frame(b);aligned,reg=register(before,after);roi=np.zeros(before.shape[:2],bool);x0,y0,x1,y1=positivebox;roi[y0:y1,x0:x1]=True
  keep(ident,aligned,after,[245,210,550,440],2,'same_camera_visible_hatching_group_new_ink_core',roi,[(250,214,290,260),(488,213,535,235)],metadata=dict(before_seconds=a,after_seconds=b,registration=reg,individual_contacts_verified=False,structure_reference_transfer=crossreg,phase_role_basis='Visible repeated new hatch marks within one identified rock surface; group-level reference only.'),structure_reference=transferred)
 # A different source work is reserved for an exploratory role test. Original
 # motion-guided curves constrain reference construction only, never model input.
 old=json.loads((SINGLE/'single_reference_manifest.json').read_text('utf8'))
 for r in old['records']:
  if r['status']!='admitted_candidate':continue
  z=np.load(SINGLE/'single_reference'/(r['id']+'.npz'));before=z['before'];after=z['after'];q=np.array(r['reference_curve']);roi=np.zeros(before.shape[:2],'uint8');cv2.polylines(roi,[np.round(q).astype('int32')],False,1,5)
  x0=max(0,int(q[:,0].min())-35);x1=min(before.shape[1],int(q[:,0].max())+36);y0=max(0,int(q[:,1].min())-22);y1=min(before.shape[0],int(q[:,1].max())+23)
  keep('V_'+r['id'],before,after,[x0,y0,x1,y1],2,'previously_reviewed_motion_plus_local_new_ink_core',roi.astype(bool),[(x0+2,y0+2,x0+22,y0+25),(x1-22,y0+2,x1-2,y0+25)],split='exploratory_other_work_test',family='liupengkai_qiushan_same_work',metadata=dict(before_seconds=r['before'],after_seconds=r['after'],registration=None,existing_reference_reused=True,geometry_guide_not_model_input=True,individual_mask_is_not_gold=True,source_url=r['source_page']))
 dataset=dict(records=records,role_names=['confirmed_paper_reference','structure_core_reference','cun_core_reference'],source_url=frames['source_url'],train_original_scenes=3,train_source_works=1,test_source_works=1,expert_pixel_gold=False,unlabelled_is_not_background=True,raw_video_decode='sequential frame-index sampling, no random-seek fragments',camera_cut_delta_forbidden=True,
  style_recognition_not_trained=True,wash_reference_available=False,independent_full_role_test_available=False,test_work_seen_in_prior_project=True,robot_connected=False)
 save('dataset_manifest.json',dataset);print('DATASET_READY',[(r['id'],r['confirmed_partial_pixel_counts']) for r in records],flush=True)
if __name__=='__main__':main()
