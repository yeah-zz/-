# -*- coding: utf-8 -*-
"""Three source groups; partial video-supported roles and distributed paper."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from cunfa_temporal_dataset_20261112 import register
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_roles_v2_20261112';OLD=LAB/'cunfa_temporal_roles_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def contrast(im):
 g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype('float32');return cv2.GaussianBlur(g,(0,0),4)-g
def corridor(shape,curves,width=4):
 m=np.zeros(shape,'uint8')
 for q in curves:cv2.polylines(m,[np.array(q,'int32')],False,1,width)
 return m>0
def main():
 assert not (ROOT/'results.json').exists(),'Frozen evaluation already exists.'
 (ROOT/'references').mkdir(exist_ok=True);frames=json.loads((ROOT/'frame_manifest.json').read_text('utf8'));records=[];font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 def frame(name,t):return np.array(Image.open(next(r['path'] for source in frames if source['id']==name for r in source['frames'] if r['t']==t)).convert('RGB'))
 def save(ident,image,before,target,paperboxes,split,family,meta):
  lc=contrast(image);oldlc=contrast(before);near=cv2.dilate(((lc>3)|(oldlc>3)).astype('uint8'),np.ones((7,7),'uint8'))>0
  rawdelta=cv2.cvtColor(before,cv2.COLOR_RGB2GRAY).astype('float32')-cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype('float32');rawdelta-=np.median(rawdelta)
  reviewed=np.zeros(target.shape,'uint8')
  for x0,y0,x1,y1 in paperboxes:reviewed[y0:y1,x0:x1]=1
  paper=(reviewed>0)&(abs(lc)<2.5)&(abs(oldlc)<2.5)&(abs(rawdelta)<6)&~near
  target[paper&(target==255)]=0
  assert (target==0).sum()>100 and (target==1).sum()>100 and (target==2).sum()>40,(ident,[(target==k).sum() for k in range(3)])
  p=ROOT/'references'/(ident+'.npz');np.savez_compressed(p,image=image,before=before,targets=target,paper_review_region=reviewed)
  col=np.array([[90,150,90],[192,128,40],[40,135,180]],'uint8');mask=np.full_like(image,[218,212,199]);overlay=image.copy()
  for k in range(3):mask[target==k]=col[k];overlay[target==k]=(.45*image[target==k]+.55*col[k]).astype('uint8')
  for suffix,im in [('input',image),('before',before),('target',mask),('overlay',overlay)]:Image.fromarray(im).save(ROOT/'references'/(ident+'_'+suffix+'.png'))
  rec=dict(id=ident,path=str(p),sha256=sha(p),split=split,source_family=family,reference_counts={str(k):int((target==k).sum()) for k in range(3)},unknown_pixels=int((target==255).sum()),paper_review_boxes=paperboxes,role_reference_checked=True,assistant_reviewed=True,expert_verified=False,complete_instance_ownership=False,automatic_predictions_as_label=False,model_input='static full context crop only',crop_not_derived_from_single_stroke=True,**meta);records.append(rec)
  board=Image.new('RGB',(1200,430),'#f4f1e8');d=ImageDraw.Draw(board)
  for j,(im,title) in enumerate([(before,'阶段前帧 / 对齐后'),(image,'静态上下文输入'),(overlay,'本轮局部参考'),(mask,'仅彩色核心参与监督')]):
   pimg=Image.fromarray(im);pimg.thumbnail((286,330));board.paste(pimg,(j*300+5,35));d.text((j*300+5,5),title,font=font(18),fill='#243e32')
  d.text((8,382),ident+'：参考依据来自视频；灰色未知。不是模型结果或专家完整分笔。',font=font(18),fill='#835039');board.save(ROOT/'references'/(ident+'_review.jpg'),quality=95)
 # Existing short-pima work remains development data. Add only reviewed paper
 # patches spread inside/outside rock faces, never turn all unknown into paper.
 old=json.loads((OLD/'dataset_manifest.json').read_text('utf8'))
 boxes=[(4,65,26,80),(30,92,48,111),(72,10,90,26),(137,15,155,28),(270,50,290,78),(170,104,184,116),(79,195,94,210),(245,185,270,200),(180,207,199,222)]
 for r in old['records']:
  if r['id'] not in ['T_cun_group1','T_cun_group2']:continue
  z=np.load(r['path']);save('M_'+r['id'][-1],z['image'],z['before'],z['targets'].copy(),boxes,'train','mixiaolu_short_pima',dict(source_url=old['source_url'],evidence='Previously reviewed same-camera new hatch cores and geometry-supported early structure; additional manually selected paper patches.',old_reference_sha256=r['sha256'],registration=r.get('registration'),structure_definition='visible outer contour and plane-turn marks'))
 # Liu 37: visible outer structure is identified at130s, then carried through a
 # same-camera registration. Newly added cun strokes are restricted to reviewed
 # motion corridors. Brush/hand and ambiguous crossings remain unknown.
 base=frame('liu37',130);curves=[[(290,435),(293,390),(302,325),(312,269),(326,215),(342,169)],[(369,160),(388,134),(402,145),(418,165),(432,195),(443,230),(451,267)],[(414,104),(439,83),(451,90),(467,114),(482,148),(492,185),(499,216)],[(329,453),(341,390),(352,335),(364,305),(377,295),(389,295),(397,313),(402,310),(415,318)]]
 structure=corridor(base.shape[:2],curves,5)&(contrast(base)>3)
 hatchcurves=[[(379,215),(361,236),(349,258),(337,288),(327,320),(316,355),(308,390),(302,425),(301,438)],[(369,216),(351,237),(341,260),(329,290),(319,325),(310,360),(303,395),(296,429)]]
 for seconds in [149.3,170]:
  image=frame('liu37',seconds);aligned,reg=register(base,image);h,w=image.shape[:2];m=np.asarray(reg['matrix']);st=cv2.warpAffine(structure.astype('uint8'),m,(w,h),flags=cv2.INTER_NEAREST)>0
  guide=cv2.warpAffine(corridor(base.shape[:2],hatchcurves,7).astype('uint8'),m,(w,h),flags=cv2.INTER_NEAREST)>0
  a=cv2.cvtColor(aligned,cv2.COLOR_RGB2GRAY).astype('float32');b=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype('float32');delta=a-b;delta-=np.median(delta);lc=contrast(image);new=(delta>10)&(lc>3)&guide
  # Explicitly exclude the brush shaft trajectory and hand in the lower right.
  yy,xx=np.indices((h,w));new&=~((xx>338)&(yy>282));nearold=cv2.dilate((contrast(aligned)>5).astype('uint8'),np.ones((3,3),'uint8'))>0;new&=~nearold
  target=np.full((h,w),255,'uint8');target[st&(lc>3)]=1;target[new]=2
  x0,y0,x1,y1=270,65,510,465;paper=[(5,12,28,90),(82,10,115,36),(135,70,155,95),(108,148,145,190),(68,280,78,308),(175,1,207,10),(6,255,18,280)]
  save('L_'+str(seconds).replace('.','_'),image[y0:y1,x0:x1],aligned[y0:y1,x0:x1],target[y0:y1,x0:x1],paper,'train','liupengkai_qiushan',dict(source_url=frames[0]['url'],before_seconds=130,after_seconds=seconds,registration=reg,crop_box=[x0,y0,x1,y1],evidence='Visible rock contours at130s; two subsequent long hatch motions. Guide constrains reference only.',formerly_test_work_now_explicit_development=True))
 # A third visually different work is held out from gradient updates. Uploader
 # identity is known; original painter identity is NOT established.
 image=frame('howto',14);before=frame('howto',6);aligned,reg=register(before,image);h,w=image.shape[:2]
 outlines=[[(370,191),(420,149),(468,109),(494,82),(519,64),(555,60),(575,50),(594,52),(614,59)],[(417,208),(439,191),(453,181),(462,183),(478,196),(494,198),(505,209)],[(511,200),(526,165),(536,143),(554,131),(578,117),(600,96),(627,70),(649,61),(668,64),(684,80),(691,102),(698,130)],[(576,181),(599,159),(618,141),(643,123),(670,115),(694,118)],[(607,198),(632,172),(646,166),(662,169),(674,169)]]
 hatch=[[(408,235),(400,246),(393,261),(389,275)],[(425,224),(416,237),(411,248)],[(442,219),(453,229),(463,247)],[(451,207),(466,212),(478,222),(487,240)],[(481,209),(495,219),(503,238)]]
 lc=contrast(image);st=corridor((h,w),outlines,5)&(lc>2.5);guide=corridor((h,w),hatch,6);a=cv2.cvtColor(aligned,cv2.COLOR_RGB2GRAY).astype('float32');b=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype('float32');delta=a-b;delta-=np.median(delta)
 new=(delta>9)&(lc>3)&guide;oldnear=cv2.dilate((contrast(aligned)>5).astype('uint8'),np.ones((3,3),'uint8'))>0;new&=~oldnear;target=np.full((h,w),255,'uint8');target[st]=1;target[new]=2
 x0,y0,x1,y1=300,35,710,300;paper=[(5,20,30,95),(50,70,72,90),(185,80,200,94),(285,58,310,73),(265,110,282,123),(216,205,244,225),(8,205,30,238),(225,1,245,10)]
 save('H_14',image[y0:y1,x0:x1],aligned[y0:y1,x0:x1],target[y0:y1,x0:x1],paper,'heldout_source_partial_reference','howto_repost_artist_unverified',dict(source_url=frames[1]['url'],before_seconds=6,after_seconds=14,registration=reg,crop_box=[x0,y0,x1,y1],artist_identity_verified=False,source_work_seen_in_prior_project=True,evidence='Initial contour stage6-8s, hatch stage9-14s. Static contour corridors reviewed in14s frame. New ink restricted to visible front-rock hatch corridors.',named_style_label_not_training_target=True))
 dataset=dict(records=records,train_source_works=2,heldout_source_works=1,heldout_is_new_blind=False,heldout_original_artist_verified=False,roles=['paper','structure','cun'],ignore=255,expert_gold=False,style_classification_trained=False,complete_stroke_trained=False,robot_connected=False,rule='All role references are assistant-reviewed partial video-supported cores. Unknown is ignored. Heldout data never become gradients or thresholds.')
 write('dataset_manifest.json',dataset);print('V2_REFERENCE_COUNTS',[(r['id'],r['reference_counts']) for r in records],flush=True)
if __name__=='__main__':main()
