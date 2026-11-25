# -*- coding: utf-8 -*-
"""Motion-guided, partial new-work references. Unknown overlap is ignored."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pima_rock_reference_20261109 import ROOT,LAB,frame,tip
from pima_single_reference_20261109 import resample
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
EVENTS=[('R01',52.286,52.419),('R02',52.452,52.486),('R03',52.553,52.619),('R04',52.653,52.719),('R05',52.753,52.920),('R06',53.186,53.287),('R07',53.353,53.553)]
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def fit(gray,before,guide):
 q=resample(guide,1.);t=np.gradient(q,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-6);normal=np.column_stack([-t[:,1],t[:,0]])
 offsets=np.arange(-5,6);coords=q[:,None,:]+normal[:,None,:]*offsets[None,:,None]
 x=coords[:,:,0].astype(np.float32);y=coords[:,:,1].astype(np.float32)
 delta=np.clip((before-gray-1)/35,0,1);ink=np.clip((cv2.GaussianBlur(gray,(0,0),6)-gray-1)/35,0,1)
 d=cv2.remap(delta,x,y,cv2.INTER_LINEAR);i=cv2.remap(ink,x,y,cv2.INTER_LINEAR)
 cost=-np.log(.02+.65*d+.33*i)+offsets[None,:]**2*.035;acc=cost[0].copy();back=[]
 for k in range(1,len(q)):
  trans=(offsets[:,None]-offsets[None,:])**2*.35;c=trans+acc[None,:];parent=c.argmin(1);acc=cost[k]+c[np.arange(len(offsets)),parent];back.append(parent)
 at=int(acc.argmin());path=[at]
 for b in back[::-1]:at=int(b[at]);path.append(at)
 path=path[::-1];curve=coords[np.arange(len(q)),path];evidence=d[np.arange(len(q)),path]
 # Do not invent endpoints in unsupported hovering intervals. At most 12px
 # may be trimmed, and all remaining low-evidence samples stay unknown.
 valid=np.flatnonzero(evidence>.08)
 if len(valid):lo,hi=int(valid[0]),int(valid[-1])+1
 else:lo,hi=0,len(curve)
 trimmed=curve[lo:hi];smoothed=cv2.GaussianBlur(trimmed[:,None,:],(1,9),1.6).reshape(-1,2);smoothed[0]=trimmed[0];smoothed[-1]=trimmed[-1];shift=smoothed-trimmed;smoothed=trimmed+shift*np.minimum(1,1.2/np.maximum(np.linalg.norm(shift,axis=1,keepdims=True),1e-6));trimmed=smoothed
 stats=dict(new_mark_support_fraction=float((evidence[lo:hi]>.08).mean()),start_trim_px=lo,end_trim_px=len(q)-hi,tip_guide_length_px=len(q),reference_smoothing_max_shift_px=1.2,curve_length_px=float(np.linalg.norm(np.diff(trimmed,axis=0),axis=1).sum()))
 return resample(trimmed),stats,delta
def main():
 out=ROOT/'references';out.mkdir(exist_ok=True);dense=json.loads((ROOT/'dense.json').read_text('utf8'));clean=frame(54.25)
 before=np.max(np.stack([frame(t) for t in [51.385,51.619,51.852]]),axis=0);after=np.max(np.stack([frame(t) for t in [54.054,54.154,54.254,54.354]]),axis=0)
 gray=cv2.cvtColor(after,cv2.COLOR_RGB2GRAY).astype(np.float32);bg=cv2.cvtColor(before,cv2.COLOR_RGB2GRAY).astype(np.float32)
 Image.fromarray(before).save(out/'group_before.png');Image.fromarray(after).save(out/'group_after.png');box=[290,285,460,395];records=[]
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(1200,230*len(EVENTS)),'#eeeae1');draw=ImageDraw.Draw(board)
 for idx,(key,a,b) in enumerate(EVENTS):
  observations=[z for z in dense if a-.001<=z['t']<=b+.001];guides=[]
  for z in observations:
   rgb=np.array(Image.open(ROOT/'dense'/z['file']));p=tip(rgb,clean)
   if p is not None and p[0]<460:guides.append(p)
  guide=np.array(guides,np.float32);q,quality,delta=fit(gray,bg,guide)
  # 1–3 frames do not resolve the start and lift of these very short contacts.
  enough=len(guides)>=4 and quality['curve_length_px']>=20 and quality['new_mark_support_fraction']>=.6
  status='reviewed_candidate' if enough else 'uncertain_excluded'
  r=dict(id=key,video_window_seconds=[a,b],observed_tip_samples=len(guides),tip_guide=guide.tolist(),reference_curve=q.tolist(),quality=quality,status=status,
   role='cun_stroke',rock_id='foreground_rock_R',group_id='R_downward_right_01',observed_order_within_group=idx+1,
   source_work='short_pima_1gc2lo9n27j5ZShwAja69iQ3N12d1c',source_artist_verified=False,style_label='uploader_title_short_pima_not_expert_verified',
   split='train' if enough else 'excluded',reference_kind='assistant_reviewed_motion_and_new_mark_candidate',expert_verified=False,
   endpoint_uncertainty_px=5,contact_sensor_verified=False,physical_pressure=None,physical_ink_volume=None,robot_speed=None,
   individual_mask_truth=False,mask_note='Shared before/after contains several strokes; overlapping corridors and old ink are unknown, not individual gold masks')
  records.append(r);xy=np.round(q).astype(int);corridor=np.zeros(gray.shape,np.uint8);cv2.polylines(corridor,[xy],False,1,9)
  x0,y0,x1,y1=box;owned=delta.copy();owned[corridor==0]=0
  np.savez_compressed(out/(key+'.npz'),curve=q,gray=gray/255,group_delta=delta,ownership_candidate=owned,unknown_overlap=np.zeros(gray.shape,np.uint8))
  overlay=Image.fromarray(after);dr=ImageDraw.Draw(overlay);dr.line([tuple(p) for p in q],fill='#1c7ca0' if enough else '#b47a47',width=2)
  panels=[Image.fromarray(before),Image.fromarray(after),overlay,Image.fromarray(np.uint8(255*(1-owned))).convert('RGB')]
  for col,im in enumerate(panels):
   im=im.crop((290,285,460,395)).resize((290,187));board.paste(im,(col*300,idx*230+26));draw.text((col*300+4,idx*230+3),[key+' 笔组前帧','笔组后帧','运动与墨迹曲线候选','墨迹归属候选·非真值'][col],font=font,fill='#243e37')
  draw.text((4,idx*230+212),f'{status} / {len(guides)} motion frames / new mark support {quality["new_mark_support_fraction"]:.0%}',font=font,fill='#555')
 # Samples closer than 7px cannot be confidently assigned from the shared delta.
 for r in records:
  z=dict(np.load(out/(r['id']+'.npz')));mask=z['ownership_candidate']>0;over=np.zeros(mask.shape,np.uint8)
  for other in records:
   if other['id']==r['id']:continue
   cv2.polylines(over,[np.round(other['reference_curve']).astype(np.int32)],False,1,7)
  unknown=(mask&(over>0));z['unknown_overlap']=unknown.astype(np.uint8);z['ownership_candidate'][unknown]=0;np.savez_compressed(out/(r['id']+'.npz'),**z)
  r['ownership_unknown_overlap_pixels']=int(unknown.sum())
 write(ROOT/'reference_manifest.json',dict(records=records,group_before_times=[51.385,51.619,51.852],group_after_times=[54.054,54.154,54.254,54.354],
  source_url='https://www.ganjingworld.com/video/1gc2lo9n27j5ZShwAja69iQ3N12d1c',source_sha256=hashlib.sha256((LAB/'pima_tutorial_search_20261109'/'short_pima.mp4').read_bytes()).hexdigest(),
  cameras_cut_at_about_seconds=50.1,comparison_across_cut=False,final_static_time=54.25,unlabelled_pixels_are_negative=False,
  frame_aggregation='per-channel maximum of listed frames; not a single raw frame; only local visible stroke regions are interpreted',
  rock_contour_stroke_order='unknown_across_camera_cut',complete_rock_reference=False,robot_connected=False))
 board.save(ROOT/'reference_review.jpg',quality=96);print([(r['id'],r['status'],r['quality']) for r in records],flush=True)
if __name__=='__main__':main()
