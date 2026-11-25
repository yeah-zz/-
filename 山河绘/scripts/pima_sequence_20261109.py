"""Build observed *groups* from real video; no invented individual stroke truth.

Short-window temporal bright quantiles suppress moving brushes/hands. Rigid
registration uses the unchanged neighbouring rock. All masks remain weak
temporal observations and carry uncertainty, never artist path/pressure truth.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,json,hashlib
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'pima_demonstration_20261109'
CONFIG={
 36:dict(times=list(range(440,761,20)),box=[390,65,625,300],polygon=[[18,220],[25,160],[55,95],[105,40],[139,15],[172,35],[202,70],[148,88],[106,143],[71,190],[56,214]],anchor=[30,200,350,450],split='train'),
 37:dict(times=list(range(100,441,20)),reference_time=300,box=[270,115,500,470],polygon=[[14,330],[24,243],[45,158],[61,85],[93,22],[116,6],[145,35],[176,93],[186,142],[160,197],[119,250],[104,293],[79,320]],anchor=[560,260,775,440],split='same_work_development'),
}
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sample(cap,t):
 cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000);ok,im=cap.read()
 if not ok:raise RuntimeError(f'Missing frame {t}')
 return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
def register(im,reference,anchor):
 x0,y0,x1,y1=anchor
 a=cv2.GaussianBlur(reference[y0:y1,x0:x1],(3,3),0).astype(np.float32)/255
 b=cv2.GaussianBlur(im[y0:y1,x0:x1],(3,3),0).astype(np.float32)/255
 warp=np.eye(2,3,dtype=np.float32)
 try:
  score,warp=cv2.findTransformECC(a,b,warp,cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,80,1e-5),None,3)
 except cv2.error:return im,0.,[0.,0.]
 if np.linalg.norm(warp[:,2])>12:return im,0.,warp[:,2].tolist()
 aligned=cv2.warpAffine(im,warp,(im.shape[1],im.shape[0]),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
 return aligned,float(score),warp[:,2].tolist()
def ink(im):
 bg=cv2.GaussianBlur(im.astype(np.float32),(0,0),7)
 return np.clip((bg-im.astype(np.float32)-2)/65,0,1)
def main():
 folder=ROOT/'sequence';folder.mkdir(exist_ok=True);records=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
 for lesson,cfg in CONFIG.items():
  cap=cv2.VideoCapture(str(ROOT/'sources'/str(lesson)/'video.mp4'));reference=sample(cap,cfg.get('reference_time',cfg['times'][-1]));snapshots=[]
  box=cfg['box'];x0,y0,x1,y1=box;roi=np.zeros((y1-y0,x1-x0),np.uint8);cv2.fillPoly(roi,[np.array(cfg['polygon'],np.int32)],1)
  for t in cfg['times']:
   local=folder/f'{lesson}_{t:04d}.png';stats=[]
   if local.exists():clean=np.array(Image.open(local))
   else:
    stack=[]
    for dt in np.linspace(0,9,16):
     frame=sample(cap,t+dt);aligned,score,shift=register(frame,reference,cfg['anchor']);stats.append(dict(score=score,translation=shift))
     if score>.85:stack.append(aligned)
    if len(stack)<8:print('REJECT_SNAPSHOT',lesson,t,len(stack),flush=True);continue
    # It is an observed window, not a single instantaneous artist state.
    clean=np.percentile(np.stack(stack),90,axis=0).astype(np.uint8);Image.fromarray(clean).save(local)
   snapshots.append(dict(time=t,gray=clean[y0:y1,x0:x1],path=str(local),registration=stats))
  cap.release()
  if not snapshots:raise RuntimeError(f'No admissible snapshots in lesson {lesson}')
  board=Image.new('RGB',(1200,290*((len(snapshots)+4)//5)),'#eeeae1');d=ImageDraw.Draw(board)
  for i,s in enumerate(snapshots):
   im=Image.fromarray(s['gray']).convert('RGB');im.thumbnail((230,250));x=(i%5)*240;y=(i//5)*290;board.paste(im,(x,y+30));d.text((x+4,y+3),f'{lesson}课 {s["time"]}–{s["time"]+9}s',font=font,fill='#222')
  board.save(ROOT/f'clean_sequence_{lesson}.jpg',quality=95)
  for a,b in zip(snapshots,snapshots[1:]):
   before=ink(a['gray']);after=ink(b['gray']);delta=np.maximum(after-before,0)*roi
   # Prevent jitter around old dark lines from becoming a new stroke target.
   old=cv2.dilate((before>.23).astype(np.uint8),np.ones((3,3),np.uint8))
   reliable=(roi>0)&(old==0);target=delta*reliable
   name=f'{lesson}_{a["time"]:04d}_{b["time"]:04d}';np.savez_compressed(folder/(name+'.npz'),before=before,after=after,target=target,valid=reliable,roi=roi,raw=b['gray'])
   records.append(dict(id=name,lesson=lesson,source_group='liupengkai_qiushan_same_painting',split=cfg['split'],
       time_windows=[[a['time'],a['time']+9],[b['time'],b['time']+9]],box=box,file=str(folder/(name+'.npz')),
       observed_order_scope='between windows only; individual strokes unknown',label_kind='automatic_temporal_increment_weak',
       assistant_content_review=False,expert_paths=None,individual_stroke_count=None,pressure=None,
       phase_candidate='outline' if (lesson==36 and b['time']<=480) or (lesson==37 and b['time']<=140) else 'cun_candidate',
       new_ink_pixels=int((target>.2).sum()),valid_fraction=float(reliable.mean())))
 save(ROOT/'sequence_manifest.json',dict(records=records,independent_test_sources=0,not_expert_gold=True,robot_connected=False))
 print('SEQUENCES',len(records),flush=True)
if __name__=='__main__':main()
