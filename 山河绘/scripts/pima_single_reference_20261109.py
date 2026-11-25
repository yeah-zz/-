# -*- coding: utf-8 -*-
"""Curate motion-assisted candidate reference curves from isolated real events.

The guide only constrains reference construction; it is never a model input.
This is assistant-reviewed supervision with measured image support, not expert
ground truth or sensor-verified brush pressure/physical contact.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from pima_stroke_inspect_20261109 import ROOT
import cv2,numpy as np,json,hashlib
from PIL import Image,ImageDraw,ImageFont
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def resample(q,spacing=1.5):
 lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))];t=np.arange(0,lengths[-1],spacing)
 t=np.r_[t,lengths[-1]];return np.column_stack([np.interp(t,lengths,q[:,k]) for k in range(2)]).astype(np.float32)
def fit_reference(delta,anchors):
 a=np.array(anchors,np.float32);ys=np.arange(round(a[0,1]),round(a[-1,1])+1);guide=np.interp(ys,a[:,1],a[:,0]);offsets=np.arange(-9,10);xs=np.round(guide[:,None]+offsets).astype(int)
 evidence=delta[ys[:,None],xs];cost=-np.log(.025+.975*evidence)+.015*offsets[None,:]**2
 accumulated=cost[0].copy();back=[]
 for i in range(1,len(ys)):
  movement=xs[i,:,None]-xs[i-1,None,:];transition=.20*movement**2;transition[abs(movement)>3]=1e5
  candidate=transition+accumulated[None,:];parent=candidate.argmin(1);accumulated=cost[i]+candidate[np.arange(len(offsets)),parent];back.append(parent)
 k=int(accumulated.argmin());path=[k]
 for parent in back[::-1]:k=int(parent[k]);path.append(k)
 path=path[::-1];xx=xs[np.arange(len(ys)),path].astype(np.float32)
 # Subpixel centre from local *new* ink, retaining the complete dry-brush gaps.
 for i,y in enumerate(ys):
  x=int(xx[i]);positions=np.arange(x-1,x+2);weights=delta[y,positions]+.01;xx[i]=float((positions*weights).sum()/weights.sum())
 line=np.column_stack([xx,ys]).astype(np.float32);strength=delta[ys,np.round(xx).astype(int)];positive=np.flatnonzero(strength>.13)
 if not len(positive):return line,dict(admissible=False,reason='no new-ink support')
 first,last=int(positive[0]),int(positive[-1]);line=line[first:last+1];strength=strength[first:last+1]
 stats=dict(support_fraction=float((strength>.1).mean()),start_trim_pixels=first,end_trim_pixels=int(len(ys)-1-last),mean_new_ink=float(strength.mean()),length_pixels=float(np.linalg.norm(np.diff(line,axis=0),axis=1).sum()))
 stats['admissible']=stats['support_fraction']>.65 and first<12 and len(ys)-1-last<12
 return resample(line),stats
def main():
 source=json.loads((ROOT/'single_candidates.json').read_text('utf8'));records=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
 board=Image.new('RGB',(1200,370*len(source)),'#f1eee6');d=ImageDraw.Draw(board)
 for row,r in enumerate(source):
  z=np.load(ROOT/'single_reference'/(r['id']+'.npz'));q,quality=fit_reference(z['delta'],r['anchors']);status='admitted_candidate' if quality['admissible'] and r['id'] not in ['S04','S05'] else 'uncertain_excluded'
  if r['id']=='S04':quality['reason']='Lower endpoint crosses an older stroke and the constructed path jumps; identity/endpoint uncertain'
  if r['id']=='S05':quality['reason']='Lift/restart and weak upper path are not sufficiently resolved; excluded'
  if status!='admitted_candidate':quality['admissible']=False
  # Keep before/after full images for provenance; model sees an event-local crop.
  x0=max(0,int(q[:,0].min())-25);x1=min(z['delta'].shape[1],int(q[:,0].max())+26);y0=max(0,int(q[:,1].min())-25);y1=min(z['delta'].shape[0],int(q[:,1].max())+26)
  box=[x0,y0,x1,y1];qq=q-np.array([x0,y0],np.float32);delta=z['delta'][y0:y1,x0:x1]
  gray=cv2.cvtColor(z['after'][y0:y1,x0:x1],cv2.COLOR_RGB2GRAY).astype(np.float32)/255
  width=[];tone=[];conf=[]
  for x,y in q:
   yi=int(round(y));xi=int(round(x));profile=z['delta'][yi,max(0,xi-4):xi+5];signal=float(profile.max());support=np.flatnonzero(profile>max(.1,signal*.35))
   width.append(float(len(support)));tone.append(signal);conf.append(signal>.16)
  np.savez_compressed(ROOT/'single_reference'/(r['id']+'_reference.npz'),curve=qq,delta=delta,gray=gray,width=np.array(width),tone=np.array(tone),appearance_valid=np.array(conf))
  r=dict(r,status=status,quality=quality,event_crop=box,reference_curve=q.tolist(),assistant_reviewed=True,expert_verified=False,
   reference_kind='motion_guided_new_ink_candidate',guide_used_only_to_construct_reference=True,physical_pressure=None,
   completeness_claim='visually single continuous motion within +/-0.2s candidate bounds' if status=='admitted_candidate' else 'unknown')
  records.append(r)
  images=[]
  for name in ['before','after']:images.append(Image.fromarray(z[name]).crop((275,195,415,470)))
  overlay=Image.fromarray(z['after']);dr=ImageDraw.Draw(overlay);dr.line([tuple(p) for p in q],fill='#0085bd',width=2);images.append(overlay.crop((275,195,415,470)))
  alpha=np.zeros(z['delta'].shape,np.uint8);cv2.polylines(alpha,[np.round(q).astype(np.int32)],False,255,1);images.append(Image.fromarray(255-alpha).convert('RGB').crop((275,195,415,470)))
  for col,im in enumerate(images):
   im.thumbnail((260,310));xx=col*300+40;yy=row*370+37;board.paste(im,(xx,yy));d.text((col*300+8,row*370+4),[r['id']+' 前帧','后帧','新增墨迹归属参考','完整曲线候选'][col],font=font,fill='#243e37')
  d.text((8,row*370+343),f'{status} / 墨迹支持 {quality["support_fraction"]:.1%} / {quality["length_pixels"]:.1f}px',font=font,fill='#555')
 board.save(ROOT/'reference_curves.jpg',quality=95)
 write(ROOT/'single_reference_manifest.json',dict(records=records,independent_test_works=0,source_families=1,robot_connected=False,
  excluded_other_windows=['37_arc_01','37_arc_02','36_long_01','36_long_02','36_split_01','36_split_02'],
  exclusion_reason='ambiguous lift/contact, original multi-contact windows, or unresolved tip detection'))
 print([(r['id'],r['status'],r['quality']) for r in records],flush=True)
if __name__=='__main__':main()
