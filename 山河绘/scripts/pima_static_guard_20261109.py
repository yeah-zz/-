# -*- coding: utf-8 -*-
"""Image-supported correction of neural proposals; RAW failures remain visible.

This is a hybrid inference guard, not evidence that the neural network learned
the correction. New works used to diagnose it are development observations.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from pima_static_model_20261109 import ROOT,StaticTracer,unit,patch_at
from pima_static_discover_20261109 import proposals,evaluate_reference
import numpy as np,cv2,json,torch,shutil,time
from PIL import Image,ImageDraw,ImageFont
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sample(field,points):
 h,w=field.shape;x=points[...,0];y=points[...,1];ok=(x>=0)&(y>=0)&(x<w-1)&(y<h-1);xx=np.clip(x,0,w-1.001);yy=np.clip(y,0,h-1.001);x0=np.floor(xx).astype(int);y0=np.floor(yy).astype(int);ax=xx-x0;ay=yy-y0
 return ((1-ax)*(1-ay)*field[y0,x0]+ax*(1-ay)*field[y0,x0+1]+(1-ax)*ay*field[y0+1,x0]+ax*ay*field[y0+1,x0+1])*ok
def guarded(net,ims,seeds,max_steps=400):
 paths=[[np.array(s['point'],np.float32)] for s in seeds];dirs=np.array([unit(np.array(s['direction'],np.float32)) for s in seeds],np.float32);active=list(range(len(seeds)));reasons=['iteration_limit']*len(seeds);gap=np.zeros(len(seeds),int);field=ims[0]
 offsets=np.linspace(-1.05,1.05,25);dist=np.array([1.5,3,5,8,11],np.float32);weights=np.array([.35,.25,.20,.12,.08]);h,w=field.shape
 for step in range(max_steps):
  if not active:break
  patches=np.stack([patch_at(ims,paths[k][-1]) for k in active])
  with torch.no_grad():d,s,_,r,e=net(torch.tensor(patches,device='cuda'),torch.tensor(dirs[active],device='cuda'));d=d.cpu().numpy();stop=s.sigmoid().cpu().numpy();rem=r.cpu().numpy();end=e.cpu().numpy()
  remaining=[]
  for j,k in enumerate(active):
   point=paths[k][-1];base=np.arctan2(dirs[k,1],dirs[k,0]);angles=base+offsets;vec=np.column_stack([np.cos(angles),np.sin(angles)]);positions=point[None,None,:]+vec[:,None,:]*dist[None,:,None];signal=sample(field,positions);support=signal@weights
   prior=vec@d[j];score=support+.055*prior-.025*np.abs(offsets);winner=int(score.argmax());best=float(support[winner])
   # A blank source cannot acquire a line merely because the network continues.
   if best<.105:
    reasons[k]='source_endpoint_or_unresolved_gap';continue
   competitive=np.flatnonzero((support>max(.18,best-.025))&(np.abs(offsets-offsets[winner])>.60))
   if len(competitive) and best<.75:
    reasons[k]='crossing_or_parallel_ambiguity';continue
   # Accept a predicted terminal correction only when the source also ends.
   if step>=3 and stop[j]>.55 and rem[j]<.18 and np.max(signal[winner,2:])<.14:
    terminal=point+end[j]
    if np.linalg.norm(end[j])<6 and sample(field,terminal[None])[0]>.08:paths[k].append(terminal)
    reasons[k]='learned_endpoint_supported';continue
   if signal[winner,0]<.08:
    gap[k]+=1
    if gap[k]>5 or np.max(signal[winner,2:])<.2:reasons[k]='unresolved_dry_gap';continue
   else:gap[k]=0
   direction=vec[winner].astype(np.float32);nxt=point+direction*1.5
   # Subpixel ridge centring uses actual local image evidence, never target paths.
   normal=np.array([-direction[1],direction[0]]);trip=nxt+np.array([-1.,0.,1.])[:,None]*normal;vals=sample(field,trip);den=vals[0]-2*vals[1]+vals[2]
   shift=float(np.clip(.5*(vals[0]-vals[2])/den,-.55,.55)) if den<-.01 else 0.;nxt=nxt+normal*shift
   if min(nxt)<1 or nxt[0]>=w-1 or nxt[1]>=h-1:reasons[k]='image_boundary';continue
   if len(paths[k])>12 and np.linalg.norm(nxt-paths[k][-10])<2:reasons[k]='loop_uncertain';continue
   dirs[k]=unit(nxt-point);paths[k].append(nxt);remaining.append(k)
  active=remaining
 return [dict(curve=np.array(q,np.float32),stop_reason=reason) for q,reason in zip(paths,reasons)]
def main():
 torch.set_num_threads(4);ck=torch.load(ROOT/'static_tracer.pt',map_location='cpu',weights_only=False);net=StaticTracer().cuda();net.load_state_dict(ck['model']);net.eval();refs=json.loads((ROOT/'external_visible_curve_references.json').read_text('utf8'));old=json.loads((ROOT/'static_discovery_report.json').read_text('utf8'));results=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
 # Keep the first external run unchanged; disclose role change for all revisions.
 original=ROOT/'initial_external_evaluation';original.mkdir(exist_ok=True)
 for p in [ROOT/'static_discovery_report.json',*ROOT.glob('*_proposals.jpg')]:
  if not (original/p.name).exists():shutil.copy2(p,original/p.name)
 for image in old['images']:
  name=image['image'];rgb=np.array(Image.open(ROOT/'external'/name).convert('RGB'));ims,sk,jnear,seeds,stats=proposals(rgb);out=guarded(net,ims,seeds);curves=[];seen=np.zeros(sk.shape,np.uint8)
  for i,o in enumerate(out):
   q=o['curve'];xy=np.round(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,sk.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,sk.shape[0]-1);length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum());support=float((sample(ims[0],q)>.1).mean());duplicate=float((seen[xy[:,1],xy[:,0]]>0).mean())>.85
   if duplicate:status='overlap_duplicate_candidate'
   elif length<10:status='short_candidate'
   elif o['stop_reason']=='crossing_or_parallel_ambiguity':status='uncertain_crossing'
   else:status='image_supported_candidate'
   if not duplicate and length>=10:cv2.polylines(seen,[xy],False,1,3)
   curves.append(dict(id=f'{name}_guarded_{i:03d}',curve=q.tolist(),seed=seeds[i],stop_reason=o['stop_reason'],status=status,length_px=length,ink_support_fraction=support))
  matches=[evaluate_reference(curves,r) for r in refs if r['image']==name]
  from pima_static_discover_20261109 import resample_reference
  for match,ref in zip(matches,[r for r in refs if r['image']==name]):
   q=resample_reference(ref['points']);nearest=np.full(len(q),np.inf);touching=0
   for c in curves:
    if c['status'] in ['short_candidate','overlap_duplicate_candidate']:continue
    a=np.array(c['curve']);dist=np.linalg.norm(q[:,None]-a[None,:],axis=2).min(1);nearest=np.minimum(nearest,dist)
    if int((dist<3).sum())>=8:touching+=1
   match['all_candidates_combined_coverage_within_3px']=float((nearest<=3).mean());match['candidate_pieces_touching_reference']=touching
  results.append(dict(image=name,stats=stats,candidates=curves,reference_matches=matches,manual_seed_given=False,reference_curve_given=False))
  over=Image.fromarray(rgb);d=ImageDraw.Draw(over)
  for i,c in enumerate(curves):
   if c['status'] in ['short_candidate','overlap_duplicate_candidate']:continue
   d.line([tuple(p) for p in c['curve']],fill=['#107b95','#996623','#388055','#875aa0'][i%4],width=2)
  board=Image.new('RGB',(rgb.shape[1]*2,rgb.shape[0]+44),'#f1eee6');board.paste(Image.fromarray(rgb),(0,44));board.paste(over,(rgb.shape[1],44));dd=ImageDraw.Draw(board);dd.text((8,6),'原始画面',font=font,fill='#243e37');dd.text((rgb.shape[1]+8,6),'墨迹约束后的候选（开发对照，非纯模型输出）',font=font,fill='#243e37');board.save(ROOT/(Path(name).stem+'_guarded.jpg'),quality=95)
  print('GUARDED',name,matches,flush=True)
 blank=np.zeros((2,100,100),np.float32);b=guarded(net,blank,[dict(point=[50,50],direction=[1,0])])[0];assert len(b['curve'])==1 and b['stop_reason']=='source_endpoint_or_unresolved_gap'
 save(ROOT/'guarded_discovery_report.json',dict(images=results,blank_source_cannot_generate_line_verified=True,
  hybrid_image_constraint=True,neural_raw_outputs_preserved=True,external_used_for_algorithm_diagnosis=True,external_is_now_development_not_blind_test=True,
  no_new_training_on_external=True,adopted=False,robot_connected=False))
 protocol=json.loads((ROOT/'protocol.json').read_text('utf8'));protocol.update(external_used_for_algorithm_revision=True,external_final_role='external development observations; initial frozen-weight run archived separately');save(ROOT/'protocol.json',protocol)
if __name__=='__main__':main()
