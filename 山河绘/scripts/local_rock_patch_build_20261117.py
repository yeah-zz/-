# -*- coding: utf-8 -*-
import sys,json,copy,hashlib,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.local_rock_patch import fit_existing_path,resample,contrast_field,sample_field
from source_stroke_width_20261104 import width_profile
from hierarchical_paint_core_20261105 import order_metrics
ROOT=LAB/'local_rock_patch_20261117'
LABELS=['主轮廓／共享边待定','内部结构转折','内部纹理候选（皴法未定）']
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def plain(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x) for x in v]
 return v
def centre_image(plan,scale=3):
 a=np.full((496*scale,496*scale,3),255,np.uint8)
 for r in plan['curves']:cv2.polylines(a,[np.rint(np.asarray(r['points'])*scale).astype('int32')],False,(30,40,35),max(1,round(.55*scale)),cv2.LINE_AA)
 return Image.fromarray(a)
def band_image(plan,profiles,scale=3):
 a=np.full((496*scale,496*scale,3),255,np.uint8)
 for i,r in enumerate(plan['curves']):
  if i not in profiles:
   cv2.polylines(a,[np.rint(np.asarray(r['points'])*scale).astype('int32')],False,(30,40,35),max(1,round(.55*scale)),cv2.LINE_AA);continue
  pr=profiles[i];q=np.array(pr['points']);ww=np.array(pr['width_px']);seen=np.array(pr['observed']);tone=np.array(pr.get('relative_contrast',np.ones(len(q))*.2))
  # Only measured cross-sections use variable width. Unknowns remain the thin baseline.
  width=np.where(seen,np.clip(ww,.55,2.8),.55)
  for k in range(len(q)-1):
   c=int(45 if seen[k] else 95);cv2.line(a,tuple(np.rint(q[k]*scale).astype(int)),tuple(np.rint(q[k+1]*scale).astype(int)),(c,c,c),max(1,round(float((width[k]+width[k+1])/2*scale))),cv2.LINE_AA)
 return Image.fromarray(a)
def crop(im,box,scale=1,size=(642,294)):
 return im.crop(tuple(round(v*scale) for v in box)).resize(size,Image.Resampling.LANCZOS)
def main():
 lock=read(ROOT/'baseline_lock.json');assert not (ROOT/'metrics.json').exists();old=read(lock['working489']);approved=read(lock['approved500']);rgb=np.array(Image.open(lock['source']).convert('RGB'));candidate=copy.deepcopy(old);ids=lock['selected_curve_ids'];protected={1,39,169,384}
 assert sha(lock['working489'])==lock['working489_sha256'] and sha(lock['approved500'])==lock['approved500_sha256']
 roles={1:(0,'outer_contour'),39:(0,'front_boundary'),169:(0,'shared_boundary_pending'),384:(0,'shared_boundary_pending'),15:(1,'mixed_bottom_outline_and_structure'),137:(1,'plane_turn_candidate'),187:(1,'plane_turn_candidate'),347:(1,'plane_turn_candidate')}
 for i in ids:
  if i not in roles:roles[i]=(2,'interior_texture_unknown_method')
 review=dict(source_sha256=lock['source_sha256'],reviewer='assistant_visual_object_review',expert_confirmed=False,not_training_truth=True,object_id=4,roles=[dict(curve=i,old_phase=old['curves'][i]['phase'],old_role=old['curves'][i].get('stroke_role'),phase=roles[i][0],role=roles[i][1]) for i in ids],notes=['39为前侧边界，先于内部纹理','15连续路径包含底边和内部转折，整条保留，不拆成短笔','169和384靠近相邻小石，共享归属待定，保留原几何','内部短纹理未识别具体皴法，不因长度短而删除'])
 write(ROOT/'local_role_review.json',review);fits={};profiles={}
 for i in ids:
  r=candidate['curves'][i];p=np.asarray(r['points']);phase,role=roles[i]
  if i not in protected:
   q,diag=fit_existing_path(rgb,p);fits[i]=diag;r['points']=q.tolist()
  else:fits[i]=dict(used=False,reason='protected outline or shared-edge pending; exact geometry kept')
  r.update(phase=phase,phase_label=LABELS[phase],stroke_role=role,role_verified=False,local_role_reviewed_by_assistant=True)
  profiles[i]=plain(width_profile(rgb,r['points'],recenter=False))
 # Replace only this object's existing contiguous action block; others stay exact.
 slots=[j for j,a in enumerate(old['actions']) if a['module']==4];assert slots==list(range(slots[0],slots[-1]+1))
 priority={i:j for j,i in enumerate([1,39,169,384,15,137,187,347])};ordered=sorted([copy.deepcopy(old['actions'][j]) for j in slots],key=lambda a:(roles[a['curve']][0],priority.get(a['curve'],100+a['sequence'])))
 for slot,act in zip(slots,ordered):
  act.update(sequence=slot+1,phase=roles[act['curve']][0],length_px=float(np.linalg.norm(np.diff(np.asarray(candidate['curves'][act['curve']]['points']),axis=0),axis=1).sum()));candidate['actions'][slot]=act
 candidate['modules'][4]['local_phase_labels']=LABELS;candidate['modules'][4]['name']='前景中部主石 · 本轮局部候选';candidate.update(id='G03_01_local_candidate',parent=lock['working489'],parent_sha256=lock['working489_sha256'],adopted=False,ready_for_robot=False,new_model_weights_trained=False,local_patch_only=True,pressure_mm=None)
 candidate['stats']=copy.deepcopy(old['stats']);candidate['stats'].update(before={'count':489},after={'count':489},accepted_continuations=0,baseline_order=order_metrics(old['actions']),hierarchy_order=order_metrics(candidate['actions']))
 nonselected=[i for i in range(489) if i not in ids];assert all(candidate['curves'][i]==old['curves'][i] for i in nonselected);assert all(candidate['actions'][j]==old['actions'][j] for j in range(489) if j not in slots)
 assert all(np.array_equal(candidate['curves'][i]['points'],old['curves'][i]['points']) for i in protected)
 assert all(np.array_equal(np.array(candidate['curves'][i]['points'])[[0,-1]],np.array(old['curves'][i]['points'])[[0,-1]]) for i in ids)
 metrics=dict(approved_visual_baseline_strokes=500,working_object_baseline_strokes=489,candidate_strokes=489,selected_paths=21,other468_curves_exactly_unchanged=True,other468_actions_exactly_unchanged=True,protected4_paths_exactly_unchanged=True,all_selected_endpoints_unchanged=True,
  fitted_paths=sum(v.get('used',False) for v in fits.values()),max_fitted_normal_shift_px=max(v.get('max_normal_shift',0) for v in fits.values()),local_phase_counts=[sum(roles[i][0]==p for i in ids) for p in range(3)],
  old_local_phase_counts=[sum(old['curves'][i]['phase']==p for i in ids) for p in range(3)],module_reentries=order_metrics(candidate['actions'])['module_reentries'],line_count_not_quality_metric=True,source_resolution=[496,496],role_review_is_assistant_only=True,unknown_texture_method=True,physical_width_calibration=False,new_neural_training=False,adopted=False,robot_connected=False)
 write(ROOT/'fits.json',plain(fits));write(ROOT/'profiles.json',dict(profiles={str(k):v for k,v in profiles.items()},source_pixel_width_not_pressure=True,unknown_cross_sections_use_baseline=True));write(ROOT/'metrics.json',metrics)
 scenes=[('G03_01_approved500',approved,500),('G03_01_baseline489',old,489),('G03_01_local_candidate',candidate,489)];box=lock['source_box'];images={}
 for name,plan,n in scenes:
  folder=ROOT/name;folder.mkdir(exist_ok=True)
  if name=='G03_01_approved500':shutil.copyfile(lock['approved500'],folder/'plan.json')
  elif name=='G03_01_baseline489':shutil.copyfile(lock['working489'],folder/'plan.json')
  else:write(folder/'plan.json',plan)
  Image.fromarray(rgb).save(folder/'source.png');im=centre_image(plan);images[name]=im;im.resize((992,992)).save(folder/'comparison.jpg');im.resize((992,992)).save(folder/'sequence_comparison.jpg');crop(im,box,3).save(folder/'rock.jpg')
 measured=band_image(candidate,profiles);measured.save(ROOT/'G03_01_local_candidate/pavilion.jpg')
 board=Image.new('RGB',(1320,1070),'#f3f0e7');dr=ImageDraw.Draw(board);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 panels=[(crop(Image.fromarray(rgb),box),'原图 · 496像素来源的局部放大'),(crop(images['G03_01_approved500'],box,3),'已认可500笔整幅中的同一局部'),(crop(images['G03_01_baseline489'],box,3),'已有489笔对象基线 · 局部21条'),(crop(images['G03_01_local_candidate'],box,3),'原图约束拟合候选 · 局部仍21条'),(crop(measured,box,3),'原图可测截面的相对墨带目标 · 非实画')]
 # Last panel: keep uncertain texture distinct from reviewed structural candidates.
 role_im=np.full((1488,1488,3),255,np.uint8)
 colors=[(155,96,35),(68,126,65),(48,121,165)]
 for i in ids:cv2.polylines(role_im,[np.rint(np.array(candidate['curves'][i]['points'])*3).astype('int32')],False,colors[roles[i][0]],2,cv2.LINE_AA)
 panels.append((crop(Image.fromarray(role_im),box,3),'助手阶段：轮廓→结构→纹理候选'))
 for j,(im,title) in enumerate(panels):
  x=j%2*660+8;y=j//2*350+8;dr.text((x,y),title,font=font(20),fill='#243f35');board.paste(im,(x,y+36))
 board.save(ROOT/'G03_01_local_candidate/rock.jpg',quality=94)
 Image.fromarray(role_im).save(ROOT/'G03_01_local_candidate/tree.jpg')
 write(ROOT/'catalog.json',dict(scenes=[dict(id=n,label={'G03_01_approved500':'认可整幅基线 · 原500笔','G03_01_baseline489':'已有对象基线 · 489笔','G03_01_local_candidate':'单块主石局部候选 · 489笔'}[n]) for n,_,_ in scenes],default='G03_01_local_candidate',robot_connected=False));write(ROOT/'execution_binding.json',dict(scene=None,journal=None,plan_sha256=None))
 assert sha(lock['working489'])==lock['working489_sha256'] and sha(lock['approved500'])==lock['approved500_sha256'];print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
