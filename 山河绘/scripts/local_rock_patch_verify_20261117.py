# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2
from scipy.spatial import cKDTree
from gongbi_training.local_rock_patch import resample
ROOT=LAB/'local_rock_patch_20261117'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def render(plan,width=.8):
 count=np.zeros((1488,1488),np.uint16)
 for r in plan['curves']:
  m=np.zeros(count.shape,np.uint8);cv2.polylines(m,[np.rint(np.asarray(r['points'])*3).astype('int32')],False,1,max(1,round(width*3)));count+=m
 return count
def main():
 lock=read(ROOT/'baseline_lock.json');old=read(lock['working489']);new=read(ROOT/'G03_01_local_candidate/plan.json');ids=set(lock['selected_curve_ids']);outside=[i for i in range(489) if i not in ids];slots=[j for j,a in enumerate(old['actions']) if a['module']==4]
 checks=dict(approved500_original_hash_unchanged=sha(lock['approved500'])==lock['approved500_sha256'],working489_original_hash_unchanged=sha(lock['working489'])==lock['working489_sha256'],approved500_copy_byte_exact=sha(ROOT/'G03_01_approved500/plan.json')==lock['approved500_sha256'],working489_copy_byte_exact=sha(ROOT/'G03_01_baseline489/plan.json')==lock['working489_sha256'],curve_count489_unchanged=len(new['curves'])==489,
  other468_curve_records_unchanged=all(old['curves'][i]==new['curves'][i] for i in outside),other468_actions_unchanged=all(a==b for j,(a,b) in enumerate(zip(old['actions'],new['actions'])) if j not in slots),each_curve_once=sorted(a['curve'] for a in new['actions'])==list(range(489)),local_phases_monotone=[new['actions'][j]['phase'] for j in slots]==sorted(new['actions'][j]['phase'] for j in slots),
  endpoints_unchanged=all(np.array_equal(np.array(old['curves'][i]['points'])[[0,-1]],np.array(new['curves'][i]['points'])[[0,-1]]) for i in ids),no_robot_ready=new['ready_for_robot']==False,no_weights_trained=new['new_model_weights_trained']==False)
 before=np.concatenate([resample(old['curves'][i]['points'],.2) for i in ids]);after=np.concatenate([resample(new['curves'][i]['points'],.2) for i in ids]);distance=cKDTree(after).query(before)[0];checks['sample_deviation_under_point8px']=float(distance.max())<.8
 rb=render(old);rn=render(new);box=lock['source_box'];mask=np.ones(rb.shape,dtype=bool);mask[box[1]*3:box[3]*3,box[0]*3:box[2]*3]=False;checks['outside_crop_raster_exact']=np.array_equal(rb[mask],rn[mask]);assert all(checks.values())
 diag=dict(engineering_checks=checks,max_baseline_sample_distance_px=float(distance.max()),constant_width_overlap_proxy_before=float((rb>1).sum()/max(1,(rb>0).sum())),constant_width_overlap_proxy_after=float((rn>1).sum()/max(1,(rn>0).sum())),artist_quality_verified=False,cun_method_recognized=False,physical_width_validated=False,geometry_visually_large_improvement=False,adopted=False,robot_connected=False)
 (ROOT/'verification.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2),encoding='utf8');(ROOT/'decision.json').write_text(json.dumps(dict(geometry_candidate='research_only',local_role_order='assistant_reviewed_example_not_model_recognition',width_targets='risk_preview_only_not_adopted',production_replaced=False,approved500_retained=True,reason='Small bounded image fitting; remaining small loops/fragmented texture and uncalibrated finite-width risk do not pass artistic or physical validation'),indent=2),encoding='utf8');print(json.dumps(diag),flush=True)
if __name__=='__main__':main()
