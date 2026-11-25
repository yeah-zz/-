# -*- coding: utf-8 -*-
"""Register final appearance to early annotations, with explicit overlap limits."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pima_rock_reference_20261109 import ROOT,LAB
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
def main():
 f=json.loads((ROOT/'frames.json').read_text('utf8'))['frames'];z=min(f,key=lambda k:abs(k['t']-306.5));rgb=np.array(Image.open(ROOT/'frames'/z['file']).convert('RGB'));Image.fromarray(rgb).save(ROOT/'finished_frame.jpg')
 early=np.array(Image.open(ROOT/'references'/'group_after.png').convert('RGB'));a=cv2.cvtColor(early,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
 sift=cv2.SIFT_create(nfeatures=5000);ka,da=sift.detectAndCompute(a,None);kb,db=sift.detectAndCompute(b,None);matches=cv2.BFMatcher().knnMatch(da,db,k=2);good=[m for m,n in matches if m.distance<.65*n.distance]
 p=np.float32([ka[m.queryIdx].pt for m in good]);q=np.float32([kb[m.trainIdx].pt for m in good]);matrix,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=5000)
 if matrix is None:raise RuntimeError('Final-state registration failed; no annotation transfer allowed')
 err=np.linalg.norm(cv2.transform(p[:,None,:],matrix)[:,0]-q,axis=1);mask=inliers[:,0].astype(bool);rms=float(np.sqrt((err[mask]**2).mean()))
 if mask.sum()<20 or rms>1.5:raise RuntimeError(f'Unreliable registration {mask.sum()} {rms}')
 manifest=json.loads((ROOT/'reference_manifest.json').read_text('utf8'));new=[];overlay=Image.fromarray(rgb);d=ImageDraw.Draw(overlay)
 for r in manifest['records']:
  curve=cv2.transform(np.array(r['reference_curve'],np.float32)[:,None,:],matrix)[:,0];new.append(dict(id=r['id'],curve=curve.tolist(),status=r['status'],final_instance_ownership_verified=False))
  d.line([tuple(v) for v in curve],fill='#2689a7' if r['status']=='reviewed_candidate' else '#ba8e58',width=2)
 overlay.save(ROOT/'final_transferred_reference.jpg',quality=96)
 (ROOT/'final_state_registration.json').write_text(json.dumps(dict(final_time_seconds=z['t'],matrix=matrix.tolist(),ransac_inliers=int(mask.sum()),median_inlier_error_px=float(np.median(err[mask])),inlier_rms_px=rms,
  records=new,transferred_geometry_only=True,individual_final_ink_ownership_verified=False,later_overpainting_present=True,
  eligible_as_expert_final_instance_gold=False,source_artist_verified=False,robot_connected=False),ensure_ascii=False,indent=2),encoding='utf8')
 print('FINAL_REGISTRATION',z['t'],int(mask.sum()),rms,flush=True)
if __name__=='__main__':main()
