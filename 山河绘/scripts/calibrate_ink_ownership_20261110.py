# -*- coding: utf-8 -*-
"""Mask threshold chosen on procedural development only, not real pictures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_ink_ownership_20261110 import *
from evaluate_ink_ownership_20261110 import decode,iou
def main():
 torch.set_num_threads(4);base=old_model();net=load_decoder();scores={x:[] for x in [.25,.35,.5,.65]}
 for seed in [881201,881202,881203,881204]:
  r=synthetic(seed);image=torch.tensor(r['image'].transpose(2,0,1)[None],device='cuda');aa=[]
  for s in r['strokes']:
   x,y=np.clip(np.floor(s['curve'][POINTS//2]/SIZE*GRID).astype(int),0,GRID-1);aa.append([0,int(x),int(y)])
  with torch.no_grad():old=base(image)
  masks,_,_,_=decode(net,old,image,torch.tensor(aa,device='cuda'));truth=[s['mask']>.5 for s in r['strokes']]
  for threshold in scores:scores[threshold].append(float(np.mean([iou(m>threshold,t) for m,t in zip(masks,truth)])))
 means={x:float(np.mean(y)) for x,y in scores.items()};selected=max(means,key=means.get);write(OUT/'mask_calibration_hr.json',dict(selected=selected,mean_development_oracle_iou=means,seeds=[881201,881202,881203,881204],real_images_used=False,final_test_seeds=[990401,990416],selection_target='full-resolution component mask IoU with correct anchors; diagnostic, not end-to-end accuracy'))
 print('CALIBRATION',selected,means,flush=True)
if __name__=='__main__':main()
