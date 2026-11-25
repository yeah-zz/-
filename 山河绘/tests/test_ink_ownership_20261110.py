# -*- coding: utf-8 -*-
"""Ownership supervision must ignore unknowns and preserve real crossings."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_ink_ownership_20261110 import *
def main():
 torch.set_num_threads(4);torch.manual_seed(7)
 target=torch.zeros((2,SIZE,SIZE),device='cuda');target[0,63,20:220]=1;target[1,20:220,100]=1;known=target.max(0).values[None].expand_as(target);valid=torch.ones_like(target);valid[:,150:190,150:190]=0;logits=torch.zeros_like(target,requires_grad=True);q=torch.ones(2,device='cuda');present=torch.ones(2,device='cuda');loss,parts=ownership_loss(logits,target,valid,known,q,present);changed=logits+((valid==0)*100);loss2,_=ownership_loss(changed,target,valid,known,q,present);assert torch.allclose(loss,loss2,atol=1e-6)
 gradient=torch.autograd.grad(loss,logits)[0];assert gradient[0,63,100]<0 and gradient[1,63,100]<0;assert gradient[0,100,100]>0 and gradient[1,63,150]>0
 r=dict(image=np.ones((SIZE,SIZE,3),np.float32),paper=np.ones((SIZE,SIZE),np.uint8),strokes=[],complete=True,source='unit_one_pixel_line',synthetic=True)
 m=np.zeros((SIZE,SIZE),np.float32);m[63,20:220]=1;r['image'][m>0]=0;r['strokes']=[dict(id='thin',curve=sample_curve([[20,63],[219,63]]),role=1,group=0,oriented=True,mask=m,mask_valid=np.ones_like(m),curve_valid=np.ones(POINTS,np.float32))]
 base=old_model();f,t,v,k,p,*_=make_batch(base,r,7,max_positive=1,include_negatives=False,full_resolution=True);assert tuple(t.shape)==(1,SIZE,SIZE) and float(t.sum())==200;old=cv2.resize(m,(GRID,GRID),interpolation=cv2.INTER_NEAREST);assert old.sum()==0
 net=load_decoder();out,q=net(f,torch.tensor(r['image'].transpose(2,0,1)[None],device='cuda'));assert tuple(out.shape)==(1,SIZE,SIZE)
 write(OUT/'loss_checks.json',dict(passed=True,unknown_pixels_ignored=True,genuine_crossing_not_forced_apart=True,other_confirmed_stroke_is_nonownership_negative=True,one_pixel_target_preserved=True,old_nearest_downsample_loses_test_line=True,output_resolution=SIZE,robot_connected=False));print('PASS: unknowns, crossings, other-instance negatives, one-pixel full-resolution targets',flush=True)
if __name__=='__main__':main()
