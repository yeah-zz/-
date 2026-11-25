# -*- coding: utf-8 -*-
"""Full-resolution mask refinement with unchanged image encoder and proposals."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_ink_ownership_20261110 import *
def main():
 torch.manual_seed(11102);torch.set_num_threads(4);base=old_model();net=InkOwnershipDecoderHR().cuda();net.load_state_dict(torch.load(OUT/'ink_decoder.pt',map_location='cpu',weights_only=False)['model'],strict=False);opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001);real=real_example();start=time.time();history=[]
 for step in range(1,401):
  use_real=step%8==0;r=augment(real,4111000+step) if use_real else synthetic(411100000+step);f,t,v,k,p,dist,old,image,tt=make_batch(base,r,step+9000,full_resolution=True);detail=image.expand(len(f),-1,-1,-1);net.train();logits,q=net(f,detail);loss,parts=ownership_loss(logits,t,v,k,q,p);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
  if step%100==0:history.append(dict(step=step,parts={a:float(b.detach()) for a,b in parts.items()}));print('HR',history[-1],flush=True)
 torch.save(dict(model={k:v.detach().cpu() for k,v in net.state_dict().items()},high_resolution=True,mask_output_size=SIZE,base_frozen=True),OUT/'ink_decoder_hr.pt');write(OUT/'detail_training.json',dict(steps=400,new_procedural_scenes=350,real_augmented_views=50,original_real_modules=1,output_size=SIZE,history=history,seconds=time.time()-start,base_encoder_and_curves_frozen=True,original_pixels_used=True,reference_curves_not_input=True,robot_connected=False));print('HR_DONE',time.time()-start,flush=True)
if __name__=='__main__':main()
