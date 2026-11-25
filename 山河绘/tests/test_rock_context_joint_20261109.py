# -*- coding: utf-8 -*-
"""Scientific protocol checks: unknown is ignored and context is actually global."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.rock_context_data import *
from gongbi_training.rock_context_model import *
import torch,numpy as np
def main():
 torch.manual_seed(11);torch.set_num_threads(4);r=real_example();t=targets(r);net=RockContextNet().cuda();x=torch.tensor(r['image'].transpose(2,0,1)[None],device='cuda',requires_grad=True);seen=[]
 hook=net.context.register_forward_pre_hook(lambda module,args:seen.append(tuple(args[0].shape)));o=net(x);hook.remove();loss,p=joint_loss(o,[t]);assert torch.isfinite(loss)
 # An ignored label must remain ignored, even if its prediction is extreme.
 ignore=torch.tensor(t['heat_valid']==0,device='cuda');changed=dict(o,anchor=o['anchor']+ignore[None,None]*9);_,pc=joint_loss(changed,[t]);assert torch.allclose(p['anchor'],pc['anchor'],atol=1e-6)
 mask_unknown=np.ones((GRID,GRID),bool)
 for a in t['anchors']:mask_unknown&=a['mask_valid']==0
 delta=torch.tensor(mask_unknown,device='cuda')[None,None]*7;changed=dict(o,pixel=o['pixel']+delta);_,pc=joint_loss(changed,[t]);assert torch.allclose(p['instance'],pc['instance'],atol=1e-6)
 a=next(a for a in t['anchors'] if a['oriented'] and (a['curve_valid']==0).any());j=int(np.flatnonzero(a['curve_valid']==0)[0]);curves=o['curves'].clone();curves[0,j*2:j*2+2,a['y'],a['x']]+=.3;_,pc=joint_loss(dict(o,curves=curves),[t]);assert torch.allclose(p['curve'],pc['curve'],atol=1e-6)
 # Verify distant image pixels can affect this prediction. This is an
 # architecture check, never evidence of semantic understanding.
 score=o['anchor'][0,0,64,64];gradient=torch.autograd.grad(score,x,retain_graph=True)[0].abs().sum(1)[0];outside=torch.ones((SIZE,SIZE),dtype=torch.bool,device='cuda');outside[112:144,112:144]=False;distant=float(gradient[outside].sum());assert distant>1e-8;assert seen==[(1,256,96)]
 counts=[len(synthetic(77100+i)['strokes']) for i in range(20)];assert min(counts)!=max(counts)
 assert (t['heat_valid']==0).sum()>1000;assert all(s['group']==-1 for s in r['strokes'] if s['role']==0)
 out=dict(passed=True,unknown_anchor_labels_ignored=True,unknown_instance_pixels_ignored=True,unknown_curve_points_ignored=True,whole_module_attention_token_shape=list(seen[0]),distant_input_gradient_sum=distant,global_context_path_not_semantic_accuracy=True,
  procedural_stroke_count_range=[min(counts),max(counts)],sparse_real_not_full_gold=True,artist_order_not_fabricated=True,robot_connected=False)
 write(ROOT/'protocol_checks.json',out);print('PASS',out,flush=True)
if __name__=='__main__':main()
