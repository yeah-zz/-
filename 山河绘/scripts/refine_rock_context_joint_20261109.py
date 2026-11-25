# -*- coding: utf-8 -*-
"""Continue joint training with a smooth learned curve decoder and anchor jitter."""
import sys,time,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.rock_context_data import *
from gongbi_training.rock_context_model import *
import torch,numpy as np
def main():
 archive=ROOT/'direct_points_attempt';archive.mkdir(exist_ok=True)
 for name in ['training_report.json','real_training_module_curves.png','liupengkai_other_work_curves.png']:
  if (ROOT/name).exists():shutil.copy2(ROOT/name,archive/name)
 torch.manual_seed(110915);torch.set_num_threads(4);ck=torch.load(ROOT/'whole_context_step1000.pt',map_location='cpu',weights_only=False);net=RockContextNet(project_curve=True).cuda();net.load_state_dict(ck['model']);opt=torch.optim.AdamW(net.parameters(),lr=.00025,weight_decay=.0001);real=real_example();start=time.time();history=[]
 for step in range(1,251):
  data=[synthetic(110915000+step*4+i) for i in range(2)]+[augment(real,110915+step*2+i) for i in range(2)];ts=[targets(r) for r in data];x=torch.tensor(np.stack([r['image'].transpose(2,0,1) for r in data]),device='cuda');net.train();out=net(x);loss,parts=joint_loss(out,ts);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  if step%50==0:
   history.append(dict(step=1000+step,loss_parts={k:float(v.detach()) for k,v in parts.items()}));write(ROOT/'refinement_progress.json',dict(history=history,seconds=time.time()-start));print('REFINE',history[-1],flush=True)
 counts=dict(ck['counts']);counts['procedural_examples']+=500;counts['real_augmented_views']+=500;torch.save(dict(model={k:v.detach().cpu() for k,v in net.state_dict().items()},step=1250,counts=counts,decoder='projected_spline_10_controls',training_source_works=['short_pima_work_only'],real_external_is_development_not_blind=True),ROOT/'whole_context_step1250.pt')
 original_seeds=set(range(110940,110944))
 for step in range(101,1001):original_seeds.update(110914000+step*8+i for i in range(4 if step<=600 else 3))
 refinement_seeds={110915000+step*4+i for step in range(1,251) for i in range(2)}
 report=json.loads((ROOT/'training_report.json').read_text('utf8'));report.update(counts=counts,selected_step=1250,procedural_unique_scenes=len(original_seeds|refinement_seeds),refinement_unique_new_scenes=len(refinement_seeds-original_seeds),refinement_history=history,refinement_seconds=time.time()-start,
  refinement_reason='Raw complete-curve predictions jittered and were unstable at nearby anchors. Differentiable cubic spline projection plus 3x3 valid-anchor neighbourhood supervision; not geometric stitching.',
  real_development_seen_for_method_revision=True,external_used_for_gradient_training=False,checkpoint_selection='fixed final budget, not best real-test checkpoint')
 write(ROOT/'training_report.json',report);print('REFINEMENT_FINISHED',time.time()-start,flush=True)
if __name__=='__main__':main()
