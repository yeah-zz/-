import sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import torch,numpy as np
from gongbi_training.scene_objects import SceneObjects,scene_metrics
from gongbi_training.scene_object_reference import sparse_scene_reference,legacy_object_to_scene
from gongbi_training.cunfa_supervision import partial_role_loss
from guohua_objects_train_20261112 import ContextObjects
from guohua_scene4_20261116 import ROOT,OLD,read,write,sha
assert not (ROOT/'results.json').exists();assert not (ROOT/'pretrain_checks.json').exists();original=read(ROOT/'references.json');p=read(ROOT/'protocol.json');write('references_pre_visual_review.json',original);write('protocol_pre_visual_review.json',p);refs=copy.deepcopy(original)
for c in refs['video_cases']:
 if c['id']=='L_149_3':
  r=next(r for r in c['regions'] if r['name']=='front_hatched_face');r.update(object='unknown',kind='uncertain_polygon',review_reason='Moving brush occludes this face; exclude before any training or predictions')
write('references.json',refs);p['reference_sha256']=sha(ROOT/'references.json');p['pretraining_review']='L_149_3/front_hatched_face changed to unknown after original-only review, before any training; original references preserved';p['new_reference_count']['video_known_regions']=11;write('protocol.json',p)
torch.set_num_threads(4);state=torch.load(OLD/'expanded_control.pt',map_location='cpu',weights_only=False)['model'];old=ContextObjects();old.load_state_dict(state);old.eval();new=SceneObjects(state);new.eval();x=torch.randn(1,3,128,128)
with torch.no_grad():a=old(x);b=new(x)
assert torch.allclose(a,b[:,[3,1,2]],atol=2e-5,rtol=2e-5);assert torch.allclose(b[:,0],torch.full_like(b[:,0],-2.));assert not any(v.requires_grad for v in new.encoder.parameters())
assert legacy_object_to_scene(np.array([[0,1,2,255]],'uint8')).tolist()==[[3,1,2,255]]
t=np.array([[1,2,3,255]],'uint8');pred=np.array([[1,2,3,0]],'uint8');m=scene_metrics(t,pred);assert m['known_foreground_mean_iou']==1 and m['support'][0]==0 and m['iou'][0] is None
z=torch.randn(1,4,4,4,requires_grad=True);tar=torch.full((1,4,4),255,dtype=torch.long);tar[:,1,1]=0;partial_role_loss(z,tar).backward();assert z.grad[0,:,tar[0]==255].count_nonzero()==0
for c in refs['video_cases']:
 scene,ink=sparse_scene_reference(c['size'],c['regions']);assert set(np.unique(scene))<={0,1,255} and np.all(ink==255)
check=dict(old_other_transfers_to_new_other=True,old_conditional_logits_preserved=True,new_open_logit_initial_minus2=True,encoder_frozen=True,legacy_background_not_invented_in_metrics=True,unknown_gradients_zero=True,occluded_region_quarantined_before_training=True,video_reference_known_regions=11,H_14_training=False,expert_truth=False,robot_connected=False);write('pretrain_checks.json',check);print(json.dumps(check),flush=True)
