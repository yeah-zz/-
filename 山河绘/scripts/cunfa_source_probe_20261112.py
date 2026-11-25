# -*- coding: utf-8 -*-
"""Small directly named reference probe. No claim of general cunfa understanding."""
import sys,json,hashlib,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,joblib
from PIL import Image
from torchvision.models import mobilenet_v3_small
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from gongbi_training.cunfa_supervision import style_target,STYLE_NAMES
from cunfa_context_probe_20261111 import preprocess,WEIGHTS
from train_cunfa_gabor_20261110 import metric
ROOT=LAB/'cunfa_role_reference_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 torch.set_num_threads(4);data=json.loads((ROOT/'source_references.json').read_text('utf8'));rows=data['records'];y=np.array([style_target(r) for r in rows]);assert len(rows)==8 and np.all(y>=0)
 protocol=dict(records=[r['id'] for r in rows],data_sha256=sha(ROOT/'source_references.json'),code_sha256=sha(__file__),weights_sha256=sha(WEIGHTS),C=1.,kernel='linear',
  head_training_only=True,encoder_frozen=True,validation='leave reference source group out: AAM/HKMA/NPM, 3 groups',no_hyperparameter_search=True,per_pixel_role_targets=0,
  status='tiny teaching-example probe, not formal or independent new-artwork validation',robot_connected=False)
 if (ROOT/'source_probe_protocol.json').exists():assert json.loads((ROOT/'source_probe_protocol.json').read_text('utf8'))==protocol
 else:save('source_probe_protocol.json',protocol)
 net=mobilenet_v3_small(weights=None);net.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True));net=net.features.cuda().eval();tensor=torch.from_numpy(np.stack([preprocess(Image.open(r['image'])) for r in rows])).cuda()
 for p in net.parameters():p.requires_grad_(False)
 with torch.inference_mode():x=torch.nn.functional.normalize(net(tensor).mean((2,3)),dim=1).cpu().numpy()
 groups=np.array([r['source_group'] for r in rows]);pred=np.full(8,-1);folds=[]
 for tr,te in LeaveOneGroupOut().split(x,y,groups):
  assert {rows[i]['source_sha256'] for i in tr}.isdisjoint({rows[i]['source_sha256'] for i in te})
  svm=SVC(C=1,kernel='linear',class_weight='balanced');svm.fit(x[tr],y[tr]);pred[te]=svm.predict(x[te]);folds.append(dict(held_source_group=str(groups[te][0]),train_ids=[rows[i]['id'] for i in tr],test_ids=[rows[i]['id'] for i in te],metrics=metric(y[te],pred[te])))
 svm=SVC(C=1,kernel='linear',class_weight='balanced');svm.fit(x,y);dest=ROOT/'source_example_head.joblib';joblib.dump(dict(model=svm,classes=STYLE_NAMES,train_ids=[r['id'] for r in rows],weights_sha256=sha(WEIGHTS),formal_validation=False,scope='directly named teaching illustrations/local regions only',production_use=False),dest)
 loaded=joblib.load(dest);assert np.array_equal(loaded['model'].predict(x),svm.predict(x))
 result=dict(original_examples=8,classes={c:int(sum(y==i)) for i,c in enumerate(STYLE_NAMES)},publishers=3,source_groups=3,metrics=metric(y,pred),folds=folds,
  predictions=[dict(id=r['id'],source_label=r['local_style'],predicted=STYLE_NAMES[int(pred[i])],publisher=r['publisher']) for i,r in enumerate(rows)],
  expert_pixel_gold=0,confirmed_roles_trained=0,complete_strokes_trained=0,old_fresh_test_retrained_or_rescored=False,role_candidate_boxes_not_targets=True,
  original_views_not_augmented_sample_counts=True,formal_validation_passed=False,adopted=False,robot_connected=False)
 save('source_probe_results.json',result);print('PROBE',json.dumps(result['metrics']),flush=True)
if __name__=='__main__':main()
