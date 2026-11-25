# -*- coding: utf-8 -*-
"""Resume evaluation of already frozen weights; never repeat training."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
from cunfa_roles_v2_train_20261112 import Net,features,conf,metrics
from cunfa_band_checked_train_20261112 import Refiner
from cunfa_band_train_20261112 import ink_metrics,widths
from gongbi_training.cunfa_probability_checks import conditional_role_check
ROOT=LAB/'cunfa_retention_v4_20261112';PREV=LAB/'cunfa_band_checked_20261112';TEACHER=LAB/'cunfa_roles_v2_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def main():
 assert not (ROOT/'results.json').exists();p=read(ROOT/'protocol.json');freeze=read(ROOT/'checkpoint_freeze.json');assert sha(PREV/'dataset_manifest.json')==p['reference_dataset_sha256'];assert sha(TEACHER/'flat_roles.pt')==p['teacher_sha256'];assert sha(PREV/'foreground_refiner.pt')==p['initial_refiner_sha256'];torch.set_num_threads(4)
 teacher=Net(False).cuda();teacher.load_state_dict(torch.load(TEACHER/'flat_roles.pt',map_location='cuda',weights_only=False)['model']);teacher.eval();models={}
 for name in p['arms']+['previous_refiner']:
  path=PREV/'foreground_refiner.pt' if name=='previous_refiner' else ROOT/(name+'.pt')
  if name!='previous_refiner':assert sha(path)==freeze[name]
  model=Refiner().cuda();model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);model.eval();models[name]=model
 ds=read(PREV/'dataset_manifest.json');results=[]
 for r in ds['records']:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);x=torch.from_numpy(features(z['image']))[None].cuda();scores={};checks={}
  with torch.inference_mode():
   old=teacher.probabilities(x);probs={'v2_frozen':old}
   for name,net in models.items():probs[name]=net(x,old)[0]
  original_class=old.argmax(1)[0].cpu().numpy();prior_class=probs['previous_refiner'].argmax(1)[0].cpu().numpy();t=z['core_targets'];suppressed=(t!=255)&(t>0)&(original_class>0)&(prior_class==0)
  for name,prob in probs.items():
   check=conditional_role_check(old,prob);assert check['max_error']<1e-5;checks[name]=check;arr=prob[0].cpu().numpy();pred=arr.argmax(0).astype('uint8');np.savez_compressed(ROOT/'predictions'/(r['id']+'_'+name+'.npz'),prediction=pred,probability=arr)
   scores[name]=dict(core_roles=metrics(conf(t,pred)),band_ink=ink_metrics(z['band_targets'],pred>0),widths=widths(pred>0,r['guides']),previously_suppressed_ink_pixels=int(suppressed.sum()),recovered_previous_suppression=int((suppressed&(pred>0)).sum()),newly_suppressed_vs_previous=int(((t!=255)&(t>0)&(prior_class>0)&(pred==0)).sum()))
  results.append(dict(id=r['id'],split=r['split'],scores=scores,semantic_conditional_checks=checks))
 result=dict(protocol=p,checkpoint_hashes=freeze,records=results,diagnostic=next(r for r in results if r['split']!='train'),evaluation_script_sha256=sha(__file__),evaluation_resumed_without_retraining=True,evaluation_check_fix='Use float64 ratios at nonzero foreground, rather than a clamp which changes conditional probabilities at low foreground.',adopted=False,robot_connected=False)
 (ROOT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result['diagnostic'],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
