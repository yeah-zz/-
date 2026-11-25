import sys,json,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import torch,numpy as np,joblib
from torchvision.models import mobilenet_v3_small
ROOT=LAB/'cunfa_resolution_context_20261116';OLD=LAB/'cunfa_fresh_validation_20261111';spec=importlib.util.spec_from_file_location('old_input',OLD/'frozen/encoder_and_preprocessing.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(4);rows=[r for r in json.loads((OLD/'test_manifest.json').read_text('utf8'))['records'] if r['admission']=='source_label_test'];arrays=[]
for r in rows:
 a,b,_=m.input_views(r);arrays.extend([m.preprocess(a),m.preprocess(b)])
net=mobilenet_v3_small(weights=None);net.load_state_dict(torch.load(m.WEIGHTS,map_location='cpu',weights_only=True));net=net.features.cuda().eval();features=[]
with torch.inference_mode():
 for i in range(0,len(arrays),12):
  z=net(torch.from_numpy(np.stack(arrays[i:i+12])).cuda()).mean((2,3));features.append(torch.nn.functional.normalize(z,dim=1).cpu().numpy())
z=np.concatenate(features);x=np.concatenate([z[::2],z[1::2]],axis=1)/np.sqrt(2);old=np.load(OLD/'features.npz')['local_plus_context'];head=joblib.load(OLD/'frozen/context_head_exploratory.joblib')['model'];oldpred=json.loads((OLD/'predictions.json').read_text('utf8'))['records'];CLASSES=['披麻皴','斧劈皴'];out=[]
for i,r in enumerate(rows):out.append(dict(id=r['id'],max_abs=float(np.abs(x[i]-old[i]).max()),cosine=float(np.dot(x[i],old[i])/(np.linalg.norm(x[i])*np.linalg.norm(old[i]))),old_prediction=oldpred[i]['predictions']['local_plus_context']['predicted'],reproduced_prediction=CLASSES[int(head.predict(x[i:i+1])[0])]))
(ROOT/'baseline_reproduction_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');np.savez_compressed(ROOT/'reproduced_baseline_features.npz',features=x);print(json.dumps(out,ensure_ascii=False),flush=True)
