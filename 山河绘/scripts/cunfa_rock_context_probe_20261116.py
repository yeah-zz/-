"""Secondary input ablation using the already-frozen larger rock-face windows."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,joblib
from PIL import Image
from torchvision.models import mobilenet_v3_small
from gongbi_training.cunfa_input_scale import preprocess
ROOT=LAB/'cunfa_input_scale_20261116';CONTEXT=LAB/'cunfa_resolution_context_20261116';OLD=LAB/'cunfa_fresh_validation_20261111';CLASSES=['披麻皴','斧劈皴']
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
assert not (ROOT/'context_probe_protocol.json').exists(),'Secondary diagnostic already frozen';rows=read(CONTEXT/'comparison_manifest.json')['records'];model=joblib.load(ROOT/'fixed_extent.joblib')['model'];p=read(ROOT/'protocol.json');write('context_probe_protocol.json',dict(frozen_before_secondary_predictions=True,model_sha256=sha(ROOT/'fixed_extent.joblib'),manifest_sha256=sha(CONTEXT/'comparison_manifest.json'),window_selection_frozen_before_scale_training=True,script_sha256=sha(__file__),arms=['standard_context','larger_rock_context'],same_classifier_no_fit=True,known_cases_only=True,not_autonomous_rock_instance_recovery=True,first_mixed_arm_diagnostic_failed_feature_reproduction_and_not_used=True))
torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);freeze=read(OLD/'freeze_manifest.json');path=next(r['path'] for r in freeze['files'] if r['path'].endswith('.pth'));net=mobilenet_v3_small(weights=None);net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));net=net.features.cuda().eval();testrows=[r for r in read(OLD/'test_manifest.json')['records'] if r['admission']=='source_label_test'];cached=np.load(ROOT/'fixed_extent_features.npz')['test'];out=[];audit=[]
for arm in ['standard_context','larger_rock_context']:
 arrays=[]
 for r in rows:
  im=Image.open(r['source_image']).convert('RGB');local=Image.open(r['image']).convert('RGB');x0,y0,x1,y1=r['source_box'];cx=(x0+x1)/2;cy=(y0+y1)/2;b=[max(0,int(cx-(x1-x0)*1.25)),max(0,int(cy-(y1-y0)*1.25)),min(im.width,int(cx+(x1-x0)*1.25)),min(im.height,int(cy+(y1-y0)*1.25))];context=im.crop(tuple(b if arm=='standard_context' else r['rock_context_box']));context.save(ROOT/'views'/(r['id']+'_'+arm+'_source.png'));arrays.extend([preprocess(local,True),preprocess(context,True)])
 zz=[]
 with torch.inference_mode():
  for i in range(0,len(arrays),12):
   z=net(torch.from_numpy(np.stack(arrays[i:i+12])).cuda()).mean((2,3));zz.append(torch.nn.functional.normalize(z,dim=1).cpu().numpy())
 z=np.concatenate(zz);x=np.concatenate([z[::2],z[1::2]],axis=1)/np.sqrt(2);pred=model.predict(x)
 for i,r in enumerate(rows):
  if arm=='standard_context':
   j=next(j for j,t in enumerate(testrows) if t['id']==r['id']);cos=float(np.dot(x[i],cached[j])/(np.linalg.norm(x[i])*np.linalg.norm(cached[j])));audit.append(dict(id=r['id'],cosine=cos,max_abs=float(np.abs(x[i]-cached[j]).max())));assert cos>1-1e-6;assert pred[i]==model.predict(cached[j:j+1])[0]
  out.append(dict(id=r['id'],arm=arm,work=r['work_id'],source_label=r['label'],prediction=CLASSES[int(pred[i])],agreement=CLASSES[int(pred[i])]==r['label']))
summary={arm:dict(regions=9,agreement_count=sum(r['agreement'] for r in out if r['arm']==arm),per_work={w:sum(r['agreement'] for r in out if r['arm']==arm and r['work']==w) for w in sorted({r['work'] for r in out})}) for arm in ['standard_context','larger_rock_context']};write('context_probe_results.json',dict(records=out,summary=summary,baseline_feature_audit=audit,original_classifier_unchanged=True,new_independent_validation=False,expert_accuracy=False,not_autonomous_recognition=True,robot_connected=False));print(json.dumps(summary,ensure_ascii=False),flush=True)
