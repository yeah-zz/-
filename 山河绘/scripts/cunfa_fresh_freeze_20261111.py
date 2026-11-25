# -*- coding: utf-8 -*-
"""Freeze existing model before collecting and predicting new-author samples."""
import sys,json,hashlib,shutil,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,joblib
from sklearn.svm import SVC
from cunfa_context_probe_20261111 import select_c,CLASSES,WEIGHTS
ROOT=LAB/'cunfa_fresh_validation_20261111';PREV=LAB/'cunfa_context_reference_20261111'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 (ROOT/'frozen').mkdir(parents=True,exist_ok=True);manifest=ROOT/'freeze_manifest.json'
 if manifest.exists():
  d=json.loads(manifest.read_text('utf8'))
  for r in d['files']:assert sha(r['path'])==r['sha256']
  print('ALREADY_FROZEN',d['frozen_at_utc']);return
 rows=json.loads((PREV/'input_manifest.json').read_text('utf8'))['rows'];z=np.load(PREV/'features.npz');groups=np.array([r['artist_group'] for r in rows]);y=np.array([CLASSES.index(r['label']) for r in rows]);files=[]
 for name in ['context_head_exploratory.joblib','protocol.json']:
  dest=ROOT/'frozen'/name;shutil.copy2(PREV/name,dest);files.append(dict(path=str(dest),sha256=sha(dest)))
 # This exact feature/preprocessing code is preserved. No new data may alter it.
 dest=ROOT/'frozen'/'encoder_and_preprocessing.py';shutil.copy2(Path(__file__).with_name('cunfa_context_probe_20261111.py'),dest);files.append(dict(path=str(dest),sha256=sha(dest)))
 for key in ['local_only','context_only']:
  c,scores=select_c(z[key],y,groups);svm=SVC(C=c,kernel='linear',class_weight='balanced');svm.fit(z[key],y);p=ROOT/'frozen'/(key+'_head.joblib');joblib.dump(dict(model=svm,trained_ids=[r['id'] for r in rows],selected_C=c,inner_training_scores=scores),p);files.append(dict(path=str(p),sha256=sha(p)))
 files.append(dict(path=str(WEIGHTS),sha256=sha(WEIGHTS)))
 d=dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,train_data_sha256=sha(PREV/'input_manifest.json'),train_ids=[r['id'] for r in rows],train_authors=sorted(set(r['artist_group'] for r in rows)),
  primary='frozen local_plus_context',input_scope='assistant selected patch + fixed2.5x neighbourhood; not whole-image detection',new_samples_never_train_or_tune=True,
  sample_admission='source explicitly names technique in work/region description; assistant visual screening before predictions; ambiguous local labels excluded or uncertain, never made gold',
  reporting='all admitted samples; per work/artist and class; known source-label agreement only; no formal expert gold',
  provisional_gate=dict(minimum_each_class_recall=.8,minimum_artist_macro_agreement=.8,minimum_accuracy_gain_over_frozen_local=.0,minimum_macro_f1_gain_over_frozen_local=.0),
  failure_policy='no post-prediction relabelling, removal or tuning; all errors retained; no adoption without expert validation',robot_connected=False)
 manifest.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8');print('FROZEN',d['frozen_at_utc'],flush=True)
if __name__=='__main__':main()
