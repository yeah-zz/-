"""Conservative unit-level rejection admission; unverified family borders ignored."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
ROOT=LAB/'cunfa_photometry_20261117'
AMBIGUOUS={'aam_outside_raveled_rope','aam_outside_bands_dragged_in_mud','aam_outside_nail_head'}
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 action=sys.argv[1] if len(sys.argv)>1 else 'freeze'
 if action=='freeze':
  path=ROOT/'rejection_scope_protocol.json';assert not path.exists(),'Scope protocol already frozen'
  p=dict(ambiguous_units=sorted(AMBIGUOUS),action='exclude these from known/outside loss and keep for review',
   reason='A separately named illustration does not by itself settle membership in the broader pima/fupi families',
   retained_outside_reference_units=['aam_outside_mi_dot','aam_outside_cloud_head','qingpu_22_0_outside','qingpu_23_0_outside'],
   claims='outside these two named example families only, not universal unknown, mixed, or identification of Mi-dot',
   frozen_before_corrected_fit=True,threshold=.5,C=1,test_scope='same5 already opened units; not independent acceptance',
   first_strict_name_negative_run_not_used_as_main_rejection_result=True,robot_connected=False)
  path.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf8');print('SCOPE_FROZEN',flush=True);return
 if action!='fit':raise ValueError('Use freeze or fit')
 import numpy as np,joblib
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 protocol=read(ROOT/'rejection_scope_protocol.json');assert not (ROOT/'rejection_scope_results.json').exists()
 units=read(ROOT/'rejection_units.json')['units'];x=np.load(ROOT/'features.npz')['rejection']
 fit=np.array([u['split']=='train' and u['id'] not in AMBIGUOUS for u in units]);test=np.array([u['split']=='test' for u in units]);y=np.array([u['known'] for u in units])
 assert fit.sum()==10 and y[fit].sum()==6
 model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',random_state=0,max_iter=2000)).fit(x[fit],y[fit])
 scores=model.predict_proba(x)[:,1];pred=(scores>=.5).astype('int64')
 records=[dict(id=u['id'],split='review_only' if u['id'] in AMBIGUOUS else u['split'],
  reference_known=None if u['id'] in AMBIGUOUS else u['known'],known_score=float(scores[i]),accept_known=bool(pred[i]),
  action='two_class_candidate' if pred[i] else 'outside_known_set_or_uncertain') for i,u in enumerate(units)]
 metrics=dict(test_units=int(test.sum()),known_test=int((test&(y==1)).sum()),outside_test=int((test&(y==0)).sum()),
  known_accepted=int((test&(y==1)&(pred==1)).sum()),known_falsely_rejected=int((test&(y==1)&(pred==0)).sum()),
  outside_rejected=int((test&(y==0)&(pred==0)).sum()),outside_falsely_accepted=int((test&(y==0)&(pred==1)).sum()))
 joblib.dump(dict(model=model,protocol=protocol,trained_ids=[units[i]['id'] for i in np.flatnonzero(fit)],production_use=False),ROOT/'rejection_scope_model.joblib')
 out=dict(protocol=protocol,scope_protocol_sha256=sha(ROOT/'rejection_scope_protocol.json'),train_known=6,train_outside=4,
  review_only_ambiguous=3,metrics=metrics,records=records,general_unknown_validation=False,model_adopted=False,robot_connected=False)
 (ROOT/'rejection_scope_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(metrics),json.dumps([r for r in records if r['split']=='test']),flush=True)
if __name__=='__main__':main()
