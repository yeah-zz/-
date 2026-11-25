import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from gongbi_training.cunfa_score_alignment import *

def main():
 rng=np.random.default_rng(17);x=rng.normal(size=(24,9));groups=np.repeat(['a','b','c','d'],6)
 y=np.tile(np.array([[0,1,0],[1,0,1],[0,0,1],[1,1,0],[0,1,1],[1,0,0]]),(4,1));y[0,1]=IGNORE
 fitted=fit_training_concepts(x,y,groups)
 assert all(f['held_illustration'] not in f['fitted_illustrations'] for f in fitted['folds'])
 changed=y.copy();changed[groups=='a']=1-changed[groups=='a'];changed[0,1]=IGNORE
 # All attribute target changes in the held illustration must leave that
 # illustration's OUT-OF-FOLD base predictions unchanged.
 other=fit_training_concepts(x,changed,groups)
 assert np.allclose(fitted['oof_decision'][groups=='a'],other['oof_decision'][groups=='a'])
 a=predict_concepts(fitted,x,True);assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
 s=scores_from_decisions(fitted,fitted['oof_decision'],True)
 head=fit_style_from_oof(s,groups,['a','b','c','d'],[0,1,0,1])
 mean=aggregate_regions(s,groups,['a','b','c','d'])
 assert np.allclose(head.named_steps['standardscaler'].mean_,mean.mean(0))
 assert len(head.named_steps['svc'].support_vectors_[0])==3
 try: fit_training_concepts(x,np.zeros_like(y),groups)
 except ValueError:pass
 else:raise AssertionError('Single-polarity folds must fail explicitly')
 result=dict(held_illustration_targets_do_not_change_oof_base_scores=True,
  train_only_normalization=True,no_image_feature_bypass_in_style_head=True,
  missing_polarities_rejected=True,finite_calibrated_scores=True,
  calibration_is_training_fit_not_independent_validation=True,robot_connected=False)
 root=LAB/'cunfa_score_alignment_20261117';root.mkdir(exist_ok=True)
 (root/'contract_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
