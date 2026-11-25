# -*- coding: utf-8 -*-
"""Calibration uses G02 labels only; reused development, not blind validation."""
import sys
sys.dont_write_bytecode=True
from train_complete_candidates_20261101 import *

def main():
    artifact=torch.load(FIT/'model.pt',map_location='cpu',weights_only=False);rows=[r for r in read(ROOT/'training_rows.json')['rows'] if r['split']=='calibration'];p,u=prediction(artifact,rows);records=[];thresholds={}
    for bucket in ['faint','ordinary']:
        indices=[i for i,r in enumerate(rows) if ('faint' if r['features']['contrast_median']<.03 else 'ordinary')==bucket];selected=1.01
        for t in PROTOCOL['accept_threshold_candidates']:
            accepted=[i for i in indices if geometry_gate(rows[i]) and p[i]>=t and u[i]<=CFG['model_uncertainty']]
            if not any(rows[i]['label']==0 for i in accepted):selected=t;break
        thresholds[bucket]=selected;eligible=[i for i in indices if geometry_gate(rows[i])];records.append(dict(bucket=bucket,threshold=selected,total=len(indices),eligible_positive=sum(rows[i]['label']==1 for i in eligible),eligible_negative=sum(rows[i]['label']==0 for i in eligible),false_accept=int(sum(rows[i]['label']==0 and p[i]>=selected and u[i]<=CFG['model_uncertainty'] for i in eligible)),sensitivity_established=any(rows[i]['label']==1 for i in eligible)))
    jsave(ROOT/'bucket_calibration.json',dict(version='contrast_bucket_r1',contrast_boundary=.03,thresholds=thresholds,records=records,model_sha256=digest(FIT/'model.pt'),calibration_parent_group=2,regression_group_used=False,all_images_reused=True,scope='Sparse nonexpert calibration; bucket false-accept count does not prove general specificity'))
    jsave(ROOT/'reference_revision_audit.json',dict(changes=[dict(image='G05_01',key='ridge_238',before=0,after=None),dict(image='G02_01',key='ridge_783',before=0,after=None)],reason='RGB supports real ink along the candidate, but single-stroke identity/return is ambiguous. Prior hard rejection was overconfident. Both retained as unknown, not converted to positives.',prior_references='iterations/r1/partial_references.json',metrics_not_directly_comparable_to_original_labels=True))
    print('COMPLETE_BUCKET_CALIBRATION',thresholds,records,flush=True)
if __name__=='__main__':main()
