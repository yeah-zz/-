# -*- coding: utf-8 -*-
"""Assistant intervals read from raw SECTIONS_0..5 before correction/prediction.
These are small geometric spot checks, not expert gold or physical tip centers.
"""
INTERVALS={
 'K01':[.5,2.5], 'K02':[2.5,4.], 'K04':[-8.5,-6.5],
 'K05':[-.5,1.5], 'K06':[-9.,-7.], 'K08':[1.5,3.5],
 'K09':[-9.5,-7.5], 'K11':[0.,1.5], 'K15':[-5.,-2.5],
 'K16':[-1.75,-.25], 'K17':[-2.5,-1.], 'K18':[-.5,1.5],
 'K19':[-2.75,-1.25], 'K20':[0.,2.], 'K22':[-1.5,.5],
 'K23':[-1.,.75], 'K24':[.5,2.5], 'K25':[-3.5,-1.5],
 'K26':[4.,6.], 'K27':[.25,2.25], 'K28':[-1.,1.],
 'K29':[0.,1.5], 'K31':[.5,2.5],
}
UNCERTAIN=['K00','K03','K07','K10','K12','K13','K14','K21','K30']

if __name__=='__main__':
    from v25_references import ROOT
    from rock_v15_common import read,jsave,digest
    from pathlib import Path
    source=ROOT/'raw_checks'/'locations.json';data=read(source)
    if set(INTERVALS)|set(UNCERTAIN)!={r['id'] for r in data['records']}:raise ValueError('Every raw check must have one decision')
    jsave(ROOT/'raw_checks'/'assistant_intervals.json',dict(intervals=INTERVALS,uncertain=UNCERTAIN,raw_locations_sha256=digest(source),label_source_sha256=digest(Path(__file__)),labelled_before_reference_correction_and_v25_predictions=True,heldout_training_images=True,not_expert=True,not_training_targets=True));print('V25_CHECKS_FROZEN',len(INTERVALS),len(UNCERTAIN),flush=True)
