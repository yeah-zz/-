# -*- coding: utf-8 -*-
"""Counterexamples for ink-supported continuation; no label-derived tuning."""
from source_constrained_strokes_20261110 import *

def row(i,q):
 q=np.array(q,float);return dict(id=i,curve=q.tolist(),length_px=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()),status='pending')

def main():
 cfg=CONFIGS[0];results={}
 for name,color in [('blank_paper',240),('uniform_dark',20)]:
  image=np.full((48,64,3),color,'uint8');mask,fields=evidence(image,cfg);paths,_,_=paths_from_mask(centre_ridges(fields,mask));assert not paths;results[name]='no paths'
 rows=[row('a',[[10,20],[12,20],[14,20],[16,20],[18,20],[20,20]]),row('b',[[23,20],[25,20],[27,20],[29,20],[31,20]])]
 ink=np.zeros((48,64),bool);ink[19:22,10:32]=True
 joined,stats=continue_inside_ink(rows,ink);assert len(joined)==1;assert not(raster(joined,ink.shape)&~ink).any();results['sampling_gap_inside_ink']='joined with supported bridge'
 ink[:,21:23]=False
 separated,_=continue_inside_ink(rows,ink);assert len(separated)==2;results['white_paper_gap']='not joined'
 rows=[row('a',[[10,20],[14,20],[18,20],[20,20]]),row('b',[[10,23],[14,23],[18,23],[20,23]])]
 ink=np.ones((48,64),bool);separated,_=continue_inside_ink(rows,ink);assert len(separated)==2;results['parallel_strokes']='kept separate'
 rows=[row('a',[[8,20],[12,20],[16,20],[20,20]]),row('b',[[23,19],[27,19],[31,19]]),row('c',[[23,21],[27,21],[31,21]])]
 separated,_=continue_inside_ink(rows,ink);assert len(separated)==3;results['equally_likely_fork']='pending, no forced continuation'
 previous=json.loads((PREV/'verification.json').read_text('utf8'))
 for p in previous['recognised_baselines_unchanged']:assert hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()==p['sha256']
 results['recognised_baselines']='unchanged'
 save('counterexample_checks.json',results);print(json.dumps(results),flush=True)

if __name__=='__main__':main()
