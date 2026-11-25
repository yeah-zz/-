# -*- coding: utf-8 -*-
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image
from cunfa_context_probe_20261111 import ROOT,OLD,PREV,evaluate,save
from train_cunfa_gabor_20261110 import features,grouped_cv,CLASSES
def main():
 start=time.time();d=json.loads((ROOT/'input_manifest.json').read_text('utf8'));rows=d['rows'];z=np.load(ROOT/'features.npz');y=np.array([CLASSES.index(r['label']) for r in rows]);artists=np.array([r['artist_group'] for r in rows]);pub=np.array([r['publisher_group'] for r in rows]);results={}
 for key in ['local_only','context_only','local_plus_context']:
  e=evaluate(z[key],y,pub,rows)
  for f in e['folds']:f['held_publisher']=f.pop('held_artist')
  e['publisher_macro_agreement']=e.pop('artist_macro_agreement');e['posthoc_stress_test_not_independent']=True;results[key]=e;save(key+'_publisher_stress.json',e)
  print('PUBLISHER',key,e['metrics'],flush=True)
 # Rerun the classical baseline on exactly the same 63 rows, never compare
 # old40 or new23 metrics directly with the joint63 experiment.
 old=np.load(OLD/'gabor_magnitude_features.npy');extra=[]
 for r in rows[40:]:extra.append(features(Image.open(r['image']))[0])
 x=np.concatenate([old[:40],np.array(extra)]);e=grouped_cv(x,y,artists,'full_gabor',rows);save('gabor_same63_author_cv.json',e)
 save('audit_summary.json',dict(posthoc=True,reused_data_not_new_test=True,same63_gabor=e['label_agreement'],same63_gabor_macro_f1=e['macro_f1'],publisher_metrics={k:dict(**r['metrics'],publisher_macro_agreement=r['publisher_macro_agreement']) for k,r in results.items()},not_used_to_tune_or_select_primary=True,robot_connected=False,elapsed_seconds=time.time()-start));print('AUDIT_COMPLETE',flush=True)
if __name__=='__main__':main()
