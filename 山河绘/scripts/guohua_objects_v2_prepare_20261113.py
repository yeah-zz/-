# -*- coding: utf-8 -*-
"""Expand development works and freeze a new filename-grouped test beforehand."""
import sys,json,hashlib,re,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from guohua_hierarchy_source_audit_20261112 import fetch
from guohua_objects_prepare_20261112 import evenly
import numpy as np
from PIL import Image
ROOT=LAB/'guohua_objects_v2_20261113';OLD=LAB/'guohua_objects_pilot_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 assert not (ROOT/'results.json').exists(),'Frozen experiment';listing=json.loads((ROOT/'full_file_listings.json').read_text('utf8'));old=json.loads((OLD/'dataset_manifest.json').read_text('utf8'));base='https://huggingface.co/datasets/Rane7/CLP_dataset/resolve/'+listing['revision']+'/';pool=[x for rows in listing['listings'].values() for x in rows];rows=[]
 for r in old['records']:
  q=dict(r);q['previous_split']=r['split']
  if r['split']=='test_collection':q['split']='train';q['previous_test_explicitly_promoted_to_development']=True
  if r['split']=='test_named':q['split']='old_named_regression'
  rows.append(q)
 selected=[]
 for artist in ['wang-jian','wang-shimin','wu-yuanzhi','xia-gui']:
  selected.extend((x,'train') for x in pool if Path(x['path']).stem.startswith(artist+'_'))
 for artist,n in [('shen-zhou',4),('wen-zhengming',4),('wang-yuanqi',4),('shitao',2)]:
  selected.extend((x,'new_test') for x in evenly(sorted([x for x in pool if Path(x['path']).stem.startswith(artist+'_')],key=lambda x:x['path']),n))
 selection=dict(prior_dataset_sha256=sha(OLD/'dataset_manifest.json'),source_revision=listing['revision'],selection_rule='All files for four additional development artists; evenly spaced 4/4/4/2 files for four different heldout artist-name candidates, before any new training or predictions',previous_eight_collection_tests_now_development=True,original_artist_attribution_unverified=True,test_images_not_previewed=True,selected=[dict(path=x['path'],split=s) for x,s in selected]);write('selection_before_download.json',selection);tasks=[]
 for item,split in selected:
  ident=Path(item['path']).stem;original=item['path'].split('/')[-2];image=ROOT/'raw'/split/(ident+'.jpg');mask=ROOT/'raw'/split/(ident+'_source.png');maskpath='clp2k/annotations/'+original+'/'+ident+'.png';tasks.extend([(base+item['path'],image),(base+maskpath,mask)]);prefix=ident.split('_')[0];work=re.sub(r'_\d+$','',ident)
  rows.append(dict(id=ident,split=split,source_split=original,image=str(image),source_mask=str(mask),image_url=base+item['path'],mask_url=base+maskpath,work_group='filename:'+work,artist_group='filename:'+prefix,grouping_evidence='filename candidates; original artist and work identity not authenticated',source_revision=listing['revision'],newly_downloaded=True))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
  futures=[executor.submit(fetch,u,p,3_000_000) for u,p in tasks]
  for future in futures:future.result()
 for r in rows:
  if not r.get('newly_downloaded'):continue
  image=Image.open(r['image']).convert('RGB');mask=Image.open(r['source_mask']);assert image.size==mask.size;raw=np.array(mask);assert raw.ndim==2 and set(np.unique(raw)).issubset(set(range(15))|{255});target=np.full(raw.shape,255,'uint8');target[np.isin(raw,old['mapping']['other_named_foreground_ids'])]=0;target[np.isin(raw,[2,13])]=1;target[raw==3]=2;path=ROOT/'raw'/r['split']/(r['id']+'_target.png');Image.fromarray(target).save(path);r.update(target=str(path),image_sha256=sha(r['image']),decoded_pixel_sha256=hashlib.sha256(np.array(image).tobytes()).hexdigest(),source_mask_sha256=sha(r['source_mask']),target_sha256=sha(path),size=list(image.size),class_pixels={str(k):int((target==k).sum()) for k in [0,1,2,255]})
 seen={};works={};artists={}
 for r in rows:
  for table,key in [(seen,r['decoded_pixel_sha256']),(works,r['work_group'])]:
   assert key not in table or table[key]==r['split'],('cross-split overlap',r['id']);table[key]=r['split']
  if r['artist_group']:
   assert r['artist_group'] not in artists or artists[r['artist_group']]==r['split'];artists[r['artist_group']]=r['split']
 assert not ({r['id'] for r in old['records']}&{r['id'] for r in rows if r['split']=='new_test'})
 manifest=dict(records=rows,mapping=old['mapping'],class_names=old['class_names'],ignore_index=255,source_void_is_not_background=True,prior_dataset_sha256=sha(OLD/'dataset_manifest.json'),prior_results_sha256=sha(OLD/'results.json'),selection_sha256=sha(ROOT/'selection_before_download.json'),new_test_group_count=len({r['work_group'] for r in rows if r['split']=='new_test'}),new_test_image_count=sum(r['split']=='new_test' for r in rows),author_groups_are_filename_candidates=True,expert_dense_annotations=False,formal_independent_acceptance=False,robot_connected=False)
 write('dataset_manifest.json',manifest);write('split_checks.json',dict(decoded_exact_duplicates_disjoint=True,filename_work_groups_disjoint=True,filename_artist_candidates_disjoint=True,new_test_not_in_previous_model_dataset=True,previous_test_reuse_explicit=True,formal_author_independence_not_claimed=True,test_preview_before_checkpoint_freeze=False))
 print('V2_DATA_READY',dict(images={s:sum(r['split']==s for r in rows) for s in sorted({r['split'] for r in rows})},works={s:len({r['work_group'] for r in rows if r['split']==s}) for s in sorted({r['split'] for r in rows})}),flush=True)
if __name__=='__main__':main()
