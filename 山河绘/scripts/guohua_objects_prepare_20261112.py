# -*- coding: utf-8 -*-
"""Corroborate rock/tree codes and prepare a grouped exploratory object pilot."""
import sys,json,hashlib,re,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from guohua_hierarchy_source_audit_20261112 import fetch
import requests,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'guohua_objects_pilot_20261112';AUDIT=LAB/'guohua_hierarchy_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def evenly(rows,n):
 return [rows[i] for i in np.linspace(0,len(rows)-1,min(n,len(rows))).round().astype(int)] if rows else []
def main():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'results.json').exists(),'Keep evaluated datasets frozen.';source=json.loads((AUDIT/'source_audit.json').read_text('utf8'));revision=source['revision'];base='https://huggingface.co/datasets/Rane7/CLP_dataset/resolve/'+revision+'/'
 # Category names are explicit in the COCO source. Check them against the exact
 # same native pixels in CLPD, not merely against similarly coloured masks.
 coco=json.loads((AUDIT/'sources/instances_validation.json').read_text('utf8'));cross=json.loads((AUDIT/'res_painting_metadata.json').read_text('utf8'));evidence=[]
 for sample in cross['sampled_images']:
  mae,ident,split=sample['nearest_clpd_observation'];assert mae==0.;r=next(r for r in source['records'] if r['id']==ident and r['source_split']==split);mask=np.array(Image.open(r['mask']));same=np.array(Image.open(AUDIT/'res_sample'/sample['file_name']).convert('RGB'));assert np.array_equal(same,np.array(Image.open(r['image']).convert('RGB')))
  for cat in [2,3,13]:
   m=np.zeros(mask.shape,'uint8')
   for a in coco['annotations']:
    if a['image_id']!=sample['image_id'] or a['category_id']!=cat:continue
    for polygon in a['segmentation']:
     if len(polygon)>=6:cv2.fillPoly(m,[np.array(polygon).reshape(-1,2).round().astype('int32')],1)
   if m.any():evidence.append(dict(work=ident,category_id=cat,category_name=next(c['name'] for c in coco['categories'] if c['id']==cat),polygon_pixels=int(m.sum()),clpd_same_code_agreement=float((mask[m>0]==cat).mean())))
 assert {r['category_id'] for r in evidence}=={2,3,13};assert min(r['clpd_same_code_agreement'] for r in evidence)>.95,evidence
 mapping=dict(rock_source_ids=[2,13],tree_source_ids=[3],other_named_foreground_ids=[1,4,5,6,7,8,9,10,11,12,14],ignored_void_ids=[0,255],basis='Explicit COCO category names corroborated on two pixel-identical CLPD examples; not an official CLPD full codebook or expert verification',evidence=evidence,expert_verified=False);write('mapping_evidence.json',mapping)
 listings={}
 for split in ['training','validation']:
  url='https://huggingface.co/api/datasets/Rane7/CLP_dataset/tree/'+revision+'/clp2k/images/'+split+'?limit=1000';r=requests.get(url,timeout=30);r.raise_for_status();listings[split]=[x for x in r.json() if x['type']=='file'];write(split+'_file_listing.json',listings[split])
 selected=[];limits={'bai-xueshi':8,'dai-jin':2,'dong-yuan':2,'li-sixun':1,'qian-xuan':2,'qiu-ying':8,'qiuying':12}
 for artist,n in limits.items():
  candidates=[x for x in listings['training'] if Path(x['path']).stem.startswith(artist+'_')];selected.extend((x,'train') for x in evenly(candidates,n))
 selected.extend((x,'validation_named') for x in listings['training'] if Path(x['path']).stem.startswith('ni-zan_'))
 selected.extend((x,'test_named') for x in evenly([x for x in listings['training'] if Path(x['path']).stem.startswith('fankuan_')],12))
 selected.extend((x,'test_collection') for x in evenly([x for x in listings['validation'] if Path(x['path']).stem.startswith('Princeton_')],8))
 rows=[];tasks=[]
 for item,split in selected:
  ident=Path(item['path']).stem;original_split=item['path'].split('/')[-2];image=ROOT/'raw'/split/(ident+'.jpg');mask=ROOT/'raw'/split/(ident+'_source.png');mp='clp2k/annotations/'+original_split+'/'+ident+'.png';tasks.extend([(base+item['path'],image),(base+mp,mask)])
  prefix=ident.split('_')[0];artist='qiu-ying' if prefix=='qiuying' else prefix;work=re.sub(r'_\d+$','',ident) if prefix!='Princeton' else ident;work=work.replace('qiuying_','qiu-ying_');rows.append(dict(id=ident,split=split,source_split=original_split,image=str(image),source_mask=str(mask),image_url=base+item['path'],mask_url=base+mp,work_group='filename:'+work,artist_group='filename:'+artist if prefix!='Princeton' else None,grouping_evidence='filename artist/title tokens; original attribution and crop lineage not independently authenticated',source_revision=revision))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(fetch,u,p,3_000_000) for u,p in tasks]
  for f in futures:f.result()
 for r in rows:
  image=Image.open(r['image']);mask=Image.open(r['source_mask']);assert image.size==mask.size;raw=np.array(mask);assert raw.ndim==2 and set(np.unique(raw)).issubset(set(range(15))|{255});target=np.full(raw.shape,255,'uint8');target[np.isin(raw,mapping['other_named_foreground_ids'])]=0;target[np.isin(raw,[2,13])]=1;target[raw==3]=2;p=ROOT/'raw'/r['split']/(r['id']+'_target.png');Image.fromarray(target).save(p);decoded=np.asarray(image.convert('RGB'));r.update(target=str(p),image_sha256=sha(r['image']),decoded_pixel_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),source_mask_sha256=sha(r['source_mask']),target_sha256=sha(p),size=list(image.size),class_pixels={str(k):int((target==k).sum()) for k in [0,1,2,255]})
 # Exact duplicate and filename lineage checks. These do not establish true
 # author independence when the source lacks a verified catalogue identifier.
 seen={};works={};artists={}
 for r in rows:
  for table,key in [(seen,r['decoded_pixel_sha256']),(works,r['work_group'])]:
   assert key not in table or table[key]==r['split'],('cross-split duplicate/work',r['id']);table[key]=r['split']
  if r['artist_group']:
   assert r['artist_group'] not in artists or artists[r['artist_group']]==r['split'];artists[r['artist_group']]=r['split']
 manifest=dict(records=rows,mapping=mapping,class_names=['other_named_object','mountain_or_rock','tree'],ignore_index=255,source_void_is_not_background=True,source_license='CLPD card CC BY-NC 4.0; source annotation quality not independently certified',selection='Fixed filename quotas and evenly spaced file order, before object-model training',artist_groups_are_filename_candidates=True,true_author_independence_verified=False,test_is_formal_acceptance=False,source_masks_not_cunfa_labels=True,robot_connected=False)
 write('dataset_manifest.json',manifest);write('dataset_checks.json',dict(all_sizes_match=True,source_void_ignored=True,exact_pixel_duplicates_disjoint=True,filename_work_groups_disjoint=True,filename_author_candidates_disjoint=True,mapping_corroborated_not_expert_gold=True,formal_acceptance=False))
 # Named local teaching references remain a separate task; do not attach their
 # names to segmentation masks or treat them as newly collected experts.
 oldrefs=json.loads((LAB/'cunfa_role_reference_20261112/source_references.json').read_text('utf8'));refs=[]
 for r in oldrefs['records']:
  refs.append(dict(id=r['id'],image=r['image'],image_sha256=sha(r['image']),source_url=r['source_url'],source_evidence=r['source_evidence'],source_group=r['source_group'],named_style=r['local_style'],status='existing_named_example_needs_local_concept_review',concepts={k:None for k in ['mark_family','curvature','width_change','surface_aligned_group_flow','overlap_order','dry_wet_evidence']},newly_collected=False,expert_local_verified=False,eligible_for_new_concept_training=False))
 write('cunfa_concept_review_queue.json',dict(records=refs,new_expert_labels=0,concept_model_trained=False,whole_work_label_propagation=False));print('OBJECT_DATASET',dict(counts={s:sum(r['split']==s for r in rows) for s in sorted({r['split'] for r in rows})},mapping_min_agreement=min(r['clpd_same_code_agreement'] for r in evidence),work_groups=len(works)),flush=True)
if __name__=='__main__':main()
