# -*- coding: utf-8 -*-
"""Inspect public paired object data before assigning labels or training."""
import sys,json,hashlib,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'guohua_hierarchy_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def fetch(url,path,limit=16_000_000):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():return
 with requests.get(url,timeout=(15,60),stream=True) as r:
  r.raise_for_status();parts=[];size=0
  for part in r.iter_content(131072):
   size+=len(part);assert size<=limit,'Unexpected large file';parts.append(part)
  path.write_bytes(b''.join(parts))
def main():
 ROOT.mkdir(exist_ok=True);repo='Rane7/CLP_dataset';meta=requests.get('https://huggingface.co/api/datasets/'+repo,timeout=30).json();revision=meta['sha'];write('clpd_repo_metadata.json',meta);base='https://huggingface.co/datasets/'+repo+'/resolve/'+revision+'/'
 fetch(base+'README.md',ROOT/'sources/CLPD_README.md',50000);records=[];tasks=[]
 for split in ['training','validation']:
  url='https://huggingface.co/api/datasets/'+repo+'/tree/'+revision+'/clp2k/images/'+split+'?limit=6';r=requests.get(url,timeout=30);r.raise_for_status();items=r.json();write('clpd_'+split+'_listing.json',items)
  for item in items:
   if item['type']!='file':continue
   image=Path(item['path']);stem=image.stem;mask='clp2k/annotations/'+split+'/'+stem+'.png';ip=ROOT/'clpd_sample'/split/image.name;mp=ROOT/'clpd_sample'/split/(stem+'_mask.png');tasks.extend([(base+item['path'],ip),(base+mask,mp)]);records.append(dict(id=stem,source_split=split,image=str(ip),mask=str(mp),image_url=base+item['path'],mask_url=base+mask))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(fetch,u,p) for u,p in tasks]
  for f in futures:f.result()
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(1200,((len(records)+2)//3)*260),'#f4f1e8');draw=ImageDraw.Draw(board)
 for i,row in enumerate(records):
  image=Image.open(row['image']);mask=Image.open(row['mask']);a=np.asarray(mask);unique=np.unique(a.reshape(-1,a.shape[2]),axis=0).tolist() if a.ndim==3 else np.unique(a).tolist();row.update(image_sha256=sha(row['image']),mask_sha256=sha(row['mask']),image_size=list(image.size),mask_size=list(mask.size),mask_mode=mask.mode,unique_codes=unique,geometry_matches=image.size==mask.size,taxonomy_verified=False,author_or_work_group_verified=False,ready_for_training=False)
  x=i%3*400;y=i//3*260;im=image.convert('RGB');im.thumbnail((190,210));board.paste(im,(x+4,y+30));view=mask.convert('RGB')
  if a.ndim==2 and len(unique)<64:
   colors=np.array([[(k*73+40)%255,(k*127+70)%255,(k*47+90)%255] for k in range(256)],'uint8');view=Image.fromarray(colors[a.astype('uint8')])
  view.thumbnail((190,210));board.paste(view,(x+204,y+30));draw.text((x+4,y+4),row['source_split']+' / '+row['id'],font=font,fill='#243e32')
 board.save(ROOT/'clpd_pair_review.jpg',quality=94)
 # COCO-like source with explicit object category names; verify what is actually
 # in the public repo rather than accepting all quantities in its README.
 ghmeta=requests.get('https://api.github.com/repos/zpc481523/res-painting/commits/main',timeout=30).json();rev=ghmeta['sha'];gh='https://raw.githubusercontent.com/zpc481523/res-painting/'+rev+'/'
 fetch(gh+'README.md',ROOT/'sources/res_painting_README.md',50000);fetch(gh+'instances_validation.json',ROOT/'sources/instances_validation.json');coco=json.loads((ROOT/'sources/instances_validation.json').read_text('utf8'))
 files=requests.get('https://api.github.com/repos/zpc481523/res-painting/git/trees/'+rev+'?recursive=1',timeout=30).json().get('tree',[]);available={Path(x['path']).name:x['path'] for x in files if x['path'].startswith('validation/') and Path(x['path']).suffix.lower() in {'.png','.jpg'}};samples=[];objboard=Image.new('RGB',(1000,550),'#f4f1e8');od=ImageDraw.Draw(objboard);names={r['id']:r['name'] for r in coco['categories']}
 for i,row in enumerate(coco['images'][:2]):
  if row['file_name'] not in available:continue
  path=ROOT/'res_sample'/row['file_name'];fetch(gh+available[row['file_name']],path);image=Image.open(path).convert('RGB');overlay=image.copy();dr=ImageDraw.Draw(overlay,'RGBA');count={}
  for a in coco['annotations']:
   if a['image_id']!=row['id'] or a['category_id'] not in [2,3,13]:continue
   name=names[a['category_id']];count[name]=count.get(name,0)+1;color={2:(150,100,35,90),3:(45,155,85,110),13:(170,130,55,90)}[a['category_id']]
   for polygon in a['segmentation']:
    if len(polygon)>=6:dr.polygon(list(zip(polygon[::2],polygon[1::2])),fill=color)
  for j,im in enumerate([image,overlay]):im.thumbnail((245,480));objboard.paste(im,(i*500+j*250,40))
  native_size=Image.open(path).size;nearest=[]
  for other in records:
   otherimage=np.asarray(Image.open(other['image']).convert('RGB'));native=np.asarray(Image.open(path).convert('RGB'))
   if otherimage.shape==native.shape:nearest.append((float(np.mean(np.abs(native.astype('float32')-otherimage.astype('float32')))),other['id'],other['source_split']))
  nearest.sort();od.text((i*500+5,5),row['file_name']+' / 源对象多边形',font=font,fill='#243e32');samples.append(dict(image_id=row['id'],file_name=row['file_name'],image_sha256=sha(path),image_size=list(native_size),declared_size=[row['width'],row['height']],size_matches=native_size==(row['width'],row['height']),object_instances=count,nearest_clpd_observation=nearest[0] if nearest else None,source_annotations_not_project_expert_gold=True))
 objboard.save(ROOT/'res_object_review.jpg',quality=94);write('res_painting_metadata.json',dict(revision=rev,categories=coco.get('categories'),images_listed=len(coco.get('images',[])),public_image_files_found=len(available),sampled_images=samples,annotations_listed=len(coco.get('annotations',[])),public_repository='https://github.com/zpc481523/res-painting',expert_quality_is_source_claim_not_project_verified=True,complete_training_split_publicly_observed=False,ready_for_training=False))
 # Save metadata and explicit mapping candidates only; unknown masks do not
 # become rock/tree labels by visual guessing.
 audit=dict(source='https://huggingface.co/datasets/'+repo,revision=revision,records=records,paired_observations=len(records),all_pairs_same_size=all(r['geometry_matches'] for r in records),license_source='Dataset card states CC BY-NC 4.0',class_id_to_name_mapping_available=False,author_split_metadata_available=False,source_claimed_sample_count_not_revalidated=True,new_cunfa_labels=0,ready_for_training=False,model_trained=False,robot_connected=False)
 write('source_audit.json',audit);print(json.dumps(dict(pairs=len(records),mask_modes=sorted({r['mask_mode'] for r in records}),codes=[r['unique_codes'] for r in records[:2]],all_pairs_same_size=audit['all_pairs_same_size'],coco_categories=coco.get('categories'),coco_images=len(coco.get('images',[]))),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
