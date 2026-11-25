# -*- coding: utf-8 -*-
"""Add development works, reserving two whole filename work groups unseen."""
import sys,json,hashlib,re,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from guohua_hierarchy_source_audit_20261112 import fetch
from PIL import Image,ImageDraw,ImageFont
import numpy as np
ROOT=LAB/'guohua_faint_sources_20261116';OLD=LAB/'guohua_objects_v2_20261113'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'selection_before_download.json').exists(),'Selection frozen'
 listing=read(OLD/'full_file_listings.json');prior=read(OLD/'dataset_manifest.json');pool=[r for rows in listing['listings'].values() for r in rows];used={r['id'] for r in prior['records']};artists={r['artist_group'] for r in prior['records']};dev=[r for r in pool if any(Path(r['path']).stem.startswith(a+'_') for a in ['tang-yin','yun-shouping','zhao-ji'])];reserve=[r for r in pool if any(Path(r['path']).stem.startswith(a+'_') for a in ['sushi','xiaoyuncun'])]
 assert len(dev)==8;assert all(Path(r['path']).stem not in used for r in dev+reserve)
 assert not ({'filename:'+Path(r['path']).stem.split('_')[0] for r in dev+reserve}&artists)
 write('selection_before_download.json',dict(revision=listing['revision'],prior_manifest_sha256=sha(OLD/'dataset_manifest.json'),development_paths=[r['path'] for r in dev],reserved_paths=[r['path'] for r in reserve],reserved_mode='metadata only, do not download or preview or run predictions this round',identity='filename candidates only; original attribution unauthenticated',same_public_dataset_not_independent_source=True,new_independent_acceptance=False))
 tasks=[];rows=[];base='https://huggingface.co/datasets/Rane7/CLP_dataset/resolve/'+listing['revision']+'/'
 for r in dev:
  ident=Path(r['path']).stem;source_split=r['path'].split('/')[-2];image=ROOT/'raw'/(ident+'.jpg');mask=ROOT/'raw'/(ident+'_source.png');maskpath='clp2k/annotations/'+source_split+'/'+ident+'.png';tasks.extend([(base+r['path'],image),(base+maskpath,mask)]);rows.append(dict(id=ident,image=str(image),source_mask=str(mask),image_url=base+r['path'],mask_url=base+maskpath,work_group='filename:'+re.sub(r'_\d+$','',ident),artist_group='filename:'+ident.split('_')[0],split='new_development_not_training'))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  fs=[ex.submit(fetch,u,p,3_000_000) for u,p in tasks]
  for f in fs:f.result()
 for r in rows:
  im=Image.open(r['image']).convert('RGB');raw=np.array(Image.open(r['source_mask']));assert im.size==(raw.shape[1],raw.shape[0]);assert set(np.unique(raw))<=(set(range(15))|{255});t=np.full(raw.shape,255,'uint8');t[np.isin(raw,prior['mapping']['other_named_foreground_ids'])]=0;t[np.isin(raw,[2,13])]=1;t[raw==3]=2;target=ROOT/'raw'/(r['id']+'_target.png');Image.fromarray(t).save(target);r.update(size=list(im.size),image_sha256=sha(r['image']),decoded_pixel_sha256=hashlib.sha256(np.array(im).tobytes()).hexdigest(),target=str(target),target_sha256=sha(target),source_mask_sha256=sha(r['source_mask']))
 assert not ({r['decoded_pixel_sha256'] for r in rows}&{r['decoded_pixel_sha256'] for r in prior['records']});assert len({r['decoded_pixel_sha256'] for r in rows})==len(rows)
 write('dataset_manifest.json',dict(records=rows,selection_sha256=sha(ROOT/'selection_before_download.json'),status='Development screening only, no source category treated as fine ink truth',reserved_image_count=len(reserve),reserved_work_groups=sorted({'filename:'+re.sub(r'_\d+$','',Path(r['path']).stem) for r in reserve}),reserved_not_downloaded=True,expert_reviewed=False,robot_connected=False))
 b=Image.new('RGB',(1200,4*355),'#f4f1e8');d=ImageDraw.Draw(b);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
 for i,r in enumerate(rows):
  im=Image.open(r['image']);im.thumbnail((570,315));x=(i%2)*600;y=(i//2)*355;b.paste(im,(x+5,y+35));d.text((x+5,y+5),r['id'][:65],font=font,fill='#294735')
 b.save(ROOT/'development_screening.jpg',quality=94);print(json.dumps(dict(development=len(rows),work_groups=len({r['work_group'] for r in rows}),reserve_metadata=len(reserve)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
