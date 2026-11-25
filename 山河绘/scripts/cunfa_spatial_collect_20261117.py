# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests
from PIL import Image
ROOT=LAB/'cunfa_spatial_pairs_20261117'
SPECS=[('wanghui',49150,2272408,'train','Wang Hui','Clearing after Rain over Streams and Mountains'),
 ('gongxian',65620,155713,'train','Gong Xian','Landscapes, first museum image'),
 ('shen_attributed',51858,152593,'fresh_check','Attributed to Shen Zhou','Landscape with Autumn Foliage')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def collect(spec):
 ident,oid,iid,split,artist,title=spec;url=f'https://collectionapi.metmuseum.org/api/collection/v1/iiif/{oid}/{iid}/main-image';p=ROOT/'sources'/(ident+'.jpg')
 metadata=requests.get(f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}',timeout=30);metadata.raise_for_status();meta=metadata.json()
 (ROOT/'sources'/(ident+'_metadata.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
 assert meta['isPublicDomain'];url=meta['primaryImage']
 if not p.exists():
  r=requests.get(url,timeout=(20,60));r.raise_for_status();assert len(r.content)<40_000_000;p.write_bytes(r.content)
 im=Image.open(p).convert('RGB');im.thumbnail((1400,1400));im.save(ROOT/'sources'/(ident+'_overview.jpg'))
 return dict(id=ident,object_id=oid,image_id=iid,split=split,artist=artist,title=title,source_url=f'https://www.metmuseum.org/art/collection/search/{oid}',image_url=url,image=str(p),image_sha256=sha(p),size=Image.open(p).size,group='met_'+str(oid),public_domain=True,role_labels_not_from_museum=True)
def main():
 (ROOT/'sources').mkdir(parents=True,exist_ok=True)
 selection=ROOT/'source_selection_before_download.json'
 if selection.exists():assert json.loads(selection.read_text('utf8'))['specs']==[list(s) for s in SPECS]
 else:selection.write_text(json.dumps(dict(specs=SPECS,selection='Two development works and one attributed work for first check; no style labels propagated; all leaves/crops of an object one group',frozen_before_download=True),ensure_ascii=False,indent=2),encoding='utf8')
 with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(collect,SPECS))
 (ROOT/'source_inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(rows,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
