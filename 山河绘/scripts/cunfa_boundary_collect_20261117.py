# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests
from PIL import Image
ROOT=LAB/'cunfa_boundary_relations_20261117'
def main():
 (ROOT/'sources').mkdir(parents=True,exist_ok=True);p=ROOT/'fresh_work_selection.json'
 if not p.exists():p.write_text(json.dumps(dict(object_id=45636,split='fresh_check',selection='Met Ni Zan Woods and Valleys of Mount Yu; whole object reserved for first check; no training or threshold selection',before_download=True),indent=2),encoding='utf8')
 r=requests.get('https://collectionapi.metmuseum.org/public/collection/v1/objects/45636',timeout=30);r.raise_for_status();m=r.json();assert m['isPublicDomain'];(ROOT/'sources/metadata.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')
 dest=ROOT/'sources/nizan.jpg'
 if not dest.exists():
  r=requests.get(m['primaryImage'],timeout=60);r.raise_for_status();assert len(r.content)<40_000_000;dest.write_bytes(r.content)
 im=Image.open(dest).convert('RGB');native=im.size;im.thumbnail((1400,1400));im.save(ROOT/'sources/nizan_overview.jpg')
 s=dict(image=str(dest),size=native,overview_size=im.size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_url=m['objectURL'],image_url=m['primaryImage'],artist=m['artistDisplayName'],title=m['title'],group='met_45636')
 (ROOT/'source_inventory.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(s,ensure_ascii=False))
if __name__=='__main__':main()
