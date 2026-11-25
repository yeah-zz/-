# -*- coding: utf-8 -*-
"""Read-only public teaching sources; hash and extract to the E-drive lab."""
import sys,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests
sys.path.insert(0,str(LAB/'guohua_system_review_20261110/deps'))
import pymupdf
ROOT=LAB/'cunfa_cross_source_20261117'
SOURCES=[
 ('qingpu','https://m.qplll.net/File/Course/QP_COURSE_PDF/中国山水画法基础.pdf','Qingpu community teaching material'),
 ('ntmofa','https://twfineartsarchive.ntmofa.gov.tw/QuarterlyFile/B000702.pdf','National Taiwan Museum of Fine Arts archive')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def collect(spec):
 ident,url,publisher=spec;dest=ROOT/'sources'/(ident+'.pdf')
 try:
  if not dest.exists():
   with requests.get(url,stream=True,timeout=(15,45)) as response:
    response.raise_for_status();assert int(response.headers.get('Content-Length','0'))<90_000_000
    size=0
    with dest.open('wb') as stream:
     for chunk in response.iter_content(262144):
      size+=len(chunk);assert size<90_000_000;stream.write(chunk)
  assert dest.read_bytes()[:4]==b'%PDF'
  doc=pymupdf.open(dest);pages=[]
  for i,page in enumerate(doc):
   text=page.get_text();(ROOT/'sources'/(ident+'_'+str(i+1)+'.txt')).write_text(text,encoding='utf8')
   pages.append(dict(page=i+1,text=text,image_count=len(page.get_images(full=True))))
  # All pages stay in the provenance inventory. Render only likely learning
  # pages plus front matter; do not infer labels from text match alone.
  selected=[p['page'] for p in pages if p['page']<=3 or (ident=='qingpu' and any(s in p['text'] for s in ['皴','勾','点苔','染'])) or (ident=='ntmofa' and any(s in p['text'] for s in ['皴法','米點皴','披麻皴']))]
  for number in selected:
   doc[number-1].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(ROOT/'pages'/(ident+'_'+str(number)+'.png'))
  return dict(id=ident,url=url,publisher=publisher,path=str(dest),sha256=sha(dest),size=dest.stat().st_size,pages=pages,rendered_pages=selected,status='downloaded',artist_verified=False,model_labels_created=False)
 except Exception as exc:return dict(id=ident,url=url,status='unavailable',reason=str(exc))
def main():
 (ROOT/'sources').mkdir(parents=True,exist_ok=True);(ROOT/'pages').mkdir(exist_ok=True)
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(collect,SOURCES))
 (ROOT/'source_inventory.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
 for r in results:print(json.dumps({k:v for k,v in r.items() if k!='pages'},ensure_ascii=False),flush=True)
 for r in results:
  if r['status']=='downloaded':
   print(r['id'],'PAGE_TOPICS',json.dumps([dict(page=p['page'],text=p['text'][:110]) for p in r['pages']],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
