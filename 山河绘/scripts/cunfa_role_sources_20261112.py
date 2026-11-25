# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests
sys.path.insert(0,str(LAB/'guohua_system_review_20261110/deps'))
import pymupdf
ROOT=LAB/'cunfa_role_reference_20261112'
def main():
 (ROOT/'sources').mkdir(exist_ok=True);url='https://hk.art.museum/Archive/resources/S_Wucius_WONG_Suggested_Class_Activities_CHIN.pdf';path=ROOT/'sources/hkma_wucius_activities.pdf'
 if not path.exists():r=requests.get(url,timeout=60);r.raise_for_status();assert r.content.startswith(b'%PDF');path.write_bytes(r.content)
 doc=pymupdf.open(path);page=doc[3];page.get_pixmap(matrix=pymupdf.Matrix(2,2)).save(ROOT/'sources/hkma_page4.png')
 records=[]
 for i,im in enumerate(page.get_images(full=True)):
  xref=im[0];a=doc.extract_image(xref);dest=ROOT/'sources'/f'hkma_p4_img{i}.{a["ext"]}';dest.write_bytes(a['image']);records.append(dict(index=i,path=str(dest),size=[a['width'],a['height']],rects=[list(r) for r in page.get_image_rects(xref)],sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
 data=dict(url=url,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pages=len(doc),used_page_one_based=4,images=records,
  source_title='王無邪 初中教學活動建議',publisher='Hong_Kong_Museum_of_Art',illustration_artist_identity_verified=False,role='pedagogical reference, not expert pixel annotation',
  stage_evidence={'outline_before_texture':'PDF page3 teaches baimiao practice, page4 teaches axe/hemp technique; instructional sequence, not original-artist recovered chronology'},text=page.get_text())
 (ROOT/'source_manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(data,ensure_ascii=False),flush=True)
 # Inspect public viewer metadata only; stop on any access failure.
 u='https://www.dpm.org.cn/Uploads/tilegenerator/dest/files/image/8831/2007/1305/img0005.xml'
 try:
  r=requests.get(u,timeout=25);r.raise_for_status();(ROOT/'sources/xia_public_viewer_metadata.xml').write_bytes(r.content);print('VIEWER_METADATA',r.text[:1800],flush=True)
 except Exception as e:print('VIEWER_UNAVAILABLE',str(e),flush=True)
if __name__=='__main__':main()
