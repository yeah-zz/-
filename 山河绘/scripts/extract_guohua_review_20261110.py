# -*- coding: utf-8 -*-
import os,sys,json,concurrent.futures
from pathlib import Path
BASE=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\guohua_system_review_20261110')
for key,suffix in {'TEMP':'temp','TMP':'temp','PIP_CACHE_DIR':'cache/pip','TORCH_HOME':'cache/torch','HF_HOME':'cache/hf','XDG_CACHE_HOME':'cache','CUDA_CACHE_PATH':'cache/cuda'}.items():os.environ[key]=str(BASE/suffix)
sys.dont_write_bytecode=True;sys.path.insert(0,str(BASE/'deps'))
import pymupdf,requests
def main():
 p=BASE/'six_principles_survey.pdf'
 if not p.exists():
  urls=['https://jcst.ict.ac.cn/cn/article/pdf/preview/10.1007/s11390-024-3408-x.pdf','https://jcst.ict.ac.cn/fileup/1000-9000/PDF/JCST-2024-2-4-3408-269.pdf','https://link.springer.com/content/pdf/10.1007/s11390-024-3408-x.pdf']
  for url in urls:
   try:
    r=requests.get(url,timeout=35,headers={'User-Agent':'Mozilla/5.0'});r.raise_for_status();assert r.content.startswith(b'%PDF');p.write_bytes(r.content);print('SURVEY_SAVED',url,len(r.content),flush=True);break
   except Exception as exc:print('SURVEY_ATTEMPT',url,str(exc)[:160],flush=True)
 summaries=[]
 for p in BASE.glob('*.pdf'):
  doc=pymupdf.open(p);out=BASE/p.stem;out.mkdir(exist_ok=True);pages=[]
  for i,page in enumerate(doc):
   txt=page.get_text(sort=True);(out/f'page_{i+1:02d}.txt').write_text(txt,encoding='utf8');pages.append(dict(page=i+1,characters=len(txt),first_lines=txt[:180].replace('\n',' ')))
  (out/'full_text.txt').write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n'+doc[i].get_text(sort=True) for i in range(len(doc))),encoding='utf8');summaries.append(dict(file=p.name,pages=len(doc),characters=sum(r['characters'] for r in pages),page_summaries=pages))
 (BASE/'pdf_inventory.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(summaries,ensure_ascii=False),flush=True)
 # Archive current public code, without executing it.
 requests_list=[]
 paths=(BASE/'mac_paths.txt').read_text('utf8').splitlines()
 for path in paths:requests_list.append(('mac_code/'+path,'https://raw.githubusercontent.com/pfeijing/mac-guohua/main/'+path))
 fpaths=(BASE/'frida_paths.txt').read_text('utf8').splitlines()
 for path in fpaths:
  if path.startswith('src/') and (any(x in path for x in ['stroke','paint','param','options']) or path.endswith('robot.py')) and path.endswith('.py'):requests_list.append(('frida_code/'+path,'https://raw.githubusercontent.com/cmubig/Frida/master/'+path))
 def fetch(item):
  name,url=item
  try:
   rr=requests.get(url,timeout=35);rr.raise_for_status();dest=BASE/name;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(rr.content);return dict(file=name,url=url,bytes=len(rr.content),saved=True)
  except Exception as exc:return dict(file=name,url=url,error=str(exc),saved=False)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(fetch,requests_list))
 (BASE/'code_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8');print('CODE_FILES',len(records),sum(r['saved'] for r in records),flush=True)
if __name__=='__main__':main()
