# -*- coding: utf-8 -*-
"""Archive publicly available primary papers/code for the requested review."""
import os,sys,json,hashlib,concurrent.futures
from pathlib import Path
BASE=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\guohua_system_review_20261110')
for key,suffix in {'TEMP':'temp','TMP':'temp','PIP_CACHE_DIR':'cache/pip','TORCH_HOME':'cache/torch','HF_HOME':'cache/hf','XDG_CACHE_HOME':'cache','CUDA_CACHE_PATH':'cache/cuda'}.items():os.environ[key]=str(BASE/suffix)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import requests
from bs4 import BeautifulSoup
FILES={
 'frida.pdf':'https://arxiv.org/pdf/2210.00664',
 'oriental_ink.pdf':'https://arxiv.org/pdf/2308.11187',
 'gwm.pdf':'https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_GWM_Towards_Scalable_Gaussian_World_Models_for_Robotic_Manipulation_ICCV_2025_paper.pdf',
 'animated.pdf':'https://graphics.uni-konstanz.de/publikationen/Tang2018AnimatedConstructionChinese/tang2017.pdf',
 'mac_guohua.pdf':'https://www.nature.com/articles/s41598-026-74735-6.pdf',
 'survey_page.html':'https://jcst.ict.ac.cn/article/doi/10.1007/s11390-024-3408-x',
 'mac_page.html':'https://www.nature.com/articles/s41598-026-74735-6',
 'mac_tree.json':'https://api.github.com/repos/pfeijing/mac-guohua/git/trees/main?recursive=1',
 'frida_readme.md':'https://raw.githubusercontent.com/cmubig/Frida/master/README.md',
 'frida_tree.json':'https://api.github.com/repos/cmubig/Frida/git/trees/master?recursive=1',
 'oriental_full.html':'https://arxiv.org/html/2308.11187v2',
 'frida_full.html':'https://arxiv.org/html/2210.00664v1'}
def fetch(item):
 name,url=item
 try:
  r=requests.get(url,timeout=70,headers={'User-Agent':'Mozilla/5.0'});pdf=name.endswith('.pdf');valid=r.status_code==200 and (r.content.startswith(b'%PDF') if pdf else len(r.content)>50)
  if valid:(BASE/name).write_bytes(r.content)
  return dict(name=name,url=url,status=r.status_code,bytes=len(r.content),saved=valid,content_type=r.headers.get('content-type'),sha256=hashlib.sha256(r.content).hexdigest() if valid else None)
 except Exception as exc:return dict(name=name,url=url,saved=False,error=str(exc))
def main():
 BASE.mkdir(exist_ok=True);records=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
  for r in pool.map(fetch,FILES.items()):records.append(r);print(json.dumps(r),flush=True)
 (BASE/'collection_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
 if (BASE/'survey_page.html').exists():
  soup=BeautifulSoup((BASE/'survey_page.html').read_text('utf8',errors='replace'),'html.parser');links=[]
  for tag in soup.find_all(['a','meta','script']):
   value=str(tag)
   if any(x in value.lower() for x in ['pdf','fulltext','3408','articleid','article_id']):links.append(value[:1800])
  (BASE/'survey_pdf_hints.txt').write_text('\n'.join(links),encoding='utf8')
 for name in ['mac_tree.json','frida_tree.json']:
  if (BASE/name).exists():
   obj=json.loads((BASE/name).read_text('utf8'));paths=[x['path'] for x in obj.get('tree',[]) if x['type']=='blob'];(BASE/name.replace('_tree.json','_paths.txt')).write_text('\n'.join(paths),encoding='utf8');print(name,paths[:70],flush=True)
if __name__=='__main__':main()
