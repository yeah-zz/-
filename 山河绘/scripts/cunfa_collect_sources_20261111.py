import sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests
from bs4 import BeautifulSoup
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_structure_20261111'
IDS=['230692','230678','232500','228140','228153','234462']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 (ROOT/'sources').mkdir(parents=True,exist_ok=True);records=[]
 for ident in IDS:
  url=f'https://www.dpm.org.cn/collection/paint/{ident}.html';p=ROOT/'sources'/(ident+'.html')
  if not p.exists():
   r=requests.get(url,timeout=45);r.raise_for_status();p.write_bytes(r.content)
  soup=BeautifulSoup(p.read_bytes(),'html.parser')
  # The article's text container, excluding keyword glossary and biography.
  candidates=soup.select('div.text')
  choices=[e for e in candidates if len(e.get_text(' ',strip=True))>120]
  main=next((e for e in choices if e.find('p')),choices[0] if choices else None)
  text=main.get_text(' ',strip=True) if main else ''
  # Save all potential article containers for human source audit if selector differs.
  item=dict(id=ident,url=url,article_candidates=[e.get_text(' ',strip=True) for e in choices],main_text=text,images=[])
  imgs=soup.select('img[data-img]');unique=[]
  for im in imgs:
   u=im.get('data-img')
   if u not in [x[0] for x in unique]:unique.append((u,im.get('data-option','')))
  # Selected leaves only; no high-resolution tile reconstruction.
  allowed={'230692':[6],'230678':[0],'232500':[0],'228140':[0,1],'228153':[0],'234462':[0,1,2,3]}[ident]
  for idx in allowed:
   if idx>=len(unique):continue
   u,title=unique[idx];dest=ROOT/'sources'/f'{ident}_{idx}.jpg'
   if not dest.exists():
    r=requests.get(u,timeout=50);r.raise_for_status();dest.write_bytes(r.content)
   with Image.open(dest) as im:size=list(im.size)
   item['images'].append(dict(url=u,path=str(dest),title=title,index=idx,size=size,sha256=sha(dest)))
  records.append(item);print(ident,'TEXT',text[:2600], 'IMAGES',[(x['index'],x['size']) for x in item['images']],flush=True)
 (ROOT/'source_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
 allim=[(r,i) for r in records for i in r['images']];out=Image.new('RGB',(1500,((len(allim)+3)//4)*510),'#f4f1e8');d=ImageDraw.Draw(out)
 for n,(r,i) in enumerate(allim):
  x=n%4*375;y=n//4*510;im=Image.open(i['path']).convert('RGB');im.thumbnail((365,455));out.paste(im,(x,y+48));d.text((x+5,y+3),r['id']+'_'+str(i['index']),font=font,fill='black');d.text((x+5,y+25),str(i['size']),font=font,fill='black')
 out.save(ROOT/'new_sources.jpg',quality=92)
if __name__=='__main__':main()
