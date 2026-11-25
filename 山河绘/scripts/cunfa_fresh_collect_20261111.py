# -*- coding: utf-8 -*-
import sys,json,hashlib,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests
from bs4 import BeautifulSoup
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_fresh_validation_20261111'
SPECS={'231531':'Wang_Shimin','234350':'Ma_Yuan','233555':'Wang_Yuanqi','231419':'Dai_Jin','229559':'Tang_Yin','231773':'Wang_Jian','230757':'Xia_Gui'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fetch(item):
 key,artist=item;url=f'https://www.dpm.org.cn/collection/paint/{key}.html';p=ROOT/'sources'/(key+'.html')
 try:
  if not p.exists():r=requests.get(url,timeout=50);r.raise_for_status();p.write_bytes(r.content)
  soup=BeautifulSoup(p.read_bytes(),'html.parser');divs=[x for x in soup.select('div.text') if len(x.get_text(' ',strip=True))>120]
  text=next((x.get_text(' ',strip=True) for x in divs if x.find('p')),divs[0].get_text(' ',strip=True) if divs else '')
  article=text.split('关键词')[0];imgs=[];seen=set()
  for im in soup.select('img[data-img]'):
   u=im.get('data-img')
   if u in seen:continue
   seen.add(u);i=len(imgs);dest=ROOT/'sources'/f'{key}_{i}.jpg'
   if not dest.exists():r=requests.get(u,timeout=45);r.raise_for_status();dest.write_bytes(r.content)
   with Image.open(dest) as a:size=list(a.size)
   imgs.append(dict(url=u,path=str(dest),sha256=sha(dest),size=size,title=im.get('data-option','')))
   if len(imgs)==2:break
  return dict(id=key,artist=artist,url=url,article=article,images=imgs,status='collected')
 except Exception as e:return dict(id=key,url=url,status='unavailable',error=str(e))
def main():
 assert (ROOT/'freeze_manifest.json').exists();(ROOT/'sources').mkdir(exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,SPECS.items()))
 (ROOT/'source_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
 for r in records:print(r['id'],r.get('artist'),r.get('article','')[-1300:],[(i['size']) for i in r.get('images',[])],flush=True)
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);allim=[(r,i) for r in records for i in r.get('images',[])];out=Image.new('RGB',(1500,((len(allim)+3)//4)*550),'#f4f1e8');d=ImageDraw.Draw(out)
 for n,(r,i) in enumerate(allim):
  x=n%4*375;y=n//4*550;im=Image.open(i['path']).convert('RGB');im.thumbnail((365,500));out.paste(im,(x,y+48));d.text((x+5,y+2),r['id']+' '+r['artist'],font=font,fill='black');d.text((x+5,y+25),str(i['size']),font=font,fill='black')
 out.save(ROOT/'source_review.jpg',quality=93)
if __name__=='__main__':main()
