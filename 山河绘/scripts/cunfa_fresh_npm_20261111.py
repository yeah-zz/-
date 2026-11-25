# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from bs4 import BeautifulSoup
import requests
from PIL import Image,ImageDraw,ImageFont
from urllib.parse import urljoin
ROOT=LAB/'cunfa_fresh_validation_20261111'
def main():
 p=ROOT/'sources/npm_5621.html';s=BeautifulSoup(p.read_bytes(),'html.parser');imgs=[x for x in s.select('img') if '/Image/GetImage?' in x.get('src','')];out=[]
 for i,tag in enumerate(imgs):
  url=urljoin('https://digitalarchive.npm.gov.tw',tag['src']);dest=ROOT/'sources'/f'npm_5621_{i}.jpg'
  if not dest.exists():
   r=requests.get(url,timeout=50);r.raise_for_status();dest.write_bytes(r.content)
  im=Image.open(dest);out.append(dict(url=url,path=str(dest),size=list(im.size),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()));print(i,im.size,flush=True)
 # Public image preview URLs from HTML only. No download form, CAPTCHA or IIIF access.
 data=dict(id='npm_5621',artist='Wen_Zhengming',url='https://digitalarchive.npm.gov.tw/Collection/Detail/5621?dep=P',title='明文徵明山水　卷',label='披麻皴',evidence='Catalogue techniques table lists hemp-fibre cun; work-level metadata, not expert crop annotation',
  text_attribution='明文徵明山水 卷。國立故宮博物院，台北，CC BY 4.0 @ www.npm.gov.tw',public_preview_only=True,images=out)
 (ROOT/'npm_source_manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
 canvas=Image.new('RGB',(1500,len(out)*330),'#f4f1e8');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
 for i,r in enumerate(out):
  im=Image.open(r['path']).convert('RGB');im.thumbnail((1480,285));canvas.paste(im,(10,i*330+35));d.text((10,i*330),str(i)+' '+str(r['size']),font=font,fill='black')
 canvas.save(ROOT/'npm_source_review.jpg',quality=93)
if __name__=='__main__':main()
