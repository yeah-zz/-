"""Bounded public tutorial-image collection, never paid originals or entire books."""
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from PIL import Image, ImageDraw, ImageFont
from .runtime import LAB

CORPUS = LAB/'rock_cunfa_v04'
PAGES = [
 'https://mskj.caa.edu.cn/info/1010/18051.htm',
 'https://www.sohu.com/a/1054023618_121124794',
 'https://www.sohu.com/a/259730799_744221',
 'https://www.sohu.com/a/300227543_764920',
 'https://www.sohu.com/a/620541556_121124711',
 'https://www.sohu.com/a/260258671_271454',
 'https://www.sohu.com/a/283639067_99908875',
 'https://www.sohu.com/a/238292082_99941818',
 'https://www.sohu.com/a/302810213_99953422',
 'https://www.sohu.com/a/436682033_271454',
 'https://www.sohu.com/a/196559761_711363',
 'https://www.sohu.com/a/248821814_683980',
 'https://www.sohu.com/a/244115867_789939',
 'https://www.xmcdw.com/html/course-13.html',
 'https://www.9900.com.tw/talk/SMobile.aspx?jid=3d1eee4a5254004ff947&stype=S101',
]
PAGES += [f'https://www.xmcdw.com/html/course-{n}.html' for n in (12,14,15,16,18,197,198,199,200,201)]
PAGES += ['https://www.muzhiart.com/cms/show-3051.html',
          'https://k.sina.cn/article_6495090570_18323378a00100dap5.html',
          'https://kknews.cc/n/8g3q28n.html']

class Images(HTMLParser):
    def __init__(self):
        super().__init__(); self.images=[];self.title='';self.in_title=False
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=='title':self.in_title=True
        if tag=='img':
            for key in ('data-src','data-original','src'):
                if d.get(key):self.images.append(d[key]);break
    def handle_endtag(self,tag):
        if tag=='title':self.in_title=False
    def handle_data(self,data):
        if self.in_title:self.title+=data

def fetch(url,referer=None):
    headers={'User-Agent':'Mozilla/5.0 (educational local image study)'}
    if referer:headers['Referer']=referer
    with urlopen(Request(url,headers=headers),timeout=35) as f:
        raw=f.read(20_000_001)
    if len(raw)>20_000_000:raise ValueError('20MB response cap')
    return raw

def sheet(records,name='raw',per_page=72):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    dest=CORPUS/'review';dest.mkdir(exist_ok=True)
    for start in range(0,len(records),per_page):
        batch=records[start:start+per_page];canvas=Image.new('RGB',(1440,((len(batch)+5)//6)*158),'#eeeeea');draw=ImageDraw.Draw(canvas)
        for k,r in enumerate(batch):
            im=Image.open(r['image']).convert('RGB');im.thumbnail((232,130))
            x=(k%6)*240;y=(k//6)*158
            canvas.paste(im,(x+(232-im.width)//2,y))
            draw.text((x+3,y+132),r['id'],font=font,fill='black')
        canvas.save(dest/f'{name}_{start//per_page+1:02d}.jpg',quality=90)

def main(argv=None):
    for folder in ('html','web','references','review'):(CORPUS/folder).mkdir(parents=True,exist_ok=True)
    tasks=[];pages=[];errors=[]
    for pi,url in enumerate([] if '--additional-only' in (argv or []) else PAGES):
        path=CORPUS/'html'/f'page_{pi:02d}.html'
        try:
            raw=path.read_bytes() if path.exists() else fetch(url)
            path.write_bytes(raw)
            html=raw.decode('utf-8',errors='replace');p=Images();p.feed(html)
            # Restrict to article images; exclude avatars, recommendations, logos.
            candidates=[]
            if 'sohu.com' in url:
                body=html.split('article-content',1)[-1].split('</article>',1)[0]
                article=Images();article.feed(body)
                candidates=[u for u in article.images if any(d in u for d in ('itc.cn','sohucs.com'))]
                if not candidates:candidates=[u for u in p.images if any(d in u for d in ('itc.cn','sohucs.com'))]
            elif 'caa.edu.cn' in url:candidates=[u for u in p.images if 'virtual_attach_file.vsb' in u or '/__local/' in u]
            elif 'xmcdw.com' in url:
                body=html.split('article-content',1)[-1].split('post-copyright',1)[0]
                article=Images();article.feed(body);candidates=article.images
            else:candidates=p.images
            candidates=list(dict.fromkeys(urljoin(url,u) for u in candidates if not u.startswith('data:')))
            pages.append(dict(id=f'page_{pi:02d}',url=url,title=p.title,images=len(candidates)))
            tasks.extend((pi,ii,url,img) for ii,img in enumerate(candidates[:100]))
            print('PAGE',pi,len(candidates),p.title[:70],flush=True)
        except Exception as exc:errors.append(dict(url=url,error=str(exc)))
    additional=Path(__file__).with_name('cunfa_public_urls.json')
    if additional.exists():
        extra=json.loads(additional.read_text(encoding='utf-8'));groups={}
        for item in extra:
            if item['page'] not in groups:groups[item['page']]=28+len(groups)
            pi=groups[item['page']]
            ii=sum(t[0]==pi for t in tasks)
            tasks.append((pi,ii,item['page'],item['url']))
    def download(task):
        pi,ii,page,url=task;identity=f'web_{pi:02d}_{ii:03d}';dest=CORPUS/'web'/(identity+'.png')
        if not dest.exists():
            raw=fetch(url,page);im=Image.open(io.BytesIO(raw));im.load();im=im.convert('RGB')
            if min(im.size)<100 or max(im.size)<220:raise ValueError('small icon/thumbnail')
            im.thumbnail((2000,2000));im.save(dest)
        im=Image.open(dest)
        return dict(id=identity,source_page=page,image_url=url,page_group=f'page_{pi:02d}',image=str(dest),width=im.width,height=im.height,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),license='Public article illustration; no open reuse license asserted')
    old=CORPUS/'web_sources.json'
    records=json.loads(old.read_text(encoding='utf-8')) if '--additional-only' in (argv or []) and old.exists() else []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(download,t):t for t in tasks}
        for future in as_completed(futures):
            try:records.append(future.result())
            except Exception as exc:
                task=futures[future];errors.append(dict(url=task[3],error=str(exc)))
    records.sort(key=lambda r:r['id']);unique={}
    for r in records:
        if r['sha256'] not in unique:unique[r['sha256']]=r
        elif unique[r['sha256']]['id']!=r['id']:unique[r['sha256']].setdefault('also_seen_on',[]).append(r['source_page'])
    records=list(unique.values())
    (CORPUS/'web_sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    (CORPUS/'collection_audit.json').write_text(json.dumps(dict(pages=pages,attempted=len(tasks),unique_images=len(records),errors=errors),ensure_ascii=False,indent=2),encoding='utf-8')
    sheet(records)
    print('COLLECTION_COMPLETE',len(records),'unique images',len(errors),'errors',str(CORPUS/'review'),flush=True)
