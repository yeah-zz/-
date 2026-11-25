# -*- coding: utf-8 -*-
"""Bounded clean-rock collection. Pipeline training/evaluation is offline."""
import sys,json,math,hashlib,io,re,shutil,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from urllib.request import Request,urlopen
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor,as_completed
from gongbi_training.cunfa_collect import Images
ROOT=LAB/'clean_strokes_v12_20261024'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
PAGES=[('artist_steps','https://www.sohu.com/a/277347183_100275234'),('wang_steps','https://www.culturechina.cn/m/167620.html'),('north_methods','https://www.zgshscjxh.com/nd.jsp?id=1588')]
PUBLIC_IMAGE_LINKS=[
 ('outline_01','b8615ac0be8341fcbc539d950b821fe0'),('outline_02','331c942d146d43c4a4629cd542472c31'),
 ('pima_01','69101840106d4058b97f382f32d77346'),('pima_02','81087be476c14a39bd822288eee4968f'),('pima_03','9d5960122ef341b9ae2258121cb225bb'),('pima_04','6fcfe77c5f8a47478391a451d27b4f73'),
 ('heye_01','42e3726a156b444683058a7d8b647448'),('heye_02','ff09d50a6582417980e1516fd58edb1f'),('heye_03','b4e79f0037834863aca5d3c6e9f73e4c'),('heye_04','eec47646bb01417aab13afc01dd54336')]
def jsave(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fetch(url,referer=None):
    headers={'User-Agent':'Mozilla/5.0'}
    if referer:headers['Referer']=referer
    with urlopen(Request(url,headers=headers),timeout=25) as response:body=response.read(18_000_001)
    if len(body)>18_000_000:raise ValueError('response exceeds 18MB cap')
    return body
def sheet(records,dest,per_page=36):
    for start in range(0,len(records),per_page):
        batch=records[start:start+per_page];board=Image.new('RGB',(1500,250*math.ceil(len(batch)/5)),'#f5f3ec');d=ImageDraw.Draw(board)
        for j,r in enumerate(batch):
            im=Image.open(r['image']).convert('RGB');im.thumbnail((288,210));x=(j%5)*300;y=(j//5)*250;board.paste(im,(x+(288-im.width)//2,y));d.text((x+4,y+213),r['id']+' '+str(r['width'])+'x'+str(r['height']),font=FONT,fill='black')
        board.save(dest/f'raw_{start//per_page+1:02d}.jpg',quality=94)
def collect():
    for name in ['html','raw','review','data','diagnostics','runs','evaluation']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    old={}
    for name in ['rock_cunfa_v04/web_sources.json','rock_multitask_v09_20261024/new_sources.json','rock_multitask_v09_20261024/new_style_probes.json']:
        path=LAB/name
        if path.exists():
            for r in json.loads(path.read_text(encoding='utf-8')):
                if r.get('sha256'):old[r['sha256']]=r.get('image',r.get('id'))
    tasks=[];errors=[];pages=[]
    for group,url in PAGES:
        try:
            dest=ROOT/'html'/(group+'.html');body=dest.read_bytes() if dest.exists() else fetch(url);dest.write_bytes(body);html=body.decode('utf-8',errors='replace')
            if 'sohu.com' in url:article=html.split('article-content',1)[-1].split('</article>',1)[0]
            else:article=html
            parser=Images();parser.feed(article);urls=list(dict.fromkeys(urljoin(url,u) for u in parser.images if not u.startswith('data:')))
            if group=='artist_steps':urls=[u for u in urls if 'sohucs.com' in u or 'itc.cn' in u]
            elif group=='wang_steps':urls=[u for u in urls if 'img.culturechina.cn' in u]
            else:urls=[u for u in urls if not any(t in u.lower() for t in ['logo','icon','banner','qrcode'])]
            pages.append(dict(group=group,url=url,candidates=len(urls)));tasks.extend((group,i,url,img) for i,img in enumerate(urls[:48]))
            print('SOURCE_PAGE',group,len(urls),flush=True)
        except Exception as e:errors.append(dict(page=url,error=str(e)))
    def download(t):
        group,index,page,url=t;identity=group+'_'+str(index).zfill(2);p=ROOT/'raw'/(identity+'.png')
        if not p.exists():
            body=fetch(url,page);im=Image.open(io.BytesIO(body));im.load();im=im.convert('RGB')
            if min(im.size)<160 or max(im.size)<250:raise ValueError('thumbnail/icon')
            im.thumbnail((2000,2000));im.save(p)
        im=Image.open(p);sha=digest(p)
        return dict(id=identity,image=str(p),width=im.width,height=im.height,sha256=sha,source_page=page,image_url=url,source_group=group,old_exact_duplicate=sha in old,annotation_status='unlabeled',underlying_book_family_verified=False)
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(download,t):t for t in tasks}
        for f in as_completed(futures):
            try:records.append(f.result())
            except Exception as e:errors.append(dict(url=futures[f][3],error=str(e)))
    # Explicit public links exposed by the article reader. The article HTML
    # uses lazy-image placeholders; do not decode or circumvent that mechanism.
    for name,identifier in PUBLIC_IMAGE_LINKS:
        try:
            url='https://5b0988e595225.cdn.sohucs.com/images/20181123/'+identifier+'.jpeg';path=ROOT/'raw'/(name+'.png')
            if not path.exists():
                im=Image.open(io.BytesIO(fetch(url,PAGES[0][1]))).convert('RGB');im.thumbnail((2000,2000));im.save(path)
            im=Image.open(path);sha=digest(path);records.append(dict(id=name,image=str(path),width=im.width,height=im.height,sha256=sha,source_page=PAGES[0][1],image_url=url,source_group='artist_steps',old_exact_duplicate=sha in old,annotation_status='unlabeled',underlying_book_family_verified=False))
        except Exception as e:errors.append(dict(id=name,error=str(e)))
    records.sort(key=lambda r:r['id']);seen={};unique=[]
    for r in records:
        if r['sha256'] in seen:r['duplicate_of']=seen[r['sha256']]
        else:seen[r['sha256']]=r['id'];unique.append(r)
    jsave(ROOT/'web_sources.json',records);jsave(ROOT/'collection_audit.json',dict(pages=pages,downloaded=len(records),exact_unique=len(unique),exact_old_duplicates=sum(r['old_exact_duplicate'] for r in records),errors=errors));sheet(unique,ROOT/'review')
    print('COLLECTION_COMPLETE',len(records),len(unique),'unique',len(errors),'errors',flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['collect']);a=parser.parse_args();collect()
