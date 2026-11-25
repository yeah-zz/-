# -*- coding: utf-8 -*-
"""Bounded new-image sourcing for joint center/affinity experiments."""
from rock_v15_common import LAB,read,jsave,digest,FONT,SMALL
from pathlib import Path
from PIL import Image,ImageDraw
from html.parser import HTMLParser
from urllib.parse import urljoin
import urllib.request
import concurrent.futures
import io,math,hashlib,re,shutil
import numpy as np,cv2
from human_mountain_v11 import paste

ROOT=LAB/'joint_center_affinity_v24_20261026'
PAGES=[('zhuge','https://www.sohu.com/a/277347183_100275234',18),('three_faces','https://www.sohu.com/a/412487984_120749958',18),('cai_fupi','https://www.sohu.com/a/993519974_100169461',8),('wang_xingtang','https://www.culturechina.cn/m/167620.html',14)]
EXPLICIT=[('cai_fupi_body',0,'https://www.sohu.com/a/993519974_100169461','https://q2.itc.cn/images01/20260307/ac60bb457be74082bcb7217df2318b5e.jpeg')]

class Images(HTMLParser):
    def __init__(self):super().__init__();self.urls=[]
    def handle_starttag(self,tag,attrs):
        if tag.lower()!='img':return
        d=dict(attrs)
        for k in ['data-src','data-original','src']:
            if d.get(k):self.urls.append(d[k])

def fetch(url,referer=None,limit=6000000):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Referer':referer or url})
    with urllib.request.urlopen(req,timeout=25) as r:
        b=r.read(limit+1)
    if len(b)>limit:raise ValueError('Per-file download cap exceeded')
    return b

def phash(im):
    a=cv2.dct(np.asarray(im.convert('L').resize((32,32)),np.float32))[:8,:8];return (a>np.median(a.reshape(-1)[1:])).reshape(-1)

def main():
    for sub in ['sources','html','contact_sheets']:(ROOT/sub).mkdir(parents=True,exist_ok=True)
    tasks=list(EXPLICIT);errors=[]
    for group,url,cap in PAGES:
        try:
            p=ROOT/'html'/(group+'.html');raw=p.read_bytes() if p.exists() else fetch(url);p.write_bytes(raw);html=raw.decode('utf-8',errors='replace')
            # Current Sohu image tags contain encoded data-src values; the
            # page also publishes original image URLs in its article payload.
            if 'sohu.com' in url:
                originals=re.findall(r'https?://[^\s"\'<>\\]+?/images(?:01)?/[^\s"\'<>\\]+?\.(?:jpeg|jpg|png)',html.replace('\\/','/'),re.I)
                originals=list(dict.fromkeys(u.replace('http://','https://',1) for u in originals if 'sohucs' in u or 'itc.cn' in u))
                if originals:
                    tasks.extend((group+'_body',i,url,u) for i,u in enumerate(originals[:cap]));print('V24_PAGE_BODY',group,len(originals),flush=True);continue
                html=html.split('article-content',1)[-1].split('</article>',1)[0]
            parser=Images();parser.feed(html);urls=[]
            for u in parser.urls:
                u=urljoin(url,u)
                if any(x in u.lower() for x in ['logo','qrcode','avatar','icon','banner']) or u.startswith('data:'):continue
                if 'sohu.com' in url and not any(t in u for t in ['sohucs','itc.cn']):continue
                if u not in urls:urls.append(u)
            tasks.extend((group,i,url,u) for i,u in enumerate(urls[:cap]));print('V24_PAGE',group,len(urls),flush=True)
        except Exception as e:errors.append(dict(page=url,error=str(e)))
    def download(t):
        group,i,page,url=t;dest=ROOT/'sources'/(group+'_'+str(i).zfill(2)+'.png')
        original_hash=None
        if not dest.exists():
            raw=fetch(url,page);im=Image.open(io.BytesIO(raw)).convert('RGB')
            if min(im.size)<120 or max(im.size)<250:raise ValueError('Small image/icon excluded')
            original_hash=hashlib.sha256(raw).hexdigest();im.thumbnail((1800,2000));im.save(dest)
        im=Image.open(dest).convert('RGB');return dict(id=dest.stem,image=str(dest),source_group=group,source_page=page,image_url=url,sha256=digest(dest),download_sha256=original_hash,width=im.width,height=im.height,license_not_established=True,local_research_only=True)
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        future={pool.submit(download,t):t for t in tasks}
        for f in concurrent.futures.as_completed(future):
            try:r=f.result();rows.append(r);print('V24_IMAGE',r['id'],r['width'],r['height'],flush=True)
            except Exception as e:errors.append(dict(url=future[f][3],error=str(e)))
    old=[]
    for folder in [LAB/'autonomous_strokes_v15_20261025'/'raw',LAB/'structure_generalization_v20_20261025'/'sources',LAB/'stroke_diagnosis_v21_20261025'/'images']:
        for p in folder.glob('*.png'):
            try:old.append((str(p),phash(Image.open(p))))
            except OSError:pass
    seen=[]
    for r in sorted(rows,key=lambda x:x['id']):
        im=Image.open(r['image']);h=phash(im);nearest=min([(int(np.count_nonzero(h!=hh)),p) for p,hh in old],default=(64,None));within=min([(int(np.count_nonzero(h!=hh)),p) for p,hh in seen],default=(64,None));seen.append((r['image'],h));r.update(old_phash_distance=nearest[0],nearest_old=nearest[1],possible_old_duplicate=nearest[0]<=7,batch_phash_distance=within[0],possible_batch_duplicate=within[0]<=5)
    rows.sort(key=lambda r:r['id'])
    if (ROOT/'source_manifest.json').exists() and not (ROOT/'initial_source_manifest.json').exists():shutil.copy2(ROOT/'source_manifest.json',ROOT/'initial_source_manifest.json')
    jsave(ROOT/'source_manifest.json',dict(records=rows,errors=errors,pages=[list(x) for x in PAGES],source_page_is_not_independent_artist=True,reviewed_before_training=True))
    for start in range(0,len(rows),20):
        sub=rows[start:start+20];board=Image.new('RGB',(1500,285*math.ceil(len(sub)/5)),'#eeeeea');d=ImageDraw.Draw(board)
        for i,r in enumerate(sub):
            x=i%5*300;y=i//5*285;d.text((x+5,y+4),r['id']+(' OLD?' if r['possible_old_duplicate'] else ''),font=SMALL,fill='black');paste(board,np.asarray(Image.open(r['image'])),x+5,y+28,290,250)
        board.save(ROOT/'contact_sheets'/('SOURCES_'+str(start//20)+'.jpg'),quality=95)
    print('V24_COLLECTED',len(rows),'possible_old',sum(r['possible_old_duplicate'] for r in rows),'errors',len(errors),flush=True)

if __name__=='__main__':main()
