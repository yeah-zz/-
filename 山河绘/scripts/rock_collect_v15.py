# -*- coding: utf-8 -*-
"""Bounded tutorial-image sourcing, provenance, deduplication, crop proposals."""
from rock_v15_common import *
import io,math,concurrent.futures,re
import numpy as np,cv2
from PIL import Image,ImageDraw
from urllib.parse import urljoin
from gongbi_training.cunfa_collect import Images,fetch
PAGES=[
 ('kuhw_1975','https://www.kuhw.com/hhjc/sshjc/2019-09-12/1975.html'),
 ('kuhw_28593','https://www.kuhw.com/hhjc/sshjc/2019-12-14/28593.html'),
 ('kuhw_29483','https://www.kuhw.com/hhjc/sshjc/2020-02-12/29483.html'),
 ('kuhw_2256','https://www.kuhw.com/hhjc/sshjc/2019-09-12/2256.html'),
 ('whb_302198','https://www.whb.cn/zhuzhan/whh/20191111/302198.html'),
 ('meiyuan_1251','https://www.meiyuanbang.com/m/newdetail/1251'),
 ('sohu_248756255','https://www.sohu.com/a/248756255_99953422'),
 ('sina_6586334590','https://k.sina.com.cn/article_6586334590_188937d7e00100ip0p.html'),
 ('airmb_77457','https://m.airmb.com/shuhua/gh/2019/1118/77457.html')]

def phash(im):
    g=np.asarray(im.convert('L').resize((32,32)),np.float32);d=cv2.dct(g)[:8,:8];b=(d>np.median(d.flatten()[1:])).flatten();return b

def sheet(rows,name):
    for start in range(0,len(rows),40):
        batch=rows[start:start+40];out=Image.new('RGB',(1500,220*math.ceil(len(batch)/5)),'#ecebe7');d=ImageDraw.Draw(out)
        for i,r in enumerate(batch):
            im=Image.open(r['image']).convert('RGB');im.thumbnail((290,188));x=(i%5)*300;y=(i//5)*220;out.paste(im,(x+(290-im.width)//2,y));d.text((x+3,y+188),r['id'],font=SMALL,fill='black')
        out.save(ROOT/'contact_sheets'/f'{name}_{start//40:02d}.jpg',quality=92)

def collect():
    for p in ['html','raw','contact_sheets','crop_proposals']:(ROOT/p).mkdir(parents=True,exist_ok=True)
    old=[]
    for p in [LAB/'rock_cunfa_v04'/'web_sources.json',LAB/'clean_strokes_v12_20261024'/'web_sources.json',LAB/'rock_multitask_v09_20261024'/'new_sources.json']:
        if p.exists():old+=read(p)
    oldhash=[]
    for r in old:
        try:oldhash.append(phash(Image.open(r['image'])))
        except (KeyError,OSError):pass
    tasks=[];pages=[];errors=[]
    for group,url in PAGES:
        try:
            path=ROOT/'html'/(group+'.html');raw=path.read_bytes() if path.exists() else fetch(url);path.write_bytes(raw);html=raw.decode('utf-8',errors='replace')
            if 'sohu.com' in url:html=html.split('article-content',1)[-1].split('</article>',1)[0]
            if 'kuhw.com' in url:html=html.split('<h1',1)[-1].split('上一篇',1)[0]
            parser=Images();parser.feed(html);urls=list(dict.fromkeys(urljoin(url,u) for u in parser.images if not u.startswith('data:') and not any(t in u.lower() for t in ['logo','icon','qrcode','avatar','banner','advert'])))
            if 'sohu.com' in url:urls=[u for u in urls if 'sohucs.com' in u or 'itc.cn' in u]
            tasks += [(group,i,url,u) for i,u in enumerate(urls[:42])];pages.append(dict(group=group,url=url,candidates=len(urls)));print('NEW_SOURCE',group,len(urls),flush=True)
        except Exception as e:errors.append(dict(page=url,error=str(e)))
    def get(t):
        group,i,page,u=t;p=ROOT/'raw'/(group+'_'+str(i).zfill(2)+'.png')
        if not p.exists():
            raw=fetch(u,page);im=Image.open(io.BytesIO(raw)).convert('RGB')
            if min(im.size)<150 or max(im.size)<250:raise ValueError('small/icon')
            im.thumbnail((2000,2200));im.save(p)
        im=Image.open(p).convert('RGB');return dict(id=p.stem,image=str(p),width=im.width,height=im.height,source_page=page,image_url=u,source_group=group,sha256=digest(p),license='not established; local research only, do not redistribute originals',expert_verified=False),phash(im)
    rows=[];hashes=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        fs={pool.submit(get,t):t for t in tasks}
        for f in concurrent.futures.as_completed(fs):
            try:r,h=f.result();rows.append(r);hashes.append(h)
            except Exception as e:errors.append(dict(url=fs[f][3],error=str(e)))
    # Deterministic IDs; duplicates are recorded rather than called new works.
    out=[];seen=[]
    for r,h in sorted(zip(rows,hashes),key=lambda x:x[0]['id']):
        r['near_duplicate_old']=any(np.count_nonzero(h!=v)<=5 for v in oldhash)
        r['near_duplicate_this_batch']=any(np.count_nonzero(h!=v)<=5 for v in seen);seen.append(h)
        im=np.asarray(Image.open(r['image']).convert('RGB'));g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);sat=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)[:,:,1];r['paper_bright_fraction']=float((g>190).mean());r['mean_saturation']=float(sat.mean());r['gray_ink_fraction']=float((g<135).mean());r['eligible_clear_candidate']=not r['near_duplicate_old'] and not r['near_duplicate_this_batch'] and r['paper_bright_fraction']>.35 and r['mean_saturation']<65 and .005<r['gray_ink_fraction']<.55;out.append(r)
    jsave(ROOT/'sources.json',out);jsave(ROOT/'collection_audit.json',dict(pages=pages,raw_unique_downloads=len(out),old_hashes_compared=len(oldhash),new_clear_candidates=sum(r['eligible_clear_candidate'] for r in out),errors=errors,source_page_not_equal_artwork_independence=True));sheet(out,'raw')
    candidates=[r for r in out if r['eligible_clear_candidate']];sheet(candidates,'new_clear')
    print('COLLECTED_V15',len(out),'NEW_CLEAR_CANDIDATES',len(candidates),flush=True)

if __name__=='__main__':collect()
