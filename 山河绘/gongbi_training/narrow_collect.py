"""Collect a new narrow-style corpus; provenance and counts stay explicit."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from .runtime import LAB

CORPUS=LAB/'narrow_corpus_v03'
UA='GongbiLineResearch/0.3 (educational local image analysis)'

def get(url,timeout=35):
    request=urllib.request.Request(url,headers={'User-Agent':UA})
    with urllib.request.urlopen(request,timeout=timeout) as response:return response.read()

def save_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def probe():
    CORPUS.mkdir(exist_ok=True)
    urls={
      'commons':'https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(dict(action='query',format='json',generator='categorymembers',gcmtitle='Category:Qiu Ying',gcmtype='file',gcmlimit=50,prop='imageinfo',iiprop='url|extmetadata',iiurlwidth=1800)),
      'jiguzuo':'https://www.jiguzuo.com/guohua/',
      'npm':'https://theme.npm.edu.tw/opendata/',
      'dpm':'https://www.dpm.org.cn/collection/paint/231522.html',
    }
    def fetch(item):
        key,url=item
        try:
            data=get(url);(CORPUS/(key+'_probe.txt')).write_bytes(data)
            print('PROBE',key,len(data),data[:80].decode('utf-8',errors='replace'),flush=True)
        except Exception as error:print('PROBE_FAILED',key,str(error),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,urls.items()))

def inspect_pages():
    for key,url in [('qinglu','https://www.jiguzuo.com/zhuanti/qinglvshanshui/'),('example','https://www.jiguzuo.com/guohua/tang-dai-hua-xia-shan-gao-yi-tu.html')]:
        body=get(url).decode('utf-8',errors='replace');(CORPUS/(key+'_probe.txt')).write_text(body,encoding='utf-8')
        links=re.findall(r'<a\b[^>]*href=[\"\x27]([^\"\x27]+)[\"\x27][^>]*>(.*?)</a>',body,re.S|re.I)
        links=[(html.unescape(h),re.sub('<[^>]+>','',t).strip()[:100]) for h,t in links]
        imgs=re.findall(r'<img\b[^>]+>',body,re.I)
        print(key,'links',json.dumps(links[-100:],ensure_ascii=True),'images',json.dumps(imgs[-12:],ensure_ascii=True),flush=True)


def cached_page(url):
    path=CORPUS/'pages'/(hashlib.sha256(url.encode()).hexdigest()+'.html')
    if path.exists():return path.read_text(encoding='utf-8')
    body=get(url).decode('utf-8',errors='replace');path.parent.mkdir(exist_ok=True)
    path.write_text(body,encoding='utf-8');return body


class Tags(HTMLParser):
    def __init__(self):super().__init__();self.images=[]
    def handle_starttag(self,tag,attrs):
        if tag!='img':return
        attrs=dict(attrs);url=attrs.get('data-original',attrs.get('src',''))
        if 'img.jiguzuo.com/guohua/' in url and '!list' not in url and '!w100' not in url:
            self.images.append((urllib.parse.urljoin('https://www.jiguzuo.com',html.unescape(url)),attrs.get('alt','')))


def collect_qinglu(max_pages=8):
    CORPUS.mkdir(exist_ok=True)
    links={}
    first='https://www.jiguzuo.com/zhuanti/qinglvshanshui/'
    body=cached_page(first)
    pages={first}
    for href in re.findall(r'href="([^"]*qinglvshanshui/[^\"]*)"',body):pages.add(urllib.parse.urljoin(first,href))
    for url in sorted(pages)[:max_pages]:
        try:
            text=cached_page(url)
            for href,title in re.findall(r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>',text,re.S):
                if '/guohua/' not in href or not href.endswith('.html'):continue
                title=html.unescape(re.sub('<[^>]+>','',title)).strip()
                if not title:continue
                page=urllib.parse.urljoin(first,href);links[page]=dict(page_url=page,title=title,discovery_category='explicit_qinglu_landscape')
            print('INDEX',url,len(links),'painting pages',flush=True)
        except Exception as error:print('INDEX_FAILED',url,str(error),flush=True)
    save_json(CORPUS/'discovered_painting_pages.json',list(links.values()))
    sources=[];failures=[]
    def parse_article(record):
        body=cached_page(record['page_url'])
        group='jgz_'+hashlib.sha256(record['page_url'].encode()).hexdigest()[:16]
        # Long scroll pages use a different layout; only full-size public
        # preview image tags are accepted, never related-list thumbnails.
        tags=Tags();tags.feed(body)
        unique={url:alt for url,alt in tags.images}
        return [dict(record,id=group+'_'+str(i).zfill(3),group=group,image_url=url,caption=alt,
                     source_type='public_qinglu_catalogue_preview',license='No explicit reuse license asserted; public preview only; see source page',
                     supervision='unlabeled',originals_downloaded=False)
                for i,(url,alt) in enumerate(unique.items())]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(parse_article,r):r for r in links.values()}
        for index,job in enumerate(as_completed(jobs)):
            try:
                found=job.result();sources.extend(found)
                if (index+1)%20==0:print('PAGES_PARSED',index+1,'/',len(jobs),'public images',len(sources),flush=True)
            except Exception as error:failures.append(dict(url=jobs[job]['page_url'],error=str(error)))
    save_json(CORPUS/'candidate_images.json',dict(records=sources,failures=failures))
    print('DISCOVERY_COMPLETE',len(links),'painting pages',len(sources),'public image URLs',flush=True)
    return sources


def download_candidates(workers=8):
    from PIL import Image,ImageOps
    from io import BytesIO
    sources=json.loads((CORPUS/'candidate_images.json').read_text(encoding='utf-8'))['records']
    successes=[];failures=[]
    def download(record):
        directory=CORPUS/'raw'/record['group'];directory.mkdir(parents=True,exist_ok=True)
        metadata=directory/(record['id']+'.json')
        if metadata.exists():
            try:
                cached=json.loads(metadata.read_text(encoding='utf-8'))
                if Path(cached['path']).exists():return cached
            except (ValueError,KeyError):pass
        raw=get(record['image_url'],50)
        if len(raw)>30_000_000:raise ValueError('Preview unexpectedly large')
        image=ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert('RGB')
        if min(image.size)<140 or max(image.size)<500:raise ValueError('Preview too small: '+str(image.size))
        path=directory/(record['id']+'.jpg');image.save(path,quality=95)
        result=dict(record,path=str(path),dimensions=list(image.size),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    downloaded_at_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
        save_json(metadata,result);return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(download,r):r for r in sources}
        for job in as_completed(jobs):
            try:successes.append(job.result())
            except Exception as error:failures.append(dict(id=jobs[job]['id'],url=jobs[job]['image_url'],error=str(error)))
            if (len(successes)+len(failures))%40==0:
                print('DOWNLOAD_PROGRESS',len(successes),'ok',len(failures),'failed',flush=True)
                save_json(CORPUS/'download_progress.json',dict(records=successes,failures=failures,total=len(sources)))
    # Exact duplicates cannot inflate the corpus count.
    unique={};duplicates=[]
    for r in sorted(successes,key=lambda x:x['id']):
        if r['sha256'] in unique:duplicates.append(dict(id=r['id'],duplicate_of=unique[r['sha256']]['id']))
        else:unique[r['sha256']]=r
    result=dict(records=list(unique.values()),download_failures=failures,exact_duplicates=duplicates,
                downloaded_files=len(successes),unique_files=len(unique),painting_page_groups=len({r['group'] for r in unique.values()}),
                note='Page groups are not guaranteed distinct artworks; details and alternate versions share parent page. Perceptual/work grouping follows before any split.')
    save_json(CORPUS/'downloaded_sources.json',result)
    print('DOWNLOAD_COMPLETE',result['unique_files'],'unique files',result['painting_page_groups'],'page groups',flush=True)

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');parser.add_argument('--inspect-pages',action='store_true');parser.add_argument('--discover',action='store_true');parser.add_argument('--download',action='store_true');args=parser.parse_args(argv)
    if args.probe:probe()
    if args.inspect_pages:inspect_pages()
    if args.discover:collect_qinglu()
    if args.download:download_candidates()
