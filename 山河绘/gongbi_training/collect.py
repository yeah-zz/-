"""Museum-source collection and group-preserving user-reference ingestion."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import urllib.request
import urllib.parse

from .runtime import LAB

ATTACHMENTS = [
 '138ede90-34c6-4d18-9986-c47455e6cf77',
 'c516efd9-cbb6-4842-9fc5-4af01bcb968b',
 'bd777ab1-18c8-4eaa-b8ab-8c3b160b868b',
 'c886330a-9aa2-4989-acf3-23b1ea40d787',
 '29c0732a-d10d-4033-ba7a-661b17ef0ad0',
 '5d1bc901-dfd6-481e-97c5-4c82b7f63add',
 '3e6a2eea-f48f-4a5e-9119-79d5dee6903e',
]

# Album leaves share a group, so related leaves cannot cross split boundaries.
CMA_SELECTION = {
 '1971.40':('core_gongbi','train','cma_1971_40'),
 '1952.283':('core_gongbi','train','cma_1952_283'),
 '1973.72':('core_gongbi','train','cma_1973_72'),
 '1997.95':('core_qinglu','validation','cma_1997_95'),
 '1917.94':('color_landscape_support','train','cma_1917_94'),
 '1971.231':('color_landscape_support','train','cma_1971_231'),
 '1975.21':('architecture_support','train','cma_1975_21'),
 '1979.27.1.8':('fine_line_support','train','cma_1979_27_album'),
 '1979.27.1.10':('fine_line_support','train','cma_1979_27_album'),
 '1998.78.4':('color_landscape_support','train','cma_song_xu_album'),
 '1998.78.5':('color_landscape_support','train','cma_song_xu_album'),
 '1998.78.13':('color_landscape_support','train','cma_song_xu_album'),
 '1954.583':('color_landscape_support','validation','cma_1954_583'),
}
MET_SELECTION = {51806:('core_qinglu','train'),40081:('core_gongbi','train'),49227:('core_gongbi','validation')}


def get(url, timeout=35):
    req = urllib.request.Request(url, headers={'User-Agent':'GongbiLineResearch/0.1 (local academic image experiment)'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return response.read()


def ingest():
    from PIL import Image
    result=[]
    folder=LAB/'data'/'raw'/'user_qianli'
    folder.mkdir(exist_ok=True)
    for i,key in enumerate(ATTACHMENTS,1):
        src=Path(r'C:\Users\Lenovo\AppData\Local\Temp')/('codex-clipboard-'+key+'.png')
        dst=folder/f'reference_{i:02d}.png'
        if not dst.exists():shutil.copy2(src,dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest()==hashlib.sha256(src.read_bytes()).hexdigest()
        with Image.open(dst) as im: size=list(im.size)
        result.append(dict(id=f'user_qianli_{i:02d}', group='wang_ximeng_qianli_jiangshan',
                           title='用户提供《千里江山图》局部 '+str(i), path=str(dst),
                           source_type='user_attachment', sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
                           dimensions=size, split='train', label_type='unlabeled',
                           notes='All seven crops belong to ONE painting; never split across train/validation.'))
    return result


def discover():
    query=urllib.parse.urlencode(dict(q='landscape',department='Chinese Art',type='Painting',limit=200,cc0=1))
    url='https://openaccess-api.clevelandart.org/api/artworks/?'+query
    file=LAB/'data'/'cma_candidates.json'
    if file.exists(): payload=json.loads(file.read_text(encoding='utf-8'))
    else:
        payload=json.loads(get(url,60))
        file.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    data=payload.get('data',[])
    summary=[dict(accession=o.get('accession_number'), title=o.get('title'),
                  creator='; '.join(c.get('description','') for c in o.get('creators',[])),
                  medium=o.get('technique'), description=o.get('wall_description'),
                  department=o.get('department'), culture=o.get('culture'),
                  license=o.get('share_license_status'), image=bool(o.get('images',{}).get('print')))
             for o in data]
    (LAB/'data'/'candidate_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CANDIDATES',len(summary),flush=True)
    return data


def download_source(obj, museum, spec):
    from PIL import Image
    if museum=='cma':
        accession=obj['accession_number']
        assert obj.get('share_license_status')=='CC0'
        style,split,group=spec
        image_info=obj['images'].get('print') or obj['images']['web']
        url=image_info['url']
        page=obj.get('url') or 'https://www.clevelandart.org/art/'+accession
        creators=obj.get('creators')
        title=obj['title']
    else:
        assert obj.get('isPublicDomain')
        accession=str(obj['objectID'])
        style,split=spec
        group='met_'+accession
        url=obj['primaryImage']
        if not url:raise ValueError('No Met primary image')
        page=obj['objectURL']
        creators=obj.get('artistDisplayName')
        title=obj['title']
    key=museum+'_'+accession.replace('.','_')
    folder=LAB/'data'/'raw'/key
    folder.mkdir(exist_ok=True)
    (folder/'metadata.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    dst=folder/'painting.jpg'
    if not dst.exists():dst.write_bytes(get(url,90))
    with Image.open(dst) as im:
        size=list(im.size)
        im.verify()
    r=dict(id=key,group=group,title=title,creator=creators,path=str(dst),source_type='museum_open_access',
           museum=museum,page_url=page,image_url=url,license='CC0 / public-domain museum image',
           sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),dimensions=size,bytes=dst.stat().st_size,
           split=split,style_tier=style,label_type='unlabeled', downloaded_at_utc=datetime.now(timezone.utc).isoformat())
    (folder/'source.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DOWNLOADED',key,size,flush=True)
    return r


def fetch_met(object_id,spec):
    folder=LAB/'data'/'raw'/('met_'+str(object_id))
    folder.mkdir(exist_ok=True)
    file=folder/'metadata.json'
    if file.exists():obj=json.loads(file.read_text(encoding='utf-8'))
    else:obj=json.loads(get(f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}',25))
    return download_source(obj,'met',spec)


def contact_sheet(records):
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    cols=4; rows=(len(records)+cols-1)//cols
    sheet=Image.new('RGB',(1440,rows*290),(244,243,236)); draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(records):
        x=(i%cols)*360;y=(i//cols)*290
        im=Image.open(r['path']).convert('RGB');im.thumbnail((340,232))
        sheet.paste(im,(x+10+(340-im.width)//2,y+10+(232-im.height)//2))
        draw.text((x+10,y+244),r['id']+' / '+r['split'],font=font,fill=(30,40,35))
        draw.text((x+10,y+267),r.get('style_tier','user examples'),font=font,fill=(60,70,60))
    sheet.save(LAB/'previews'/'collection_contact_sheet.png')


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--discover',action='store_true')
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args(argv)
    records=ingest()
    (LAB/'data'/'user_sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print('USER_REFERENCES',len(records),'ONE_ARTWORK_GROUP',flush=True)
    data=discover()
    if not args.download:return
    lookup={o['accession_number']:o for o in data}
    failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={}
        for accession,spec in CMA_SELECTION.items():
            if accession in lookup:
                futures[pool.submit(download_source,lookup[accession],'cma',spec)]=accession
        for oid,spec in MET_SELECTION.items():futures[pool.submit(fetch_met,oid,spec)]=str(oid)
        for future in as_completed(futures):
            try:records.append(future.result())
            except Exception as exc:
                error=dict(source=futures[future],error=str(exc))
                failures.append(error);print('SOURCE_FAILED',json.dumps(error),flush=True)
    records.sort(key=lambda r:r['id'])
    groups={}
    for r in records:
        assert groups.setdefault(r['group'],r['split'])==r['split']
    manifest=dict(records=records,download_failures=failures,
                  painting_groups=len(groups),user_reference_group_count=1,
                  requested_line_policy='structure-first; exclude dense foliage, dots, water ripples, textile grain and text',
                  no_human_dense_ground_truth=True)
    (LAB/'data'/'sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    contact_sheet(records)
    print('COLLECTION_READY',len(records),'images',len(groups),'painting/album groups',flush=True)
