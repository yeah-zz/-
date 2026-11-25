"""Download public museum/LOC candidates to E; no training or image inference."""
from probe_public_sources_20261029 import ROOT,one
from rock_v15_common import jsave,read,digest
import urllib.request,json,concurrent.futures,io,time
from PIL import Image,ImageDraw,ImageFont,ImageOps
from pathlib import Path
def get(url,cap=18000000):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 ShanHeHui research'})
    with urllib.request.urlopen(req,timeout=30) as r:data=r.read(cap+1)
    if len(data)>cap:raise ValueError('Download cap exceeded')
    return data
def prepare():
    try:
        loc=json.loads(get('https://www.loc.gov/resource/gdcwdl.wdl_13536/manifest.json'));jsave(ROOT/'source_metadata'/'loc_iiif.json',loc)
        canvases=loc.get('sequences',[{}])[0].get('canvases',[])
    except Exception as e:
        meta=read(ROOT/'source_metadata'/'loc.json');resource=meta['resources'][0];count=resource['segment_count'];base=resource['image'].split('/full/')[0]
        canvases=[{'images':[{'resource':{'service':{'@id':base.rsplit(':',1)[0]+':'+str(i+1).zfill(3)}}}]} for i in range(count)]
        jsave(ROOT/'source_metadata'/'loc_manifest_fallback.json',dict(error=str(e),published_representative_image=resource['image'],segment_count=count,source='page image service IDs derived from published resource metadata; each download verified separately'))
    rows=[]
    # Contact sheet selection is by image content/clarity, before inference.
    for i,c in enumerate(canvases):
        if i<4 or i%2:continue
        image=c['images'][0]['resource'];service=image.get('service',{});service=service[0] if isinstance(service,list) else service
        base=service.get('@id',service.get('id',''))
        url=base+'/full/!1800,1800/0/default.jpg' if base else image['@id']
        rows.append(dict(id=f'loc_{i+1:03d}',image_url=url,source_page='https://www.loc.gov/resource/gdcwdl.wdl_13536/?sp='+str(i+1),source_group='LOC same illustrated book',title='Mustard Seed Garden manual page '+str(i+1),license='LOC item access unrestricted',kind='manual_page',artwork_group='LOC_wdl_13536'))
    cma=read(ROOT/'source_metadata'/'cma.json')['data']
    for r in cma:
        if not any('China' in c for c in r.get('culture',[])) or r.get('share_license_status')!='CC0':continue
        if not r.get('images',{}).get('web') or r.get('classification') not in [None,'Chinese Art','Painting','Paintings']:
            if 'painting' not in str(r.get('technique','')).lower() and 'ink' not in str(r.get('technique','')).lower():continue
        if any(k in r['title'].lower() for k in ['brush holder','teacup']):continue
        rows.append(dict(id='cma_'+str(r['id']),image_url=r['images']['web']['url'],highres_url=r['images'].get('print',{}).get('url'),source_page=r['url'],source_group='Cleveland Museum of Art',title=r['title'],license='CC0',kind='painting',artwork_group=str(r.get('accession_number','')).split('.')[0]+'.'+str(r.get('accession_number','')).split('.')[1] if len(str(r.get('accession_number','')).split('.'))>1 else str(r['id'])))
    jsave(ROOT/'candidates_metadata.json',rows)
    return rows
def download(row):
    try:
        dest=ROOT/'candidates'/(row['id']+'.jpg');dest.parent.mkdir(exist_ok=True)
        if not dest.exists():dest.write_bytes(get(row['image_url']))
        im=Image.open(dest).convert('RGB');im.thumbnail((1800,1800),Image.Resampling.LANCZOS)
        work=ROOT/'inputs'/(row['id']+'.png');work.parent.mkdir(exist_ok=True);im.save(work)
        return dict(row,image=str(work),source_file=str(dest),size=list(im.size),sha256=digest(work),status='ok')
    except Exception as e:return dict(row,status='failed',error=str(e))
def contact(rows):
    good=[r for r in rows if r['status']=='ok'];f=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',15)
    for offset in range(0,len(good),30):
        group=good[offset:offset+30];board=Image.new('RGB',(1600,300*((len(group)+4)//5)),'#eeeae1');d=ImageDraw.Draw(board)
        for j,r in enumerate(group):
            x=j%5*320;y=j//5*300;d.text((x+3,y+2),r['id'],font=f,fill='black')
            with Image.open(r['image']) as im:
                im.thumbnail((312,270),Image.Resampling.LANCZOS);board.paste(im,(x+4+(312-im.width)//2,y+24+(270-im.height)//2))
        board.save(ROOT/f'candidates_{offset//30+1:02d}.jpg',quality=90)
if __name__=='__main__':
    rows=prepare();results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(download,rows):
            results.append(r);print('CANDIDATE',r['id'],r['status'],flush=True)
    jsave(ROOT/'candidates_manifest.json',dict(records=results,used_for_training=False,inference_not_yet_run=True));contact(results)
    print('CONTACT_READY',len([r for r in results if r['status']=='ok']),flush=True)
