# -*- coding: utf-8 -*-
from rock_v15_common import *
import re,urllib.request,urllib.parse,time
import numpy as np,cv2
from PIL import Image,ImageDraw
ROOT=LAB/'structure_generalization_v20_20261025'
SOURCES=[('caa_steps','https://mskj.caa.edu.cn/info/1010/18051.htm','CAA educational step-by-step; two artwork sequences'),('mooc_gu_he','https://mooc1.chaoxing.com/mooc-ans/ztnodedetailcontroller/visitnodedetail?_fromV2_=&_from_=&courseId=215129993&knowledgeId=366199475&rtag=','Textbook chapter Gu Kunbo / He Tianjian'),('mooc_stones','https://mooc1.chaoxing.com/mooc-ans/ztnodedetailcontroller/visitnodedetail?_fromV2_=&_from_=&courseId=93122072&knowledgeId=93127117&rtag=','Different landscape teaching course')]

def fetch(url,max_bytes=10000000):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Referer':url})
    with urllib.request.urlopen(req,timeout=40) as response:
        data=response.read(max_bytes+1)
        if len(data)>max_bytes:raise ValueError('Download exceeds task cap')
        return data,response.headers.get_content_type()

def phash(im):
    a=np.asarray(im.convert('L').resize((32,32)),np.float32);q=cv2.dct(a)[:8,:8];return (q>np.median(q[1:])).reshape(-1)

def main():
    (ROOT/'sources').mkdir(parents=True,exist_ok=True);out=[];errors=[]
    for key,url,desc in SOURCES:
        try:
            data,_=fetch(url);html=data.decode('utf-8',errors='replace');(ROOT/'sources'/(key+'.html')).write_text(html,encoding='utf-8');tags=re.findall(r'<img\b[^>]*>',html,re.I);urls=[]
            for tag in tags:
                matches=re.findall(r'(?:src|data-original|data-src)\s*=\s*[\"\']([^\"\']+)',tag,re.I)
                for item in matches:
                    item=item.replace('&amp;','&');image_url=urllib.parse.urljoin(url,item)
                    if key=='caa_steps' and not re.search(r'01|02|03|04|05|06|07|\.jpg',tag,re.I):continue
                    if key=='caa_steps' and '/__local/' not in image_url:continue
                    if key.startswith('mooc') and not ('ananas' in image_url or 'chaoxing' in image_url):continue
                    if key.startswith('mooc') and not re.search(r'\.(?:jpg|png|jpeg)|origin|download|object|img',image_url,re.I):continue
                    if image_url not in urls:urls.append(image_url)
            for i,image_url in enumerate(urls[:25]):
                try:
                    image_data,kind=fetch(image_url);from io import BytesIO
                    im=Image.open(BytesIO(image_data)).convert('RGB')
                    if im.width<130 or im.height<90:continue
                    file=ROOT/'sources'/(key+'_'+str(i).zfill(2)+'.png');im.save(file);out.append(dict(id=file.stem,image=str(file),width=im.width,height=im.height,source_page=url,image_url=image_url,source_group=key,source_description=desc,sha256=digest(file),original_download_sha256=__import__('hashlib').sha256(image_data).hexdigest(),expert_verified=False,split='external_unassigned',independent_artwork_not_verified=True));print('V20_DOWNLOAD',file.stem,im.size,flush=True)
                except Exception as e:errors.append(dict(source=image_url,error=str(e)))
        except Exception as e:errors.append(dict(source=url,error=str(e)))
    old=[];raw=LAB/'autonomous_strokes_v15_20261025'/'raw'
    for p in raw.glob('*'):
        try:old.append((str(p),phash(Image.open(p))))
        except Exception:pass
    for r in out:
        h=phash(Image.open(r['image']));dist=[(int(np.count_nonzero(h!=v)),p) for p,v in old];distance,path=min(dist) if dist else (64,None);r.update(nearest_old_phash_hamming=distance,nearest_old_image=path,possible_duplicate_old=distance<=7)
    jsave(ROOT/'source_manifest.json',dict(records=out,errors=errors,download_sources=SOURCES,images_are_new_to_this_round_not_independent_artworks_guaranteed=True))
    board=Image.new('RGB',(1500,300*int(np.ceil(len(out)/4))+40),'#f6f3ec');d=ImageDraw.Draw(board);d.text((6,4),'v20 新来源候选图（未用于训练，需筛查图注/重复/可见笔画）',font=FONT,fill='black')
    from human_mountain_v11 import paste
    for i,r in enumerate(out):
        x=(i%4)*375;y=40+(i//4)*300;d.text((x+5,y),r['id']+' '+str((r['width'],r['height'])),font=SMALL,fill='black');paste(board,np.asarray(Image.open(r['image'])),x+5,y+25,365,270)
    board.save(ROOT/'SOURCE_CONTACT.jpg',quality=94);print('V20_SOURCES',len(out),'errors',len(errors),flush=True)
if __name__=='__main__':main()
