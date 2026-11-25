from v20_sources import *
from io import BytesIO
manifest=read(ROOT/'source_manifest.json');url=SOURCES[0][1];html=(ROOT/'sources'/'caa_steps.html').read_text(encoding='utf-8');tags=re.findall(r'<img\b[^>]*>',html,re.I)
for tag in tags:
    title=re.search(r'title=[\"\'](0[1-7])\.jpg',tag,re.I);src=re.search(r'\bsrc=[\"\']([^\"\']+)',tag,re.I)
    if not title or not src:continue
    identity='caa_steps_'+title.group(1)
    if any(r['id']==identity for r in manifest['records']):continue
    image_url=urllib.parse.urljoin(url,src.group(1).replace('&amp;','&'))
    try:
        data,_=fetch(image_url);im=Image.open(BytesIO(data)).convert('RGB');file=ROOT/'sources'/(identity+'.png');im.save(file);manifest['records'].append(dict(id=identity,image=str(file),width=im.width,height=im.height,source_page=url,image_url=image_url,source_group='caa_steps',source_description='CAA educational step-by-step, 01-04 same artwork,05-07 same artwork',sha256=digest(file),expert_verified=False,split='external_unassigned',independent_artwork_not_verified=True));print('V20_DOWNLOAD',identity,im.size,flush=True)
    except Exception as e:manifest['errors'].append(dict(source=image_url,error=str(e)))
jsave(ROOT/'source_manifest.json',manifest)
