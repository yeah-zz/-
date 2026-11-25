# -*- coding: utf-8 -*-
"""Read public zoom-viewer tiles for four existing ROIs, preserving provenance."""
import sys,json,hashlib,math,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_role_reference_20261112';PREV=LAB/'cunfa_fresh_validation_20261111'
BASE='https://www.dpm.org.cn/Uploads/tilegenerator/dest/files/image/8831/2007/1305/img0005_files/13/'
W,H,T,OV=4268,4095,256,1
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fetch(xy):
 x,y=xy;p=ROOT/'tiles'/f'{x}_{y}.jpg';u=BASE+p.name
 if not p.exists():r=requests.get(u,timeout=30);r.raise_for_status();assert r.content[:2]==b'\xff\xd8';p.write_bytes(r.content)
 return dict(x=x,y=y,url=u,path=str(p),sha256=sha(p),size=list(Image.open(p).size))
def main():
 (ROOT/'tiles').mkdir(exist_ok=True);(ROOT/'native').mkdir(exist_ok=True)
 data=json.loads((PREV/'test_manifest.json').read_text('utf8'));rows=[r for r in data['records'] if r['work_id']=='230757'];old=Image.open(rows[0]['source_image']).convert('RGB');sx=W/old.width;sy=H/old.height
 requests_needed=set();boxes=[]
 for r in rows:
  b=r['source_box'];native=[math.floor(b[0]*sx),math.floor(b[1]*sy),math.ceil(b[2]*sx),math.ceil(b[3]*sy)];boxes.append((r,native))
  requests_needed.update((x,y) for x in range(native[0]//T,(native[2]-1)//T+1) for y in range(native[1]//T,(native[3]-1)//T+1))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:tiles=list(pool.map(fetch,sorted(requests_needed)))
 lookup={(r['x'],r['y']):r for r in tiles};records=[];font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 for r,b in boxes:
  x0,y0,x1,y1=b;out=Image.new('RGB',(x1-x0,y1-y0))
  for x in range(x0//T,(x1-1)//T+1):
   for y in range(y0//T,(y1-1)//T+1):
    im=Image.open(lookup[x,y]['path']).convert('RGB');ox=OV if x else 0;oy=OV if y else 0;corew=min(T,W-x*T);coreh=min(T,H-y*T);core=im.crop((ox,oy,ox+corew,oy+coreh));out.paste(core,(x*T-x0,y*T-y0))
  p=ROOT/'native'/(r['id']+'.png');out.save(p);ref=Image.open(r['image']).convert('RGB');small=out.resize(ref.size,Image.Resampling.LANCZOS);a=np.array(ref).astype(float);z=np.array(small).astype(float);corr=float(np.corrcoef(a.mean(2).ravel(),z.mean(2).ravel())[0,1]);assert corr>.85,(r['id'],corr)
  rec=dict(id=r['id'],old_image=r['image'],old_sha256=sha(r['image']),old_box=r['source_box'],native_image=str(p),native_sha256=sha(p),native_box=b,scale=[sx,sy],native_size=list(out.size),old_size=list(ref.size),thumbnail_native_gray_correlation=corr,
   same_work_and_region=True,new_independent_sample=False,local_named_style_confirmed=False,source_url='https://www.dpm.org.cn/collection/paint/230757.html',source='Public zoom viewer metadata and its documented DeepZoom tile URL pattern; no restricted access or super-resolution')
  records.append(rec);board=Image.new('RGB',(1100,640),'#f4f1e8');d=ImageDraw.Draw(board);d.text((16,10),r['id']+'：旧预览放大 → 同位置原生像素',font=font(24),fill='#233e32');ref=ref.resize((470,500));out.thumbnail((510,500));board.paste(ref,(15,65));board.paste(out,(555,65));d.text((16,580),'左：旧预览插值，仅供对比；右：公开放大原图，不是AI增强或模型输出。',font=font(20),fill='#475d4d');d.text((16,612),'故宫博物院《雪堂客话图》；同图复查，不计作新增测试样本。',font=font(19),fill='#475d4d');board.save(ROOT/'native'/(r['id']+'_comparison.jpg'),quality=94)
 manifest=dict(source_size=[W,H],thumbnail_size=list(old.size),overlap=OV,tile_size=T,level=13,metadata_sha256=sha(ROOT/'sources/xia_public_viewer_metadata.xml'),viewer_code_sha256=sha(ROOT/'sources/xia_public_viewer.html'),downloaded_tiles=tiles,records=records,old_test_unchanged=True,robot_connected=False)
 (ROOT/'native_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print('NATIVE_READY',[(r['id'],r['native_size'],round(r['thumbnail_native_gray_correlation'],3)) for r in records],flush=True)
if __name__=='__main__':main()
