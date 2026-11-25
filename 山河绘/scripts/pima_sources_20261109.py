"""Inspect publicly playable teacher videos before admitting any stroke labels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure, LAB
configure()
import requests,json,concurrent.futures,re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image,ImageDraw,ImageFont
import cv2
ROOT=LAB/'pima_demonstration_20261109'
IDS={29:'1gp5i20c7rr6MaCTGoD0VquRf1k91c',30:'1gpdao8hkle3F5rtCdRD4dfz61rs1c',35:'1h5vhv3aojk6X9pF0DnGmgq9W1a01c',36:'1h61em88ail7GNxGXI9TUa6Xl1s91c',37:'1h5vi07sks9KFNZ72hjMX9VrP1331c',40:'1h5vhvejj0k4q2fA9zcp03oAM1tv1c'}
def inspect(item):
 n,vid=item;out=ROOT/'sources'/str(n);out.mkdir(parents=True,exist_ok=True)
 page='https://www.ganjingworld.com/video/'+vid
 r=requests.get(page,timeout=40);r.raise_for_status();r.encoding='utf8';(out/'page.html').write_text(r.text,encoding='utf8')
 soup=BeautifulSoup(r.text,'html.parser');meta=json.loads(soup.select_one('script[type="application/ld+json"]').string)
 (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
 # Only use public media explicitly embedded in the current page; no member-only assets.
 src=soup.select_one('video source')
 if src is None:return n,'no publicly embedded source'
 master=src['src'];r=requests.get(master,timeout=30);r.raise_for_status()
 choices=[x for x in r.text.splitlines() if x and not x.startswith('#')]
 target=next((x for x in choices if '480' in x),choices[0]);playlist=urljoin(master,target)
 r=requests.get(playlist,timeout=30);r.raise_for_status();(out/'playlist.m3u8').write_text(r.text,encoding='utf8')
 assert '#EXT-X-KEY' not in r.text
 lines=[x for x in r.text.splitlines() if x and not x.startswith('#')]
 assert len(set(lines))==1,'Only single-object public byte-range media supported'
 video=out/'video.mp4'
 if not video.exists():
  r=requests.get(urljoin(playlist,lines[0]),timeout=90);r.raise_for_status();video.write_bytes(r.content)
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);board=Image.new('RGB',(1281,550),'#eeeae1');d=ImageDraw.Draw(board)
 for i,f in enumerate([.05,.2,.4,.6,.8,.95]):
  t=duration*f;cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read()
  if ok:
   im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.save(out/f'frame_{i}.jpg');im.thumbnail((427,240));x=(i%3)*427;y=(i//3)*275;board.paste(im,(x,y+30));d.text((x+5,y+4),f'第{n}课 {t:.1f}秒',fill='#222',font=font)
 cap.release();board.save(out/'overview.jpg',quality=92)
 print('SOURCE',n,round(duration),video.stat().st_size,flush=True)
 return n,'downloaded public source, observations unreviewed'
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for result in pool.map(inspect,IDS.items()):print(result,flush=True)
