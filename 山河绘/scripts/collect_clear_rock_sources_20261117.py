# -*- coding: utf-8 -*-
"""Collect a publicly embedded, unencrypted teaching stream for visual audit."""
import sys,json,hashlib,time
from pathlib import Path
from urllib.parse import urljoin
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.source_identity import check_previously_seen
import requests,cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'clear_rock_sources_20261117'
PAGE='https://www.ganjingworld.com/video/1gga38p7dgh3iKX2kYUhFzF7k1ns1c'
def write(name,value):(ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'frames').mkdir(exist_ok=True)
    # This guard is now before media requests; the first run's late discovery
    # remains explicitly recorded in source_preflight.json, never rewritten as early.
    prior=[]
    for path in (LAB/'pima_independent_20261109').glob('*/metadata.json'):
        item=json.loads(path.read_text('utf8'))
        prior.append(dict(url=item['url'],metadata=str(path),title=item['name']))
    preflight=check_previously_seen(PAGE,prior)
    write('source_intake_preflight.json',dict(**preflight,checked_before_current_media_requests=True,
        known_source_use='native_resolution_supplement_only',not_a_new_work_claim=True))
    selection=json.loads((ROOT/'fuchun_media_selection.json').read_text('utf8'))
    playlist=urljoin(selection['master'],'v720p/index.m3u8')
    response=requests.get(playlist,timeout=30);response.raise_for_status();text=response.text
    assert '#EXT-X-KEY' not in text,'Encrypted media is not used'
    (ROOT/'fuchun_720.m3u8').write_text(text,encoding='utf8')
    media=[line for line in text.splitlines() if line and not line.startswith('#')]
    assert len(set(media))==1,'Review a multi-object stream before downloading'
    url=urljoin(playlist,media[0]);video=ROOT/'fuchun_720.ts'
    if not video.exists():
        response=requests.get(url,stream=True,timeout=(20,40));response.raise_for_status()
        partial=ROOT/'fuchun_720.part';total=0
        with partial.open('wb') as f:
            for block in response.iter_content(1024*1024):
                total+=len(block)
                if total>250*1024*1024:raise RuntimeError('Review before downloading more than 250 MiB')
                f.write(block)
        partial.replace(video)
    print('MEDIA_BYTES',video.stat().st_size,flush=True)
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    if not 1<fps<100:raise RuntimeError('Invalid frame rate')
    wanted=[0,30,60,90,120,180,240,360,480,600,900,1200,1500,1800,1875]
    frames={round(t*fps):t for t in wanted};rows=[];idx=0
    while cap.grab():
        if idx in frames:
            ok,bgr=cap.retrieve()
            if not ok:raise RuntimeError('Frame retrieval failed')
            t=frames[idx];path=ROOT/f'frames/fuchun_{t:04d}.png'
            Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(path)
            rows.append(dict(requested_seconds=t,frame_index=idx,nominal_seconds=idx/fps,path=str(path),sha256=sha(path),size=[bgr.shape[1],bgr.shape[0]]))
            print('FRAME',t,flush=True)
        idx+=1
    cap.release()
    board=Image.new('RGB',(1600,270*((len(rows)+3)//4)),'#f5f2e9');d=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for i,row in enumerate(rows):
        im=Image.open(row['path']);im.thumbnail((400,230));x=i%4*400;y=i//4*270
        board.paste(im,(x,y+32));d.text((x+8,y+4),f"{row['nominal_seconds']:.1f}秒 · 原生{row['size'][0]}×{row['size'][1]}",font=font,fill='#254237')
    board.save(ROOT/'fuchun_overview.jpg',quality=94)
    result=dict(source_url=PAGE,source_page_sha256=sha(ROOT/'fuchun.html'),metadata=str(ROOT/'fuchun_metadata.json'),
                media_url=url,playlist_url=playlist,video_path=str(video),video_sha256=sha(video),fps=fps,
                sequential_decoded_frames=idx,duration_nominal_seconds=idx/fps,frames=rows,
                frame_sampling='sequential decode; indices and nominal times, not verified original playback speed',
                painter_identity_verified=False,local_cunfa_verified=False,training_admitted=False,
                split='development_visual_audit_only',new_independent_validation=False,robot_connected=False)
    write('fuchun_manifest.json',result);print('DONE',len(rows),idx/fps,flush=True)
if __name__=='__main__':main()
