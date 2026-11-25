"""Collect public, provenance-tracked teaching observations; never infer gold order."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure, LAB
configure()
import json, hashlib, concurrent.futures, shutil
from urllib.parse import urljoin
import requests, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = LAB / 'pima_demonstration_20261109'
RAW = ROOT / 'raw'
URL = 'https://media1-us-east.cloudokyo.cloud/video/v11/74/3c/70/743c7020-4a4b-4ab3-addf-ab87109870b4/playlist_480p.m3u8'

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')

def playlist():
    r = requests.get(URL, timeout=30); r.raise_for_status()
    (RAW/'playlist_480p.m3u8').write_text(r.text, encoding='utf8')
    assert '#EXT-X-KEY' not in r.text, 'Encrypted streams are not handled'
    entries, time, duration = [], 0., None
    for line in r.text.splitlines():
        if line.startswith('#EXTINF:'): duration = float(line.split(':')[1].split(',')[0])
        elif line and not line.startswith('#'):
            entries.append(dict(index=len(entries), time=time, duration=duration, url=urljoin(URL,line)))
            time += duration
    save(RAW/'segments.json', entries)
    return entries

def fetch(entry):
    # This source uses byte-range HLS over one complete fragmented MP4.
    # Cache the shared object once, rather than downloading it for each range.
    path=RAW/Path(entry['url']).name
    if not path.exists():
        previous=RAW/'segment_0007.ts'
        if previous.exists():
            shutil.copy2(previous,path)
        else:
            r=requests.get(entry['url'],timeout=60);r.raise_for_status();path.write_bytes(r.content)
    return path

def main():
    RAW.mkdir(parents=True,exist_ok=True)
    entries=playlist()
    chosen=[min(entries,key=lambda e:abs(e['time']-t)) for t in range(20,1175,90)]
    cache={entry['url']:fetch(entry) for entry in chosen[:1]}
    assert len({e['url'] for e in chosen})==1, 'Add per-URL caching for a new source'
    paths=[cache[e['url']] for e in chosen]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    board=Image.new('RGB',(1281,280*((len(chosen)+2)//3)), '#eeeae1');draw=ImageDraw.Draw(board)
    observations=[]
    for i,(entry,path) in enumerate(zip(chosen,paths)):
        cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_MSEC,entry['time']*1000);ok,bgr=cap.read();cap.release()
        if not ok:raise RuntimeError(f'Cannot decode {path}')
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));out=ROOT/f"overview_{entry['index']:04d}.jpg";im.save(out)
        thumb=im.resize((427,240));x=(i%3)*427;y=(i//3)*280;board.paste(thumb,(x,y+30))
        draw.text((x+8,y+3),f"{entry['time']:.1f}秒 / 片段 {entry['index']}",font=font,fill='#252525')
        observations.append(dict(**entry,local=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frame=str(out),label_status='unreviewed',expert_paths=None,physical_pressure=None))
    board.save(ROOT/'video_overview.jpg',quality=94)
    save(ROOT/'observations.json',dict(source_page='https://www.ganjingworld.com/video/1h5vhvrokus1JWo6gFiWQFshE1iv1c',
        source_author='刘鹏凯彩墨山水',style_label='披麻皴（作者标题）',source_group='liupengkai_qiushan_lesson31',
        split='train_development_only',independent_test_sources=0,frames=observations,robot_connected=False))
    # Only remove redundant generated downloads after confirming byte identity.
    original=paths[0];digest=hashlib.sha256(original.read_bytes()).hexdigest()
    for duplicate in RAW.glob('segment_*.ts'):
        assert duplicate.resolve().is_relative_to(RAW.resolve())
        if hashlib.sha256(duplicate.read_bytes()).hexdigest()==digest:duplicate.unlink()
    print('OVERVIEW',len(observations),ROOT/'video_overview.jpg',flush=True)

if __name__=='__main__':main()
