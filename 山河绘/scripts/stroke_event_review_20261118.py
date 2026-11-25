# -*- coding: utf-8 -*-
"""Replay source frames and record what their brush motion can support."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import LAB, configure

configure()

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = LAB / 'stroke_event_review_20261118'
PROJECT = Path(__file__).resolve().parents[1]
SOURCES = {
    'L': ('pima_demonstration_20261109/sources/37/video.mp4', 30),
    'F': ('clear_rock_sources_20261117/fuchun_720.ts', 25),
}
EVENTS = [
    dict(id='L_descent', source='L', frames=[6759, 6779, *range(6793, 6807), 6883],
         box=[290, 165, 425, 430], title='岩面弯线的下行运笔',
         note='6795至6804帧可见笔锋沿岩面弯线下移。上端与旧墨迹重合，下端运动模糊且紧邻回笔。6883帧已经经过多次动作，不能用于分离这一笔的新增墨迹。',
         visible_motion_frames=[6795, 6804], role='岩面内部运笔', rejection='起落和新增墨迹身份未核清'),
    dict(id='L_foot', source='L', frames=list(range(6564, 6655, 6)),
         box=[275, 330, 465, 470], title='石脚重叠与重复触纸',
         note='笔锋多次往返于已有密集线条的石脚，几次移动的边界难以区分。',
         visible_motion_frames=None, role='石脚运笔，单笔待定', rejection='多次往返与旧线重叠'),
    dict(id='F_trees', source='F', frames=list(range(6100, 6201, 5)),
         box=[125, 150, 600, 550], title='树群附近的遮挡',
         note='手与笔遮挡局部，运笔从山面移到树群附近，无法稳定判断墨迹属于岩面还是枝叶。',
         visible_motion_frames=None, role='待定', rejection='手部遮挡，树石归属待定'),
]


def read(path):
    return json.loads(Path(path).read_text('utf8'))


def digest(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def extract():
    target = ROOT / 'frames'
    target.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, (relative, fps) in SOURCES.items():
        video = LAB / relative
        selected = {i: event for event in EVENTS if event['source'] == name for i in event['frames']}
        cap = cv2.VideoCapture(str(video))
        try:
            assert abs(cap.get(cv2.CAP_PROP_FPS) - fps) < 0.02
            for index in range(max(selected) + 1):
                assert cap.grab(), (name, index)
                if index not in selected:
                    continue
                ok, frame = cap.retrieve()
                assert ok
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image.crop(selected[index]['box']).save(target / f'{name}{index}.png')
        finally:
            cap.release()
        provenance[name] = dict(path=str(video), sha256=digest(video), fps=fps, previously_seen=True)
    return provenance


def main():
    provenance = extract()
    records = []
    for event in EVENTS:
        row = {k: v for k, v in event.items() if k != 'frames'}
        row.update(landed_at=None, lifted_at=None, ink_ownership_mask=None, complete_stroke=False,
                   training_allowed=False, frames=[])
        for index in event['frames']:
            relative = f'frames/{event["source"]}{index}.png'
            row['frames'].append(dict(index=index, nominal_seconds=index / provenance[event['source']]['fps'],
                                      path=relative, sha256=digest(ROOT / relative)))
        records.append(row)
    lock = read(LAB / 'local_rock_patch_20261117/baseline_lock.json')
    baseline = {k: digest(lock[k]) == lock[k + '_sha256'] for k in ('approved500', 'working489')}
    assert all(baseline.values())
    checks = read(LAB / 'two_work_bands_20261118/checkpoint_freeze.json')
    assert all(digest(LAB / 'two_work_bands_20261118' / f'{key}.pt') == value for key, value in checks.items())
    save(ROOT / 'source_events.json', dict(records=records, videos=provenance,
         decode='sequential OpenCV decode; seconds are nominal indexed playback time',
         atlas_group='two previously seen works', complete_training_strokes=0,
         independent_validation=False, new_model_trained=False, baseline_hashes=baseline, robot_connected=False,
         source_intake=[dict(url='https://www.artmuseum.cuhk.edu.hk/zh/training/resource/detail/32', result='馆方目录可核对，披麻皴视频未取得可逐帧检查的材料'),
                        dict(url='https://www.bilibili.com/video/BV1HC4y187tE/', result='页面返回412，未取得视频帧')]))
    payload = json.dumps(records, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>山石示范逐帧复核</title><style>body{font:17px/1.65 system-ui;max-width:980px;margin:auto;padding:24px;background:#f5f2e8;color:#243d35}button,select,input{font:inherit;max-width:100%}button{margin:5px 6px 5px 0}img{display:block;height:420px;max-width:100%;object-fit:contain;object-position:left;image-rendering:pixelated}@media(max-width:550px){img{height:320px}}.panel{border:1px solid #b7b8aa;padding:16px}</style><h1>山石示范逐帧复核</h1><p>两件既有示范，三处观察窗口。一段可见下行运笔。完整落笔、提笔和新增墨迹归属仍待核实。</p><div class="panel"><label for="event">选择片段</label> <select id="event"></select><p id="note"></p><img id="frame" alt="示范原始视频裁片"><p id="time"></p><button id="prev">上一帧</button><button id="next">下一帧</button><button id="play">慢放</button><input type="range" id="position" aria-label="浏览所保存的帧" min="0" value="0"></div><p><a href="source_events.json">查看帧号、来源和排除原因</a></p><script>const rows=DATA;const q=id=>document.getElementById(id);const event=q('event'),position=q('position');let timer=null;rows.forEach((r,i)=>event.add(new Option(r.title,i)));function render(){const r=rows[+event.value],f=r.frames[+position.value];q('frame').src=f.path;q('time').textContent='源帧 '+f.index+' · '+f.nominal_seconds.toFixed(3)+' 秒';q('note').textContent=r.note;q('prev').disabled=+position.value===0;q('next').disabled=+position.value===+position.max}function stop(){clearInterval(timer);timer=null;q('play').textContent='慢放'}function change(){stop();position.max=rows[+event.value].frames.length-1;position.value=0;render()}event.onchange=change;position.oninput=render;q('prev').onclick=()=>{position.stepDown();render()};q('next').onclick=()=>{position.stepUp();render()};q('play').onclick=()=>{if(timer!==null){stop();return}q('play').textContent='暂停';timer=setInterval(()=>{if(+position.value===+position.max){stop();return}position.stepUp();render()},250)};change();</script></html>'''
    (ROOT / 'index.html').write_text(page.replace('DATA', payload), encoding='utf8')
    report = PROJECT / 'reports/20261118_stroke_event_review.md'
    report.write_text('''# 山石示范逐帧复核

**推进目标**　核查单笔运笔、起落与墨迹归属，为完整笔触参考准备可追溯证据。

**实际新增**　顺序解码两件既有视频，归档3个窗口的54张裁片。刘鹏凯示范第6795至6804帧可见笔锋沿弯曲岩面下移，可用于运动复核。保留原生坐标、来源、帧号、时间、哈希和浏览器逐帧回放。

**剩余问题**　下行片段的起点与旧线交叠，末端紧随回笔，不能确认唯一的一次落笔、提笔和墨迹归属。另两段分别有石脚多次往返和树群附近的遮挡。3段均未纳入完整单笔训练。馆方与小斧劈候选未取得可逐帧使用的新视频。

**独立验证**　未进行。两件原视频此前已查看，此轮只增加可审查的运动片段和排除案例。冻结模型及500／489笔基线哈希核查通过，机械臂离线。
''', encoding='utf8')
    print(json.dumps(dict(events=len(records), frames=sum(len(r['frames']) for r in records),
                          baseline_hashes=baseline, report=str(report)), ensure_ascii=False))


def serve():
    import os
    import serve_regional_presence_20261117 as preview

    manifest = read(ROOT / 'source_events.json')
    preview.ROOT = ROOT
    preview.ALLOWED = {'index.html', 'source_events.json'} | {
        frame['path'] for event in manifest['records'] for frame in event['frames']
    }
    with preview.ThreadingHTTPServer(('127.0.0.1', 0), preview.Handler) as server:
        save(ROOT / 'server.json', dict(url=f'http://127.0.0.1:{server.server_port}/', pid=os.getpid()))
        server.serve_forever()


if __name__ == '__main__':
    if '--serve' in sys.argv:
        serve()
    else:
        main()
