# -*- coding: utf-8 -*-
"""Known-artwork regression of source-guided continuity and tree attachment."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_faint_outline_branch_20261122 import CASES, save_json
from evaluate_new_artwork_replay_20261122 import coverage, sha
from gongbi_training.continuity_attachment import route_between_areas, source_ridge_support
from gongbi_training.runtime import LAB
from source_ink import extract


OUT = LAB/'continuity_attachment_gate_20261122'
WINDOWS = [
    [
        ('远峰可见外缘','outline',[318,24,414,92],[327,57,349,86],[385,28,413,48]),
        ('中央岩面凸缘','outline',[221,297,308,360],[224,340,248,357],[277,301,306,323]),
        ('瀑布左侧岩线','outline',[315,266,346,388],[319,272,339,294],[319,365,338,386]),
        ('右侧细枝接回主干','tree',[413,252,456,345],[420,264,442,283],[434,311,455,332]),
        ('贴邻岩面到同一主干','rock_control',[384,256,456,345],[389,266,409,285],[434,311,455,332]),
    ],
    [
        ('左岸山石外沿','outline',[15,178,101,253],[17,184,33,201],[77,227,98,245]),
        ('近景岩石上沿','outline',[391,451,500,526],[395,506,416,523],[464,455,490,471]),
        ('右侧高树细枝接回主干','tree',[501,74,545,146],[509,78,523,97],[517,121,531,141]),
        ('右侧枝脚邻近岩面','rock_control',[470,400,592,490],[560,419,583,450],[493,425,518,447]),
    ],
]
# These visible branches are checked on the source photo. No contour points
# enter route selection, which receives broad boxes and source evidence only.
BRANCH_REFS = [
    [[427,275],[432,279],[436,285],[437,291],[438,296],[441,304],[443,311],[442,319]],
    [[514,85],[517,91],[519,96],[521,101],[521,110],[523,121],[525,134]],
]


def panel(rgb, target, attempts):
    label,kind,crop,start,end = target
    x0,y0,x1,y1 = crop
    base = rgb[y0:y1,x0:x1]
    scale = min(340/(x1-x0),200/(y1-y0))
    size = (round((x1-x0)*scale),round((y1-y0)*scale))
    columns = []
    for name,key in [('原图与起止区域',None),('正式墨迹掩膜','strong'),('弱墨掩膜','weak'),
                     ('加入源图横截面证据','ridge')]:
        im = Image.fromarray(base).resize(size,Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(im)
        def point(p):
            x,y = p
            return (round((x-x0)*scale),round((y-y0)*scale))
        if key is None:
            for box,c in ((start,'#ffcd66'),(end,'#5bd2db')):
                draw.rectangle([point(box[:2]),point((box[2]-1,box[3]-1))],outline=c,width=2)
        else:
            result = attempts[key]
            if len(result['points'])>1:
                draw.line([point(p) for p in result['points']],
                          fill='#db941f' if result['status']=='ambiguous' else '#42e4df',width=2)
        columns.append((name,im))
    return columns


def run():
    protocol = OUT/'protocol.json'
    record = dict(source_hashes=[sha(c['file']) for c in CASES], boxes=WINDOWS, branches=BRANCH_REFS,
                  source_function_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/continuity_attachment.py'),
                  scope='Both artworks used as known-image regression. Source-reviewed endpoint windows. No expert truth.')
    if protocol.exists():
        assert json.loads(protocol.read_text('utf8'))==record
    else:
        save_json(protocol,record)
    all_cases = []
    for case,targets,branch in zip(CASES,WINDOWS,BRANCH_REFS):
        rgb = np.array(Image.open(case['file']).convert('RGB'))
        raw = extract(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        board = Image.new('RGB',(1455,245*len(targets)+30),'#f5f1e7')
        draw = ImageDraw.Draw(board)
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
        refs = dict(case['traces'])
        result = []
        for row,target in enumerate(targets):
            name,kind,crop,start,end = target
            enhanced = source_ridge_support(rgb,raw['contrast'],raw['weak_mask'],start,end)
            masks = dict(strong=raw['mask'],weak=raw['weak_mask'],ridge=enhanced)
            attempts = {}
            for key,mask in masks.items():
                route = route_between_areas(rgb,mask,crop=crop,start_box=start,end_box=end)
                points = route['points'].tolist()
                trace = refs.get(name) if kind == 'outline' else branch if kind == 'tree' else None
                scores = coverage(rgb.shape[:2],[(name,trace)],[dict(points=points)] if points else []) if trace else []
                attempts[key] = dict(points=points,status=route['status'],reference=scores[0] if scores else None,
                                     alternatives=len(route['routes']))
            for col,(title,im) in enumerate(panel(rgb,target,attempts)):
                x,y = 6+col*362,row*245
                draw.text((x,y+5),f'{name} · {title}',font=font,fill='#26382b')
                board.paste(im,(x,y+30))
            result.append(dict(name=name,kind=kind,attempts=attempts))
        draw.text((7,board.height-25),'青色是候选路线，橙色表示存在竞争路线。没有路线时保留原图。',font=font,fill='#26382b')
        board.save(OUT/f"{case['id']}_continuity_comparison.png")
        all_cases.append(dict(id=case['id'],results=result))
        for row in result:
            print(case['id'],row['name'],{k:(v['status'],v['reference']) for k,v in row['attempts'].items()},flush=True)
    parents = json.loads((LAB/'local_rock_patch_20261117/baseline_lock.json').read_text('utf8'))
    hashes = {key:sha(Path(parents[key]))==parents[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    save_json(OUT/'results.json',dict(cases=all_cases,baseline_hashes=hashes,
              adopted=False,robot_connected=False,independent_acceptance=False,protocol_sha256=sha(protocol)))


if __name__=='__main__':
    run()
