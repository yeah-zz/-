# -*- coding: utf-8 -*-
"""Known-artwork regression: rock-face edges and direction at trunk attachments."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_continuity_attachment_20261122 import BRANCH_REFS, CASES, OUT as PRIOR, WINDOWS
from evaluate_new_artwork_replay_20261122 import coverage, sha
from evaluate_faint_outline_branch_20261122 import save_json
from gongbi_training.rock_boundary_branch_direction import check_branch_entry, trace_rock_boundary
from gongbi_training.runtime import LAB, ROOT


OUT = LAB/'rock_boundary_branch_direction_20261122'
# Light-side signs follow the local rock faces in the original review windows.
POLARITIES = [[-1,1,-1],[1,-1]]
# The axis describes the reviewed woody direction downstream of a junction.
TRUNK_DIRECTIONS = [
    {'右侧细枝接回主干':[[439,299],[448,332]],
     '贴邻岩面到同一主干':[[439,299],[448,332]]},
    {'右侧高树细枝接回主干':[[521,104],[524,145]],
     '右侧枝脚邻近岩面':[[487,407],[511,441]]},
]


def draw_row(board, rgb, crop, row, label, old, new, kind, direction=None):
    x0,y0,x1,y1 = crop
    local = rgb[y0:y1,x0:x1]
    scale = min(335/(x1-x0),198/(y1-y0))
    size = (round((x1-x0)*scale),round((y1-y0)*scale))
    foot = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)

    def point(xy):
        return tuple(np.rint((np.asarray(xy)-[x0,y0])*scale).astype(int))

    for col,(name,item) in enumerate((('原图',None),('上轮路线',old),('岩面或方向核查',new))):
        tile = Image.fromarray(local).resize(size,Image.Resampling.NEAREST)
        d = ImageDraw.Draw(tile)
        if item is not None and len(item['points'])>1:
            color = '#e99b36' if item['status'] in ('ambiguous','direction_conflict') else '#42e4df'
            d.line([point(p) for p in item['points']],fill=color,width=2)
        if col==2 and direction:
            d.line([point(p) for p in direction],fill='#f27365',width=2)
        x,y=col*360+8,row*244
        board.paste(tile,(x,y+29))
        displayed = name
        if col==2 and kind!='outline':
            status = {'direction_supported':'方向相符','direction_conflict':'方向冲突',
                      'attachment_uncertain':'交接待定','no_route':'无路线','ambiguous':'有竞争路线'}
            displayed = status[new['status']]
        foot.text((x,y+6),f'{label} · {displayed}',font=font,fill='#263a32')


def run():
    OUT.mkdir(exist_ok=True)
    previous = json.loads((PRIOR/'results.json').read_text('utf8'))
    assert sha(PRIOR/'protocol.json')==previous['protocol_sha256']
    protocol = OUT/'protocol.json'
    settings = dict(prior_results_sha256=sha(PRIOR/'results.json'),
                    image_sha256=[sha(c['file']) for c in CASES],
                    tool_sha256=sha(ROOT/'gongbi_training/rock_boundary_branch_direction.py'),
                    light_sides=POLARITIES,trunk_axes=TRUNK_DIRECTIONS,
                    scope='Known-image regression with source-reviewed windows and branch axes, held-out not performed.')
    if protocol.exists():
        assert json.loads(protocol.read_text('utf8'))==settings
    else:
        save_json(protocol,settings)
    cases = []
    for i,(case,targets,prior) in enumerate(zip(CASES,WINDOWS,previous['cases'])):
        rgb = np.array(Image.open(case['file']).convert('RGB'))
        refs = dict(case['traces'])
        sheet = Image.new('RGB',(1084,244*len(targets)+33),'#f5f1e7')
        result = []
        edge_i = 0
        for index,((name,kind,crop,start,end),row) in enumerate(zip(targets,prior['results'])):
            assert name==row['name']
            old = row['attempts']['ridge']
            direction = TRUNK_DIRECTIONS[i].get(name)
            if kind=='outline':
                route,support = trace_rock_boundary(rgb,crop=crop,start_box=start,end_box=end,
                                                   lighter_side=POLARITIES[i][edge_i])
                points = route['points'].tolist()
                ref = coverage(rgb.shape[:2],[(name,refs[name])],[dict(points=points)] if points else [])[0]
                x0,y0,x1,y1 = crop
                candidate = dict(status=route['status'],points=points,reference=ref,
                                 source_edge_pixels=int(support[y0:y1,x0:x1].sum()))
                edge_i += 1
            else:
                assessed = check_branch_entry(old['points'],end,direction)
                candidate = dict(old,**assessed)
                if old['status']=='ambiguous' and candidate['status']=='direction_supported':
                    candidate['status']='ambiguous'
            draw_row(sheet,rgb,crop,index,name,old,candidate,kind,direction)
            result.append(dict(name=name,kind=kind,before=old,after=candidate))
            print(case['id'],name,old['status'],old.get('reference'),candidate['status'],
                  candidate.get('reference'),candidate.get('cosine'),flush=True)
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
        ImageDraw.Draw(sheet).text((8,sheet.height-25),'青色为候选，橙色为方向冲突或存在竞争路线，红色为人工复核的主干延伸方向。',font=font,fill='#263a32')
        sheet.save(OUT/f"{case['id']}_boundary_direction.png")
        cases.append(dict(id=case['id'],rows=result))
    parents = json.loads((LAB/'local_rock_patch_20261117/baseline_lock.json').read_text('utf8'))
    checks = {key:sha(Path(parents[key]))==parents[key+'_sha256'] for key in ('approved500','working489')}
    assert all(checks.values())
    save_json(OUT/'results.json',dict(cases=cases,baseline_hashes=checks,protocol_sha256=sha(protocol),
                                     independent_acceptance=False,adopted=False,robot_connected=False))


if __name__=='__main__':
    run()
