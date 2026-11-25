# -*- coding: utf-8 -*-
"""Find natural omission proposals while keeping the direction result hidden."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_faint_ink_continuation_20261121 import MODULE as CONTINUATION, ROOT as CONTINUATION_RUN
from evaluate_faint_ink_direction_filter_20261121 import MODULE, ROOT as DEVELOPED
from evaluate_ink_paper_extension_20261121 import write_json
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_width_20261119 import predict_extra
from validate_faint_ink_direction_20261121 import ROOT as PREVIOUS
from gongbi_training.faint_ink_continuation import recover_faint_ink
from gongbi_training.ink_paper_join_guard import protect_ink_joins
from gongbi_training.ink_paper_persistent_stop import persistent_paper_extension
from gongbi_training.ink_paper_width import separate_narrow_paper


ROOT = LAB/'faint_ink_natural_trial_20261121'
ALBUM = LAB/'cunfa_cross_source_20261117/original_images'
VIEWS = LAB/'guohua_faint_sources_20261116/views'
SOURCES = dict(outlines=ALBUM/'qingpu_18_0.jpeg', rocks=ALBUM/'qingpu_19_1.jpeg',
               far_rocks=ALBUM/'qingpu_23_0.jpeg', banks=ALBUM/'qingpu_25_0.jpeg',
               hills=VIEWS/'zhao-ji_autumn-colors_3_input.png',
               tree=VIEWS/'yun-shouping_sunset-along-the-floral-embankment_part_4_input.png')


def grouped(mask):
    merged = cv2.dilate(mask.astype(np.uint8), np.ones((3,3), np.uint8))
    _, ids = cv2.connectedComponents(merged, connectivity=8)
    groups = []
    for index in np.unique(ids[mask]):
        ys, xs = np.where(mask & (ids == index))
        pad = 7
        box = [max(0,int(xs.min())-pad), max(0,int(ys.min())-pad),
               min(mask.shape[1],int(xs.max())+pad+1), min(mask.shape[0],int(ys.max())+pad+1)]
        groups.append(dict(box=box, xy=[[int(x),int(y)] for y,x in zip(ys,xs)]))
    return sorted(groups,key=lambda item: (-len(item['xy']), item['box']))


def boards(name, image, groups, destination=None):
    destination = ROOT if destination is None else destination
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for page_start in range(0,len(groups),9):
        page = groups[page_start:page_start+9]
        board = Image.new('RGB',(1100,220*((len(page)+2)//3)), '#f5f1e7')
        pen = ImageDraw.Draw(board)
        for index,item in enumerate(page):
            x0,y0,x1,y1 = item['box']
            left, top = 366*(index%3)+6, 220*(index//3)+5
            tile = Image.fromarray(image).crop(item['box'])
            scale = min(173/tile.width,181/tile.height)
            tile = tile.resize((round(tile.width*scale),round(tile.height*scale)),Image.Resampling.NEAREST)
            board.paste(tile,(left,top+24))
            marked = tile.copy()
            d = ImageDraw.Draw(marked)
            for x,y in item['xy']:
                u,v=(x-x0)*scale,(y-y0)*scale
                d.rectangle((u,v,u+scale-1,v+scale-1),outline='#f5842d',width=1)
            board.paste(marked,(left+179,top+24))
            pen.text((left,top),f"{name} {page_start+index+1:02d} · {len(item['xy'])}像素",font=font,fill='#263c34')
        board.save(destination/f'{name}_source_{page_start//9+1}.png')


def main():
    ROOT.mkdir(exist_ok=True)
    assert sha(MODULE)==read(DEVELOPED/'protocol.json')['module']
    assert sha(CONTINUATION)==read(CONTINUATION_RUN/'protocol.json')['module']
    previous = read(PREVIOUS/'protocol.json')
    known = {info['sha256'] for info in previous['images'].values()}
    inputs = {name:dict(path=str(path),sha256=sha(path)) for name,path in SOURCES.items()}
    assert len({info['sha256'] for info in inputs.values()})==len(inputs)
    assert all(info['sha256'] not in known for info in inputs.values())
    manifest = dict(images=inputs, direction_rule=sha(MODULE),continuation_rule=sha(CONTINUATION),
                    scope='Old continuation proposals only. Direction rule is not run until references are frozen.')
    path = ROOT/'screen_protocol.json'
    if path.exists():
        assert read(path)==manifest
    else:
        write_json(path,manifest)
    out = {}
    for name,(image,original) in predict_extra(manifest).items():
        narrow, _, _ = separate_narrow_paper(image,original)
        continuous, _, _ = persistent_paper_extension(image,original,narrow)
        baseline, _, _ = protect_ink_joins(image,original,narrow,continuous,span=3,check_dark_edge=True)
        proposed, additions = recover_faint_ink(image,baseline,limit=6,allow_tips=True,min_path_contrast=12)
        np.savez_compressed(ROOT/f'{name}.npz',baseline=baseline,proposed=proposed,additions=additions)
        groups = grouped(additions)
        out[name] = dict(pixels=int(additions.sum()),groups=groups)
        boards(name,image,groups[:18])
    write_json(ROOT/'screen.json',out)
    print({name:(value['pixels'],len(value['groups'])) for name,value in out.items()})


if __name__ == '__main__':
    main()
