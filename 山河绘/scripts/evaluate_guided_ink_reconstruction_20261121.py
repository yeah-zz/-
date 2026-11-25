# -*- coding: utf-8 -*-
"""Extract two guided stroke candidates from continuous original ink bands."""
import cv2
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_whole_scene_repair_20261121 import ROOT as WHOLE
from evaluate_object_stroke_gate_20261121 import ROOT as GATE, trace_evidence
from gongbi_training.ink_band_reconstruction import reconstruct_ink_path
from gongbi_training.local_rock_patch import contrast_field
from validate_faint_ink_narrow_support_holdout_20261121 import segment


LAB = segment.LAB
ROOT = LAB/'guided_ink_reconstruction_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/ink_band_reconstruction.py'
RIDGE_WEIGHTS = {'balanced':.35,'strong':1.9}
TARGETS = (
    dict(id='rock_ridge',title='山面内部转折线',module=2,role='structure',
         start=[267,308],end=[281,334],crop=[258,299,292,346]),
    dict(id='pine_right',title='松树右斜枝',module=8,role='branch',
         start=[295,138],end=[362,170],crop=[285,129,372,180]),
    dict(id='pine_left',title='松树左枝对照',module=8,role='branch',
         start=[225,155],end=[283,155],crop=[218,141,291,163]),
)


def to_mask(path, shape):
    canvas = np.zeros(shape,np.uint8)
    if len(path)>1:
        cv2.polylines(canvas,[np.asarray(path,dtype=np.int32)],False,1,1)
    return canvas


def describe(path, mask, trace, contrast):
    pixels = to_mask(path,mask.shape)
    evidence = {f'radius{radius}':trace_evidence(trace,pixels,radius)
                for radius in (1,3)}
    evidence['path_points'] = len(path)
    evidence['length_px'] = float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()) if len(path)>1 else 0.0
    evidence['on_mask'] = int(((pixels>0)&(mask>0)).sum())
    evidence['mean_local_contrast'] = float(contrast[pixels>0].mean()) if pixels.any() else None
    return evidence


def draw_views(rgb, coverage, plotted):
    height = 245
    board = Image.new('RGB',(1080,height*len(plotted)+29),'#f5f1e7')
    pen = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    columns = [(None,'原图'),('plan','489笔轨迹'),('mask','仅掩膜路径'),('weighted','原图加权路径')]
    for row, (spec, unweighted, path) in enumerate(plotted):
        left,top,right,bottom = spec['crop']
        size = (right-left,bottom-top)
        scale = min(250/size[0],199/size[1])
        resized = tuple(round(v*scale) for v in size)
        for col,(mode,title) in enumerate(columns):
            if mode=='plan':
                view = np.full((size[1],size[0],3),250,np.uint8)
                view[coverage[top:bottom,left:right]] = [32,91,72]
            else:
                view = rgb[top:bottom,left:right].copy()
                curve = unweighted if mode=='mask' else path
                if mode and len(curve)>1:
                    coords = np.asarray(curve,dtype=np.int32)-[left,top]
                    coords = coords.astype(np.int32)
                    cv2.polylines(view,[coords],False,(238,122,30) if mode=='mask' else (28,157,185),1)
            x,y=col*270+7,row*height+6
            pen.text((x,y),f"{spec['title']} · {title}",font=font,fill='#263c34')
            board.paste(Image.fromarray(view).resize(resized,Image.Resampling.NEAREST),(x,y+27))
    pen.text((8,board.height-24),'橙色沿掩膜寻路，蓝色使用原图墨线证据。两种候选共用人工选定的起止点。',font=font,fill='#263c34')
    board.save(ROOT/'comparison.png')


def main():
    ROOT.mkdir(exist_ok=True)
    bench = segment.read(WHOLE/'protocol.json')
    gate = segment.read(GATE/'references.json')
    sources = bench['images']['G03_01']
    assert segment.sha(sources['path'])==sources['sha256']
    gate_result = segment.read(GATE/'results.json')
    assert segment.sha(GATE/'references.json')==gate_result['reference_sha256']
    evidence_file = WHOLE/'G03_01.npz'
    assert segment.sha(evidence_file)==gate['inputs']['g_scene']['frozen_array_sha256']
    lock = ROOT/'protocol.json'
    segment.lock_file(lock,dict(source=sources,evidence_sha256=segment.sha(evidence_file),
                     reference_sha256=segment.sha(GATE/'references.json'),targets=list(TARGETS),
                     scope='Manually selected crop and endpoint hints only. No interior reference trace input.'))
    rules = ROOT/'rules.json'
    segment.lock_file(rules,dict(module_sha256=segment.sha(MODULE),weights=RIDGE_WEIGHTS,
                                scope='Weights compared on development examples.'))
    with Image.open(sources['path']) as image, np.load(evidence_file) as cached:
        rgb = np.array(image.convert('RGB'))
        mask = cached['baseline']
        coverage = cached['plan_coverage']
    contrast = contrast_field(rgb)
    refs = {r['id']:r['trace'] for r in gate['rows']}
    assessments,plotted,candidates = {},[],[]
    for spec in TARGETS:
        common = dict(crop=spec['crop'])
        unweighted = reconstruct_ink_path(rgb,mask,spec['start'],spec['end'],source_weighted=False,**common)
        weighted = reconstruct_ink_path(rgb,mask,spec['start'],spec['end'],
                                        ridge_weight=RIDGE_WEIGHTS['balanced'],**common)
        dark_weighted = reconstruct_ink_path(rgb,mask,spec['start'],spec['end'],
                                             ridge_weight=RIDGE_WEIGHTS['strong'],**common)
        trace = refs[spec['id']]
        assessments[spec['id']] = dict(old={f'radius{radius}':trace_evidence(trace,coverage,radius)
                                                   for radius in (1,3)},
                                      mask_only=describe(unweighted,mask,trace,contrast),
                                      strong_source_weight=describe(dark_weighted,mask,trace,contrast),
                                      source_weighted=describe(weighted,mask,trace,contrast))
        if assessments[spec['id']]['source_weighted']['radius3']['connected']:
            candidates.append(dict(id=spec['id'],module=spec['module'],role=spec['role'],
                                   points=weighted.tolist(),sequence_verified=False))
        plotted.append((spec,unweighted,weighted))
    draw_views(rgb,coverage,plotted)
    segment.write_json(ROOT/'candidate_paths.json',dict(paths=candidates,coordinates='G03_01 source pixels',
                       standalone_preview=True,adopted=False,robot_connected=False))
    frozen = segment.read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:segment.sha(frozen[key])==frozen[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    segment.write_json(ROOT/'results.json',dict(metrics=assessments,baseline_hashes=hashes,
                       protocol_sha256=segment.sha(lock),rules_sha256=segment.sha(rules),
                       independent_acceptance=False,robot_connected=False,adopted=False))
    print(assessments)


if __name__=='__main__':
    main()
