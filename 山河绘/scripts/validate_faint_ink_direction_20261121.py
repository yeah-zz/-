# -*- coding: utf-8 -*-
"""Freeze source references before testing direction-constrained faint ink."""
from pathlib import Path

import numpy as np

from evaluate_faint_ink_continuation_20261121 import MODULE as CONTINUATION, ROOT as CONTINUATION_RUN
from evaluate_faint_ink_direction_filter_20261121 import MODULE, ROOT as DEVELOPED
from evaluate_ink_paper_extension_20261121 import write_json
from evaluate_ink_paper_regions_20261118 import render
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_width_20261119 import predict_extra
from validate_ink_paper_join_guard_20261121 import ROOT as EARLIER
from gongbi_training.faint_ink_continuation import recover_faint_ink
from gongbi_training.faint_ink_direction_filter import filter_faint_ink_direction
from gongbi_training.ink_paper_join_guard import protect_ink_joins
from gongbi_training.ink_paper_persistent_stop import persistent_paper_extension
from gongbi_training.ink_paper_width import separate_narrow_paper


ROOT = LAB/'faint_ink_direction_validation_20261121'
ALBUM = LAB/'cunfa_cross_source_20261117/original_images'
SOURCES = dict(brush=ALBUM/'qingpu_15_1.jpeg', mountain=ALBUM/'qingpu_22_1.jpeg',
               textured=LAB/'guohua_faint_sources_20261116/views/yun-shouping_sunset-along-the-floral-embankment_part_1_input.png')
SPECS = [
    dict(id='B_tip', image='brush', title='笔锋渐淡', kind='ink', box=[621,342,681,367],
         trace=[[626,360],[632,360],[635,359],[641,358],[647,357],[650,356],
                [656,355],[659,354],[665,353],[668,352]]),
    dict(id='B_paper', image='brush', title='笔锋周围纸面', kind='paper', box=[621,337,681,368],
         polygon=[[628,346],[671,339],[673,343],[630,350]]),
    dict(id='M_ridge', image='mountain', title='淡山脊', kind='ink', box=[188,59,240,92],
         trace=[[198,83],[201,81],[204,79],[207,78],[210,76],[213,76],
                [216,75],[219,73],[222,72],[225,70],[228,68]]),
    dict(id='M_paper', image='mountain', title='山脊外纸面', kind='paper', box=[188,57,240,92],
         polygon=[[199,74],[206,68],[214,67],[224,60],[222,64],[212,72],[204,73]]),
    dict(id='T_near', image='textured', title='山面旁浅纸纹', kind='paper', box=[96,61,148,122],
         polygon=[[103,69],[128,69],[133,79],[125,93],[108,94],[100,86]]),
    dict(id='T_open', image='textured', title='开阔纸纹', kind='paper', box=[285,396,366,451],
         polygon=[[301,409],[350,407],[353,432],[309,439],[297,428]]),
]
CONTROLLED = dict(image='brush', box=[609,354,639,368],
                  trace=[[612,363],[614,362],[617,362],[620,361],[623,361],[626,360],[632,360]],
                  erased_columns=[619,620,621,622])


def polygon_mask(shape, polygon):
    import cv2
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 1)
    return mask.astype(bool)


def line_mask(shape, trace):
    guide = np.asarray(trace)
    columns = np.arange(guide[0,0], guide[-1,0]+1)
    rows = np.rint(np.interp(columns,guide[:,0],guide[:,1])).astype(int)
    mask = np.zeros(shape, bool)
    for offset in (-1,0,1):
        mask[rows+offset,columns] = True
    return mask


def fixtures():
    development = read(DEVELOPED/'protocol.json')
    assert development['module']==sha(MODULE)
    assert sha(DEVELOPED/'protocol.json')==read(DEVELOPED/'results.json')['protocol_sha256']
    assert read(CONTINUATION_RUN/'protocol.json')['module']==sha(CONTINUATION)
    used = {entry['source_sha256'] for entry in read(EARLIER/'protocol.json')['images'].values()}
    images = {key:dict(path=str(path),sha256=sha(path)) for key,path in SOURCES.items()}
    assert len({info['sha256'] for info in images.values()})==len(images)
    assert all(info['sha256'] not in used for info in images.values())
    locked = dict(images=images,specs=SPECS,controlled=CONTROLLED,development=sha(DEVELOPED/'results.json'),
                  code={str(path):sha(path) for path in (CONTINUATION,MODULE)}, independent_reviewer=False,
                  scope='Previously unused images for the faint-ink rule. Source references fixed before inference.')
    protocol = ROOT/'protocol.json'
    if protocol.exists():
        assert read(protocol)==locked
    else:
        write_json(protocol,locked)
    return protocol,locked


def compare(image, baseline):
    raw, proposed = recover_faint_ink(image,baseline,limit=6,allow_tips=True,min_path_contrast=12)
    protected, accepted, refused = filter_faint_ink_direction(image,baseline,raw)
    np.testing.assert_array_equal(proposed,accepted | refused)
    return dict(baseline=baseline, continuation=raw, direction=protected)


def main():
    ROOT.mkdir(exist_ok=True)
    protocol, locked = fixtures()
    results, images, counts = {}, {}, {}
    for name,(image,original) in predict_extra(locked).items():
        narrow, _, _ = separate_narrow_paper(image,original)
        extended, _, _ = persistent_paper_extension(image,original,narrow)
        baseline, _, _ = protect_ink_joins(image,original,narrow,extended,span=3,check_dark_edge=True)
        variants = dict(frozen=original,**compare(image,baseline))
        images[name] = image, variants
        counts[name] = {key:int(((pred>0)&(baseline==0)).sum()) for key,pred in variants.items()}
        np.savez_compressed(ROOT/f'{name}.npz',**variants)
    panels = []
    for spec in locked['specs']:
        image, variants = images[spec['image']]
        region = (line_mask(image.shape[:2],spec['trace']) if spec['kind']=='ink'
                  else polygon_mask(image.shape[:2],spec['polygon']))
        stats = {}
        for key,pred in variants.items():
            ink = (pred>0)&region
            stats[key] = dict(pixels=int(ink.sum()), covered_columns=int(ink.any(axis=0).sum()),
                              new_pixels=int((ink&(variants['baseline']==0)).sum()))
        results[spec['id']] = dict(kind=spec['kind'],pixels=int(region.sum()),
                                   columns=int(region.any(axis=0).sum()),scores=stats)
        panels.append((spec,image,region,variants))

    control = locked['controlled']
    image, original = images[control['image']]
    trace = line_mask(image.shape[:2],control['trace'])
    omitted = np.zeros(image.shape[:2],bool)
    omitted[:,control['erased_columns']] = trace[:,control['erased_columns']]
    interrupted = original['baseline'].copy()
    interrupted[omitted] = 0
    lost = omitted & (original['baseline']>0)
    checked = compare(image,interrupted)
    control_result = dict(original_erased_pixels=int(lost.sum()),
                          original_erased_columns=int(lost.any(axis=0).sum()),
                          recovered_columns={name:int(((label>0)&lost).any(axis=0).sum())
                                             for name,label in checked.items()},
                          recovered_pixels={name:int(((label>0)&lost).sum()) for name,label in checked.items()})
    panels.append((dict(title='受控短断口',box=control['box']),image,omitted,
                   dict(original=original['baseline'],**checked)))
    render(panels[:-1],ROOT/'comparison.png',['原图参考','冻结模型','补线基线','直接续接','方向筛选'])
    render([panels[-1]],ROOT/'controlled.png',['原图参考','原基线','受控漏提','直接续接','方向筛选'])
    np.savez_compressed(ROOT/'controlled.npz',omitted=omitted,lost=lost,**checked)

    approved = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {k:sha(approved[k])==approved[k+'_sha256'] for k in ('approved500','working489')}
    assert all(hashes.values())
    output = dict(regions=results,controlled=control_result,additions=counts,baseline_hashes=hashes,
                  protocol_sha256=sha(protocol),independent_trial=True,independent_acceptance=False,
                  adopted=False,robot_connected=False)
    write_json(ROOT/'results.json',output)
    print(counts)
    for name,row in results.items():
        print(name,row)
    print(control_result)


if __name__ == '__main__':
    main()
