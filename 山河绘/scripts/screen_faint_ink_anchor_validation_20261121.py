# -*- coding: utf-8 -*-
"""Screen new images with the old direction rule before testing anchor evidence."""
import numpy as np
from PIL import Image

from evaluate_faint_ink_anchor_guard_20261121 import DIRECTION, MODULE, NATURAL, OPTIONS, ROOT as DEVELOPED
from evaluate_ink_paper_extension_20261121 import write_json
from evaluate_ink_paper_seams_20261118 import LAB, read, sha
from evaluate_ink_paper_width_20261119 import predict_extra
from screen_faint_ink_natural_20261121 import boards, grouped
from validate_faint_ink_direction_20261121 import compare
from gongbi_training.ink_paper_join_guard import protect_ink_joins
from gongbi_training.ink_paper_persistent_stop import persistent_paper_extension
from gongbi_training.ink_paper_width import separate_narrow_paper


ROOT = LAB/'faint_ink_anchor_validation_20261121'
MUSEUM = LAB/'cunfa_fresh_validation_20261111/sources'
ALBUM = LAB/'cunfa_cross_source_20261117/original_images'
SOURCES = {
    'mountain': (MUSEUM/'231531_0.jpg',[30,125,460,560]),
    'branches': (MUSEUM/'234350_0.jpg',[170,285,930,825]),
    'silk': (MUSEUM/'npm_5621_10_content.png',None),
    'folded': (ALBUM/'qingpu_21_2.jpeg',[17,92,610,431]),
    'rounded': (ALBUM/'qingpu_22_0.jpeg',[13,81,572,470]),
}


def main():
    ROOT.mkdir(exist_ok=True)
    development = read(DEVELOPED/'protocol.json')
    assert development['module']==sha(MODULE)
    assert development['options']['core']==OPTIONS['core']
    assert read(NATURAL/'screen_protocol.json')['direction_rule']==sha(DIRECTION)
    excluded = {item['sha256'] for item in read(NATURAL/'screen_protocol.json')['images'].values()}
    images = {}
    for name,(source,box) in SOURCES.items():
        assert sha(source) not in excluded
        path = source
        if box:
            path = ROOT/f'{name}_input.png'
            if not path.exists():
                with Image.open(source) as image:
                    image.convert('RGB').crop(box).save(path)
        images[name] = dict(path=str(path),sha256=sha(path),source=str(source),crop=box,source_sha256=sha(source))
    manifest = dict(images=images,anchor_rule=sha(MODULE),options=OPTIONS['core'],
                    scope='Screened with the antecedent direction rule only. Anchor results remain hidden.')
    protocol = ROOT/'screen_protocol.json'
    if protocol.exists():
        assert read(protocol)==manifest
    else:
        write_json(protocol,manifest)
    results = {}
    for name,(image,original) in predict_extra(manifest).items():
        narrow, _, _ = separate_narrow_paper(image,original)
        persistent, _, _ = persistent_paper_extension(image,original,narrow)
        baseline, _, _ = protect_ink_joins(image,original,narrow,persistent,span=3,check_dark_edge=True)
        variants = compare(image,baseline)
        added = (baseline==0)&(variants['direction']>0)
        groups = grouped(added)
        np.savez_compressed(ROOT/f'{name}.npz',baseline=baseline,direction=variants['direction'],additions=added)
        results[name] = dict(additions=int(added.sum()),groups=groups)
        boards(name,image,groups[:18],destination=ROOT)
    write_json(ROOT/'screen.json',results)
    print({name:(v['additions'],len(v['groups'])) for name,v in results.items()})


if __name__ == '__main__':
    main()
