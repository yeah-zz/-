# -*- coding: utf-8 -*-
"""Review all prior-rule proposals from the supplemental image screen."""
import numpy as np
from PIL import Image

from validate_ink_paper_join_guard_20261121 import DEVELOPED, MODULE, ROOT, read, sha, write_json
from gongbi_training.ink_paper_join_guard import protect_ink_joins


SCREEN = ROOT/'screened'
REFERENCES = {
    'round_215_166': ('paper', '两条弧线之间的连续亮纸'),
    'round_470_218': ('unknown', '房屋细线间的浅灰边缘，归属待核'),
    'shaded_235_141': ('paper', '两条较暗的纵向笔迹间有连续亮带'),
    'shaded_241_244': ('unknown', '位于墨线向下交汇前的亮灰过渡'),
}


def main():
    manifest = read(SCREEN/'inputs.json')
    coordinates = read(SCREEN/'proposals.json')
    assert set(REFERENCES) == set(coordinates)
    assert sha(MODULE) == read(DEVELOPED/'protocol.json')['module']
    assert sha(ROOT/'protocol.json') == read(ROOT/'results.json')['protocol_sha256']
    locked = dict(input_sha=sha(SCREEN/'inputs.json'), coords_sha=sha(SCREEN/'proposals.json'),
                  rule_sha=sha(MODULE), arrays={key:sha(SCREEN/f'{key}.npz') for key in manifest['images']},
                  preguard_reference={key:dict(label=label,basis=basis) for key,(label,basis) in REFERENCES.items()},
                  scope='Selected from old proposals and reviewed on source before evaluating the new rule.')
    protocol = SCREEN/'review_protocol.json'
    if protocol.exists():
        assert read(protocol) == locked
    else:
        write_json(protocol, locked)
    points, counts = [], {}
    for name, info in manifest['images'].items():
        assert sha(info['path']) == info['sha256']
        with Image.open(info['path']) as source:
            image = np.array(source.convert('RGB'))
        with np.load(SCREEN/f'{name}.npz') as saved:
            original, narrow, before, proposed = (saved[key] for key in ('original','narrow','prior','proposed'))
        _, accepted, rejected = protect_ink_joins(image, original, narrow, before, span=3, check_dark_edge=True)
        np.testing.assert_array_equal(accepted | rejected, proposed)
        counts[name] = dict(proposed=int(proposed.sum()), accepted=int(accepted.sum()), restored=int(rejected.sum()))
        selected = {key:point['xy'] for key,point in coordinates.items() if point['image']==name}
        assert set(map(tuple,selected.values())) == {(int(x),int(y)) for y,x in np.argwhere(proposed)}
        for key,(x,y) in selected.items():
            label, basis = REFERENCES[key]
            points.append(dict(id=key,image=name,xy=[x,y],label=label,basis=basis,accepted=bool(accepted[y,x])))
    write_json(SCREEN/'review_results.json',dict(points=points, counts=counts, protocol_sha256=sha(protocol),
                                                independent_acceptance=False, adopted=False))
    print(counts)
    print(points)


if __name__ == '__main__':
    main()
