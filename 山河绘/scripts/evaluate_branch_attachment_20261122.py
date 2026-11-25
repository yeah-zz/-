# -*- coding: utf-8 -*-
"""Compare image-derived trunk attachments with the prior manually zoned search."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_whole_scene_repair_20261121 import ARCHIVE, ROOT as WHOLE
from evaluate_automatic_ink_routing_20261121 import metric, role_hint
from evaluate_guided_ink_reconstruction_20261121 import TARGETS, to_mask
from evaluate_object_stroke_gate_20261121 import ROOT as GATE
from evaluate_structural_ink_ownership_20261122 import ROOT as PRIOR, ZONES, rectangle, tree_outline
from gongbi_training.automatic_ink_routing import automatic_ink_route
from gongbi_training.branch_attachment import trace_branch_attachment
from gongbi_training.local_rock_patch import contrast_field
from gongbi_training.structural_ink_routing import route_owned_ink
from gongbi_training.structure_candidate_rank import rank_owned_candidates
from validate_faint_ink_narrow_support_holdout_20261121 import segment


LAB = segment.LAB
ROOT = LAB / 'branch_attachment_20261122'
# Source review shows a light window separating the upper and lower paths.
LEFT_VISIBLE_LIMB = [[254, 152], [257, 153], [259, 154], [262, 155], [266, 154], [269, 153], [272, 153]]
LEFT_PAPER_XY = [[x, y] for x in range(260, 266) for y in (150, 151)]
JUNCTION_BOXES = {'pine_left': [279, 150, 287, 158], 'pine_right': [292, 136, 302, 145]}
CROP_OFFSETS = [(-3, 0), (3, 0), (0, -3), (0, 3)]


def junction_in_box(point, box):
    return bool(point and box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3])


def overview(image, rows):
    output = Image.new('RGB', (1120, 509), '#f5f1e7')
    draw = ImageDraw.Draw(output)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    for row, (target, before, after, revised) in enumerate(rows):
        left, top, right, bottom = target['crop']
        x0, y0 = row * 235, 6
        scale = min(265 / (right - left), 195 / (bottom - top))
        size = (round((right - left) * scale), round((bottom - top) * scale))
        for column, (name, result) in enumerate((('原图', None), ('上轮人工端区', before),
                                                ('自动找交接处', after), ('原图复核参考', None))):
            tile = Image.fromarray(image[top:bottom, left:right]).resize(size, Image.Resampling.NEAREST)
            layer = ImageDraw.Draw(tile)
            def local(point):
                return tuple(np.rint((np.asarray(point) - [left, top]) * scale).astype(int))
            if result and len(result['points']):
                color = '#f1a737' if result['status'] == 'ambiguous' else '#22b8d1'
                layer.line(list(map(local, result['points'])), fill=color, width=2)
            if column == 2 and after['junctions']:
                x, y = local(after.get('junction', after['junctions'][0]['junction']))
                layer.ellipse((x-4, y-4, x+4, y+4), outline='#f07158', width=2)
            if column == 3:
                layer.line(list(map(local, revised)), fill='#22b8d1', width=2)
                a, b, c, d = JUNCTION_BOXES[target['id']]
                layer.rectangle([local((a, b)), local((c-1, d-1))], outline='#f07158', width=2)
                if target['id'] == 'pine_left':
                    for point in LEFT_PAPER_XY:
                        x, y = local(point)
                        layer.point((x, y), fill='#944eaa')
            x = y0 + column * 280
            draw.text((x, x0 + 4), f"{target['title']} · {name}", font=font, fill='#263c34')
            output.paste(tile, (x, x0 + 30))
    draw.text((7, 480), '红圈为自动定位，红框为原图复核范围。紫点标记左枝白缝，蓝线为路径或参考。', font=font, fill='#263c34')
    output.save(ROOT / 'comparison.png')


def main():
    ROOT.mkdir(exist_ok=True)
    source = ARCHIVE / 'G03_01/source.png'
    mask_file = WHOLE / 'G03_01.npz'
    ref_file = GATE / 'references.json'
    base_lock = segment.read(LAB / 'local_rock_patch_20261117/baseline_lock.json')
    baseline_hashes = {key: segment.sha(base_lock[key]) == base_lock[key + '_sha256']
                       for key in ('approved500', 'working489')}
    assert all(baseline_hashes.values())
    prior = segment.read(PRIOR / 'results.json')
    assert segment.sha(PRIOR / 'protocol.json') == prior['protocol_sha256']
    module = Path(__file__).resolve().parents[1] / 'gongbi_training/branch_attachment.py'
    protocol = ROOT / 'protocol.json'
    segment.lock_file(protocol, {'source': segment.sha(source), 'mask': segment.sha(mask_file),
                                 'module': segment.sha(module), 'refs': segment.sha(ref_file),
                                 'prior': segment.sha(PRIOR / 'results.json'), 'junction_boxes': JUNCTION_BOXES,
                                 'left_ink': LEFT_VISIBLE_LIMB, 'left_paper': LEFT_PAPER_XY,
                                 'scope': 'Known color painting; crop given, junction side inferred. Development only.'})
    crop_protocol = ROOT / 'crop_probe_protocol.json'
    segment.lock_file(crop_protocol, {'main_protocol_sha256': segment.sha(protocol), 'offsets': CROP_OFFSETS,
                                      'scope': 'Fixed offsets on the same development crops.'})
    with Image.open(source) as im, np.load(mask_file) as cached:
        image, ink = np.array(im.convert('RGB')), cached['baseline']
    plan = segment.read(base_lock['working489'])
    owner = tree_outline(plan, ink.shape)
    refs = {item['id']: np.array(item['trace']) for item in segment.read(ref_file)['rows']}
    revised = refs['pine_left'].copy()
    support = np.array(LEFT_VISIBLE_LIMB)
    inside = (revised[:, 0] > support[0, 0]) & (revised[:, 0] < support[-1, 0])
    revised[inside, 1] = np.rint(np.interp(revised[inside, 0], support[:, 0], support[:, 1])).astype(int)
    field = contrast_field(image)
    scores, views, sensitivity = {}, [], {}
    for target in TARGETS[1:]:
        key = target['id']
        box = JUNCTION_BOXES[key]
        zone = ZONES[key]
        crop = target['crop']
        hint = role_hint(plan, target, ink.shape)
        base = automatic_ink_route(image, ink, crop=crop, hints=hint, gate_options=(8, 3, 2, .03), include_routes=True)
        owned = route_owned_ink(image, ink, owner, crop=crop, start_zone=rectangle(ink.shape, zone['start']),
                                end_zone=rectangle(ink.shape, zone['end']), hints=hint)
        old = rank_owned_candidates([base, owned], owner, zone['start'], zone['end'])
        assert metric(old, ink, refs[key], field) == {k: v for k,v in prior['scores'][key]['ranked'].items()
                                                    if k != 'shortlist'}
        new = trace_branch_attachment(image, ink, owner, crop=crop, hints=hint)
        j = new.get('junction', new['junctions'][0]['junction'] if new['junctions'] else None)
        reviewed_trace = revised if key == 'pine_left' else refs[key]
        result = {name: {'frozen_reference': metric(route, ink, refs[key], field),
                         'reviewed_visible_limb': metric(route, ink, reviewed_trace, field)}
                  for name, route in (('prior', old), ('automatic_attachment', new))}
        result['junction'] = j
        result['junction_candidates'] = new['junctions']
        result['junction_in_reviewed_box'] = junction_in_box(j, box)
        if key == 'pine_left':
            paper = np.array(LEFT_PAPER_XY)
            for name, route in (('prior', old), ('automatic_attachment', new)):
                pixels = to_mask(route['points'], ink.shape)
                result[name]['confirmed_paper_hits'] = int(pixels[paper[:, 1], paper[:, 0]].sum())
        scores[key] = result
        views.append((target, old, new, reviewed_trace))
        sensitivity[key] = []
        for dx, dy in CROP_OFFSETS:
            shifted = tuple(c + offset for c, offset in zip(crop, (dx, dy, dx, dy)))
            probe = trace_branch_attachment(image, ink, owner, crop=shifted, hints=hint)
            output = metric(probe, ink, reviewed_trace, field)
            sensitivity[key].append(dict(offset=(dx, dy), junction=probe.get('junction'),
                                         within_box=junction_in_box(probe.get('junction'), box),
                                         covered=output['radius3']['covered_near'],
                                         connected=output['radius3']['connected'], status=probe['status']))
    overview(image, views)
    segment.write_json(ROOT / 'results.json', dict(scores=scores, protocol_sha256=segment.sha(protocol),
                       baseline_hashes=baseline_hashes, independent_acceptance=False, adopted=False,
                       robot_connected=False))
    segment.write_json(ROOT / 'crop_sensitivity.json', dict(scores=sensitivity,
                       protocol_sha256=segment.sha(crop_protocol)))
    for key, row in scores.items():
        print(key, row['junction'], row['junction_in_reviewed_box'], {
            stage: [row[stage][ref]['radius3']['covered_near'] for ref in ('frozen_reference','reviewed_visible_limb')]
            for stage in ('prior','automatic_attachment')})
    print('crop probe', {key: sum(row['within_box'] for row in rows) for key, rows in sensitivity.items()})


if __name__ == '__main__':
    main()
