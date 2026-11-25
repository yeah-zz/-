# -*- coding: utf-8 -*-
"""Compare frozen paper seam rules on source-reviewed teaching images."""
import hashlib
import json
import sys
from pathlib import Path

from evaluate_ink_paper_seams_20261118 import LAB, UPSTREAM, read, sha
from gongbi_training.ink_paper_context import separate_contextual_paper

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = LAB / 'paper_seam_transfer_20261118'
SOURCES = LAB / 'cunfa_role_reference_20261112'
MODULES = Path(__file__).resolve().parents[1] / 'gongbi_training'
V1 = LAB / 'paper_seam_trial_20261118'
V2 = LAB / 'paper_context_trial_20261118'
MODELS = LAB / 'cunfa_roles_v2_20261112'

# Native image coordinates, selected by reviewing source pixels with the
# coordinate grid and closeups. Uncertain points are never scored as paper.
POINTS = {
    'aam_long_hemp': {
        'paper': [(326,316),(362,286),(396,284),(469,411),(472,425),(489,421),(505,410),(519,320),(399,352),(406,344)],
        'ink': [(292,312),(324,263),(358,342),(373,324),(369,364),(421,374),(443,359),(489,343),(411,400),(516,336)],
        'uncertain': [(389,370),(410,379),(445,305),(427,328),(365,305),(489,291),(410,335),(461,333),(492,359),(513,373)],
    },
    'hkma_hemp': {
        'paper': [(54,54),(58,65),(63,76),(66,85),(82,63),(128,105),(126,89),(159,81),(164,92),(139,82)],
        'ink': [(29,31),(27,46),(39,76),(54,98),(65,104),(78,34),(105,49),(121,66),(140,65),(155,75)],
        'uncertain': [(89,53),(101,67),(108,73),(94,76),(71,91),(67,93),(85,73),(26,65),(158,73),(169,84)],
    },
    'hkma_axe': {
        'paper': [(130,151),(215,155),(211,177),(201,149)],
        'ink': [(148,139),(156,136),(185,142),(178,176),(182,174),(197,188),(205,185),(136,164),(183,148),(171,198)],
        'uncertain': [(155,156),(160,165),(171,177),(191,166),(159,182),(207,173),(132,182),(173,148),(179,156),(180,162),(188,180),(149,186)],
    },
}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def lock_artifacts():
    roles = read(UPSTREAM / 'protocol.json')
    first = read(V1 / 'protocol.json')
    second = read(V2 / 'protocol.json')
    assert sha(V1 / 'protocol.json') == second['base_protocol_sha256']
    pairs = [
        (MODELS / 'flat_roles.pt', roles['teacher_sha256']),
        (UPSTREAM / 'band_paper.pt', first['candidate_sha256']),
        (MODULES / 'ink_paper_seams.py', first['filter_sha256']),
        (MODULES / 'ink_paper_context.py', second['context_filter_sha256']),
    ]
    assert all(sha(path) == digest for path, digest in pairs)
    return {str(path): digest for path, digest in pairs}


def freeze():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT / 'frozen.json').exists() and not (ROOT / 'results.json').exists()
    catalog = read(SOURCES / 'source_references.json')['records']
    examples = []
    for name, points in POINTS.items():
        record = next(r for r in catalog if r['id'] == name)
        path = Path(record['image'])
        assert sha(path) == record['crop_sha256']
        with Image.open(path) as image:
            positions = [xy for entries in points.values() for xy in entries]
            assert len(positions) == len(set(positions))
            assert all(0 <= x < image.width and 0 <= y < image.height for x, y in positions)
        examples.append(dict(id=name, image=str(path), sha256=sha(path), points=points,
                             source_url=record['source_url'], source_group=record['source_group']))
    write(ROOT / 'frozen.json', dict(examples=examples, artifacts=lock_artifacts(),
         scope='pre-existing project assets, new to the seam-rule evaluation; assistant source-only point review',
         metrics='ink and paper points are scored; uncertain points and unmarked pixels are not ground truth',
         expert_verified=False, independent_stroke_validation=False))
    print('References and artifact hashes frozen before inference.')


def compare_board(panels):
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    width, height = 285, 235
    board = Image.new('RGB', (width * 4, height * len(panels) + 32), '#f5f1e7')
    draw = ImageDraw.Draw(board)
    colors = {'ink': '#1f8661', 'paper': '#236ea9', 'uncertain': '#855da1'}
    for i, (label, image, points, arrays, box) in enumerate(panels):
        for j, (title, mask) in enumerate(zip(['原图与参考点', '冻结模型', '拆分规则', '加纸色复核'], [None, *arrays])):
            view = image.copy()
            if mask is not None:
                view[mask] = (0.55 * image[mask] + 0.45 * np.array([39, 139, 111])).astype(np.uint8)
            crop = Image.fromarray(view).crop(box)
            scale = min((width - 18) / crop.width, (height - 36) / crop.height)
            crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.NEAREST)
            if mask is None:
                dots = ImageDraw.Draw(crop)
                for kind, coords in points.items():
                    for x, y in coords:
                        xx, yy = (x - box[0] + 0.5) * scale, (y - box[1] + 0.5) * scale
                        dots.ellipse((xx - 3, yy - 3, xx + 3, yy + 3), outline=colors[kind], width=2)
            draw.text((j * width + 7, i * height + 4), f'{label} · {title}', font=font, fill='#263c34')
            board.paste(crop, (j * width + 7, i * height + 30))
    draw.text((8, board.height - 30), '参考圈  绿＝墨迹  蓝＝纸面  紫＝待定    模型列绿色覆盖＝预测墨迹', font=font, fill='#263c34')
    board.save(ROOT / 'comparison.png')


def evaluate():
    assert not (ROOT / 'results.json').exists(), 'Retain original measurement; use a separate trial for changes.'
    frozen_path = ROOT / 'frozen.json'
    frozen = read(frozen_path)
    assert frozen['artifacts'] == lock_artifacts()
    import torch
    from cunfa_roles_v2_train_20261112 import Net, features
    from cunfa_band_checked_train_20261112 import Refiner

    torch.set_num_threads(4)
    teacher = Net(False).eval()
    teacher.load_state_dict(torch.load(MODELS / 'flat_roles.pt', map_location='cpu', weights_only=False)['model'])
    refiner = Refiner().eval()
    refiner.load_state_dict(torch.load(UPSTREAM / 'band_paper.pt', map_location='cpu', weights_only=False)['model'])
    result, panels = [], []
    views = [(270,240,553,450), (0,0,175,146), (120,125,218,207)]
    names = ['长披麻示例', '披麻练习', '斧劈练习']
    for example, label, box in zip(frozen['examples'], names, views):
        path = Path(example['image'])
        assert sha(path) == example['sha256']
        image = np.array(Image.open(path).convert('RGB'))
        with torch.inference_mode():
            x = torch.from_numpy(features(image))[None]
            probabilities, _ = refiner(x, teacher.probabilities(x))
            prediction = probabilities[0].argmax(0).numpy().astype(np.uint8)
        context, removed, withheld = separate_contextual_paper(image, prediction)
        simple = prediction.copy()
        simple[removed | withheld] = 0
        variants = {'frozen': prediction, 'seam': simple, 'context': context}
        row = {'id':example['id'], 'source_group':example['source_group'], 'scores':{}, 'points':[]}
        for tag, points in example['points'].items():
            for x, y in points:
                row['points'].append(dict(label=tag, xy=[x,y], ink={name:bool(pred[y,x]) for name,pred in variants.items()}))
        for name in variants:
            row['scores'][name] = {tag: {'n': len(points), 'ink': sum(bool(variants[name][y,x]) for x,y in points)}
                                   for tag, points in example['points'].items()}
            row['scores'][name]['removed_pixels'] = int(((prediction > 0) & (variants[name] == 0)).sum())
        result.append(row)
        panels.append((label, image, example['points'], [pred > 0 for pred in variants.values()], box))
        np.savez_compressed(ROOT / (example['id'] + '.npz'), prediction=prediction, seam=simple, context=context,
                            removed=removed, withheld=withheld)
    compare_board(panels)
    lock = read(LAB / 'local_rock_patch_20261117/baseline_lock.json')
    baseline = {key: sha(lock[key]) == lock[key+'_sha256'] for key in ('approved500','working489')}
    assert all(baseline.values())
    write(ROOT/'results.json',dict(examples=result, frozen_sha256=sha(frozen_path), baseline_hashes=baseline,
         new_model_trained=False, adopted=False, robot_connected=False))
    print(json.dumps({row['id']:row['scores'] for row in result}, ensure_ascii=False))


if __name__ == '__main__':
    {'freeze': freeze, 'evaluate': evaluate}[sys.argv[1]]()
