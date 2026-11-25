# -*- coding: utf-8 -*-
"""Compare the frozen direction rule on source-reviewed natural omissions."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from screen_faint_ink_natural_20261121 import LAB, MODULE, ROOT, read, sha, write_json
from gongbi_training.faint_ink_direction_filter import filter_faint_ink_direction


# One-based proposal indices refer to the source review crops. Groups selected
# from the old unfiltered candidates. Unlisted pixels in a group stay unknown.
REFERENCES = [
    ('rocks', 2, 'faint', [1,2], '沿岩面上连续变淡的灰色短线'),
    ('banks', 2, 'faint', [1,2,3], '岸边浅色斜线有前后连续的墨迹'),
    ('tree', 1, 'ink', list(range(7,16)), '沿明显的深色横向墨迹'),
    ('tree', 6, 'paper', list(range(1,9)), '浅纸面纹理，没有可辨连续墨线'),
    ('tree', 15, 'paper', list(range(1,7)), '枝条外侧的浅底色纹理'),
    ('tree', 16, 'paper', list(range(1,7)), '细枝右侧的背景纸纹'),
]
REVIEWED = {
    'rocks': [2],
    'far_rocks': [1,3,5],
    'banks': [2,5],
    'hills': [3,12,17],
    'tree': [1,2,3,6,11,13,15,16,17],
}


def references(source):
    labels = {}
    for name, group, label, indices, basis in REFERENCES:
        for index in indices:
            assert 0 < index <= len(source[name]['groups'][group-1]['xy'])
            labels[name,group,index] = (label,basis)
    output = []
    for name, groups in REVIEWED.items():
        for group in groups:
            for index, (x,y) in enumerate(source[name]['groups'][group-1]['xy'],1):
                label,basis = labels.get((name,group,index), ('unknown','原图淡纹归属无法稳定确认'))
                output.append(dict(image=name, group=group, point=index, xy=[x,y], label=label,basis=basis))
    return output


def draw_examples(images, examples):
    panels = [('rocks',2,'岩面短淡线'), ('banks',2,'岸边淡线'), ('tree',1,'横向墨迹'),
              ('tree',6,'浅纸纹'), ('tree',15,'枝条旁纸面'), ('tree',16,'纸面细纹')]
    board = Image.new('RGB',(1030,195*len(panels)+32),'#f5f1e7')
    pen = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for row,(name,group,title) in enumerate(panels):
        image, variants = images[name]
        items = [p for p in examples if p['image']==name and p['group']==group]
        xs,ys = zip(*(p['xy'] for p in items))
        x0,y0,x1,y1 = min(xs)-7,min(ys)-7,max(xs)+8,max(ys)+8
        scale = min(241//(x1-x0),159//(y1-y0))
        for col,(key,caption) in enumerate([(None,'原图'),('baseline','原提取'),('proposed','直接续接'),('direction','方向筛选')]):
            crop = image[y0:y1,x0:x1].copy()
            if key:
                roles = variants[key][y0:y1,x0:x1]
                old = variants['baseline'][y0:y1,x0:x1]>0
                add = (roles>0)&~old
                crop[old] = (.62*crop[old]+.38*np.array([34,137,111])).astype(np.uint8)
                crop[add] = (.35*crop[add]+.65*np.array([38,149,200])).astype(np.uint8)
            tile = Image.fromarray(crop).resize(((x1-x0)*scale,(y1-y0)*scale),Image.Resampling.NEAREST)
            d = ImageDraw.Draw(tile)
            for item in items:
                if item['label']=='unknown':
                    continue
                px,py = item['xy']
                x,y = (px-x0)*scale,(py-y0)*scale
                outline = '#f08721' if item['label'] in ('faint','ink') else '#d45085'
                d.rectangle((x,y,x+scale-1,y+scale-1),outline=outline,width=1)
            left,top=257*col+4,row*195+2
            pen.text((left,top),f'{title} · {caption}',font=font,fill='#263c34')
            board.paste(tile,(left,top+24))
    pen.text((8,board.height-27),'橙框为墨线参考，粉框为纸面参考。蓝色覆盖为补点，绿色覆盖为原墨迹预测。',font=font,fill='#263c34')
    board.save(ROOT/'comparison.png')


def main():
    source = read(ROOT/'screen.json')
    manifest = read(ROOT/'screen_protocol.json')
    assert manifest['direction_rule']==sha(MODULE)
    reviewed = references(source)
    protocol = ROOT/'review_protocol.json'
    locked = dict(screen_sha=sha(ROOT/'screen.json'), inputs_sha=sha(ROOT/'screen_protocol.json'),
                  arrays={name:sha(ROOT/f'{name}.npz') for name in manifest['images']},
                  references=reviewed, scope='Proposal-screened natural examples, reviewed before the direction output.')
    if protocol.exists():
        assert read(protocol)==locked
    else:
        write_json(protocol,locked)

    images, totals = {}, {}
    for name,meta in manifest['images'].items():
        assert meta['sha256']==sha(meta['path'])
        with Image.open(meta['path']) as opened:
            image = np.array(opened.convert('RGB'))
        with np.load(ROOT/f'{name}.npz') as data:
            baseline, proposed, additions = data['baseline'], data['proposed'], data['additions']
        direction, added, rejected = filter_faint_ink_direction(image,baseline,proposed)
        np.testing.assert_array_equal(added | rejected, additions)
        totals[name] = dict(previous=int(additions.sum()),direction=int(added.sum()))
        images[name] = image, dict(baseline=baseline,proposed=proposed,direction=direction)
        np.savez_compressed(ROOT/f'{name}_direction.npz',roles=direction,added=added)
    results = []
    for row in reviewed:
        x,y = row['xy']
        _, variants = images[row['image']]
        flags = {key:bool(roles[y,x]) for key,roles in variants.items()}
        assert not flags['baseline'] and flags['proposed']
        results.append(dict(row,ink=flags))
    metrics = {}
    for label in ('faint','ink','paper','unknown'):
        subset = [r for r in results if r['label']==label]
        metrics[label] = dict(total=len(subset), baseline=0, proposed=len(subset),
                              direction=sum(r['ink']['direction'] for r in subset))
    draw_examples(images,results)
    lock = read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:sha(lock[key])==lock[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    output = dict(points=results,metrics=metrics,additions=totals,baseline_hashes=hashes,
                  protocol_sha256=sha(protocol),independent_trial=True,independent_acceptance=False,
                  adopted=False,robot_connected=False)
    write_json(ROOT/'results.json',output)
    print(totals)
    print(metrics)
    for key,group in [(name,group) for name,group,_,_,_ in REFERENCES]:
        print(key,group,[(p['point'],p['label'],p['ink']['direction']) for p in results
                         if p['image']==key and p['group']==group])


if __name__ == '__main__':
    main()
