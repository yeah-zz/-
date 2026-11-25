# -*- coding: utf-8 -*-
"""Compare a narrow profile with and without generating additional candidates."""
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_faint_ink_color_texture_20261121 import LAB, MODULE as COLOR, ROOT as COLOR_RUN, evaluate, inputs, segment
from gongbi_training.faint_ink_narrow import bridge_faint_ink_narrow


ROOT = LAB/'faint_ink_narrow_profile_20261121'
MODULE = Path(__file__).resolve().parents[1]/'gongbi_training/faint_ink_narrow.py'
PADDING = 24


def apply_profile(rgb, original, direction, current, allow_new, *, bridge=bridge_faint_ink_narrow):
    """Tile large source images with the same halo as the component evaluation."""
    height,width = original.shape
    output = current.copy()
    for top in range(0,height,512):
        for left in range(0,width,512):
            bottom,right = min(top+512,height),min(left+512,width)
            x,y = max(0,left-PADDING),max(0,top-PADDING)
            context = np.s_[y:bottom+PADDING,x:right+PADDING]
            local = np.s_[top-y:bottom-y,left-x:right-x]
            proposals = direction[context]
            roles = original[context]
            if not ((proposals>0) & (roles==0)).any():
                continue
            patch = bridge(rgb[context],roles,proposals,allow_new=allow_new)[local]
            window = output[top:bottom,left:right]
            fill = (window==0) & (patch>0)
            window[fill] = patch[fill]
    return output


def main():
    ROOT.mkdir(exist_ok=True)
    color_protocol = segment.read(COLOR_RUN/'protocol.json')
    assert color_protocol['module']==segment.sha(COLOR)
    sources,labels = inputs()
    selected = {row['image'] for row in labels}
    sources = {name:meta for name,meta in sources.items() if name in selected}
    lock = ROOT/'protocol.json'
    segment.lock_file(lock,dict(module=segment.sha(MODULE),color=segment.sha(COLOR_RUN/'protocol.json'),
                      inputs={name:[segment.sha(meta[0]),segment.sha(meta[1])] for name,meta in sources.items()},
                      labels=labels,scope='Development comparison on known source references.'))
    totals,scores,images,panels,points = {},{},{},[],[]
    added_positions = {}
    shown = ('暖色画面淡横线','横线外的浅色底材','青绿山面斜线','圆石弧线','岩层横线','上一轮剩余印刷网纹')
    for name, (source,mask,stages) in sources.items():
        saved = evaluate(source,mask,stages)
        with Image.open(source) as image, np.load(mask) as data:
            rgb = np.array(image.convert('RGB'))
            basis,direction = data['baseline'],data['direction']
        color = saved['adaptive']
        filter_only = apply_profile(rgb,basis,direction,color,False)
        generate = apply_profile(rgb,basis,direction,color,True)
        variants = dict(baseline=basis,color=color,filter_only=filter_only,generate=generate)
        totals[name] = {k:int(((m>0)&(color==0)).sum()) for k,m in variants.items()}
        totals[name]['generation_outside_direction'] = int(((generate>0)&(color==0)&(direction==0)).sum())
        added_positions[name] = np.argwhere((generate>0)&(color==0))[:,::-1].tolist()
        np.testing.assert_array_equal(generate[color>0], color[color>0])
        for row in (row for row in labels if row['image']==name):
            label,trace,title = (row[k] for k in ('label','trace','title'))
            scores[title] = dict(label=label,n=len(trace),hits={key:sum(bool(m[y,x]) for x,y in trace)
                                                               for key,m in variants.items()})
            if label=='faint':
                scores[title]['segment'] = {key:segment.score_segment(m,trace) for key,m in variants.items()}
            if title in shown:
                images[name] = rgb,variants
                group = len(panels)+1
                panels.append((name,group,title))
                mark = 'paper' if label=='paper_review' else label
                points.extend(dict(image=name,group=group,xy=xy,label=mark) for xy in trace)
        if name=='color_holdout/wang':
            for x in range(96,111):
                print('warm',x,{key:int(m[87,x]) for key,m in dict(direction=direction,**variants).items()},flush=True)
            np.savez_compressed(ROOT/'warm_line.npz',**variants,direction=direction)
        print(name,totals[name],flush=True)
    panels.sort(key=lambda row:shown.index(row[2]))
    columns = ((None,'原图'),('color','上轮设色方案'),('filter_only','细截面复核'),('generate','重找细线候选'))
    segment.render(images,points,ROOT/'comparison.png',sections=panels,columns=columns)
    locks = segment.read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    hashes = {key:segment.sha(locks[key])==locks[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    segment.write_json(ROOT/'results.json',dict(scores=scores,totals=totals,added_positions=added_positions,
                       baseline_hashes=hashes,
                       protocol_sha256=segment.sha(lock),independent_acceptance=False,
                       adopted=False,robot_connected=False))
    for title,row in scores.items():
        if row['label'] in ('faint','paper_review'):
            print(title,row,flush=True)
    print('paper',[(k,v) for k,v in scores.items() if v['label']=='paper' and v['hits']['generate']>0],flush=True)


if __name__=='__main__':
    main()
