# -*- coding: utf-8 -*-
"""Trace missed outlines through extraction stages and test tree-context tiles."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_new_artwork_replay_20261122 import (CATALOGUE, CHECKPOINT, CORE, SCENE, SOURCE, TRACES,
                                                   catchment, coverage, sample, sha)
from gongbi_training.runtime import LAB, ROOT, configure
from gongbi_training.new_artwork_detail_candidates import guarded_tree_hints, supplemental_ink
from source_ink import extract


OUT = LAB/'faint_outline_branch_probe_20261122'
CASES = [
    dict(id='zhai',file=SOURCE,url=CATALOGUE,traces=TRACES, cores=CORE,
         paper=[[184,61,268,129],[225,133,273,167]], fresh=False),
    dict(id='wang',file=OUT/'wang_gai_original.jpg',url='https://www.clevelandart.org/art/1975.64.2',
         traces=[('左岸山石外沿',[[18,189],[30,193],[43,200],[55,209],[66,221],[76,230],[90,238]]),
                 ('近景岩石上沿',[[399,515],[416,505],[432,497],[442,482],[460,472],[481,464]])],
         cores=[('右侧高树细枝',2,[513,88,521,121]),('右侧近景树干',2,[462,346,483,378]),
                ('前景岩面',1,[507,502,537,525]),('左岸岩面',1,[34,227,49,251]),('上方开阔区',0,[268,43,331,83])],
         paper=[[263,43,331,83],[112,351,169,391]],fresh=True)
]


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf8')


def regions(pred, candidate, accepted, case):
    rows = []
    for name,category,(x0,y0,x1,y1) in case['cores']:
        yx = np.s_[y0:y1,x0:x1]
        a = pred[yx]
        b = a.copy()
        b[accepted[yx]] = 2
        rows.append(dict(name=name,expected=int(category),pixels=int(a.size),
                         baseline_correct=int((a==category).sum()),accepted_correct=int((b==category).sum()),
                         baseline_tree=int((a==2).sum()),unfiltered_tree_change=int(candidate[yx].sum()),
                         accepted_tree_change=int(accepted[yx].sum())))
    return rows


def infer_field(net, image, device, sizes):
    import torch
    from guohua_objects_train_20261112 import pack

    fields = []
    for size in sizes:
        data,_,(x,y,w,h,ow,oh) = pack(image,size=size)
        with torch.inference_mode():
            values = net(torch.from_numpy(data)[None].to(device)).softmax(1)[0].cpu().numpy()
        fields.append(np.stack([cv2.resize(plane[y:y+h,x:x+w],(ow,oh)) for plane in values]))
    return np.mean(fields,axis=0)


def tile_field(net, image, device, width):
    height, total_width = image.shape[:2]
    ys = list(range(0,max(1,height-width),width//2))+[max(0,height-width)]
    xs = list(range(0,max(1,total_width-width),width//2))+[max(0,total_width-width)]
    weights = np.maximum(.15,np.hanning(width).astype('float32'))
    kernel = weights[:,None]*weights[None,:]
    total = np.zeros((4,height,total_width),np.float32)
    counts = np.zeros((height,total_width),np.float32)
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            crop = image[y:y+width,x:x+width]
            field = infer_field(net,crop,device,(256,))
            use = kernel[:crop.shape[0],:crop.shape[1]]
            total[:,y:y+width,x:x+width] += field * use
            counts[y:y+width,x:x+width] += use
    assert (counts>0).all()
    return total/counts


def trace_stages(extracted, traces):
    result = {}
    for key in ('weak_mask','mask','skeleton','raster'):
        map_dist = cv2.distanceTransform((extracted[key]==0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        result[key] = []
        for name, trace in traces:
            x,y = sample(trace).T
            hits = map_dist[y,x] <= 2.5
            result[key].append(dict(name=name,covered=int(hits.sum()),points=len(hits)))
    return result


def fit_figure(figure):
    im = Image.fromarray(figure)
    im.thumbnail((375,420),Image.Resampling.LANCZOS)
    return im


def make_figure(rgb, pred, candidate, accepted, baseline, added, case):
    h,w = rgb.shape[:2]
    overlay = cv2.addWeighted(rgb,.55,SCENE[pred],.45,0)
    # Show unfiltered tree votes separately even when the confidence gate abstains.
    marks = overlay.copy()
    marks[candidate] = [58,190,211]
    marks[accepted] = [27,107,190]
    raw = np.full_like(rgb,255)
    raw[baseline['raster']>0] = [32,49,40]
    merged = raw.copy()
    added_map = catchment((h,w),[r['points'] for r in added],0)>0
    merged[added_map&(baseline['raster']==0)] = [46,156,165]
    ref = Image.fromarray(rgb.copy())
    pen = ImageDraw.Draw(ref)
    for _,trace in case['traces']:
        pen.line([tuple(point) for point in trace],fill='#63dcda',width=2)
    for _,_,box in case['cores']:
        pen.rectangle(box,outline='#f3c562',width=2)
    columns = [('原图检查位置',np.asarray(ref)),('现有笔画',raw),('新增淡墨候选',merged),
               ('原分区与局部树木证据',marks)]
    board = Image.new('RGB',(1555,481),'#f5f1e7')
    d = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for i,(title,im) in enumerate(columns):
        x = 9+i*388
        d.text((x,5),title,font=font,fill='#233a30')
        board.paste(fit_figure(im),(x,30))
    d.text((9,453),'青色笔画是补提候选。分区上的浅青表示局部双尺度支持为树，深蓝表示通过置信门槛。',font=font,fill='#233a30')
    board.save(OUT/f"{case['id']}_comparison.png")


def main():
    import_path = Path(__file__)
    details = ROOT/'gongbi_training/new_artwork_detail_candidates.py'
    lock = dict(cases=[dict(c,path_sha256=sha(c['file'])) for c in CASES], checkpoint_sha256=sha(CHECKPOINT),
                rule_sha256=sha(details), runner_sha256=sha(import_path), strong=16, weak=9,
                scope='First artwork development. Second artwork frozen local review, visual references before predictions.')
    protocol=OUT/'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text('utf8'))==json.loads(json.dumps(lock,default=str))
    else:
        save_json(protocol,lock)

    configure()
    import torch
    from gongbi_training.scene_objects import SceneObjects

    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=SceneObjects().to(device)
    net.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['model'])
    net.eval()
    rows=[]
    for case in CASES:
        rgb=np.array(Image.open(case['file']).convert('RGB'))
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        baseline=extract(bgr)
        weaker, added=supplemental_ink(bgr,baseline)
        shape=bgr.shape[:2]
        full=infer_field(net,rgb,device,(256,512))
        left=tile_field(net,rgb,device,128)
        right=tile_field(net,rgb,device,224)
        pred=full.argmax(0)
        candidate,accepted=guarded_tree_hints(full,left,right)
        addition=catchment(shape,[r['points'] for r in added],0)>0
        addition &= baseline['raster']==0
        paper=[int(addition[y0:y1,x0:x1].sum()) for x0,y0,x1,y1 in case['paper']]
        line=dict(stages=trace_stages(baseline,case['traces']),
                  baseline_paths=len(baseline['paths']),weaker_paths=len(weaker['paths']),
                  added_paths=len(added), added_ink_pixels=int(addition.sum()),paper_pixels=paper,
                  baseline=coverage(shape,case['traces'],baseline['paths']),
                  weaker=coverage(shape,case['traces'],weaker['paths']),
                  guarded=coverage(shape,case['traces'],baseline['paths']+added))
        objects=dict(regions=regions(pred,candidate,accepted,case),candidate_pixels=int(candidate.sum()),
                     accepted_pixels=int(accepted.sum()))
        np.savez_compressed(OUT/f"{case['id']}_candidate_maps.npz",baseline=pred,
                            tree_local=candidate,tree_accepted=accepted,added_ink=addition)
        rows.append(dict(id=case['id'],url=case['url'],fresh=case['fresh'],line=line,objects=objects))
        make_figure(rgb,pred,candidate,accepted,baseline,added,case)
        print(case['id'],json.dumps(rows[-1],ensure_ascii=False),flush=True)
    baseline_lock=json.loads((LAB/'local_rock_patch_20261117/baseline_lock.json').read_text('utf8'))
    hashes={key:sha(Path(baseline_lock[key]))==baseline_lock[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    save_json(OUT/'results.json',dict(cases=rows,baseline_hashes=hashes,protocol_sha256=sha(protocol),
              expert_annotation=False,independent_acceptance=False,adopted=False,robot_connected=False))


if __name__=='__main__':
    main()
