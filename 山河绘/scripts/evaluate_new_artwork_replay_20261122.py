"""Frozen extraction and source-reviewed offline playback on CMA 1986.49.10."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gongbi_training.runtime import LAB, ROOT as PROJECT, configure
from source_ink import extract


OUT = LAB / 'new_artwork_replay_20261122'
SOURCE = OUT / 'museum_original.jpg'
CHECKPOINT = LAB / 'guohua_scene4_mapping_fix_20261116/mapping_fixed.pt'
CATALOGUE = 'https://www.clevelandart.org/art/1986.49.10'
COLOR = [(75,139,165),(177,97,49),(56,118,82),(144,98,157),(102,117,38),(100,124,149)]
SCENE = np.array([[238,235,219],[184,127,75],[87,147,100],[129,105,167]],np.uint8)
SCENE_NAMES = ['空景','山石','树木','其他']

# Raster coordinates associated with the original image, defined before prediction.
CORE = [
    ('远山淡面',1,[363,84,382,105]),('中央岩面',1,[274,338,291,370]),
    ('松针核心',2,[88,424,98,452]),('左上树冠',2,[181,167,201,185]),
    ('右侧细枝',2,[429,272,439,291]),('上方空景',0,[184,61,268,129]),
]
TRACES = [
    ('远峰可见外缘',[[336,78],[340,66],[350,59],[364,51],[378,42],[391,36],[402,32]]),
    ('中央岩面凸缘',[[232,352],[243,344],[252,328],[264,320],[276,312],[287,309],[299,316]]),
    ('瀑布左侧岩线',[[326,280],[331,290],[332,312],[333,340],[330,361],[327,380]]),
]
MODULES = [
    ('右上远山','rock',[[306,154],[324,74],[352,44],[395,25],[434,57],[457,105],[453,202],[416,216],[374,216],[350,190]]),
    ('左上坡树群','tree',[[118,216],[118,191],[130,174],[145,177],[152,139],[172,130],[185,147],[204,151],[217,170],[220,187],[196,200],[171,198],[154,221]]),
    ('瀑布左侧主山石','rock',[[177,276],[198,211],[233,213],[261,224],[289,244],[308,238],[330,260],[343,342],[335,403],[360,445],[345,471],[297,484],[250,492],[226,469],[188,458],[176,404]]),
    ('右侧树群','tree',[[361,326],[378,302],[397,291],[414,307],[429,293],[420,272],[440,255],[456,255],[456,379],[437,397],[418,388],[417,350],[386,337]]),
    ('近景左侧松树群','tree',[[27,501],[27,367],[46,337],[59,338],[69,371],[75,404],[88,401],[99,417],[109,390],[125,379],[137,401],[131,451],[119,493],[103,526],[87,545],[64,534]]),
    ('前景溪边石组','rock',[[126,469],[173,468],[211,464],[236,458],[267,472],[297,474],[324,475],[358,463],[403,459],[448,475],[455,537],[406,545],[363,518],[310,515],[253,522],[193,516],[152,510],[120,516]]),
]
TRUNK = [
    (1,[[147,211],[151,194],[166,180],[176,170],[189,166]]),
    (3,[[433,357],[436,336],[433,309],[437,287],[444,270]]),
    (4,[[44,499],[49,472],[52,442],[55,414],[54,395],[53,366]]),
    (4,[[76,521],[81,498],[82,472],[86,441],[88,425]]),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def freeze():
    protocol = dict(accession='1986.49.10', author='翟大坤', catalogue=CATALOGUE,
                    image_sha256=sha(SOURCE), checkpoint_sha256=sha(CHECKPOINT),
                    extractor_sha256=sha(PROJECT/'source_ink.py'), cores=CORE, traces=TRACES, modules=MODULES,
                    trunk=TRUNK, scales=[256,512], threshold=.78, reference_tolerance_px=2.5,
                    reference_type='Assistant source review; coarse regions and visible segments, no expert labels.',
                    no_tuning_on_image=True, checked_source_accession_in_local_catalogues=True,
                    independent_author_corpus_audit=False)
    path = OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text('utf-8')) == json.loads(json.dumps(protocol))
    else:
        write('protocol.json',protocol)
    return path


def frozen_object_map(rgb):
    configure()
    import torch
    from guohua_objects_train_20261112 import pack
    from gongbi_training.scene_objects import SceneObjects

    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SceneObjects().to(device)
    net.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['model'])
    net.eval()
    predictions = []
    for size in (256,512):
        inputs,_,(x,y,w,h,ow,oh) = pack(rgb,size=size)
        with torch.inference_mode():
            p = net(torch.from_numpy(inputs)[None].to(device)).softmax(1)[0].cpu().numpy()
        predictions.append(np.stack([cv2.resize(row[y:y+h,x:x+w],(ow,oh)) for row in p]))
    return np.mean(predictions,axis=0).argmax(0)


def sample(points, count=None):
    p = np.asarray(points,dtype=float)
    arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    u = np.linspace(0,arc[-1],count or max(2,int(arc[-1])+1))
    return np.rint(np.stack([np.interp(u,arc,p[:,i]) for i in range(2)],axis=1)).astype(int)


def catchment(shape, paths, radius):
    canvas = np.zeros(shape,np.uint8)
    for path in paths:
        cv2.polylines(canvas,[np.rint(path).astype(np.int32)],False,1,2*radius+1)
    return canvas


def region_mask(shape, points):
    canvas = np.zeros(shape,np.uint8)
    cv2.fillPoly(canvas,[np.asarray(points,dtype=np.int32)],1)
    return canvas


def line_image(shape, rows, color=False):
    out = np.full((*shape,3),255,np.uint8)
    for item in rows:
        ink = COLOR[item['module']] if color else (31,42,37)
        cv2.polylines(out,[np.rint(item['points']).astype(np.int32)],False,ink,1,cv2.LINE_AA)
    return out


def review(rows, shape):
    areas = [region_mask(shape,polygon) for _,_,polygon in MODULES]
    blocked = region_mask(shape,[[37,12],[153,12],[154,118],[38,121]])
    blocked[:4,:] = blocked[-4:,:] = blocked[:,:4] = blocked[:,-4:] = 1
    structural = catchment(shape,[trace for _,trace in TRACES],5)
    trunk = {i:catchment(shape,[p for m,p in TRUNK if m==i],4) for i in (1,3,4)}
    kept, pending = [], []
    for row in rows:
        x,y = sample(row['points']).T
        if blocked[y,x].mean() > .15:
            pending.append(dict(id=row['id'],reason='题款或图框邻域'))
            continue
        votes = np.array([mask[y,x].mean() for mask in areas])
        module = int(votes.argmax())
        other = float(np.delete(votes,module).max())
        _,kind,_ = MODULES[module]
        if votes[module] < .78 or other > .18:
            pending.append(dict(id=row['id'],reason='对象范围外或跨界待核'))
            continue
        if kind == 'tree':
            woody = trunk[module][y,x].mean()
            phase = 0 if woody >= .68 and row['length_px'] >= 12 else 1 if woody >= .25 and row['length_px'] >= 10 else 2
        else:
            primary = structural[y,x].mean()
            phase = 0 if primary >= .45 and row['length_px'] >= 12 else 1
        kept.append(dict(row,module=module,phase=phase))
    return kept, pending


def ordered(rows):
    from hierarchical_paint_core_20261105 import order_metrics

    # Preserve the same block barrier and within-phase distance/length policy
    # as the archived object planner, with this source's reviewed object order.
    cursor = np.array([0.,599.])
    actions = []
    for mid, _ in enumerate(MODULES):
        for phase in range(3):
            remaining = [i for i,s in enumerate(rows) if (s['module'],s['phase'])==(mid,phase)]
            while remaining:
                def travel(i):
                    return np.linalg.norm(np.asarray(rows[i]['points'])[[0,-1]]-cursor,axis=1)
                i = max(remaining,key=lambda j:rows[j]['length_px']**.7/(1+float(travel(j).min())/50))
                remaining.remove(i)
                d = travel(i)
                reverse = bool(d[1]<d[0])
                cursor = np.asarray(rows[i]['points'][0 if reverse else -1])
                actions.append(dict(curve=i,module=mid,phase=phase,reverse=reverse,
                                    stroke_id=rows[i]['id'],air_travel_px=float(min(d)),length_px=rows[i]['length_px']))
    assert sorted(action['curve'] for action in actions)==list(range(len(rows)))
    metrics = order_metrics(actions)
    phase_violations = sum(a['module']==b['module'] and a['phase']>b['phase'] for a,b in zip(actions,actions[1:]))
    metrics['phase_inversions'] = phase_violations
    assert phase_violations==0 and metrics['module_reentries']==0
    return actions,metrics


def coverage(shape, traces, rows):
    mask = catchment(shape,[row['points'] for row in rows],0)
    dist = cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    results = []
    for name,trace in traces:
        x,y = sample(trace).T
        covered = dist[y,x] <= 2.5
        results.append(dict(name=name,covered=int(covered.sum()),points=len(covered),complete=bool(covered.all())))
    return results


def figures(rgb,pred,auto,kept):
    shape = pred.shape
    overlay = cv2.addWeighted(rgb,.52,SCENE[pred],.48,0)
    board = Image.new('RGB',(4*469+10,677),'#f5f1e6')
    pen = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    columns = [('原图',rgb),('冻结模型直接分区',overlay),('自动提取输出',line_image(shape,auto)),
               ('按原图复核后的分块候选',line_image(shape,kept,True))]
    for j,(title,im) in enumerate(columns):
        x = 8+j*469
        pen.text((x,8),title,fill='#21372b',font=font)
        board.paste(Image.fromarray(im),(x,35))
    pen.text((8,639),f'自动输出 {len(auto)} 笔。复核后 {len(kept)} 笔进入回放。末列颜色表示对象分块。',fill='#21372b',font=font)
    board.save(OUT/'whole_comparison.png')
    review = Image.fromarray(rgb.copy())
    pen = ImageDraw.Draw(review)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(_,_,polygon) in enumerate(MODULES):
        pen.line([tuple(p) for p in polygon+[polygon[0]]], fill=COLOR[i], width=2)
    for i,(name,cls,box) in enumerate(CORE,1):
        pen.rectangle(box,outline='#fde795',width=2)
        pen.text((box[0],box[1]-14),f'{i} {name}',font=small,fill='#fbf8e5',stroke_width=1,stroke_fill='#18271e')
    for _,trace in TRACES:
        pen.line([tuple(p) for p in trace],fill='#3ce1dc',width=2)
    review.save(OUT/'review_references.png')


def main():
    protocol = freeze()
    shape_expected = (600,462)
    rgb = np.array(Image.open(SOURCE).convert('RGB'))
    assert rgb.shape[:2]==shape_expected
    shape = rgb.shape[:2]
    pred = frozen_object_map(rgb)
    raw = extract(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    auto = [dict(id=f'p{i:04}',points=row['points'],length_px=row['length_px'],
                 closed=row['closed']) for i,row in enumerate(raw['paths'])]
    assert auto
    # Archive unedited predictions and stroke geometry before applying source review.
    np.savez_compressed(OUT/'frozen_inference.npz',object_map=pred,mask=raw['mask'],raster=raw['raster'])
    write('automatic_curves.json',dict(curves=auto,stats=raw['stats']))
    kept,pending = review(auto,shape)
    assert len(kept)+len(pending)==len(auto)
    actions,order = ordered(kept)
    assert len({a['stroke_id'] for a in actions})==len(kept)
    by_stroke={s['id']:s for s in auto}
    assert all(s['points']==by_stroke[s['id']]['points'] for s in kept)
    modules = []
    for i,(name,kind,_) in enumerate(MODULES):
        phases = [sum(row['module']==i and row['phase']==phase for row in kept) for phase in range(3)]
        modules.append(dict(name=name,kind=kind,id=i,count=sum(phases),phase_counts=phases))
    cores = []
    for name,cls,(x0,y0,x1,y1) in CORE:
        sub=pred[y0:y1,x0:x1]
        cores.append(dict(name=name,ref=SCENE_NAMES[cls],correct=float((sub==cls).mean()),
                          classes=dict(zip(SCENE_NAMES,np.bincount(sub.ravel(),minlength=4).tolist()))))
    baseline_lock=json.loads((LAB/'local_rock_patch_20261117/baseline_lock.json').read_text('utf8'))
    hashes={key:sha(Path(baseline_lock[key]))==baseline_lock[key+'_sha256'] for key in ('approved500','working489')}
    assert all(hashes.values())
    results = dict(source=CATALOGUE,protocol_sha256=sha(protocol),automatic_paths=len(auto),
                   compact_marks=len(raw['dots']),pending=len(pending),scheduled=len(kept),
                   pending_reasons=dict(Counter(p['reason'] for p in pending)),
                   source_cores=cores,automatic_contours=coverage(shape,TRACES,auto),
                   reviewed_contours=coverage(shape,TRACES,kept),modules=modules,order=order,
                   independent_acceptance=False,expert_annotation=False,
                   semantic_roles_experimental=True,adopted=False,robot_connected=False,baseline_hashes=hashes)
    plan=dict(source='museum_original.jpg',width=shape[1],height=shape[0],curves=kept,modules=modules,
              actions=actions,pending=pending,ready_for_robot=False)
    figures(rgb,pred,auto,kept)
    write('offline_plan.json',plan)
    write('results.json',results)
    template = Path(__file__).with_name('new_artwork_replay_viewer.html').read_text('utf8')
    embedded = json.dumps(plan,ensure_ascii=False).replace('<','\\u003c')
    assert template.count('__OFFLINE_PLAN_JSON__')==1
    (OUT/'index.html').write_text(template.replace('__OFFLINE_PLAN_JSON__',embedded),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
