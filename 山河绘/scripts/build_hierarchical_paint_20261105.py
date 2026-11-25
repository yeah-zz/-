# -*- coding: utf-8 -*-
"""Build auditable block-and-layer plans from the latest conservative extraction."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,shutil,colorsys
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from hierarchical_paint_core_20261105 import object_ownership,recover_continuity,structural_roles,hierarchy_schedule,order_metrics
from module_stroke_core_20261104 import samples,length
from landscape_stroke_refine_20261104 import deduplicate

ROOT=LAB/'hierarchical_painting_20261105'
PRIOR=LAB/'context_curve_selection_20261105'
MASKS=LAB/'module_stroke_planner_20261104/outputs'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)

def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def read(p):return json.loads(p.read_text('utf8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def color(i):return tuple(round(x*190) for x in colorsys.hsv_to_rgb((i*.618)%1,.75,1))

def draw(curves,actions=None,count=None,colored=False):
    im=np.full((992,992,3),255,np.uint8)
    acts=actions if actions is not None else [dict(curve=i,module=r['module']) for i,r in enumerate(curves)]
    for a in acts if count is None else acts[:count]:
        row=curves[a['curve']];p=np.asarray(row['points'])
        cv2.polylines(im,[np.rint(p*2).astype(np.int32)],False,color(a['module']) if colored else (35,42,37),1,cv2.LINE_AA)
    return Image.fromarray(im)

def control_route(curves):
    left=set(range(len(curves)));cursor=np.array([0.,495.]);out=[]
    while left:
        def score(i):
            p=curves[i]['points'];d=min(np.linalg.norm(p[0]-cursor),np.linalg.norm(p[-1]-cursor))
            return length(p)/(1+d/20)
        i=max(left,key=score);left.remove(i);p=curves[i]['points'];rev=np.linalg.norm(p[-1]-cursor)<np.linalg.norm(p[0]-cursor)
        out.append(dict(curve=i,module=curves[i]['module'],air_travel_px=float(np.linalg.norm((p[-1] if rev else p[0])-cursor))))
        cursor=p[0] if rev else p[-1]
    return out

def build(ident):
    folder=ROOT/ident;folder.mkdir(parents=True,exist_ok=True)
    parent=PRIOR/ident/'candidate.json';d=read(parent);source=Path(d['source'])
    assert sha(source)==d['source_sha256']
    native=Image.open(source).convert('RGB');rgb=np.array(native.resize((496,496),Image.Resampling.LANCZOS))
    factors=np.array([496/native.width,496/native.height])
    # Known-error learned duplicate decisions from round1 are NOT deployed here.
    original=[np.asarray(r['points'],float)*factors for r in d['input_rows']]
    original += [np.asarray(p,float)*factors for p in d['added_native_paths']]
    maskroot=MASKS/ident;manifest=read(maskroot/'sam_manifest_seeded.json')
    masks_source=Path(manifest['source']);assert sha(masks_source)==manifest['source_sha256']
    cached=np.array(Image.open(masks_source).convert('RGB').resize((496,496),Image.Resampling.LANCZOS))
    alignment=float(np.abs(cached.astype(float)-rgb.astype(float)).mean())
    if alignment>1.5:raise ValueError(f'Mask source alignment mismatch {ident}: {alignment}')
    with np.load(maskroot/'sam_masks_seeded.npz') as masks:
        modules,ownership=object_ownership(rgb,original,masks,manifest['records'])
    dedup,dupd=deduplicate(rgb,original,[r['module'] for r in ownership])
    own=[dict(module=r['module'],uncertain=any(ownership[i]['uncertain'] for i in r['source_curve_ids'])) for r in dedup]
    joined,diag=recover_continuity(rgb,[r['points'] for r in dedup],own)
    for r in joined:r['source_curve_ids']=[j for i in r['source_curve_ids'] for j in dedup[i]['source_curve_ids']]
    structural_roles(rgb,joined,modules);route,module_order=hierarchy_schedule(joined,modules)
    baseline=control_route([dict(points=p,module=o['module']) for p,o in zip(original,ownership)])
    # Partition proves provenance; sampled coverage checks absorption/retention.
    assert sorted(i for r in joined for i in r['source_curve_ids'])==list(range(len(original)))
    tree=cKDTree(np.concatenate([samples(r['points'],.3) for r in joined]))
    distances=tree.query(np.concatenate([samples(p,.3) for p in original]))[0]
    assert distances.max()<1.5
    geom=lambda pp:dict(count=len(pp),short_under_5px=sum(length(p)<5 for p in pp),
        long_over_25px=sum(length(p)>25 for p in pp),median_length_px=float(np.median([length(p) for p in pp])))
    stats=dict(id=ident,before=geom(original),after=geom([r['points'] for r in joined]),
        duplicate_candidates_absorbed=len(dupd['events']),accepted_continuations=len(diag['accepted']),
        ambiguous_continuations=len(diag['pending']),baseline_order=order_metrics(baseline),hierarchy_order=order_metrics(route),
        modules=len(modules),object_proposals=sum(m['sam_key'] is not None for m in modules),
        unresolved_regions=sum(m['sam_key'] is None for m in modules),uncertain_owned_input=sum(o['uncertain'] for o in ownership),
        source_sample_coverage_within_1p5px=float((distances<=1.5).mean()),
        source_sample_max_distance_px=float(distances.max()),mask_source_mean_pixel_difference=alignment,
        artist_line_accuracy=None,rock_instance_accuracy=None,cun_style_accuracy=None,
        new_model_weights_trained=False,physical_quality_verified=False,ready_for_robot=False)
    assert stats['hierarchy_order']['module_reentries']==0
    descriptions=[];overlay=(rgb*.75+255*.25).astype(np.uint8)
    for m in modules:
        mid=m['id'];contours,_=cv2.findContours(m['mask'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=[cv2.approxPolyDP(c,1.5,True).reshape(-1,2).tolist() for c in contours if cv2.contourArea(c)>10]
        desc={k:v for k,v in m.items() if k!='mask'}
        desc.update(name=f'{"对象候选" if m["sam_key"] else "待定区"} {mid+1:02}',outlines=contour,
            planned_strokes=sum(r['module']==mid for r in joined),order=module_order.index(mid)+1)
        descriptions.append(desc)
        cv2.drawContours(overlay,contours,-1,color(mid),1)
        cv2.putText(overlay,str(desc['order']),tuple(np.rint(m['center']).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,color(mid),1,cv2.LINE_AA)
    plan=dict(schema='hierarchical-image-plan-v1',id=ident,width=496,height=496,source=str(source),source_sha256=sha(source),
        parent=str(parent),parent_sha256=sha(parent),mask_manifest_sha256=sha(maskroot/'sam_manifest_seeded.json'),
        modules=descriptions,module_order=module_order,curves=joined,actions=route,stats=stats,
        coordinate_system='496px image space; no robot coordinates',ready_for_robot=False,
        camera_connected=False,cun_styles_trained=False,structural_roles_verified=False,
        learned_relation_filter_applied=False,object_lock=True,phase_order=[0,1,2],
        limitations=['Object regions are SAM proposals, not verified stone instances',
                    'Interior long curves are candidates, not recognized cun styles',
                    'Coverage tests check retained inputs, not missing true lines',
                    'No physical ink model, brush calibration or robot execution'])
    save(folder/'plan.json',plan);save(folder/'statistics.json',stats);save(folder/'joining_diagnostics.json',diag)
    save(folder/'deduplication.json',dupd)
    Image.fromarray(rgb).save(folder/'source.png');Image.fromarray(overlay).save(folder/'modules.png')
    draw(joined).save(folder/'centerlines.png');draw(joined,colored=True).save(folder/'module_lines.png')
    before=[dict(points=p,module=o['module']) for p,o in zip(original,ownership)]
    board=Image.new('RGB',(1680,510),'#f4f2e9');pd=ImageDraw.Draw(board)
    panels=[(Image.fromarray(rgb),'原图'),(draw(before),'之前中心线'),(draw(joined),'连续笔画候选'),(draw(joined,colored=True),'按对象候选分块（颜色非皴法）')]
    for i,(im,label) in enumerate(panels):board.paste(im.resize((402,402),Image.Resampling.LANCZOS),(i*420+8,45));pd.text((i*420+8,10),label,font=FONT,fill='#263e34')
    pd.text((10,462),f'{ident}：{len(original)} → {len(joined)}笔；原输入采样覆盖保留，尚未验证真实主线完整度；仅离线规划。',font=FONT,fill='#263e34')
    board.save(folder/'comparison.jpg',quality=95)
    board=Image.new('RGB',(1260,480),'#f4f2e9');pd=ImageDraw.Draw(board)
    # Same curve set isolates ordering from extraction changes.
    global_control=control_route(joined);n=min(100,len(route))
    for i,(im,label) in enumerate([(Image.fromarray(overlay),'自动区域与顺序（实例待验证）'),
        (draw(joined,global_control,n,True),f'全局距离排序：前{n}笔'),(draw(joined,route,n,True),f'分块分层排序：前{n}笔')]):
        board.paste(im.resize((402,402),Image.Resampling.LANCZOS),(i*420+8,45));pd.text((i*420+8,10),label,font=FONT,fill='#263e34')
    board.save(folder/'sequence_comparison.jpg',quality=95)
    print('PLAN',json.dumps(stats,ensure_ascii=False),flush=True)
    return dict(id=ident,plan_sha256=sha(folder/'plan.json'),stats=stats)

def main():
    cv2.setNumThreads(2);ROOT.mkdir(exist_ok=True)
    records=[build(i) for i in ['G03_01','G05_03','T02']]
    save(ROOT/'catalog.json',dict(scenes=records,default='G03_01',robot_connected=False))
    binding=ROOT/'execution_binding.json'
    if not binding.exists():save(binding,dict(scene=None,plan_sha256=None,journal=None,note='No execution bound; playback is never hardware progress'))
    shutil.copy2(Path(__file__).with_name('hierarchical_paint_platform_20261105.html'),ROOT/'index.html')
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['hierarchical_paint_core_20261105.py','build_hierarchical_paint_20261105.py',
                 'hierarchical_paint_server_20261105.py','hierarchical_paint_platform_20261105.html']:
        p=Path(__file__).with_name(name)
        if p.exists():shutil.copy2(p,snap/name)
    print('PLATFORM',ROOT/'index.html',flush=True)

if __name__=='__main__':main()
