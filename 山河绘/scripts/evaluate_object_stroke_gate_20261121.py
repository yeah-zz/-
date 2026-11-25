# -*- coding: utf-8 -*-
"""Review original ink across a complete object and check frozen gap repair."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_whole_scene_repair_20261121 import ARCHIVE, ROOT as WHOLE
from evaluate_ink_paper_width_20261119 import predict_extra
from validate_faint_ink_narrow_support_holdout_20261121 import run_variants, segment, support_lock


LAB = segment.LAB
ROOT = LAB/'object_stroke_gate_20261121'
GUIDES = {
    'cliff_edge': ('中央山石外轮廓','contour', [[301,310],[304,311],[307,314],[308,318],[309,323],[311,328]]),
    'rock_cun': ('中央山石坡面皴线候选','cun', [[224,320],[229,324],[233,328],[236,333],[238,340],[240,346]]),
    'rock_ridge': ('山面内部转折线','structure', [[267,308],[270,313],[273,320],[277,326],[281,334]]),
    'pine_left': ('松树左侧枝干','branch', [[225,155],[234,154],[241,154],[245,152],[251,153],[258,150],
                                                        [265,150],[271,153],[278,153],[283,155]]),
    'pine_right': ('松树右侧斜枝','branch', [[295,138],[303,139],[312,141],[321,141],[327,146],
                                                        [333,153],[338,159],[346,161],[354,164],[362,170]]),
    'rock_open': ('山石外的邻近留白','paper', [[318,311],[319,315],[320,320]]),
    'pine_open': ('松枝下的留白','paper', [[350,140],[358,142],[367,143]]),
    'rock_color': ('相邻山面空隙','paper', [[251,329],[253,334],[256,337]]),
}
LEGACY = ('A_contour_02','A_cun_02','B_cun_06')
PALETTE = dict(contour='#ec8526',cun='#14a6cb',structure='#e6ab39',branch='#dc8327',paper='#cc4678',
               unknown='#b8a22d')
TILE_BOXES = {
    'g_scene': [(135,282,328,385),(218,128,386,187)],
    'reference_rock': [(0,10,184,155)],
}


def center_trace(points):
    line = np.asarray(points,dtype=float)
    xy = [tuple(np.rint(line[0]).astype(int))]
    for a,b in zip(line,line[1:]):
        count = max(1,int(np.ceil(np.abs(b-a).max())))
        xy.extend(tuple(np.rint(a+(b-a)*step/count).astype(int)) for step in range(1,count+1))
    return [[int(x),int(y)] for x,y in dict.fromkeys(xy)]


def references():
    rows = [dict(id=k,title=title,label=label,image='g_scene',trace=center_trace(points),
                 origin='assistant source-only visual trace') for k,(title,label,points) in GUIDES.items()]
    legacy = segment.read(LAB/'rock_skill_offline_20261110/visible_reference.json')
    for rec in legacy['records']:
        if rec['id'] not in LEGACY:
            continue
        curve = rec['curve'][5:-5]
        rows.append(dict(id=rec['id'],title={'A_contour_02':'示范山石右侧外缘',
                         'A_cun_02':'示范山石细皴线甲','B_cun_06':'示范山石细皴线乙'}[rec['id']],
                         label=rec['role'],image='reference_rock',trace=center_trace(curve),
                         origin='archived visible source trace, credible stroke candidate'))
    return rows


def sources():
    return {'g_scene': ARCHIVE/'G03_01/source.png',
            'reference_rock': LAB/'whole_rock_evidence_20261117/views/rock_native.png'}


def review_board(rows):
    ROOT.mkdir(exist_ok=True)
    board = Image.new('RGB',(960,760),'#f5f1e7')
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    pen = ImageDraw.Draw(board)
    tiles = [('g_scene',TILE_BOXES['g_scene'][0],'中央主山石'),
             ('g_scene',TILE_BOXES['g_scene'][1],'松枝'),('reference_rock',TILE_BOXES['reference_rock'][0],'示范山石')]
    for i,(name,box,title) in enumerate(tiles):
        left,top = (i%2)*480+8,(i//2)*380+8
        with Image.open(sources()[name]) as image:
            crop = image.crop(box).convert('RGB')
        scale = min(462/crop.width,330/crop.height)
        tile = crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
        overlay = ImageDraw.Draw(tile)
        legend = []
        for index,row in enumerate(rows,1):
            if row['image']!=name:
                continue
            coords = [(round((x-box[0])*scale),round((y-box[1])*scale)) for x,y in row['trace']]
            if len(coords)>1:
                overlay.line(coords,fill=PALETTE[row['label']],width=2)
            if coords and all(0<=xy<limit for xy,limit in zip(coords[0],tile.size)):
                overlay.text((min(coords[0][0]+3,tile.width-20),max(1,coords[0][1]-17)),str(index),font=font,
                             fill=PALETTE[row['label']],stroke_width=1,stroke_fill='white')
                legend.append((index,row))
        pen.text((left,top),title,font=font,fill='#263c34')
        board.paste(tile,(left,top+26))
        for j,(index,row) in enumerate(legend):
            legend_y = top+tile.height+34+(j//2)*25
            if legend_y+20 >= (i//2+1)*380:
                break
            pen.text((left+(j%2)*228,legend_y),f"{index} {row['title']}",font=font,fill=PALETTE[row['label']])
    board.save(ROOT/'source_guides.png')


def longest_gap(hits):
    longest, run = 0,0
    for hit in hits:
        run = 0 if hit else run+1
        longest = max(longest,run)
    return longest


def trace_evidence(trace, mask, radius=1):
    xy = np.asarray(trace,dtype=np.int32)
    x0,y0 = np.maximum(xy.min(axis=0)-radius-1,0)
    x1,y1 = np.minimum(xy.max(axis=0)+radius+2,mask.shape[::-1])
    points = xy-[x0,y0]
    band = np.zeros((y1-y0,x1-x0),np.uint8)
    cv2.polylines(band,[points.astype(np.int32)],False,1,1)
    tolerance = np.ones((2*radius+1,2*radius+1),np.uint8)
    corridor = cv2.dilate(band,tolerance).astype(bool)
    local = mask[y0:y1,x0:x1]
    pixels = (local>0)&corridor
    near_ink = cv2.dilate(pixels.astype(np.uint8),tolerance).astype(bool)
    hits = near_ink[points[:,1],points[:,0]]
    _,connected = cv2.connectedComponents(pixels.astype(np.uint8),connectivity=8)
    def neighbors_at(point):
        x,y = point
        return set(connected[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1].ravel()) - {0}
    link = bool(neighbors_at(points[0]) & neighbors_at(points[-1]))
    return dict(connected=link,trace_points=len(xy),covered_near=int(hits.sum()),
                max_run_missing=longest_gap(hits),pixels_in_corridor=int(pixels.sum()))


def audit():
    rows = references()
    frozen = support_lock()
    inputs = {name:dict(path=str(path),sha256=segment.sha(path)) for name,path in sources().items()}
    bench = segment.read(WHOLE/'results.json')
    assert segment.sha(WHOLE/'protocol.json')==bench['protocol_sha256']
    for key,old_path in (('g_scene',WHOLE/'G03_01.npz'),):
        inputs[key]['frozen_array_sha256'] = segment.sha(old_path)
    ref_lock = ROOT/'references.json'
    segment.lock_file(ref_lock,dict(algorithms=frozen,inputs=inputs,rows=rows,tolerance_px=1,
                      scope='Source-checked object paths. Stroke identity and style remain unverified.'))
    corrections = {'rock_color':dict(label='unknown',title='山面淡色与墨边交界',
                  reason='原图放大复核发现参考线部分贴近可见墨边，撤出留白负例。')}
    segment.lock_file(ROOT/'reference_corrections.json',
                      dict(original_reference_sha256=segment.sha(ref_lock),corrections=corrections))
    rows = [dict(row,**corrections.get(row['id'],{})) for row in rows]
    review_board(rows)
    with Image.open(sources()['g_scene']) as scene, np.load(WHOLE/'G03_01.npz') as masks:
        rgb = np.array(scene.convert('RGB'))
        scene_variants = {key:masks[key] for key in ('baseline','color','span5')}
        scene_variants['plan'] = masks['plan_coverage'].astype(np.uint8)
    options = segment.lock_options()
    image,roles = predict_extra(dict(images={'reference_rock':inputs['reference_rock']}))['reference_rock']
    baseline,direction,previous = segment.old_stages(image,roles,options)
    _,rock_variants = run_variants(Image.fromarray(image),baseline,direction,previous)
    images = {'g_scene':(rgb,scene_variants),
              'reference_rock':(image,{key:rock_variants[key] for key in ('baseline','color','span5')})}
    scores,panels,points,plan_sensitivity = {},[],[],{}
    for index,row in enumerate(rows,1):
        _,variants = images[row['image']]
        trace = row['trace']
        label = row['label']
        scores[row['id']] = dict(label=label,title=row['title'],image=row['image'],scope=row['origin'],
                                 variants={k:trace_evidence(trace,v) for k,v in variants.items()})
        if row['image']=='g_scene' and label not in ('paper','unknown'):
            plan_sensitivity[row['id']] = trace_evidence(trace,variants['plan'],radius=3)
        panels.append((row['image'],index,row['title']))
        style = label if label in ('paper','unknown') else 'ink'
        points.extend(dict(image=row['image'],group=index,xy=xy,label=style) for xy in trace)
    columns = ((None,'原图'),('baseline','冻结提取'),('color','设色方案'),('span5','5像素跨度'))
    segment.render(images,points,ROOT/'stroke_comparison.png',sections=panels,columns=columns)
    scene_panels = [panel for panel in panels if panel[0]=='g_scene']
    plan_columns = ((None,'原图'),('baseline','冻结提取'),('span5','5像素跨度'),('plan','489笔轨迹'))
    segment.render(images,points,ROOT/'plan_comparison.png',sections=scene_panels,columns=plan_columns,
                   standalone=('plan',))
    out = dict(scores=scores,plan_sensitivity_radius3=plan_sensitivity,
               corrections_sha256=segment.sha(ROOT/'reference_corrections.json'),
               reference_sha256=segment.sha(ref_lock),baseline_hashes=bench['baseline_hashes'],
               independent_acceptance=False,robot_connected=False,adopted=False)
    segment.write_json(ROOT/'results.json',out)
    for key,row in scores.items():
        print(key,{name:(v['covered_near'],v['max_run_missing'],v['connected'],v['pixels_in_corridor'])
                   for name,v in row['variants'].items()})


if __name__=='__main__':
    audit()
