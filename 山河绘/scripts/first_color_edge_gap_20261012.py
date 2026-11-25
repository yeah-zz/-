"""Replan painting 01 fills with a one-millimetre estimated edge gap."""
import copy
import csv
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from web.catalog import ARTWORKS
from scripts.consolidate_selected_artworks import digest, stroke_data, index_for
from ink_control import refresh_stats
from shuimo import press_to_width_mm

spec = importlib.util.spec_from_file_location('fill_helpers', ROOT/'artifacts/五幅局部修订_20261010/revise.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
TARGET = ROOT / ARTWORKS[0][3]
TASK = TARGET.parent
PACKAGE = TASK.parent
OUT = ROOT / 'artifacts/第一幅色带净间隙1mm_20261012'
GAP = 1.0


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def encode(v):
    return json.dumps(v, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def edges(stroke):
    p = np.asarray(stroke['points'])
    return [(a,b,press_to_width_mm(max(stroke['press']))) for a,b in zip(p[:-1],p[1:]) if np.linalg.norm(a-b)>1e-8]


def point_dist(points, a, b):
    """Distances in image pixels, each point against each retained segment."""
    delta = b-a
    t = np.clip(np.sum((points[:,None,:]-a[None,:,:])*delta, axis=2) /
                np.maximum(np.sum(delta*delta, axis=1), 1e-12), 0, 1)
    return np.linalg.norm(points[:,None,:]-(a[None,:,:]+t[:,:,None]*delta),axis=2)


def clearance(q, retained, width, scale):
    if not retained:
        return np.ones(len(q),dtype=bool)
    a,b,widths = np.array([e[0] for e in retained]),np.array([e[1] for e in retained]),np.array([e[2] for e in retained])
    required = ((width+widths)/2+GAP)/scale
    return np.all(point_dist(q,a,b)>=required[None,:]-1e-7,axis=1)


def segment_gap(p, retained, width, scale):
    if not retained:
        return float('inf')
    a,b,widths = np.array([e[0] for e in retained]),np.array([e[1] for e in retained]),np.array([e[2] for e in retained])
    # Endpoint projection gives exact line-segment distance unless they cross.
    dist = np.minimum(point_dist(p,a,b).min(0), point_dist(np.vstack((a,b)),p[:1],p[1:]).reshape(2,-1).min(0))
    def cross(x,y):return x[...,0]*y[...,1]-x[...,1]*y[...,0]
    v=p[1]-p[0];w=b-a
    intersect=(cross(v,a-p[0])*cross(v,b-p[0])<0)&(cross(w,p[0]-a)*cross(w,p[1]-a)<0)
    dist[intersect]=0
    return float(np.min(dist*scale-(width+widths)/2))


def main():
    baseline=TARGET.read_bytes();old=read(TARGET);bundle=copy.deepcopy(old)
    module=bundle['workflow']['modules']['color'];original=module['strokes'];scale=module['mm_per_px']
    assert set(s['press'][0] for s in original)=={.85}
    assert not bundle['workflow'].get('edge_gap_revision'), 'Already replanned'
    width=press_to_width_mm(.85);pitch=width+GAP
    eligible=lambda s:s.get('nominal_pitch_mm') is not None
    fixed=[s for s in original if not eligible(s)]
    retained=[e for s in fixed for e in edges(s)]
    groups=list(dict.fromkeys((s['color'],s['drawing_region']) for s in original if eligible(s)))
    w,height=bundle['ink_job']['meta']['image_size'];shape=(height,w)
    replacements={};removed=set();region_report=[];min_gap=float('inf')
    for gi,(color,region) in enumerate(groups):
        source=[s for s in original if eligible(s) and (s['color'],s['drawing_region'])==(color,region)]
        template=source[0];horizontal=template['path_style']=='left-to-right-lift-return'
        area=h.footprint(source,shape,scale)
        scan=area.T.copy() if horizontal else area
        safe=(cv2.distanceTransform(scan,cv2.DIST_L2,5)>=width/(2*scale)).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(safe)
        paths=[]
        for component in range(1,count):
            mask=(labels==component).astype(np.uint8);x0,y0,ww,hh,_=stats[component]
            for column in np.arange(x0,x0+ww,pitch/scale):
                q=np.column_stack((np.full(int(math.ceil(hh/.2)),column),np.arange(y0,y0+hh,.2)))
                for part in h.segments(q,mask):
                    if horizontal:part=part[:,::-1].copy()
                    good=clearance(part,retained,width,scale)
                    changes=np.diff(np.r_[False,good,False].astype(int))
                    for a,z in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
                        p=part[a:z]
                        if len(p)<2:continue
                        # Check continuous geometry, not just sampled points.
                        while len(p)>2 and segment_gap(p[[0,-1]],retained,width,scale)<GAP-1e-6:
                            p=p[1:-1]
                        gap=segment_gap(p[[0,-1]],retained,width,scale)
                        if gap<GAP-1e-6 or np.linalg.norm(p[-1]-p[0])*scale<.20:continue
                        if np.linalg.norm(p[-1]-p[0])*scale>module['long_stroke_reload']['max_length_mm']:raise ValueError('Fill segment too long')
                        points=p[[0,-1]].tolist();s=copy.deepcopy(template)
                        s.update(id=f'edgegap1-{gi}-{len(paths):04d}',points=points,
                                 press=[.85,.85],speed=[template['speed'][0]]*2,
                                 pressure_mm=.85,target_width_mm=[width]*2,
                                 nominal_pitch_mm=pitch,target_edge_gap_mm=GAP,
                                 parent_stroke_id=template['id'],gap_design_component=component,
                                 joined_passes=1)
                        refresh_stats(s,scale);paths.append(s);retained.extend(edges(s));min_gap=min(min_gap,gap)
        region_report.append(dict(color=color,region=region,before=len(source),after=len(paths)))
        replacements[source[0]['id']]=paths;removed.update(s['id'] for s in source)
    result=[]
    for s in original:
        if s['id'] in replacements:result.extend(replacements[s['id']])
        if s['id'] not in removed:result.append(copy.deepcopy(s))
    assert result and len({s['id'] for s in result})==len(result)
    assert [s for s in result if not eligible(s)]==fixed
    assert bundle['ink_job']==old['ink_job']
    assert min_gap>=GAP-1e-6
    module.update(strokes=result,stroke_count=len(result))
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    revision=dict(date=stamp,source_sha256=hashlib.sha256(baseline).hexdigest(),
                  previous_center_pitch_mm=.9,target_edge_gap_mm=GAP,
                  modeled_brush_width_mm=width,nominal_center_pitch_mm=pitch,
                  pressure_mm=.85,measured_on_paper=False,regions=region_report,
                  minimum_model_edge_gap_mm=min_gap,
                  note='按色带边缘留1mm空隙重排成片上色。0.85mm下压对应现有模型笔宽2.89mm，主中心线间距3.89mm；实际笔宽和洇色仍需实测。水纹等独立非铺色笔触保留。')
    plan=bundle['workflow'];plan['edge_gap_revision']=revision
    plan['color_geometry'].update(pitch_mm=pitch,pitch_px=pitch/scale,target_edge_gap_mm=GAP,modeled_brush_width_mm=width,pressure_range_mm=[.85,.85])
    hashed=digest(stroke_data(bundle))
    for key in ('source_revision','web_installation'):
        plan[key]['stroke_data_sha256']=hashed
    OUT.mkdir(parents=True,exist_ok=True)
    candidate=TASK/'painting.gap-candidate.json';candidate.write_bytes(encode(bundle))
    try:
        from web.site_settings import prepared_artwork
        from run_json_painting import manifest_for
        from run_color_experiment import module_lines
        from color_profile import ColorProfile
        with patch.object(socket.socket,'connect',side_effect=AssertionError('Offline only')):
            prepared=prepared_artwork(candidate,ROOT/'web/data/site_settings.json')
            manifest_for(prepared[2],prepared[1],prepared[3])
            m=prepared[1]['modules']['color'];profile=ColorProfile(**m['motion_candidate']).to_dict()
            for color in ('blue','green'):
                lines,skipped=module_lines(m,color,profile);assert not skipped and lines
        order=index_for(bundle);counts={k:len(v) for k,v in order.items()}
        report=dict(offline_check='passed',hardware_started=False,active_run_unchanged=True,
                    effective_from='next-start',counts=counts,revision=revision,stroke_data_sha256=hashed)
        writes={TARGET:encode(bundle),TASK/'色带1mm净间隙校验.json':encode(report)}
        for p in TASK.glob('*_计划.json'):writes[p]=encode(plan)
        p=TASK/'绘画顺序索引.json';v=read(p);v.update(counts=counts,stages=order,stroke_data_sha256=hashed);writes[p]=encode(v)
        p=TASK/'Web当前应用记录.json';v=read(p);v.update(counts=counts,stroke_data_sha256=hashed,edge_gap_revision=revision);writes[p]=encode(v)
        table=io.StringIO(newline='');writer=csv.writer(table);writer.writerow(['阶段','阶段内笔序','笔触ID','完整部位','类别','原笔触ID','长度mm'])
        for stage,rows in order.items():
            for row in rows:writer.writerow([stage,row['order'],row['id'],row['region'],row['role'],row['parent_stroke_id'],row['length_mm']])
        writes[TASK/'逐笔顺序索引.csv']=table.getvalue().encode('utf-8-sig')
        catalog=ROOT/'待画作品/画稿清单.json';v=read(catalog)
        row=next(row for row in v['artworks'] if row['painting_json']==ARTWORKS[0][3]);row.update(counts=counts,stroke_data_sha256=hashed,edge_gap_revision=revision)
        writes[catalog]=encode(v)
        preview=PACKAGE/'预览与下压标注';png=h.render(bundle)
        for name in ('web_preview.png','执行路径预览_非实测线宽.png','任务预览_线宽色彩为估计.png','黑线加蓝绿_实际路径.png','色带间隙1mm预览.png'):writes[preview/name]=png
        board=Image.new('RGB',(1600,720),'#faf8f0');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
        for x,source,title in [(20,old,'之前：中心线0.9mm，模型色带重叠'),(820,bundle,'现在：模型色带边缘留1mm空隙')]:
            pic=Image.open(io.BytesIO(h.render(source)));pic.thumbnail((760,575));board.paste(pic,(x,55));draw.text((x,15),title,font=font,fill='#34463a')
        draw.text((20,665),'下压保持0.85mm；估计笔宽2.89mm，中心线3.89mm。实际留白会受笔宽与洇色影响。',font=font,fill='#34463a')
        buffer=io.BytesIO();board.save(buffer,format='PNG');writes[preview/'色带间隙1mm前后对照.png']=buffer.getvalue()
        p=PACKAGE/'阅读说明.md';writes[p]=(p.read_text(encoding='utf-8')+'\n\n## 当前铺色间隙（2026-10-12）\n\n'+revision['note']+f'\n当前蓝色{counts["blue"]}笔、绿色{counts["green"]}笔。新画稿下次启动使用；已载入的当前运行不变。\n').encode('utf-8')
        backup=PACKAGE/'历史版本'/(stamp+'_色带净间隙1mm前');backup.mkdir(parents=True,exist_ok=False)
        originals={p:p.read_bytes() if p.exists() else None for p in writes}
        for p,data in originals.items():
            if data is not None:
                saved=backup/p.relative_to(ROOT);saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(data)
        (backup/'painting.json').write_bytes(baseline)
        assert TARGET.read_bytes()==baseline
        done=[]
        try:
            for p,data in writes.items():
                assert (p.read_bytes() if p.exists() else None)==originals[p]
                temp=p.with_name(p.name+'.gap.tmp');temp.write_bytes(data);temp.replace(p);done.append(p)
            assert read(TARGET)==bundle
        except Exception:
            for p in reversed(done):
                if originals[p] is None:p.unlink(missing_ok=True)
                else:p.write_bytes(originals[p])
            raise
        (OUT/'painting.json').write_bytes(encode(bundle));(OUT/'report.json').write_bytes(encode(report))
        print(json.dumps(report|dict(backup=str(backup)),ensure_ascii=False))
    finally:candidate.unlink(missing_ok=True)


if __name__=='__main__':main()
