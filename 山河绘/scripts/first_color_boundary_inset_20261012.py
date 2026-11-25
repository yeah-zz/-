"""Clip existing color centerlines to brush-width-aware interior masks."""
import copy
import csv
from datetime import datetime
import hashlib
import importlib.util
import io
from pathlib import Path
import socket
import sys
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.first_color_edge_gap_20261012 import h,read,encode,TARGET,TASK,PACKAGE
from scripts.consolidate_selected_artworks import digest,stroke_data,index_for
from scripts.directional_rock_color import REGIONS
from scripts.source_aligned_washes import dense_path,points_inside
from ink_control import refresh_stats
from shuimo import press_to_width_mm

OUT=ROOT/'artifacts/第一幅上色边界内收_20261012'
BLEED=.8
RASTER_RESERVE=.30


def face_polygons():
    # Read only the reviewed geometry constant; importing this old command
    # would initialize its obsolete build/install entry points.
    import ast
    text=(ROOT/'artifacts/第一幅扩大色面_20261011/revise.py').read_text(encoding='utf-8')
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FACES' for t in n.targets))
    return ast.literal_eval(node.value)


def main():
    baseline=TARGET.read_bytes();old=read(TARGET);bundle=copy.deepcopy(old)
    module=bundle['workflow']['modules']['color'];colors=module['strokes'];scale=module['mm_per_px']
    assert {p for s in colors for p in s['press']}=={.85}
    assert set(s.get('nominal_pitch_mm') for s in colors if s.get('nominal_pitch_mm') is not None)=={1.}
    assert not bundle['workflow'].get('boundary_inset_revision'), 'Already inset'
    ink=stroke_data(bundle)['black'];w,height=bundle['ink_job']['meta']['image_size'];shape=(height,w)
    ratio=bundle['workflow']['source_revision']['source_transform']['scale']
    offset=np.asarray(bundle['workflow']['source_revision']['source_transform']['offset'])
    bounds={};faces=face_polygons()
    for region,_,poly in REGIONS:
        mask=h.polygon(shape,np.asarray(poly)*ratio+offset)
        bounds[region]=np.maximum(bounds.get(region,np.zeros(shape,np.uint8)),mask)
    for region,polys in faces.items():
        for poly in polys:bounds[region]=np.maximum(bounds.get(region,np.zeros(shape,np.uint8)),h.polygon(shape,poly))
    island=next(s for s in ink if s.get('drawing_region')=='亭下山岛' and s.get('role')=='coarse')
    # The island contour is open at the waterline. Explicitly select its lower
    # side, so an outside piece above the contour cannot survive distance checks.
    crest=np.asarray(island['points']);order=np.argsort(crest[:,0]);xs=np.arange(w)
    top=np.interp(xs,crest[order,0],crest[order,1]);yy=np.arange(height)[:,None]
    inside=(yy>=top[None,:])&(xs[None,:]>=crest[:,0].min())&(xs[None,:]<=crest[:,0].max())
    bounds['亭下山岛']&=inside.astype(np.uint8)
    black=np.zeros(shape,np.uint8)
    for s in ink:
        if s.get('role')=='title' or s.get('drawing_region') in ('整棵松树','湖面水纹','小舟'):continue
        cv2.polylines(black,[np.rint(s['points']).astype(np.int32)],False,1,max(1,int(np.ceil(s['width_mm']/scale))))
    black_clear=cv2.distanceTransform(1-black,cv2.DIST_L2,5)*scale
    width=press_to_width_mm(.85);center_inset=width/2+BLEED+RASTER_RESERVE
    eligible=lambda s:s.get('role') in ('terrain_wash','source_face_wash')
    groups=list(dict.fromkeys((s['color'],s['drawing_region']) for s in colors if eligible(s)))
    safe_masks={};allowed_masks={}
    for color,region in groups:
        source=copy.deepcopy([s for s in colors if eligible(s) and (s['color'],s['drawing_region'])==(color,region)])
        for s in source:s['press']=[.3]*len(s['points'])
        area=h.footprint(source,shape,scale)
        if region in bounds:area&=bounds[region]
        distance=cv2.distanceTransform(area,cv2.DIST_L2,5)*scale
        safe_masks[(color,region)]=((distance>=center_inset)&(black_clear>=center_inset)).astype(np.uint8)
        allowed_masks[(color,region)]=area
    result=[];removed=[];modified=[];before_bad=0;after_bad=0;minimum_black_clear=float('inf')
    for s in colors:
        if not eligible(s):result.append(copy.deepcopy(s));continue
        key=(s['color'],s['drawing_region']);allowed=allowed_masks[key];safe=safe_masks[key]
        original_foot=h.footprint([s],shape,scale)
        before_bad+=int(np.count_nonzero(original_foot&(1-allowed)))
        q=dense_path(s['points'],.05/scale)
        good=points_inside(q,safe);transitions=np.diff(np.r_[False,good,False].astype(int))
        parts=[]
        for a,z in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)):
            part=q[a:z]
            if len(part)<2 or np.linalg.norm(part[-1]-part[0])*scale<.25:continue
            # Current fills are straight; clipping retains the same scan column
            # and 1mm phase. Each interrupted part receives a separate lift.
            p=part[[0,-1]].tolist()
            assert points_inside(dense_path(p,.02/scale),safe).all()
            t=copy.deepcopy(s);t.update(points=p,press=[.85,.85],speed=[s['speed'][0]]*2,
                                       target_width_mm=[width]*2)
            if not good.all():t.update(id=s['id']+'-inset-%d'%len(parts),revision_parent_id=s['id'])
            refresh_stats(t,scale)
            footprint=h.footprint([t],shape,scale)
            violations=int(np.count_nonzero(footprint&(1-allowed)))
            assert violations==0,(s['id'],'brush footprint outside allowed interior')
            assert not np.any(footprint&black),(s['id'],'brush footprint overlaps black outline')
            after_bad+=violations
            qd=np.rint(dense_path(p,.02/scale)).astype(int)
            minimum_black_clear=min(minimum_black_clear,float(black_clear[qd[:,1],qd[:,0]].min())-width/2)
            parts.append(t)
        if not parts:removed.append(s['id'])
        elif not good.all():modified.append(s['id'])
        result.extend(parts)
    assert result and len({s['id'] for s in result})==len(result)
    assert bundle['ink_job']==old['ink_job']
    assert [s for s in result if not eligible(s)]==[s for s in colors if not eligible(s)]
    assert all(s.get('nominal_pitch_mm')==1. for s in result if eligible(s))
    module.update(strokes=result,stroke_count=len(result))
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    revision=dict(date=stamp,source_sha256=hashlib.sha256(baseline).hexdigest(),
                  pressure_mm=.85,center_pitch_mm=1.,modeled_width_mm=width,
                  bleed_allowance_mm=BLEED,raster_reserve_mm=RASTER_RESERVE,
                  centerline_inset_mm=center_inset,minimum_model_black_edge_clearance_mm=minimum_black_clear,
                  removed_ids=removed,clipped_ids=modified,before_strokes=len(colors),after_strokes=len(result),
                  before_outside_design_pixels=before_bad,after_outside_design_pixels=after_bad,
                  island_selected_side='below_actual_black_contour',
                  note='山石上色按0.85mm下压的估计半笔宽内收，额外保留0.8mm洇色/定位余量与0.30mm栅格余量；亭下山岛只保留实际黑轮廓内侧。1mm中心线间距不变；实物笔宽与渗色仍需验证。')
    plan=bundle['workflow'];plan['boundary_inset_revision']=revision
    plan['color_geometry'].update(brush_width_mm=width,boundary_bleed_allowance_mm=BLEED,boundary_centerline_inset_mm=center_inset)
    hashed=digest(stroke_data(bundle))
    for key in ('source_revision','web_installation'):plan[key]['stroke_data_sha256']=hashed
    OUT.mkdir(parents=True,exist_ok=True)
    candidate=TASK/'painting.boundary-candidate.json';candidate.write_bytes(encode(bundle))
    try:
        from web.site_settings import prepared_artwork
        from run_json_painting import manifest_for
        with patch.object(socket.socket,'connect',side_effect=AssertionError('Offline only')):
            prepared=prepared_artwork(candidate,ROOT/'web/data/site_settings.json');manifest_for(prepared[2],prepared[1],prepared[3])
        order=index_for(bundle);counts={k:len(v) for k,v in order.items()}
        report=dict(offline_check='passed',hardware_started=False,effective_from='next-start',
                    counts=counts,revision=revision,stroke_data_sha256=hashed,
                    full_modeled_terrain_brush_footprints_inside=True,black_strokes_unchanged=True)
        writes={TARGET:encode(bundle),TASK/'当前上色边界校验.json':encode(report)}
        for p in TASK.glob('*_计划.json'):writes[p]=encode(plan)
        for name in ('绘画顺序索引.json','Web当前应用记录.json'):
            p=TASK/name;v=read(p);v.update(counts=counts,stroke_data_sha256=hashed,boundary_inset_revision=revision)
            if name=='绘画顺序索引.json':v['stages']=order
            writes[p]=encode(v)
        table=io.StringIO(newline='');writer=csv.writer(table);writer.writerow(['阶段','阶段内笔序','笔触ID','完整部位','类别','原笔触ID','长度mm'])
        for stage,rows in order.items():
            for row in rows:writer.writerow([stage,row['order'],row['id'],row['region'],row['role'],row['parent_stroke_id'],row['length_mm']])
        writes[TASK/'逐笔顺序索引.csv']=table.getvalue().encode('utf-8-sig')
        p=ROOT/'待画作品/画稿清单.json';v=read(p);row=next(row for row in v['artworks'] if row['folder']==PACKAGE.name)
        row.update(counts=counts,stroke_data_sha256=hashed,boundary_inset_revision=revision);writes[p]=encode(v)
        preview=PACKAGE/'预览与下压标注';png=h.render(bundle)
        for name in ('web_preview.png','执行路径预览_非实测线宽.png','任务预览_线宽色彩为估计.png','黑线加蓝绿_实际路径.png','上色边界内收后.png'):writes[preview/name]=png
        board=Image.new('RGB',(1600,770),'#faf8f0');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
        for x,b,label in [(20,old,'修正前：路径离轮廓太近'),(820,bundle,'修正后：按笔宽内收并留余量')]:
            pic=Image.open(io.BytesIO(h.render(b)));pic.thumbnail((760,560));board.paste(pic,(x,50));draw.text((x,10),label,font=font,fill='#34463a')
            crop=Image.open(io.BytesIO(h.render(b))).crop((1050,615,1375,745));crop.thumbnail((760,160));board.paste(crop,(x,580))
        draw.text((20,733),'下方放大亭下山岛；笔距1mm、下压0.85mm。预览为笔宽模型，实物仍需验证。',font=font,fill='#34463a')
        buf=io.BytesIO();board.save(buf,format='PNG');writes[preview/'上色边界内收前后对照.png']=buf.getvalue()
        p=PACKAGE/'阅读说明.md';writes[p]=(p.read_text(encoding='utf-8')+'\n\n## 上色边界内收（2026-10-12）\n\n'+revision['note']+f'\n当前蓝色{counts["blue"]}笔、绿色{counts["green"]}笔。已核对整个估计笔刷足迹处于设计内侧，不能保证真实毛笔无洇色。下一次启动生效。\n').encode('utf-8')
        backup=PACKAGE/'历史版本'/(stamp+'_上色边界内收前');backup.mkdir(parents=True,exist_ok=False)
        originals={p:p.read_bytes() if p.exists() else None for p in writes}
        for p,data in originals.items():
            if data is not None:
                saved=backup/p.relative_to(ROOT);saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(data)
        (backup/'painting.json').write_bytes(baseline);assert TARGET.read_bytes()==baseline
        done=[]
        try:
            for p,data in writes.items():
                assert (p.read_bytes() if p.exists() else None)==originals[p]
                tmp=p.with_name(p.name+'.boundary.tmp');tmp.write_bytes(data);tmp.replace(p);done.append(p)
            assert read(TARGET)==bundle
        except Exception:
            for p in reversed(done):
                if originals[p] is None:p.unlink(missing_ok=True)
                else:p.write_bytes(originals[p])
            raise
        (OUT/'report.json').write_bytes(encode(report))
        print(encode(dict(counts=counts,removed=len(removed),clipped=len(modified),before_outside=before_bad,after_outside=after_bad,model_clearance_mm=minimum_black_clear,offline_check='passed',backup=str(backup))).decode('utf-8'))
    finally:candidate.unlink(missing_ok=True)


if __name__=='__main__':main()
