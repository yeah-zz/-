"""Build/validate/install five full-face color plans with black silhouettes."""
import argparse
import ast
import copy
import csv
from datetime import datetime
import hashlib
import io
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.first_color_edge_gap_20261012 import h,read,encode
from scripts.consolidate_selected_artworks import digest,stroke_data,index_for
from scripts.source_aligned_washes import dense_path,points_inside
from scripts.directional_rock_color import REGIONS as FIRST_REGIONS
from scripts.first_color_boundary_inset_20261012 import face_polygons
from ink_control import InkPolicy,refresh_stats,split_long_strokes,build_plan
from geometry import CanvasMap,check_waypoints
from painting_session import session_lock
from run_mountain_v4_robot import safe_transfers
from shuimo import press_to_width_mm,ink_palette_index
from web.catalog import ARTWORKS
import job

OUT=ROOT/'artifacts/五幅完整色面黑轮廓_20261012'
BLEED=.5
RASTER=.25


def constant(path,name):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    return ast.literal_eval(node.value)


def convert_outlines(bundle,number):
    meta=bundle['ink_job']['meta'];scale=meta['ink_execution']['mm_per_px']
    original=meta['ink_execution']['base_strokes'];colors=bundle['workflow']['modules']['color']['strokes']
    policy=InkPolicy(**meta['ink_execution']['policy']);converted=[];used=[]
    for color in colors:
        if color.get('role')!='distant_color_outline':continue
        s=copy.deepcopy(original[0]);p=color['points'];n=len(p)
        for key in ('source_edge_index','source_path_ids','source_width_mm','title_character','title_component','title_radical'):
            s.pop(key,None)
        s.update(id='black-'+color['id'],name='黑墨山体外轮廓',role='far_outline',points=copy.deepcopy(p),
                 press=[.15]*n,speed=[40.]*n,pal_index=ink_palette_index(),layer=1,tail_mm=0.,
                 drawing_region=color['drawing_region'],source_color_stroke_id=color['id'],
                 recipe='black-mountain-silhouette',target_width_mm=[press_to_width_mm(.15)]*n)
        parts=split_long_strokes([refresh_stats(s,scale)],scale,policy)
        for i,part in enumerate(parts):
            part['id']=s['id']+('-part-%d'%i if len(parts)>1 else '')
            part['parent_stroke_id']=s['id'];converted.append(part)
        used.append(color['id'])
    body=[s for s in original if s.get('role')!='title'];title=[s for s in original if s.get('role')=='title']
    pending=converted.copy();result=[]
    for i,s in enumerate(body):
        result.append(s)
        region=s.get('drawing_region')
        if not any(t.get('drawing_region')==region for t in body[i+1:]):
            result.extend(t for t in pending if t.get('drawing_region')==region)
            pending=[t for t in pending if t.get('drawing_region')!=region]
    result.extend(pending);result.extend(title)
    assert [s for s in result if not s.get('source_color_stroke_id')]==original
    meta['ink_execution']['base_strokes']=result;meta['strokes']=len(result)
    bundle['workflow']['modules']['outline'].update(strokes=copy.deepcopy(result),stroke_count=len(result))
    return used,converted


def build_one(number,target,faces_by,far):
    old=read(target);bundle=copy.deepcopy(old)
    source=old
    if number==1:
        # Restore the broader approved face seeds; the recent clipping was too
        # sparse for the user's latest full-coverage request.
        source=read(target.parent.parent/'历史版本/20261012_203558_上色边界内收前/painting.json')
    module=bundle['workflow']['modules']['color'];scale=module['mm_per_px']
    colors=copy.deepcopy(source['workflow']['modules']['color']['strokes'])
    moved,added=convert_outlines(bundle,number)
    ink=stroke_data(bundle)['black'];w,height=bundle['ink_job']['meta']['image_size'];shape=(height,w)
    eligible=lambda s:s.get('role') in ('terrain_wash','source_face_wash') and '水纹' not in s.get('drawing_region','')
    terrain=[s for s in colors if eligible(s)]
    regions=list(dict.fromkeys(s['drawing_region'] for s in terrain))
    faces=copy.deepcopy(face_polygons() if number==1 else faces_by[number])
    if number==1:
        tr=bundle['workflow']['source_revision']['source_transform'];ratio=tr['scale'];off=np.asarray(tr['offset'])
        for region,_,polygon in FIRST_REGIONS:
            faces.setdefault(region,[]).append((np.asarray(polygon)*ratio+off).tolist())
        faces['主峰'].append([(845,232),(889,207),(902,211),(931,229),(946,252),(962,271),(975,278),(962,297),(926,288),(904,267),(877,241)])
    for region,poly,*_ in far.get(number,[]):faces.setdefault(region,[]).append(poly)
    if number==5:
        # These are the nine separately reviewed terrace tops. Fill each top
        # rather than expanding a single envelope across the vertical risers.
        faces.setdefault('整片梯田与田埂',[]).extend(copy.deepcopy(h.FIELDS))
    outer=np.zeros(shape,np.uint8);objects=np.zeros(shape,np.uint8)
    for s in ink:
        role=s.get('role');region=s.get('drawing_region','')
        structural=role in ('coarse','far','far_outline','mountain_outline') or (role=='near_outline' and s['length_mm']>=12)
        if structural and not any(name in region for name in ('树','题字','水纹')):
            cv2.polylines(outer,[np.rint(s['points']).astype(np.int32)],False,1,max(1,int(np.ceil(s['width_mm']/scale))))
    for region in ('亭子','台阶','小舟','完整渔舟','题字'):
        ss=[s for s in ink if s.get('drawing_region')==region]
        if not ss:continue
        pts=np.vstack([s['points'] for s in ss]);lo=np.floor(pts.min(0)-.5/scale).astype(int);hi=np.ceil(pts.max(0)+.5/scale).astype(int)
        cv2.rectangle(objects,tuple(lo),tuple(hi),1,-1)
    outer_clear=cv2.distanceTransform(1-outer,cv2.DIST_L2,5)*scale
    replacements={};generated=[];stats=[];audits=[]
    for ri,region in enumerate(regions):
        selected=[s for s in terrain if s['drawing_region']==region]
        seeds={}
        for color in ('blue','green'):
            ss=copy.deepcopy([s for s in selected if s['color']==color])
            for s in ss:s['press']=[.3]*len(s['points'])
            seeds[color]=h.footprint(ss,shape,scale)
        area=np.maximum(seeds['blue'],seeds['green'])
        for polygon in faces.get(region,[]):area|=h.polygon(shape,polygon)
        area&=1-objects
        # The first island must be filled below its exact drawn crest.
        if number==1 and region=='亭下山岛':
            crest=np.asarray(next(s for s in ink if s.get('drawing_region')==region and s.get('role')=='coarse')['points'])
            ix=np.argsort(crest[:,0]);xs=np.arange(w);top=np.interp(xs,crest[ix,0],crest[ix,1])
            area&=((np.arange(height)[:,None]>=top[None,:])&(xs[None,:]>=crest[:,0].min())&(xs[None,:]<=crest[:,0].max())).astype(np.uint8)
        # Retain the established green areas; cover the remainder of an expanded
        # rock face in blue. Green-only subjects (grass/terraces) remain green.
        dominant='green' if all(s['color']=='green' for s in selected) else 'blue'
        green=area.copy() if dominant=='green' else seeds['green']&area
        blue=np.zeros(shape,np.uint8) if dominant=='green' else area&(1-green)
        # A face interface may meet under the brush. Only the outer envelope and
        # black silhouette require bleed clearance; internal textures are kept.
        dist=cv2.distanceTransform(area,cv2.DIST_L2,5)*scale
        paths=[];coverage=np.zeros(shape,np.uint8);target_area=area.copy()
        for color,choice in [('green',green),('blue',blue)]:
            templates=[s for s in selected if s['color']==color]
            if not choice.any():continue
            template=copy.deepcopy(templates[0] if templates else selected[0]);template['color']=color
            pressure=float(module['motion_candidate']['press_mm']);width=press_to_width_mm(pressure)
            inset=width/2+BLEED+RASTER
            safe=((dist>=inset)&(outer_clear>=inset)&(choice>0)).astype(np.uint8)
            horizontal=template.get('path_style')=='left-to-right-lift-return' or region=='柳树下草坡'
            pitch=1. if number==1 else float(template.get('nominal_pitch_mm',.9))
            scan=safe.T.copy() if horizontal else safe
            if not scan.any():continue
            yy,xx=np.nonzero(scan);startx=float(xx.min());endx=float(xx.max());y0=int(yy.min());y1=int(yy.max())
            for x in np.arange(startx,endx+1e-6,pitch/scale):
                ys=np.arange(y0,y1+.2,.2);q=np.column_stack((np.full(len(ys),x),ys))
                for part in h.segments(q,scan):
                    length=(part[-1,1]-part[0,1])*scale
                    if length<.30:continue
                    ends=np.linspace(part[0,1],part[-1,1],max(1,int(np.ceil(length/(55 if horizontal else 20))))+1)
                    for a,z in zip(ends[:-1],ends[1:]):
                        points=[[float(x),float(a)],[float(x),float(z)]]
                        if horizontal:points=[p[::-1] for p in points]
                        s=copy.deepcopy(template);n=len(points)
                        s.update(id=f'fullblack-{number}-{ri}-{color}-{len(paths):04d}',points=points,press=[pressure]*n,
                                 speed=[template['speed'][0]]*n,pressure_mm=pressure,
                                 width_mm=width,brush_width_mm_estimate=width,target_width_mm=[width]*n,
                                 role='terrain_wash',nominal_pitch_mm=pitch,joined_passes=1,
                                 parent_stroke_id=template['id'],path_style='left-to-right-lift-return' if horizontal else 'top-to-bottom-lift-return')
                        refresh_stats(s,scale)
                        assert points_inside(dense_path(points,.02/scale),safe).all()
                        footprint=h.footprint([s],shape,scale)
                        assert not np.any(footprint&(1-area)),(number,region,'outside envelope')
                        assert not np.any(footprint&outer),(number,region,'crosses black silhouette')
                        paths.append(s);coverage|=footprint
            audits.append(dict(color=color,region=region,full_brush_inside_envelope=True,black_silhouette_not_crossed=True))
        old_coverage=h.footprint([s for s in old['workflow']['modules']['color']['strokes'] if eligible(s) and s['drawing_region']==region],shape,scale)
        stats.append(dict(region=region,before_strokes=sum(eligible(s) and s['drawing_region']==region for s in old['workflow']['modules']['color']['strokes']),after_strokes=len(paths),
                          before_area_mm2=float(old_coverage.sum()*scale**2),after_area_mm2=float(coverage.sum()*scale**2),
                          target_area_mm2=float(target_area.sum()*scale**2)))
        replacements[region]=paths;generated.extend(paths)
    # Keep the chosen color/subject order. Conversion removes colored silhouettes
    # from the color stage rather than drawing them twice.
    kept=[copy.deepcopy(s) for s in colors if not eligible(s) and s.get('role')!='distant_color_outline']
    allcolors=kept+generated;order=bundle['workflow']['execution_order'].get('color_subjects',regions)
    allcolors.sort(key=lambda s:(order.index(s.get('drawing_region')) if s.get('drawing_region') in order else len(order)))
    assert not any('outline' in s.get('role','') for s in allcolors)
    assert [s for s in allcolors if s.get('role')!='terrain_wash']==[s for s in sorted(kept,key=lambda s:(order.index(s.get('drawing_region')) if s.get('drawing_region') in order else len(order))) if s.get('role')!='terrain_wash']
    module.update(strokes=allcolors,stroke_count=len(allcolors))
    assert allcolors
    meta=bundle['ink_job']['meta'];meta['image']=bundle['workflow']['reference_image']='../参考原图.png'
    mapper=CanvasMap(meta['image_size']);policy=InkPolicy(**meta['ink_execution']['policy'])
    steps,_=build_plan(ink,ink_palette_index(),scale,policy);waypoints,_=job.steps_to_waypoints(steps,mapper)
    bundle['ink_job']['waypoints']=safe_transfers(waypoints);assert check_waypoints(bundle['ink_job']['waypoints'])[0]
    revision=dict(date='2026-10-12',source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  all_mountain_outlines_black=True,converted_color_outline_ids=moved,added_black_strokes=len(added),
                  regions=stats,brush_footprint_checks=audits,bleed_allowance_mm=BLEED,raster_reserve_mm=RASTER,
                  pressure_mm=module['motion_candidate']['press_mm'],
                  note='五幅统一黑墨勾勒山体外轮廓，蓝绿尽量铺满已有完整岩面；仅在外轮廓和建筑周围按当前估计半笔宽加0.5mm洇色余量内收。内部纹理不再扩大为大片禁涂区。各幅既定下压、笔距、速度与配色主体保留。')
    plan=bundle['workflow'];plan['complete_faces_black_outlines_revision']=revision
    plan['boundary_inset_revision']=dict(status='superseded-by-complete-face-planning',current_bleed_allowance_mm=BLEED)
    hashed=digest(stroke_data(bundle))
    for key in ('source_revision','web_installation'):plan[key]['stroke_data_sha256']=hashed
    return old,bundle,revision


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    faces=constant(ROOT/'artifacts/其余四幅完整单向色面_20261011/revise.py','FACES_BY_NUMBER')
    far=constant(ROOT/'artifacts/二至五幅色面补全_20261010/prepare.py','FAR')
    records=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
    board=Image.new('RGB',(1700,5*660),'#faf8f0');draw=ImageDraw.Draw(board)
    for number,item in enumerate(ARTWORKS[:5],1):
        target=ROOT/item[3];old,bundle,revision=build_one(number,target,faces,far)
        folder=OUT/str(number);folder.mkdir(exist_ok=True)
        candidate=folder/'painting.json';candidate.write_bytes(encode(bundle))
        from web.site_settings import prepared_artwork
        from run_json_painting import manifest_for
        with patch.object(socket.socket,'connect',side_effect=AssertionError('Offline only')):
            prepared=prepared_artwork(candidate,ROOT/'web/data/site_settings.json');manifest_for(prepared[2],prepared[1],prepared[3])
        counts={k:len(v) for k,v in index_for(bundle).items()}
        report=dict(number=number,title=item[1],counts=counts,revision=revision,offline_check='passed',hardware_started=False,
                    painting_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),stroke_data_sha256=digest(stroke_data(bundle)))
        (folder/'report.json').write_bytes(encode(report));(folder/'after.png').write_bytes(h.render(bundle));(folder/'before.png').write_bytes(h.render(old))
        for x,prefix,label in [(20,'before','修改前'),(870,'after','完整色面 + 黑色外轮廓')]:
            pic=Image.open(folder/(prefix+'.png'));pic.thumbnail((800,565));board.paste(pic,(x,(number-1)*660+65));draw.text((x,(number-1)*660+20),f'{number:02d} {item[1]} · {label}',font=font,fill='#34463a')
        records.append(report)
        print(encode(dict(number=number,counts=counts,black_outlines_added=revision['added_black_strokes'],offline_check='passed')).decode('utf-8'),flush=True)
    board.save(OUT/'五幅调整前后.png');(OUT/'report.json').write_bytes(encode(records))


def install():
    import urllib.request
    def state():
        with urllib.request.urlopen('http://127.0.0.1:8765/api/state',timeout=5) as response:return __import__('json').load(response)
    st=state();assert not st.get('parametersLocked') and st['status'] in ('idle','complete','stopped','saved','error'),st
    records=read(OUT/'report.json');writes={};catalog_path=ROOT/'待画作品/画稿清单.json';catalog=read(catalog_path)
    for number,item in enumerate(ARTWORKS[:5],1):
        target=ROOT/item[3];folder=OUT/str(number);bundle=read(folder/'painting.json');report=records[number-1]
        assert hashlib.sha256(target.read_bytes()).hexdigest()==report['revision']['source_sha256']
        assert hashlib.sha256((folder/'painting.json').read_bytes()).hexdigest()==report['painting_sha256']
        task=target.parent;package=task.parent;plan=bundle['workflow'];writes[target]=encode(bundle)
        writes[task/'black_only.json']=encode(bundle['ink_job'])
        for p in task.glob('*_黑线任务.json'):writes[p]=encode(bundle['ink_job'])
        for p in task.glob('*_计划.json'):writes[p]=encode(plan)
        index=index_for(bundle);writes[task/'绘画顺序索引.json']=encode(dict(format='shanhe-stroke-order-index-v1',counts=report['counts'],stages=index,stroke_data_sha256=report['stroke_data_sha256'],title_after_all_colors=True))
        table=io.StringIO(newline='');writer=csv.writer(table);writer.writerow(['阶段','阶段内笔序','笔触ID','完整部位','类别','原笔触ID','长度mm'])
        for stage,rows in index.items():
            for row in rows:writer.writerow([stage,row['order'],row['id'],row['region'],row['role'],row['parent_stroke_id'],row['length_mm']])
        writes[task/'逐笔顺序索引.csv']=table.getvalue().encode('utf-8-sig')
        p=task/'Web当前应用记录.json';v=read(p);v.update(counts=report['counts'],stroke_data_sha256=report['stroke_data_sha256'],complete_faces_black_outlines_revision=report['revision']);writes[p]=encode(v)
        writes[task/'完整色面黑轮廓校验.json']=encode(report)
        for name in ('web_preview.png','执行路径预览_非实测线宽.png','任务预览_线宽色彩为估计.png','黑线加蓝绿_实际路径.png','当前作画阶段预览.png','完整色面黑轮廓.png'):writes[package/'预览与下压标注'/name]=(folder/'after.png').read_bytes()
        blank=copy.deepcopy(bundle);blank['workflow']['modules']['color']['strokes']=[];writes[package/'预览与下压标注/纯勾线.png']=h.render(blank)
        p=package/'阅读说明.md';writes[p]=(p.read_text(encoding='utf-8')+'\n\n## 当前完整色面与黑轮廓（2026-10-12）\n\n'+report['revision']['note']+'\n笔宽及洇色仍为估计；下一次启动使用本次画稿。详细校验见任务资料/完整色面黑轮廓校验.json。\n').encode('utf-8')
        row=next(row for row in catalog['artworks'] if row['folder']==package.name);row.update(counts=report['counts'],stroke_data_sha256=report['stroke_data_sha256'],complete_faces_black_outlines_revision=report['revision'])
    writes[catalog_path]=encode(catalog)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S');backup=ROOT/'archive'/('五幅完整色面黑轮廓前_'+stamp)
    originals={p:p.read_bytes() if p.exists() else None for p in writes}
    for p,data in originals.items():
        if data is not None:
            q=backup/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
    with session_lock(ROOT/'robot_runs'):
        assert not (ROOT/'robot_runs/blue_wash_motion.lock').exists()
        done=[]
        try:
            for p,data in writes.items():
                assert (p.read_bytes() if p.exists() else None)==originals[p]
                tmp=p.with_name(p.name+'.fullblack.tmp');tmp.write_bytes(data);tmp.replace(p);done.append(p)
            assert all(p.read_bytes()==data for p,data in writes.items())
        except Exception:
            for p in reversed(done):
                if originals[p] is None:p.unlink(missing_ok=True)
                else:p.write_bytes(originals[p])
            raise
    (OUT/'installation.json').write_bytes(encode(dict(status='installed',backup=str(backup),artworks=records)))
    print('Five artworks installed; backup: '+str(backup))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','install']);args=parser.parse_args()
    {'build':build,'install':install}[args.action]()
