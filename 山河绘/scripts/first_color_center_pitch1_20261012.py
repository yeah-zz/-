"""Rebuild painting 01 from its 0.9mm baseline with 1mm center pitch."""
import copy
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.first_color_edge_gap_20261012 import h,read,encode,TARGET,TASK,PACKAGE
from scripts.consolidate_selected_artworks import digest,stroke_data,index_for
from ink_control import refresh_stats
from shuimo import press_to_width_mm


def main():
    original_folder=PACKAGE/'历史版本/20261012_195446_色带净间隙1mm前'
    source=original_folder/'painting.json'
    baseline=TARGET.read_bytes();old=read(source);bundle=copy.deepcopy(old)
    module=bundle['workflow']['modules']['color'];strokes=module['strokes'];scale=module['mm_per_px']
    assert len(strokes)==852 and set(s['press'][0] for s in strokes)=={.85}
    eligible=lambda s:s.get('nominal_pitch_mm') is not None
    groups=list(dict.fromkeys((s['color'],s['drawing_region']) for s in strokes if eligible(s)))
    w,height=bundle['ink_job']['meta']['image_size'];shape=(height,w)
    replacements={};removed=set();regions=[];spacing_checks=0
    for gi,(color,region) in enumerate(groups):
        originals=[s for s in strokes if eligible(s) and (s['color'],s['drawing_region'])==(color,region)]
        template=originals[0];horizontal=template['path_style']=='left-to-right-lift-return'
        # Reconstruct the same designed color area used by the original 0.9mm
        # planner; pressure changes must not expand this design boundary.
        geometry=copy.deepcopy(originals)
        for s in geometry:s['press']=[.3]*len(s['points'])
        area=h.footprint(geometry,shape,scale)
        if horizontal:area=area.T.copy()
        safe=(cv2.distanceTransform(area,cv2.DIST_L2,5)>=press_to_width_mm(.3)/(2*scale)).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(safe);paths=[]
        for ci in range(1,count):
            component=(labels==ci).astype(np.uint8);x0,y0,ww,hh,_=stats[ci]
            columns=np.arange(x0,x0+ww,1./scale)
            if len(columns)>1:
                assert np.allclose(np.diff(columns)*scale,1.,rtol=0,atol=1e-10)
                spacing_checks+=len(columns)-1
            for x in columns:
                ys=np.arange(y0,y0+hh,.2);q=np.column_stack((np.full(len(ys),x),ys))
                for part in h.segments(q,component):
                    length=(part[-1,1]-part[0,1])*scale
                    if length<.08:continue
                    maximum=55 if horizontal else 20
                    ends=np.linspace(part[0,1],part[-1,1],max(1,int(np.ceil(length/maximum)))+1)
                    for start,end in zip(ends[:-1],ends[1:]):
                        points=[[float(x),float(start)],[float(x),float(end)]]
                        if horizontal:points=[point[::-1] for point in points]
                        s=copy.deepcopy(template)
                        s.update(id=f'pitch1-{gi}-{len(paths):04d}',points=points,
                                 press=[.85,.85],speed=[template['speed'][0]]*2,
                                 nominal_pitch_mm=1.,pressure_mm=.85,
                                 target_width_mm=[press_to_width_mm(.85)]*2,
                                 parent_stroke_id=template['id'],pitch_design_component=ci)
                        refresh_stats(s,scale);paths.append(s)
        assert paths,(color,region)
        replacements[originals[0]['id']]=paths;removed.update(s['id'] for s in originals)
        regions.append(dict(color=color,region=region,before=len(originals),after=len(paths)))
    result=[]
    for s in strokes:
        if s['id'] in replacements:result.extend(replacements[s['id']])
        if s['id'] not in removed:result.append(copy.deepcopy(s))
    assert len({s['id'] for s in result})==len(result)
    assert [s for s in result if not eligible(s)]==[s for s in strokes if not eligible(s)]
    assert bundle['ink_job']==old['ink_job']
    module.update(strokes=result,stroke_count=len(result))
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    revision=dict(date=stamp,previous_center_pitch_mm=.9,center_pitch_mm=1.,
                  pressure_mm=.85,spacing_checks=spacing_checks,regions=regions,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  note='按用户最新要求：成片上色中心线间距1.00mm；撤销色带边缘留1mm空隙方案，恢复原0.9mm版的色面范围后重排。下压保持0.85mm，原速度、方向、补蘸设置保留。')
    plan=bundle['workflow'];plan['center_pitch_revision']=revision
    plan['color_geometry'].update(pitch_mm=1.,pitch_px=1./scale,pressure_range_mm=[.85,.85])
    hashed=digest(stroke_data(bundle))
    for key in ('source_revision','web_installation'):plan[key]['stroke_data_sha256']=hashed
    candidate=TASK/'painting.pitch1-candidate.json';candidate.write_bytes(encode(bundle))
    try:
        from web.site_settings import prepared_artwork
        from run_json_painting import manifest_for
        with patch.object(socket.socket,'connect',side_effect=AssertionError('Offline only')):
            prepared=prepared_artwork(candidate,ROOT/'web/data/site_settings.json')
            manifest_for(prepared[2],prepared[1],prepared[3])
        order=index_for(bundle);counts={k:len(v) for k,v in order.items()}
        report=dict(offline_check='passed',hardware_started=False,effective_from='next-start',
                    counts=counts,revision=revision,stroke_data_sha256=hashed)
        writes={TARGET:encode(bundle),TASK/'中心线间距1mm校验.json':encode(report)}
        for p in TASK.glob('*_计划.json'):writes[p]=encode(plan)
        p=TASK/'绘画顺序索引.json';v=read(p);v.update(counts=counts,stages=order,stroke_data_sha256=hashed);writes[p]=encode(v)
        p=TASK/'Web当前应用记录.json';v=read(p);v.pop('edge_gap_revision',None);v.update(counts=counts,stroke_data_sha256=hashed,center_pitch_revision=revision);writes[p]=encode(v)
        table=io.StringIO(newline='');writer=csv.writer(table);writer.writerow(['阶段','阶段内笔序','笔触ID','完整部位','类别','原笔触ID','长度mm'])
        for stage,rows in order.items():
            for row in rows:writer.writerow([stage,row['order'],row['id'],row['region'],row['role'],row['parent_stroke_id'],row['length_mm']])
        writes[TASK/'逐笔顺序索引.csv']=table.getvalue().encode('utf-8-sig')
        p=ROOT/'待画作品/画稿清单.json';v=read(p);row=next(row for row in v['artworks'] if row['folder']==PACKAGE.name)
        row.pop('edge_gap_revision',None);row.update(counts=counts,stroke_data_sha256=hashed,center_pitch_revision=revision);writes[p]=encode(v)
        p=TASK/'色带1mm净间隙校验.json'
        if p.exists():
            v=read(p);v.update(status='superseded',superseded_by='中心线间距1mm校验.json');writes[p]=encode(v)
        preview=PACKAGE/'预览与下压标注';png=h.render(bundle)
        for name in ('web_preview.png','执行路径预览_非实测线宽.png','任务预览_线宽色彩为估计.png','黑线加蓝绿_实际路径.png','中心线间距1mm预览.png'):writes[preview/name]=png
        original_readme=original_folder/'待画作品'/PACKAGE.name/'阅读说明.md'
        assert original_readme.exists()
        writes[PACKAGE/'阅读说明.md']=(original_readme.read_text(encoding='utf-8')+'\n\n## 当前上色中心线间距（2026-10-12）\n\n'+revision['note']+f'\n蓝色{counts["blue"]}笔、绿色{counts["green"]}笔；通过离线校验，下次启动生效。\n').encode('utf-8')
        backup=PACKAGE/'历史版本'/(stamp+'_中心线间距1mm前');backup.mkdir(parents=True,exist_ok=False)
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
                temp=p.with_name(p.name+'.pitch.tmp');temp.write_bytes(data);temp.replace(p);done.append(p)
            saved=read(TARGET)
            assert saved==bundle and 'edge_gap_revision' not in saved['workflow']
        except Exception:
            for p in reversed(done):
                if originals[p] is None:p.unlink(missing_ok=True)
                else:p.write_bytes(originals[p])
            raise
        print(json.dumps(report|dict(backup=str(backup)),ensure_ascii=False))
    finally:candidate.unlink(missing_ok=True)


if __name__=='__main__':main()
