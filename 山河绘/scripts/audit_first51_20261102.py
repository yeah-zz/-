# -*- coding: utf-8 -*-
"""Offline audit of emitted strokes, fragmentation, and brush-footprint overlap.

No robot imports, execution, or modification of the paused drawing task.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from geometry import CanvasMap
from bind_full_ink_legacy_20261102 import padded_mapping
import job

LAB=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022')
TASK=LAB/'selected_complete_G03_01_20261102'/'fresh_paper_reload3_dwell2'
OUT=LAB/'execution_coverage_audit_20261102'
LOG=TASK/'G03_01_complete_full_job_runs'/'20261102_215121_118426_e7be48c9_diagnostic.jsonl'


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def length(p):return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())


def footprint_audit(paths,width_mm):
    """Scenario coverage, not a physical ink model or measured brush width."""
    pts=np.concatenate(paths);lo=pts.min(axis=0)-width_mm*2
    size=np.ceil((pts.max(axis=0)-lo+width_mm*2)/.05).astype(int)+1
    covered=np.zeros((size[1],size[0]),np.uint16)
    novel=[];sum_area=0
    thickness=max(1,int(round(width_mm/.05)))
    for p in paths:
        mask=np.zeros(covered.shape,np.uint8)
        q=np.rint((p-lo)/.05).astype(np.int32)
        cv2.polylines(mask,[q],False,1,thickness,cv2.LINE_8)
        own=mask.astype(bool);area=int(own.sum());sum_area+=area
        novel.append(float((own&(covered==0)).sum()/max(1,area)))
        covered+=mask
    union=int((covered>0).sum())
    return dict(assumed_width_mm=width_mm,measured_brush_width=False,
        overlap_fraction_of_sum_areas=1-union/max(1,sum_area),
        fraction_of_union_painted_multiple_times=float((covered>=2).sum()/max(1,union)),
        segments_with_under_25_percent_new_area=sum(v<.25 for v in novel),
        segment_novel_fractions=novel)


def run():
    OUT.mkdir(exist_ok=True)
    data=json.loads((TASK/'G03_01_complete_full_job.json').read_text('utf-8'))
    rows=data['meta']['ink_execution']['base_strokes'];mapper=CanvasMap(data['meta']['image_size'])
    xy=[mapper.to_robot(s['points']) for s in rows]
    lengths=np.array([length(p) for p in xy])
    actual=defaultdict(list)
    for line in LOG.read_text('utf-8').splitlines():
        r=json.loads(line);wp=r.get('waypoint') or {}
        if (r.get('event')=='command_submit' and r.get('command','').startswith('MovL') and
                r.get('action')=='stroke' and r.get('stroke_number',99999)<=51 and
                wp.get('tag','').startswith(('down:','draw:'))):
            actual[r['stroke_number']].append(wp)
    mismatches=[]
    for i,s in enumerate(rows[:51],1):
        expected=[p for p in job.stroke_waypoints(mapper,s) if p['tag'].startswith(('down:','draw:'))]
        a=actual[i]
        if len(a)!=len(expected) or any(x['tag']!=y['tag'] or
                not np.allclose([x[k] for k in ('x','y','z')],[y[k] for k in ('x','y','z')],rtol=0,atol=1e-7)
                for x,y in zip(a,expected)):
            mismatches.append(i)
    prefix=xy[:51];points=np.concatenate(prefix)
    rest=sorted(range(51,len(rows)),key=lambda i:(-lengths[i],i))
    if set(rest)!=set(range(51,len(rows))) or len(rest)!=len(set(rest)):
        raise AssertionError('Remaining path coverage changed')
    new51=[xy[i] for i in rest[:51]]
    statistics=dict(completed_segments=51,total_segments=len(rows),
        completed_length_mm=float(lengths[:51].sum()),first51_median_mm=float(np.median(lengths[:51])),
        first51_under_1mm=int(sum(lengths[:51]<1)),first51_under_2mm=int(sum(lengths[:51]<2)),
        total_under_2mm=int(sum(lengths<2)),all_median_mm=float(np.median(lengths)),
        first51_bounds_mm=np.ptp(points,axis=0).tolist(),
        whole_bounds_mm=np.ptp(np.concatenate(xy),axis=0).tolist(),
        paper_command_sequence_mismatch_strokes=mismatches,
        paper_command_count=sum(map(len,actual.values())),
        audit_is_of_submitted_commands_not_optical_motion=True,
        proposed_next51_length_mm=float(sum(length(p) for p in new51)),
        proposed_next51_bounds_mm=np.ptp(np.concatenate(new51),axis=0).tolist(),
        new_order_only_preserves_all_remaining_geometry=True,
        brush_stroke_deduplication_completed=False,robot_connected=False,motion_started=False)
    statistics['proposal_is_order_change_only']=True
    audits=[footprint_audit(prefix,w) for w in [.5,1.,1.5]]
    save(OUT/'audit.json',dict(statistics=statistics,brush_footprint_scenarios=audits))
    save(OUT/'proposed_remaining_order.json',dict(proposal_only=True,executable_job=False,
        completed_prefix_untouched=list(range(51)),remaining_indices=rest,
        skipped_indices=[],policy='Longer remaining paths first, no deletion or new connections',
        requires_further_stroke_level_dedup_review=True))
    source=Image.open(LAB/'selected_complete_G03_01_20261102'/'source.png').convert('RGB')
    w,h=source.size;pw,ph=mapper.canvas_size_mm();_,mult,offset,_=padded_mapping(w,h,pw,ph)
    native=[(np.array(s['points'])-offset)/mult for s in rows]
    def render(indices,with_source=False,previous=False):
        canvas=np.array(source) if with_source else np.full((h,w,3),255,np.uint8)
        if previous:
            for p in native[:51]:cv2.polylines(canvas,[np.rint(p).astype(np.int32)],False,(180,180,180),1,cv2.LINE_AA)
        for i in indices:
            p=np.rint(native[i]).astype(np.int32)
            cv2.polylines(canvas,[p],False,(210,35,25) if with_source else (30,55,160),1,cv2.LINE_AA)
            cv2.circle(canvas,tuple(p[0]),1,(215,30,30),-1)
        return Image.fromarray(canvas)
    old=render(list(range(51)));old.save(OUT/'actual_first51.png')
    first_on_original=render(list(range(51)),True);first_on_original.save(OUT/'actual_first51_on_original.png')
    proposed=render(rest[:51],False,True);proposed.save(OUT/'remaining_long_first51.png')
    image=Image.new('RGB',(1800,680),'#f5f3eb');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',23)
    cards=[(first_on_original,'已执行前51段：原图中的位置'),(old,'已执行前51段：红点为落笔点'),(proposed,'离线顺序建议：余下长线先画')]
    for i,(im,label) in enumerate(cards):
        x=i*600+12;draw.text((x,12),label,font=font,fill='#252925');image.paste(im.resize((576,576)),(x,52))
    draw.text((12,638),'仅诊断与离线顺序对照；未续画、未删除原有线条，重复覆盖仍需另行修正。',font=font,fill='#252925')
    image.save(OUT/'first51_diagnosis.jpg',quality=94)
    (OUT/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>前51段执行诊断</title><style>body{{font:18px system-ui;max-width:1500px;margin:24px auto;padding:0 20px;background:#f5f3eb;color:#252925}}img{{width:100%}}li{{margin:10px 0}}</style><h1>G03_01 前51段执行诊断 · 当前保持暂停</h1><img src="first51_diagnosis.jpg"><ul><li>51段共{statistics['completed_length_mm']:.1f}mm，中位长度{statistics['first51_median_mm']:.2f}mm；27段短于2mm。它们是图结构碎段，不是51条画家完整笔画。</li><li>全图1,670段中，有{statistics['total_under_2mm']}段短于2mm。继续按当前分段逐段落笔会增加抬落、端点积墨和覆盖。</li><li>逐条核对日志：前51段纸上指令与计划顺序不符的段数为{len(mismatches)}。这是命令记录审计，不能替代视觉/运动测量。</li><li>右图只展示余下长线优先的顺序建议：不重画已完成51段，不删除剩余线条。它能先展开构图，但并未解决笔触级去重和碎段合并。</li><li>笔宽0.5/1.0/1.5mm的重复覆盖计算仅为几何假设，当前笔宽尚无实测值，不据此自动删线。</li></ul><p><a href="audit.json">详细统计与笔宽假设</a> · <a href="proposed_remaining_order.json">离线顺序草案</a></p>''',encoding='utf-8')
    print(json.dumps(statistics,ensure_ascii=False,indent=2))
    print(json.dumps([{k:v for k,v in a.items() if k!='segment_novel_fractions'} for a in audits],ensure_ascii=False,indent=2))


if __name__=='__main__':run()
