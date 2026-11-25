"""Simplify existing ink details and retain 90% of estimated filled-color footprint."""
from pathlib import Path
from datetime import datetime
import sys,json,shutil,heapq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,cv2
import job,ink_control,shuimo,vision,watercolor,imgio
from geometry import CanvasMap,check_waypoints
from run_mountain_v4_robot import safe_transfers


def select_coverage(strokes,shape,scale,target=.90):
    footprints=[];union=np.zeros(shape[0]*shape[1],bool)
    for s in strokes:
        pts=np.rint(s['points']).astype(int)
        rad=int(np.ceil(1/scale))+2
        lo=np.maximum(pts.min(0)-rad,0);hi=np.minimum(pts.max(0)+rad+1,[shape[1],shape[0]])
        tile=np.zeros((hi[1]-lo[1],hi[0]-lo[0]),np.uint8)
        cv2.polylines(tile,[pts-lo],False,1,max(1,round(1/scale)))
        y,x=np.where(tile);indices=(y+lo[1])*shape[1]+x+lo[0]
        footprints.append(indices);union[indices]=True
    wanted=int(np.ceil(union.sum()*target));covered=np.zeros_like(union)
    heap=[(-len(f),i) for i,f in enumerate(footprints)];heapq.heapify(heap)
    selected=set();count=0
    while heap and count<wanted:
        _,i=heapq.heappop(heap);f=footprints[i];gain=int(np.count_nonzero(~covered[f]))
        if heap and gain < -heap[0][0]:heapq.heappush(heap,(-gain,i));continue
        if not gain:break
        selected.add(i);covered[f]=True;count+=gain
    return [s for i,s in enumerate(strokes) if i in selected],count/max(1,int(union.sum()))


def main():
    root=Path(__file__).resolve().parents[1];src=root/'输出/层峦松水_快速起落_20260929_162006'
    out=root/'输出'/('层峦松水_简约铺色_'+datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir(exist_ok=False)
    p=json.loads((src/'层峦松水_快速起落_计划.json').read_text(encoding='utf8'))
    _,meta=job.load_job(src/'层峦松水_快速起落_黑线任务.json')
    shutil.copy2(src/'参考原图.png',out/'参考原图.png');p['reference_image']=str(out/'参考原图.png');meta['image']=p['reference_image']
    work,_=vision.prepare_image(vision.load_image(p['reference_image']),work_size=1800)
    scale=p['mm_per_px'];snap=meta['ink_execution'];old=snap['base_strokes']
    # Preserve width-building rails and structural lines; short twig detail has a lower cutoff.
    h,w=work.shape[:2];base=[]
    for s in old:
        x,y=np.mean(s['points'],axis=0)/[w,h]
        pine=(x<.45 and .35<y<.72)
        if s.get('role')=='source_broad_rail' or s['length_mm']>=(2.4 if pine else 4.):base.append(s)
    policy=ink_control.InkPolicy(**snap['policy'])
    base=ink_control.split_long_strokes(base,scale,policy);base=ink_control.order_for_ink(base,scale,policy)
    for i,s in enumerate(base):s['id']=f'outline-{i:05d}'
    steps,_=ink_control.build_plan(base,shuimo.ink_palette_index(),scale,policy)
    snap['base_strokes']=base;meta['strokes']=len(base)
    outline=p['modules']['outline'];outline['strokes']=[v['stroke'] for v in steps if v['kind']=='stroke'];outline['stroke_count']=len(outline['strokes'])
    m=p['modules']['color'];oldcolors=m['strokes'];colors=[];report={}
    for color in ('blue','green'):
        selected,coverage=select_coverage([s for s in oldcolors if s['color']==color],work.shape,scale)
        colors+=selected;report[color]={'strokes':len(selected),'retained_estimated_footprint':coverage}
    m['strokes']=colors;m['stroke_count']=len(colors);m['reload_strokes_per_load']=4
    p['simplification']={'source_black_count':len(old),'black_count':len(base),'source_color_count':len(oldcolors),'colors':report,
        'note':'Short ink texture removed; pine branches and broad rails prioritized. Retains 90% of previous estimated color footprint. Reload limit4; existing length/time limits unchanged.'}
    mapper=CanvasMap(meta['image_size']);wp,_=job.steps_to_waypoints(steps,mapper);wp=safe_transfers(wp);assert check_waypoints(wp)[0]
    prefix=out/'层峦松水_简约铺色';job.save_job(str(prefix)+'_黑线任务.json',wp,meta)
    Path(str(prefix)+'_计划.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf8')
    canvas,_=watercolor.render_image(work,m,include_ink=False)
    for s in outline['strokes']:
        pts=np.rint(s['points']).astype(int)
        for i,(a,b) in enumerate(zip(pts,pts[1:])):
            width=(shuimo.press_to_width_mm(s['press'][i])+shuimo.press_to_width_mm(s['press'][i+1]))/2
            cv2.line(canvas,tuple(a),tuple(b),(35,35,35),max(1,round(width/scale)),cv2.LINE_AA)
    imgio.imwrite(str(prefix)+'_预览.png',canvas)
    print('OUTPUT',out);print(p['simplification'])


if __name__=='__main__':main()
