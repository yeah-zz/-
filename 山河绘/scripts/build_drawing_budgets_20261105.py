# -*- coding: utf-8 -*-
"""Keep complete extraction immutable; build comparable 700/500/300 plans."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import copy,json,hashlib,shutil
from collections import Counter
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from module_stroke_core_20261104 import samples,length
from hierarchical_paint_core_20261105 import hierarchy_schedule,order_metrics
from drawing_budget_core_20261105 import select_budgets

ROOT=LAB/'drawing_budgets_20261105';BASE=LAB/'hierarchical_painting_20261105'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def drawing(rows,box=(0,0,496,496),color=(30,43,35),scale=2):
    x,y,x1,y1=box;im=np.full(((y1-y)*scale,(x1-x)*scale,3),255,np.uint8)
    for r in rows:
        p=(np.asarray(r['points'])-np.array([x,y]))*scale
        cv2.polylines(im,[np.rint(p).astype(np.int32)],False,color,2,cv2.LINE_AA)
    return Image.fromarray(im)

def curve_stats(rows):
    ll=[length(np.asarray(r['points'],float)) for r in rows]
    return dict(count=len(rows),short_under_5px=sum(t<5 for t in ll),long_over_25px=sum(t>25 for t in ll),
                centerline_length_px=sum(ll),median_length_px=float(np.median(ll)),phases=dict(Counter(str(r['phase']) for r in rows)))

def raster_coverage(full,part):
    mask=np.zeros((992,992),np.uint8);kept=mask.copy()
    for rows,target in [(full,mask),(part,kept)]:
        for r in rows:cv2.polylines(target,[np.rint(np.asarray(r['points'])*2).astype(np.int32)],False,1,1)
    return float(np.count_nonzero((mask>0)&(cv2.dilate(kept,np.ones((3,3),np.uint8))>0))/max(1,np.count_nonzero(mask)))

def compare(folder,ident,source,variants,box=(0,0,496,496),name='comparison'):
    # Same geometry, same display width across ALL extraction/budget panels.
    boxes=[('原图',Image.open(source).convert('RGB').crop(box)),('完整稿 · '+str(len(variants['full']))+'笔',drawing(variants['full'],box))]
    boxes += [(f'{b}笔预算 · 实际{len(variants[b])}笔',drawing(variants[b],box)) for b in (700,500,300)]
    change=drawing(variants['full'],box,color=(211,102,96));ret=drawing(variants[500],box)
    a=np.array(change);k=np.array(ret);dark=k[:,:,0]<200;a[dark]=k[dark]
    boxes.append(('500笔：黑色保留／红色暂不绘制',Image.fromarray(a)))
    board=Image.new('RGB',(1572,1180),'#f4f1e8');pd=ImageDraw.Draw(board)
    for k,(title,im) in enumerate(boxes):
        col=k%3;row=k//3;x=col*524+12;y=row*562+45
        ratio=min(496/im.width,496/im.height);im=im.resize((round(im.width*ratio),round(im.height*ratio)),Image.Resampling.LANCZOS)
        board.paste(im,(x,y));pd.text((x,y-33),title,font=FONT,fill='#263e34')
    pd.text((12,1138),f'{ident}｜三档为实画细节预算，不改变完整提取稿；均为等宽中心线预览，非实际毛笔效果。',font=SMALL,fill='#263e34')
    board.save(folder/(name+'.jpg'),quality=96)
    if name=='comparison':
        compact=Image.new('RGB',(1572,585),'#f4f1e8');cd=ImageDraw.Draw(compact)
        for k,b in enumerate((700,500,300)):
            im=drawing(variants[b]);compact.paste(im.resize((496,496),Image.Resampling.LANCZOS),(k*524+12,45))
            cd.text((k*524+12,10),f'{b}笔预算 · 实际{len(variants[b])}笔',font=FONT,fill='#263e34')
        cd.text((12,550),'实画候选取舍对照｜省略不代表原线条错误；未验证实际笔宽、墨量或实画用时。',font=SMALL,fill='#263e34')
        compact.save(folder/'tiers.jpg',quality=96)

def build(ident):
    parent=BASE/ident/'plan.json';before_hash=sha(parent);original=read(parent);full=original['curves']
    selections=select_budgets(full);folder=ROOT/ident;folder.mkdir(exist_ok=True)
    variants={'full':full};records=[];summary=[]
    allcases=[('full',list(range(len(full))),None)]+[(str(b),selections[b]['selected_ids'],selections[b]) for b in (700,500,300)]
    for label,ids,selection in allcases:
        scene=ident+'_'+label;dest=ROOT/scene;dest.mkdir(exist_ok=True)
        plan=copy.deepcopy(original);rows=[copy.deepcopy(full[i]) for i in ids]
        for j,r in enumerate(rows):r['complete_curve_id']=ids[j]
        route,order=hierarchy_schedule([{**r,'points':np.asarray(r['points'],float)} for r in rows],plan['modules'])
        for a in route:a['stroke_id']=f'{scene}_c{ids[a["curve"]]:05}'
        # Fair order comparison: retain original full-plan order on SAME selected set.
        oldactions=[a for a in original['actions'] if a['curve'] in set(ids)]
        stats={**original['stats'],'before':curve_stats(full),'after':curve_stats(rows),
            'baseline_order':order_metrics(oldactions),'hierarchy_order':order_metrics(route),
            'complete_input_count':len(full),'requested_budget':int(label) if label!='full' else None,
            'omitted_for_drawing':len(full)-len(rows),'input_raster_coverage_proxy':raster_coverage(full,rows),
            'coverage_is_not_extraction_accuracy':True,'physical_time_estimate':None,
            'inherited_join_count':original['stats']['accepted_continuations'],
            'accepted_continuations':original['stats']['accepted_continuations'],
            'new_joins_in_this_round':0,'short_curve_omission_is_not_complete_stroke_reconstruction':True}
        if selection:stats.update({k:selection[k] for k in ['protected_count','budget_exceeded_for_structure','budget_not_filled_due_to_input_size']})
        plan.update(id=scene,parent=str(parent),parent_sha256=before_hash,curves=rows,actions=route,module_order=order,stats=stats,
            complete_extraction_unchanged=True,selection=selection,ready_for_robot=False,camera_connected=False,
            detail_budget_is_not_line_classification=True,pressure_calibration_available=False)
        for m in plan['modules']:
            m['planned_strokes']=sum(r['module']==m['id'] for r in rows)
            m['order']=order.index(m['id'])+1 if m['id'] in order else None
        write(dest/'plan.json',plan);write(dest/'statistics.json',stats)
        shutil.copy2(BASE/ident/'source.png',dest/'source.png')
        shutil.copy2(BASE/ident/'modules.png',dest/'modules.png')
        image=drawing(rows);image.save(dest/'centerlines.png')
        variants[int(label) if label!='full' else 'full']=rows
        record=dict(id=scene,label=f'{ident} · '+('完整提取稿' if label=='full' else f'{label}笔预算')+f'（{len(rows)}笔）',
                    plan_sha256=sha(dest/'plan.json'),stats=stats)
        records.append(record);summary.append(dict(tier=label,**stats))
    compare(folder,ident,BASE/ident/'source.png',variants)
    if ident=='G03_01':
        for key,box in [('pavilion',(108,198,231,299)),('rock',(178,285,326,389)),('tree',(150,75,480,245))]:
            compare(folder,ident,BASE/ident/'source.png',variants,box,key)
    for label,_,_ in allcases:
        dest=ROOT/(ident+'_'+label)
        shutil.copy2(folder/'comparison.jpg',dest/'comparison.jpg')
        # This link deliberately shows the same-data budget selection, not a
        # fabricated old/new sorting control.
        shutil.copy2(folder/'tiers.jpg',dest/'sequence_comparison.jpg')
    assert sha(parent)==before_hash
    write(folder/'selection.json',dict(complete_plan=str(parent),complete_sha256=before_hash,selections=selections,
        full_plan_unchanged=True,selected_geometry_unchanged=True))
    write(folder/'summary.json',summary)
    print('BUDGET',ident,[(r['id'],r['stats']['after']['count'],r['stats']['after']['short_under_5px']) for r in records],flush=True)
    return records

def main():
    ROOT.mkdir(exist_ok=True);cv2.setNumThreads(2)
    scenes=[]
    for ident in ('G03_01','G05_03','T02'):scenes+=build(ident)
    write(ROOT/'catalog.json',dict(scenes=scenes,default='G03_01_500',robot_connected=False))
    binding=ROOT/'execution_binding.json'
    if not binding.exists():write(binding,dict(scene=None,plan_sha256=None,journal=None))
    template=Path(__file__).with_name('hierarchical_paint_platform_20261105.html').read_text('utf8')
    template=template.replace("o.textContent=({G03_01:","o.textContent=s.label||({G03_01:")
    template=template.replace("plan=p;scene=id;sourceImage", "plan=p;scene=id;$('scene').value=id;sourceImage")
    template=template.replace('山河绘 · 笔序工作台','山河绘 · 实画预算对照')
    template=template.replace('输入细段 → 连续候选','完整稿 → 本档实画候选').replace('原图支持的断段连接','沿用的断段连接（本轮未新增）')
    template=template.replace('同块离开后再次返回','同线集分块顺序检查').replace('查看同线集排序对照','查看700／500／300笔对照')
    template=template.replace('区域是自动分割候选，尚未验证为独立山石；',
        '这是实画细节取舍，完整稿未删除；省略线条不代表误检。区域是未验证的对象候选；')
    (ROOT/'index.html').write_text(template,encoding='utf8')
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['drawing_budget_core_20261105.py','build_drawing_budgets_20261105.py','hierarchical_paint_server_20261105.py']:
        shutil.copy2(Path(__file__).with_name(name),snap/name)
    print('RESULT',ROOT,flush=True)
if __name__=='__main__':main()
