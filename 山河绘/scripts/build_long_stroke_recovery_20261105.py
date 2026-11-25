# -*- coding: utf-8 -*-
"""Full-input continuity experiment, never a drawing-detail budget."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,copy,shutil,colorsys
from collections import Counter
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from module_stroke_core_20261104 import samples,length
from hierarchical_paint_core_20261105 import hierarchy_schedule,order_metrics
from long_stroke_recovery_20261105 import recover_long_strokes

ROOT=LAB/'long_stroke_recovery_20261105';BASE=LAB/'hierarchical_painting_20261105'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf8')
def stats(rows):
    ll=np.array([length(np.asarray(r['points'])) for r in rows])
    return dict(count=len(rows),short_under_5px=int((ll<5).sum()),long_over_25px=int((ll>25).sum()),
        median_length_px=float(np.median(ll)),centerline_length_px=float(ll.sum()))
def drawing(rows,box=(0,0,496,496),colored=False,marks=False):
    x0,y0,x1,y1=box;scale=3;im=np.full(((y1-y0)*scale,(x1-x0)*scale,3),255,np.uint8)
    for i,r in enumerate(rows):
        q=np.asarray(r['points']);p=np.rint((q-[x0,y0])*scale).astype(np.int32)
        color=tuple(round(v*170) for v in colorsys.hsv_to_rgb((i*.618)%1,.8,1)) if colored else (30,43,35)
        cv2.polylines(im,[p],False,color,2,cv2.LINE_AA)
        if marks and np.linalg.norm(q[0]-q[-1])>.1:
            cv2.circle(im,tuple(p[0]),2,(211,134,43),-1,cv2.LINE_AA)
    return im
def panel(folder,ident,rgb,old,new,diag,key='comparison',box=(0,0,496,496)):
    x0,y0,x1,y1=box
    change=drawing(new,box)
    for a in diag['accepted']:
        p=(np.asarray(a['bridge'])-[x0,y0])*3
        cv2.polylines(change,[np.rint(p).astype(np.int32)],False,(20,165,75),4,cv2.LINE_AA)
        center=p[len(p)//2];cv2.circle(change,tuple(np.rint(center).astype(int)),5,(20,165,75),1,cv2.LINE_AA)
    ims=[(rgb[y0:y1,x0:x1],'原图'),(drawing(old,box,True,True),'之前分笔（颜色分笔／橙点起笔）'),
         (drawing(new,box,True,True),'恢复后分笔（同样显示宽度）'),(change,'绿色：本轮连接／共享段拼接位置')]
    board=Image.new('RGB',(2080,610),'#f4f1e8');d=ImageDraw.Draw(board)
    for k,(im,title) in enumerate(ims):
        image=Image.fromarray(im);ratio=min(496/image.width,496/image.height)
        image=image.resize((round(image.width*ratio),round(image.height*ratio)),Image.Resampling.LANCZOS)
        board.paste(image,(k*520+10,43));d.text((k*520+10,10),title,font=FONT,fill='#263e34')
    d.text((10,553),f'{ident}｜整幅 {len(old)} → {len(new)} 笔；本轮不省略输入细节。颜色不表示皴法。',font=SMALL,fill='#263e34')
    d.text((10,582),'连接为原图支持的离线候选，尚无逐笔专家真值；中心线预览，非毛笔实画。',font=SMALL,fill='#263e34')
    board.save(folder/(key+'.jpg'),quality=96)
    if key=='comparison':Image.fromarray(drawing(new)).save(folder/'centerlines.png')

def build(ident):
    folder=ROOT/ident;folder.mkdir(exist_ok=True);parent=BASE/ident/'plan.json';digest=sha(parent)
    source=BASE/ident/'source.png';rgb=np.array(Image.open(source).convert('RGB'));old=read(parent);rows=old['curves']
    new,diag=recover_long_strokes(rgb,rows)
    for r in new:
        ids=r['complete_input_ids'];r['source_curve_ids']=[j for i in ids for j in rows[i]['source_curve_ids']]
        phase=min(rows[i]['phase'] for i in ids)
        r.update(phase=phase,phase_label=old['curves'][ids[np.argmin([rows[i]['phase'] for i in ids])]]['phase_label'],
            role_verified=False,length_px=length(r['points']),
            source_support=float(np.mean([rows[i]['source_support'] for i in ids])),
            boundary_fraction=float(np.mean([rows[i]['boundary_fraction'] for i in ids])))
    q=np.concatenate([samples(np.asarray(r['points']),.25) for r in rows])
    tree=cKDTree(np.concatenate([samples(r['points'],.25) for r in new]));distance=tree.query(q)[0]
    if distance.max()>.8:raise ValueError('Original geometry lost beyond overlap tolerance')
    route,order=hierarchy_schedule(new,old['modules']);before=stats(rows);after=stats(new)
    report=dict(id=ident,before=before,after=after,new_joins=len(diag['accepted']),join_types=diag['types'],
        pending=len(diag['pending']),joins_in_uncertain_same_region=diag['joins_in_uncertain_same_region'],
        old_sample_distance_quantiles_px=np.quantile(distance,[.5,.95,.99,1]).tolist(),
        old_sample_coverage_within_0p8px=float((distance<=.8).mean()),
        no_detail_budget_or_length_pruning=True,expert_stroke_identity_accuracy=None,
        new_weights_trained=False,physical_improvement_verified=False,ready_for_robot=False,
        baseline_order=order_metrics(old['actions']),hierarchy_order=order_metrics(route),
        accepted_continuations=len(diag['accepted']),modules=old['stats']['modules'],
        object_proposals=old['stats']['object_proposals'],unresolved_regions=old['stats']['unresolved_regions'])
    plan=copy.deepcopy(old);plan.update(id=ident,curves=new,actions=route,module_order=order,stats=report,
        parent=str(parent),parent_sha256=digest,continuity_not_budgeting=True,ready_for_robot=False,
        artist_stroke_identity_verified=False,new_weights_trained=False)
    for m in plan['modules']:
        m['planned_strokes']=sum(r['module']==m['id'] for r in new);m['order']=order.index(m['id'])+1
    write(folder/'plan.json',plan);write(folder/'statistics.json',report);write(folder/'diagnostics.json',diag)
    shutil.copy2(source,folder/'source.png');shutil.copy2(BASE/ident/'modules.png',folder/'modules.png')
    panel(folder,ident,rgb,rows,new,diag)
    if ident=='G03_01':
        for key,box in [('rock',(178,285,326,389)),('pavilion',(108,198,231,299)),('tree',(150,75,480,245))]:
            panel(folder,ident,rgb,rows,new,diag,key,box)
    shutil.copy2(folder/'comparison.jpg',folder/'sequence_comparison.jpg')
    assert sha(parent)==digest
    print('LONG_RECOVERY',json.dumps(report,ensure_ascii=False),flush=True)
    return dict(id=ident,label=f'{ident} · 长笔触恢复（{len(new)}笔）',plan_sha256=sha(folder/'plan.json'),stats=report)

def main():
    ROOT.mkdir(exist_ok=True);cv2.setNumThreads(2)
    records=[build(i) for i in ('G03_01','G05_03','T02')]
    write(ROOT/'catalog.json',dict(scenes=records,default='G03_01',robot_connected=False))
    if not (ROOT/'execution_binding.json').exists():write(ROOT/'execution_binding.json',dict(scene=None,journal=None,plan_sha256=None))
    page=Path(__file__).with_name('hierarchical_paint_platform_20261105.html').read_text('utf8')
    page=page.replace("o.textContent=({G03_01:","o.textContent=s.label||({G03_01:")
    page=page.replace("plan=p;scene=id;sourceImage","plan=p;scene=id;$('scene').value=id;sourceImage")
    page=page.replace('山河绘 · 笔序工作台','山河绘 · 长笔触恢复').replace('输入细段 → 连续候选','完整输入 → 长笔触候选')
    page=page.replace('查看同线集排序对照','查看整幅连接对照')
    page=page.replace('区域是自动分割候选，尚未验证为独立山石；',
        '本轮未按笔数删减细节；绿色对照标出连接位置，尚无真实分笔准确率。区域仍是待验证的对象候选；')
    (ROOT/'index.html').write_text(page,encoding='utf8')
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ('long_stroke_recovery_20261105.py','build_long_stroke_recovery_20261105.py'):
        shutil.copy2(Path(__file__).with_name(name),snap/name)
if __name__=='__main__':main()
