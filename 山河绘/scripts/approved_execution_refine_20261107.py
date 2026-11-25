# -*- coding: utf-8 -*-
"""Preserve accepted paths; audit source joins, object order and finite-width risk.

All values are image-space hypotheses, never pressure/velocity robot commands.
This is deterministic offline planning with no new training or controller access.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,copy,hashlib,shutil
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from landscape_stroke_refine_20261104 import deduplicate
from long_stroke_recovery_20261105 import recover_long_strokes
from module_stroke_core_20261104 import samples,length
from hierarchical_paint_core_20261105 import hierarchy_schedule,order_metrics
from source_stroke_width_20261104 import width_profile

ROOT=LAB/'approved_execution_refine_20261107';BASE=LAB/'drawing_budgets_20261105'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
WIDTHS=(.6,1.2,2.0)

def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf8')

def mask_for(path,width,scale=3):
    m=np.zeros((496*scale,496*scale),np.uint8)
    cv2.polylines(m,[np.rint(np.asarray(path)*scale).astype(np.int32)],False,1,max(1,round(width*scale)),cv2.LINE_8)
    return m

def hit_start(path,field,width=1.2,scale=3):
    p=samples(path,.2);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    q=p[arc<=min(1.2,arc[-1])]
    xy=np.rint(q*scale).astype(int);xy=np.clip(xy,0,496*scale-1)
    return float((field[xy[:,1],xy[:,0]]>0).mean())

def schedule_contact_aware(curves,modules,reference_route,width=1.2):
    """Object/phase order fixed; change ONLY within-phase choice and orientation."""
    skeleton_order=[]
    for a in reference_route:
        k=(a['module'],a['phase'])
        if k not in skeleton_order:skeleton_order.append(k)
    cursor=np.array([0.,495.]);route=[];recent=[]
    for mid,phase in skeleton_order:
        todo=[i for i,r in enumerate(curves) if r['module']==mid and r['phase']==phase]
        while todo:
            field=np.zeros((1488,1488),np.uint8)
            for p in recent[-12:]:field|=mask_for(p,width)
            choices=[]
            for i in todo:
                r=curves[i];p=np.asarray(r['points']);options=[]
                for reverse in (False,True):
                    q=p[::-1] if reverse else p;risk=hit_start(q,field,width);travel=float(np.linalg.norm(q[0]-cursor))
                    options.append((risk*6+travel/80,risk,travel,reverse))
                _,risk,travel,rev=min(options)
                priority=length(p)**.65/(1+travel/35)/(1+4*risk)
                choices.append((priority,-i,i,risk,travel,rev))
            _,_,i,risk,travel,rev=max(choices);todo.remove(i);r=curves[i]
            p=np.asarray(r['points']);q=p[::-1] if rev else p
            route.append(dict(sequence=len(route)+1,stroke_id=f's{i:05}',curve=i,module=mid,phase=phase,
                reverse=bool(rev),air_travel_px=travel,length_px=length(p),texture_bundle=None,
                predicted_recent_start_contact_fraction=risk,direction_artist_verified=False))
            recent.append(q);cursor=q[-1]
    assert sorted(a['curve'] for a in route)==list(range(len(curves)))
    return route

def coverage_audit(curves,route,width):
    counts=np.zeros((1488,1488),np.uint16);recent=[];starts=[];startends=[]
    for a in route:
        p=np.asarray(curves[a['curve']]['points']);p=p[::-1] if a['reverse'] else p
        field=np.zeros(counts.shape,np.uint8)
        for q in recent[-12:]:field|=mask_for(q,width)
        starts.append(hit_start(p,field,width));startends.append([p[0].tolist(),p[-1].tolist()])
        counts+=mask_for(p,width);recent.append(p)
    painted=int((counts>0).sum());overlap=int((counts>1).sum())
    return dict(assumed_width_496px=width,distinct_stroke_overlap_area_px=overlap/9,
        overlap_fraction_of_painted_area=overlap/max(1,painted),
        starts_touching_last12=sum(x>0 for x in starts),mean_start_contact_fraction=float(np.mean(starts)),
        start_end_locations=startends,not_ink_deposition_simulation=True,physical_calibration=None)

def draw(rows,color=(30,43,35)):
    out=np.full((992,992,3),255,np.uint8)
    for r in rows:cv2.polylines(out,[np.rint(np.asarray(r['points'])*2).astype(np.int32)],False,color,1,cv2.LINE_AA)
    return Image.fromarray(out)

def draw_profiles(profiles,route,taper=True):
    """Image width illustration. No assumed pressure-to-width mapping."""
    canvas=np.full((1488,1488,3),255,np.uint8);uncertain=np.zeros((1488,1488),np.uint8)
    for a in route:
        r=profiles[a['curve']];q=np.asarray(r['points']);w=np.asarray(r['width_px']).copy();obs=np.asarray(r['observed'])
        if a['reverse']:q=q[::-1];w=w[::-1];obs=obs[::-1]
        if taper and np.linalg.norm(q[0]-q[-1])>.1:
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))];ramp=min(1.2,arc[-1]/3)
            envelope=np.minimum(np.clip(arc/max(ramp,1e-6),0,1),np.clip((arc[-1]-arc)/max(ramp,1e-6),0,1))
            w*=.25+.75*envelope
        for k in range(len(q)-1):
            d=q[k+1]-q[k];n=np.array([-d[1],d[0]])/max(np.linalg.norm(d),1e-8)
            poly=np.array([q[k]+n*w[k]/2,q[k+1]+n*w[k+1]/2,q[k+1]-n*w[k+1]/2,q[k]-n*w[k]/2])*3
            cv2.fillConvexPoly(canvas,np.rint(poly).astype(np.int32),(35,42,37),cv2.LINE_AA)
            if not (obs[k] and obs[k+1]):cv2.line(uncertain,tuple(np.rint(q[k]*3).astype(int)),tuple(np.rint(q[k+1]*3).astype(int)),1,2)
    return Image.fromarray(canvas),uncertain

def build(ident):
    folder=ROOT/ident;folder.mkdir(exist_ok=True);source=BASE/ident/'plan.json';digest=sha(source);old=read(source)
    rgb=np.array(Image.open(BASE/ident/'source.png').convert('RGB'))
    rows=[{**r,'points':np.asarray(r['points'],float)} for r in old['curves']]
    dedup,duplicate_diag=deduplicate(rgb,[r['points'] for r in rows],[r['module'] for r in rows])
    ds=[]
    for r in dedup:
        ids=r['source_curve_ids'];phase=min(rows[i]['phase'] for i in ids);base=next(rows[i] for i in ids if rows[i]['phase']==phase)
        ds.append({**base,'points':r['points'],'accepted_baseline_ids':ids,'source_curve_ids':ids})
    joined,join_diag=recover_long_strokes(rgb,ds,max_gap=4.)
    for r in joined:
        ids=[j for i in r['complete_input_ids'] for j in ds[i]['accepted_baseline_ids']]
        phase=min(rows[i]['phase'] for i in ids);base=next(rows[i] for i in ids if rows[i]['phase']==phase)
        r.update(accepted_baseline_ids=ids,phase=phase,phase_label=base['phase_label'],role_verified=False,
            source_curve_ids=[v for i in ids for v in rows[i]['source_curve_ids']],length_px=length(r['points']))
    raw_route,order=hierarchy_schedule(joined,old['modules']);route=schedule_contact_aware(joined,old['modules'],raw_route)
    audit_old={str(w):coverage_audit(rows,old['actions'],w) for w in WIDTHS}
    audit_reference={str(w):coverage_audit(joined,raw_route,w) for w in WIDTHS}
    audit_new={str(w):coverage_audit(joined,route,w) for w in WIDTHS}
    # Same-line-set control: if scheduling raises its own assumed-width metric,
    # retain the original schedule. Do not claim a gain by deleting linework.
    used_contact_policy=audit_new['1.2']['starts_touching_last12']<audit_reference['1.2']['starts_touching_last12']
    if not used_contact_policy:route=raw_route;audit_new=audit_reference
    before=np.concatenate([samples(r['points'],.25) for r in rows]);after=np.concatenate([samples(r['points'],.25) for r in joined])
    distances=cKDTree(after).query(before)[0];assert distances.max()<=1.3
    accounted=sorted(i for r in joined for i in r['accepted_baseline_ids']);assert accounted==list(range(len(rows)))
    stats={**old['stats'],'before':dict(count=len(rows)), 'after':dict(count=len(joined)),
        'accepted_continuations':len(join_diag['accepted']), 'baseline_order':order_metrics(old['actions']), 'hierarchy_order':order_metrics(route)}
    plan=copy.deepcopy(old);plan.update(id=ident,parent=str(source),parent_sha256=digest,curves=joined,actions=route,module_order=order,stats=stats,
        adopted=False,ready_for_robot=False,new_weights_trained=False,pressure_mm=None,
        artist_stroke_identity_verified=False,contact_aware_schedule_used=used_contact_policy,
        no_additional_detail_budget=True,baseline_sample_max_distance_px=float(distances.max()))
    for m in plan['modules']:
        m['order']=order.index(m['id'])+1 if m['id'] in order else 0;m['planned_strokes']=sum(r['module']==m['id'] for r in joined)
    save(folder/'plan.json',plan);save(folder/'duplicates.json',duplicate_diag);save(folder/'joins.json',join_diag)
    profiles=[width_profile(rgb,r['points'],recenter=False) for r in joined]
    save(folder/'width_targets.json',dict(profiles=profiles,physical_width_measurement=False,pressure_mm=None,
        target_only=True,endpoint_taper='optional display envelope; not calibrated brush execution'))
    band,unknown=draw_profiles(profiles,route,True);band.save(folder/'width_target.png')
    display=draw(joined);changes=np.array(display)
    for e in join_diag['accepted']:cv2.polylines(changes,[np.rint(np.asarray(e['bridge'])*2).astype(np.int32)],False,(35,145,75),3,cv2.LINE_AA)
    heat=np.array(display);c=np.zeros((1488,1488),np.uint16)
    for r in joined:c+=mask_for(r['points'],1.2)
    h=cv2.resize((c>1).astype(np.uint8),(992,992),interpolation=cv2.INTER_NEAREST)>0;heat[h]=[205,95,45]
    panels=[(Image.fromarray(rgb),'原图'),(draw(rows),f'已认可基线 {len(rows)}笔'),(display,f'保留细节的连续性候选 {len(joined)}笔'),
        (Image.fromarray(changes),'绿：基于原图的保守修正'),(band,'原图相对笔宽＋起收笔目标 · 非实画'),(Image.fromarray(heat),'橙：假设1.2像素笔宽的交叠区域')]
    board=Image.new('RGB',(1560,1150),'#f4f1e8');d=ImageDraw.Draw(board)
    for j,(im,label) in enumerate(panels):
        x=j%3*520;y=j//3*555;board.paste(im.resize((496,496),Image.Resampling.LANCZOS),(x+10,y+43));d.text((x+10,y+10),label,font=FONT,fill='#263e34')
    d.text((10,1117),'未新增删线预算；保守几何修正不等于模型突破。图像笔宽与起收笔为目标，未实测、未映射毛笔下压。',font=SMALL,fill='#263e34')
    board.save(folder/'comparison.jpg',quality=95);shutil.copy2(folder/'comparison.jpg',folder/'sequence_comparison.jpg')
    for name in ('source.png','modules.png'):shutil.copy2(BASE/ident/name,folder/name)
    summary=dict(id=ident,before=len(rows),after=len(joined),duplicate_paths_removed=len(rows)-len(ds),
        conservative_joins=len(join_diag['accepted']),join_types=join_diag['types'],detail_budget_removals=0,
        baseline_max_deviation_px=float(distances.max()),baseline_samples_near_candidate=float((distances<=.8).mean()),
        module_reentries=order_metrics(route)['module_reentries'],contact_policy_used=used_contact_policy,
        before_risk=audit_old,same_curve_set_control_risk=audit_reference,after_risk=audit_new,
        width_observed_sample_fraction=float(np.mean(np.concatenate([r['observed'] for r in profiles]))),
        original_unchanged=sha(source)==digest,new_training=False,physical_improvement_verified=False)
    save(folder/'summary.json',summary);assert sha(source)==digest
    print('PRESERVED',ident,len(rows),'->',len(joined),'starts',audit_reference['1.2']['starts_touching_last12'],'->',audit_new['1.2']['starts_touching_last12'],flush=True)
    return dict(id=ident,label=f'{ident} · 保留画面{len(joined)}笔候选',plan_sha256=sha(folder/'plan.json')),summary

def main():
    ROOT.mkdir(exist_ok=True);cv2.setNumThreads(2);records=[];results=[]
    for ident in ('G03_01_500','G05_03_500','T02_500'):
        r,s=build(ident);records.append(r);results.append(s)
    save(ROOT/'catalog.json',dict(scenes=records,default='G03_01_500',robot_connected=False));save(ROOT/'results.json',results)
    save(ROOT/'execution_binding.json',dict(scene=None,journal=None,plan_sha256=None))
    page=Path(__file__).with_name('hierarchical_paint_platform_20261105.html').read_text('utf8')
    page=page.replace('o.textContent=({G03_01:','o.textContent=s.label||({G03_01:').replace('plan=p;scene=id;sourceImage',"plan=p;scene=id;$('scene').value=id;sourceImage")
    page=page.replace('输入细段 → 连续候选','认可基线 → 保留细节候选').replace('原图支持的断段连接','保守续接候选（未实画验证）')
    page=page.replace('<footer>','<footer><p><a href="/report.html">查看笔宽、交叠与原图对照</a></p>')
    (ROOT/'index.html').write_text(page,encoding='utf8')
    html='''<!doctype html><meta charset="utf-8"><title>认可基线上的离线修正</title><style>body{font:16px/1.7 Microsoft YaHei,system-ui;background:#f4f1e8;color:#243e33;padding:24px;max-width:1560px;margin:auto}img{width:100%}pre{white-space:pre-wrap;background:white;padding:16px;font-size:13px}section{margin:40px 0}a{color:#286246}</style><h1>保留你认可的画面，修正连续性与落笔顺序</h1><p>本轮以认可500笔版为输入，没有再次缩减细节预算。修正是既有原图证据与几何约束的保守应用，无新模型训练；连接并未证明为画家同一笔。分块仍是自动对象候选，皴法类别尚未验证。</p><p>图像笔宽是原图横截面估计，测不清的位置保留未知／插值标记。0.6／1.2／2.0像素为敏感性假设，不是当前毛笔实测。起收笔包络仅是离线目标，没有换算Z下压、行笔速度或含墨量。交叠面积不等于实际晕染，改变笔序也不会消除固有几何重叠。</p><p><a href="/">打开分块分层笔序回放（非实画）</a></p>'''
    for r in results:
        i=r['id'];brief={k:v for k,v in r.items() if k not in ('before_risk','same_curve_set_control_risk','after_risk')}
        risk={name:{w:{k:v for k,v in value.items() if k!='start_end_locations'} for w,value in r[name].items()} for name in ('before_risk','same_curve_set_control_risk','after_risk')}
        html+=f'<section><h2>{i} · {r["before"]} → {r["after"]}笔</h2><img src="/{i}/comparison.jpg"><pre>{json.dumps(brief,ensure_ascii=False,indent=2)}\n{json.dumps(risk,ensure_ascii=False,indent=2)}</pre></section>'
    (ROOT/'report.html').write_text(html,encoding='utf8');(ROOT/'source_snapshot').mkdir(exist_ok=True);shutil.copy2(Path(__file__),ROOT/'source_snapshot'/Path(__file__).name)

if __name__=='__main__':main()
