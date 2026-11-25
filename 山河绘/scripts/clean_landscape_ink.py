"""Repair traced burrs in the three source landscapes, without changing color."""
from pathlib import Path
import sys,json,copy,hashlib,shutil
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from web.catalog import ARTWORKS
from painting_session import save
from ink_control import refresh_stats
from artwork_order import ink_payload
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.validate_source_washes import sample_segments

PROTECTED_ROLES={'title','pavilion','stairs','boat','water','branch','willow_fine'}
REPAIR_ROLES={'near_outline','mountain_outline','far_outline','foliage_outline','texture'}
MINIMUM_MM={'texture':4.5,'near_outline':2.8,'mountain_outline':2.5,'far_outline':3.2,'foliage_outline':0}
MAX_DEVIATION_MM=.85


def smooth(stroke,scale):
    s=copy.deepcopy(stroke);p=np.asarray(s['points'],float)
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]*scale
    if arc[-1]<2 or len(p)<5:return s
    closed=np.linalg.norm(p[-1]-p[0])*scale<.15
    # Preserve endpoints; no spline overshoot and no connection between strokes.
    t=np.linspace(0,arc[-1],max(5,int(np.ceil(arc[-1]/.20))+1))
    q=np.column_stack([np.interp(t,arc,p[:,axis]) for axis in (0,1)])
    filtered=gaussian_filter1d(q,sigma=1.6,axis=0,mode='wrap' if closed else 'nearest')
    delta=filtered-q;norm=np.linalg.norm(delta,axis=1)*scale
    filtered=q+delta*np.minimum(1,.3/np.maximum(norm,1e-9))[:,None]
    filtered[0]=p[0];filtered[-1]=p[-1]
    simplified=cv2.approxPolyDP(filtered.astype(np.float32).reshape(-1,1,2),.5/scale,closed).reshape(-1,2).astype(float)
    if closed:
        # Rotate the simplified loop back to its original starting landmark.
        idx=np.argmin(np.linalg.norm(simplified-p[0],axis=1));simplified=np.roll(simplified,-idx,axis=0)
        simplified=np.vstack([p[0],simplified,p[-1]])
    else:simplified[0]=p[0];simplified[-1]=p[-1]
    new_arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(simplified,axis=0),axis=1))]*scale
    u=np.linspace(0,new_arc[-1],max(2,int(np.ceil(new_arc[-1]/.25))+1))
    points=np.column_stack([np.interp(u,new_arc,simplified[:,axis]) for axis in (0,1)])
    dense_old=sample_segments(p,.05/scale);dense_new=sample_segments(points,.05/scale)
    forward=cKDTree(dense_old).query(dense_new)[0].max()*scale
    backward=cKDTree(dense_new).query(dense_old)[0].max()*scale
    if max(forward,backward)>MAX_DEVIATION_MM:return s
    s['points']=points.tolist()
    parameter=u/max(u[-1],1e-9)*arc[-1]
    s['press']=np.interp(parameter,arc,s['press']).tolist()
    s['speed']=np.interp(parameter,arc,s['speed']).tolist()
    s['trace_cleanup']=dict(method='bounded-smoothing-and-polyline-simplification',max_deviation_mm=round(max(forward,backward),4),original_id=s['id'])
    return refresh_stats(s,scale)


def audit_cleanup(candidate,baseline):
    candidate=Path(candidate);baseline=Path(baseline)
    new=json.loads(candidate.read_text(encoding='utf-8'));old=json.loads(baseline.read_text(encoding='utf-8'))
    before=stroke_data(old);after=stroke_data(new)
    assert before['color']==after['color'],'上色路径发生改变'
    scale=new['ink_job']['meta']['ink_execution']['mm_per_px']
    originals={s['id']:s for s in before['black']};current={s['id']:s for s in after['black']}
    assert len(current)==len(after['black']) and set(current)<=set(originals)
    maximum=0;modified=0
    for sid,prior in originals.items():
        if prior['role'] in PROTECTED_ROLES:assert current[sid]==prior,'受保护结构发生改变'
        elif sid not in current:
            assert prior['role'] in REPAIR_ROLES
            assert prior['length_mm']<MINIMUM_MM[prior['role']] or (prior['role']=='texture' and prior['length_mm']<10)
    for sid,s in current.items():
        prior=originals[sid]
        assert len(s['points'])==len(s['press'])==len(s['speed'])
        assert s['role']==prior['role'] and s['drawing_region']==prior['drawing_region']
        if s['points']!=prior['points']:
            assert s['role'] in REPAIR_ROLES
            p=np.asarray(s['points']);q=np.asarray(prior['points'])
            assert np.allclose(p[[0,-1]],q[[0,-1]],atol=1e-5)
            a=sample_segments(p,.05/scale);b=sample_segments(q,.05/scale)
            deviation=max(cKDTree(a).query(b)[0].max(),cKDTree(b).query(a)[0].max())*scale
            assert deviation<=MAX_DEVIATION_MM+.001
            assert min(s['press'])>=min(prior['press'])-1e-8 and max(s['press'])<=max(prior['press'])+1e-8
            maximum=max(maximum,float(deviation));modified+=1
        length=np.linalg.norm(np.diff(s['points'],axis=0),axis=1).sum()*scale
        assert abs(length-s['length_mm'])<1e-5
    assert new['workflow']['source_revision']['stroke_data_sha256']==digest(stroke_data(new))
    assert new['workflow']['modules']['outline']['strokes']==after['black']
    from scripts.order_selected_artworks import ORDERS
    number=new['workflow']['source_revision']['artwork_number']
    ranks=[ORDERS[number].index(s['drawing_region']) for s in after['black']]
    assert ranks==sorted(ranks)
    from run_json_painting import load_artwork
    from web.execution import prepare
    prepare(candidate);load_artwork(candidate)
    return dict(robot_connected=False,black_before=len(before['black']),black_after=len(after['black']),
        removed_strokes=len(originals)-len(current),smoothed_strokes=modified,max_path_deviation_mm=round(maximum,4),
        colors_exactly_preserved=True,title_buildings_boat_branches_and_water_exactly_preserved=True,
        drawing_order_preserved=True,waypoints_regenerated_and_runtime_validation_passed=True,
        removed_by_role=dict(Counter(originals[sid]['role'] for sid in set(originals)-set(current))))


def build(number):
    path=ROOT/ARTWORKS[number][3];package=path.parent.parent
    from web.execution import prepare
    prepare(path)
    old=json.loads(path.read_text(encoding='utf-8'));new=copy.deepcopy(old)
    scale=old['ink_job']['meta']['ink_execution']['mm_per_px'];ink=stroke_data(old)['black'];clean=[];removed=[]
    # A short segment may be a necessary link between two long contours.
    # Preserve those links instead of leaving a new gap in a mountain ridge.
    endpoints=[];owners=[]
    for s in ink:
        if s['role'] in ('near_outline','mountain_outline','far_outline') and s['length_mm']>=4:
            endpoints.extend(np.asarray(s['points'])[[0,-1]]);owners.extend([s['id'],s['id']])
    endpoint_tree=cKDTree(endpoints)
    for s in ink:
        role=s['role']
        if role in PROTECTED_ROLES:clean.append(copy.deepcopy(s));continue
        p=np.asarray(s['points']);span=np.linalg.norm(np.ptp(p,axis=0))*scale
        tiny_loop=role=='texture' and np.linalg.norm(p[-1]-p[0])*scale<.7 and span<3 and s['length_mm']<10
        bridge=False
        if role in ('near_outline','mountain_outline','far_outline'):
            neighbors=endpoint_tree.query_ball_point(p[[0,-1]],.8/scale)
            ends=[{owners[i] for i in hits if owners[i]!=s['id']} for hits in neighbors]
            bridge=bool(ends[0] and ends[1] and any(a!=b for a in ends[0] for b in ends[1]))
        if (s['length_mm']<MINIMUM_MM.get(role,0) and not bridge) or tiny_loop:
            removed.append(dict(id=s['id'],role=role,length_mm=s['length_mm'],region=s['drawing_region']));continue
        clean.append(smooth(s,scale) if role in REPAIR_ROLES else copy.deepcopy(s))
    new['ink_job']=ink_payload(old['ink_job'],clean)
    plan=new['workflow'];plan['modules']['outline'].update(strokes=copy.deepcopy(clean),stroke_count=len(clean))
    revision=plan['source_revision'];revision['black_trace_cleanup']=dict(method='remove-short-fragments-and-smooth-trace-burrs',thresholds_mm=MINIMUM_MM,
        max_deviation_mm=MAX_DEVIATION_MM,black_before=len(ink),black_after=len(clean),removed_strokes=removed,
        protected_roles=sorted(PROTECTED_ROLES),colors_exactly_preserved=True)
    text='提线修复：清除山石和远山短碎线、小纹理闭环，有限幅度修顺轮廓凸起；题字、建筑、台阶、舟体、主枝与柳叶细线逐笔保留；上一版全部蓝绿上色保持不变。'
    revision['note']=revision['note'].split('提线修复：')[0]+text
    revision['stroke_data_sha256']=digest(stroke_data(new));plan['note']=revision['note']
    pending=path.parent/'待应用_提线清理';pending.mkdir(exist_ok=True)
    save(pending/'painting.json',new)
    result=audit_cleanup(pending/'painting.json',path);save(pending/'提线修复校验.json',result)
    from scripts.three_source_washes import previews
    previews(old,new,pending,ARTWORKS[number][1],black_cleanup=True)
    # Preserve the diagnostic-only pure-ink preview alongside the full version.
    blank=copy.deepcopy(new);blank['workflow']['modules']['color']['strokes']=[]
    from scripts.source_wash_comparison import render
    render(blank).save(pending/'纯勾线.png')
    save(pending/'应用记录.json',dict(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False,
        policy='Apply reviewed trace cleanup only with idle session and wash locks absent'))
    (pending/'提线修复说明.md').write_text('# '+ARTWORKS[number][1]+' · 提线修复\n\n'+text+'\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(ARTWORKS[number][1],json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    for number in (1,2,3):build(number)
