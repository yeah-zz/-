"""Independently check saved contact segments and protected original strokes."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree
from scripts.consolidate_selected_artworks import digest,stroke_data


def sample_segments(points,step):
    # Deliberately separate from the generator's sampler.
    result=[]
    for a,b in zip(points[:-1],points[1:]):
        a=np.asarray(a,float);b=np.asarray(b,float)
        n=max(1,int(np.ceil(np.linalg.norm(b-a)/step)))
        result.extend(a+(b-a)*t for t in np.linspace(0,1,n+1))
    return np.asarray(result)


def audit_candidate(candidate,baseline):
    candidate=Path(candidate);baseline=Path(baseline)
    new=json.loads(candidate.read_text(encoding='utf-8'));old=json.loads(baseline.read_text(encoding='utf-8'))
    ink=new['ink_job']['meta']['ink_execution']['base_strokes'];old_ink=old['ink_job']['meta']['ink_execution']['base_strokes']
    original={s['id']:s for s in old_ink}
    assert set(original)=={s['id'] for s in ink},'勾线笔触ID发生改变'
    for s in ink:
        prior=original[s['id']]
        assert s['role']==prior['role']
        assert np.allclose(s['points'],prior['points'],rtol=0,atol=1e-8),'原勾线轨迹发生改变'
        assert s['press']==prior['press'] and s['speed']==prior['speed'],'原勾线下压或速度发生改变'
    colors=new['workflow']['modules']['color']['strokes'];old_colors=old['workflow']['modules']['color']['strokes']
    kept=lambda ss:{s['id']:s for s in ss if s.get('role')!='terrain_wash'}
    assert kept(colors)==kept(old_colors),'叶团或已确认水纹改变'
    revision=new['workflow']['source_revision']
    assert revision['stroke_data_sha256']==digest(stroke_data(new))
    scale=new['ink_job']['meta']['ink_execution']['mm_per_px'];tr=revision['source_transform']
    pixel_mm=scale*tr['scale'];radius=int(np.ceil(revision['terrain_centerline_inset_mm']/pixel_mm))
    masks=np.load(candidate.parent/'原图色带禁涂掩膜.npz')
    trunk=masks['trunk'];pavilion=masks['pavilion'];major_lines=masks['line_clear'];h,w=trunk.shape
    washes=[s for s in colors if s.get('role')=='terrain_wash']
    sampled=0;minimum=float('inf');max_length=0.;pigments={'blue':[],'green':[]};painted_area={}
    distances={}
    for s in washes:
        assert s['color'] in pigments
        assert all(v==.30 for v in s['press'])
        assert len(s['points'])==len(s['press'])==len(s['speed'])
        p=np.asarray(s['points']);length=np.linalg.norm(np.diff(p,axis=0),axis=1).sum()*scale
        assert abs(length-s['length_mm'])<1e-5
        assert length<=50.001,'接触路径超过单笔长度限制'
        max_length=max(max_length,length)
        dense=sample_segments(p,.05/scale)
        local=(dense-np.asarray(tr['offset']))/tr['scale'];q=np.rint(local).astype(int)
        assert np.all((q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h))
        sampled+=len(q);i=s['terrain_region_index'];j=s['source_ribbon_index']
        assert np.all(masks['safe_%d'%i][q[:,1],q[:,0]])
        assert np.all(masks['ribbon_%d'%j][q[:,1],q[:,0]]),'接触段或连接段穿过色带留白'
        assert not np.any(trunk[q[:,1],q[:,0]])
        assert not np.any(pavilion[q[:,1],q[:,0]])
        assert not np.any(major_lines[q[:,1],q[:,0]])
        if i not in distances:
            area=masks['envelope_%d'%i]
            # Zero padding also enforces the original image's cropped edge.
            padded=np.pad(area,1)
            distances[i]=cv2.distanceTransform(padded,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
        clearance=distances[i][q[:,1],q[:,0]].min()*pixel_mm
        assert clearance+pixel_mm>=revision['terrain_centerline_inset_mm'],'估计笔宽与内收余量不足'
        minimum=min(minimum,float(clearance));pigments[s['color']].append(local)
        # Estimated visible footprint, for reporting coverage, never rendering a
        # fictitious watercolor texture or reducing pressure in the preview.
        footprint=painted_area.setdefault(i,np.zeros((h,w),np.uint8))
        cv2.polylines(footprint,[np.rint(local).astype(np.int32)],False,1,max(1,round(s['brush_width_mm_estimate']/pixel_mm)))
    assert washes and all(pigments.values())
    blue=np.vstack(pigments['blue']);green=np.vstack(pigments['green'])
    gap=float(cKDTree(blue).query(green)[0].min()*pixel_mm)
    assert gap+.20>=revision['terrain_blue_green_centerline_gap_mm'],'蓝绿色带间距不足'
    from artwork_order import ink_stages
    from scripts.order_selected_artworks import ORDERS
    parts=ink_stages(new['ink_job'],new['workflow'])
    assert len(parts['title'])==13 and all(s['role']=='title' for s in parts['title'])
    assert len([s for s in ink if s.get('role')=='bark_texture'])==3
    for pigment in ('blue','green'):
        ranks=[ORDERS[0].index(s['drawing_region']) for s in colors if s['color']==pigment]
        assert ranks==sorted(ranks),'同色完整部位顺序改变'
    coverage=[dict(region_index=i,estimated_coverage=round(float((m&masks['envelope_%d'%i]).sum()/max(1,masks['envelope_%d'%i].sum())),4))
              for i,m in sorted(painted_area.items())]
    counts=dict(black=len(parts['black']),**dict(Counter(s['color'] for s in colors)),title=len(parts['title']))
    def terrain_area(bundle):
        width,height=bundle['ink_job']['meta']['image_size'];footprint=np.zeros((height,width),np.uint8)
        for st in bundle['workflow']['modules']['color']['strokes']:
            if st.get('role')=='terrain_wash':
                cv2.polylines(footprint,[np.rint(st['points']).astype(np.int32)],False,1,max(1,round(st['brush_width_mm_estimate']/scale)))
        return int(footprint.sum())
    before_area=terrain_area(old);after_area=terrain_area(new)
    return dict(robot_connected=False,counts=counts,total_strokes=sum(counts.values()),
        estimated_terrain_footprint_before_px=before_area,estimated_terrain_footprint_after_px=after_area,
        estimated_terrain_footprint_ratio=round(after_area/before_area,3) if before_area else None,
        candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        black_canopy_water_and_title_preserved=True,all_terrain_pressures_mm=.30,
        title_last=True,complete_subject_order=True,terrain_contact_samples_checked=sampled,
        contact_check_step_mm=.05,terrain_paths_and_return_connectors_inside_saved_masks=True,
        trunk_and_pavilion_excluded=True,coarse_contours_excluded=True,
        minimum_centerline_to_region_boundary_mm=round(minimum,4),
        estimated_footprint_half_width_mm=round(1.3133333333333332/2,4),
        modeled_bleed_allowance_mm=1.5,minimum_blue_green_centerline_gap_mm=round(gap,4),
        max_terrain_contact_length_mm=round(max_length,4),estimated_region_coverage=coverage,
        actual_brush_width_and_bleed_need_measurement=True)


if __name__=='__main__':
    from web.catalog import ARTWORKS
    from painting_session import save
    baseline=ROOT/ARTWORKS[0][3];candidate=baseline.parent/'待应用_顺皴刷色'/'painting.json'
    report=audit_candidate(candidate,baseline)
    save(candidate.parent/'真实轨迹离线校验.json',report)
    print(json.dumps(report,ensure_ascii=False))
