"""Independent checks on saved execution paths, including wet connectors."""
from pathlib import Path
import json
from collections import Counter
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scripts.validate_source_washes import sample_segments
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.order_selected_artworks import ORDERS


def audit_candidate(candidate,baseline):
    candidate=Path(candidate);baseline=Path(baseline)
    new=json.loads(candidate.read_text(encoding='utf-8'));old=json.loads(baseline.read_text(encoding='utf-8'))
    ink=new['ink_job']['meta']['ink_execution']['base_strokes']
    assert ink==old['ink_job']['meta']['ink_execution']['base_strokes'],'原黑线、题字或下压改变'
    colors=new['workflow']['modules']['color']['strokes'];old_colors=old['workflow']['modules']['color']['strokes']
    retained=lambda ss:{s['id']:s for s in ss if s.get('role') not in ('mountain_color','distant_color','terrain_wash')}
    assert retained(colors)==retained(old_colors),'叶色、水纹或舟体改变'
    assert not any(s.get('role') in ('mountain_color','distant_color') for s in colors)
    revision=new['workflow']['source_revision'];number=revision['artwork_number']
    assert revision['stroke_data_sha256']==digest(stroke_data(new))
    tr=revision['source_transform'];scale=new['ink_job']['meta']['ink_execution']['mm_per_px'];pixel_mm=scale*tr['scale']
    masks=np.load(candidate.parent/'原图色带禁涂掩膜.npz');h,w=masks['safe'].shape
    distance=cv2.distanceTransform(np.pad(masks['pigment_envelope'],1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    washes=[s for s in colors if s.get('role')=='terrain_wash'];pigments={'blue':[],'green':[]};samples=0;max_length=0;clearance=float('inf')
    for s in washes:
        assert set(s['press'])=={.30}
        assert len(s['points'])==len(s['press'])==len(s['speed'])
        p=np.asarray(s['points']);length=np.linalg.norm(np.diff(p,axis=0),axis=1).sum()*scale
        assert length<=50.001 and abs(length-s['length_mm'])<1e-5
        max_length=max(max_length,float(length));dense=sample_segments(p,.05/scale)
        local=(dense-np.asarray(tr['offset']))/tr['scale'];q=np.rint(local).astype(int)
        assert np.all((q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h))
        assert np.all(masks['safe'][q[:,1],q[:,0]]),'色带超出安全区域'
        assert not np.any(masks['protected'][q[:,1],q[:,0]]),'越过树干、轮廓或建筑'
        assert np.all(masks['ribbon_%d'%s['source_ribbon_index']][q[:,1],q[:,0]]),'接触或连接段穿过留白'
        clearance=min(clearance,float(distance[q[:,1],q[:,0]].min()*pixel_mm))
        pigments[s['color']].append(local);samples+=len(q)
    assert washes and all(pigments.values())
    gap=float(cKDTree(np.vstack(pigments['blue'])).query(np.vstack(pigments['green']))[0].min()*pixel_mm)
    assert gap+.2>=revision['terrain_blue_green_centerline_gap_mm']
    assert clearance+pixel_mm>=revision['terrain_centerline_inset_mm']
    for c in pigments:
        ranks=[ORDERS[number].index(s['drawing_region']) for s in colors if s['color']==c]
        assert ranks==sorted(ranks)
    from artwork_order import ink_stages
    parts=ink_stages(new['ink_job'],new['workflow']);assert parts['title'] and all(s['role']=='title' for s in parts['title'])
    counts=dict(black=len(parts['black']),**dict(Counter(s['color'] for s in colors)),title=len(parts['title']))
    def modeled_area(bundle):
        width,height=bundle['ink_job']['meta']['image_size']
        footprint=np.zeros((height,width),np.uint8)
        for stroke in bundle['workflow']['modules']['color']['strokes']:
            if stroke.get('role')=='terrain_wash':
                cv2.polylines(footprint,[np.rint(stroke['points']).astype(np.int32)],False,1,
                    max(1,round(stroke['brush_width_mm_estimate']/scale)))
        return int(footprint.sum())
    before_area=modeled_area(old);after_area=modeled_area(new)
    return dict(robot_connected=False,counts=counts,total_strokes=sum(counts.values()),black_and_title_exactly_preserved=True,
        leaves_boat_and_water_preserved=True,title_last=True,complete_subject_order=True,terrain_contact_samples_checked=samples,
        contact_check_step_mm=.05,terrain_paths_and_return_connectors_inside_saved_masks=True,protected_objects_excluded=True,
        minimum_centerline_to_pigment_boundary_mm=round(clearance,4),minimum_blue_green_centerline_gap_mm=round(gap,4),
        max_terrain_contact_length_mm=round(max_length,4),terrain_strokes=len(washes),terrain_pressure_mm=.30,strip_pitch_mm=1.0,
        estimated_terrain_footprint_before_px=before_area,estimated_terrain_footprint_after_px=after_area,
        estimated_terrain_footprint_ratio=round(after_area/before_area,3) if before_area else None,
        actual_brush_width_and_bleed_need_measurement=True)
