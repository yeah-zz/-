# -*- coding: utf-8 -*-
"""Offline finite-brush drawability audit. Does not load robot configuration.

Widths passed here are labelled assumptions unless a current physical profile
is supplied. Geometric coverage never predicts real ink darkness or drying.
"""
from collections import Counter
import hashlib
import json
import numpy as np
import cv2
from brush_footprint_20261103 import footprint_audit

def length(p):
    return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())

def map_saved_job(document):
    """Use only saved image corners and robot corners, not today's live config."""
    meta=document['meta'];w,h=meta['image_size']
    src=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
    corners=np.array(meta['canvas_corners'],np.float32)
    matrix=cv2.getPerspectiveTransform(src,corners)
    ex=corners[1]-corners[0];ex=ex/np.linalg.norm(ex)
    ey=np.array([-ex[1],ex[0]])
    if ey@(corners[3]-corners[0])<0:ey=-ey
    axes=np.array([ex,ey]).T
    local=(corners-corners[0])@axes
    origin=local.min(0);size=local.max(0)-origin
    def convert(p):
        xy=cv2.perspectiveTransform(np.asarray(p,np.float32).reshape(-1,1,2),matrix).reshape(-1,2)
        return (xy-corners[0])@axes-origin
    paths=[convert(r['points']) for r in meta['ink_execution']['base_strokes']]
    return paths,dict(canvas_mm=size.astype(float).tolist(),matrix=matrix.tolist(),
        local_origin=origin.astype(float).tolist(),local_axes=axes.tolist(),
        saved_geometry_only=True,encoder_trajectory_measured=False)

def exact_segment_repeats(paths):
    seen={};repeat=0;repeat_mm=0.;cross_curve_repeat=0
    for i,p in enumerate(paths):
        for a,b in zip(p[:-1],p[1:]):
            dist=float(np.linalg.norm(b-a))
            if dist<1e-8:continue
            key=tuple(sorted((tuple(np.round(a,4)),tuple(np.round(b,4)))))
            if key in seen:
                repeat+=1;repeat_mm+=dist;cross_curve_repeat+=int(seen[key]!=i)
            else:seen[key]=i
    return dict(repeated_identical_segments=repeat,repeated_identical_length_mm=repeat_mm,
                repeated_identical_segments_across_curves=cross_curve_repeat,
                tolerance_mm=.0001,near_parallel_duplicates_not_tested_here=True)

def summarize_geometry(paths, estimated_width):
    lens=np.array([length(p) for p in paths])
    return dict(curves=len(paths),total_centerline_length_mm=float(lens.sum()),
        min_length_mm=float(lens.min()),median_length_mm=float(np.median(lens)),
        under_1mm=int((lens<1).sum()),under_2mm=int((lens<2).sum()),
        shorter_than_legacy_estimated_width=int((lens<estimated_width).sum()),
        legacy_estimated_width_mm=estimated_width,legacy_width_is_current_measurement=False,
        **exact_segment_repeats(paths))

def drawability_gate(geometry, current_profile=None):
    """No automatic fallback from a legacy display width to a physical width.

    This gate is an offline admission report, not a modification to the frozen
    executor. Dense test results and material identity are needed before a
    profile can even be assessed; straight-line continuity alone is insufficient.
    """
    required=['brush_id','paper_id','ink_condition','width_p95_mm',
              'start_diameter_p95_mm','end_diameter_p95_mm','dense_patch_passed']
    missing=[k for k in required if current_profile is None or current_profile.get(k) is None]
    if missing:
        return dict(ready_for_robot=False,status='current_ink_footprint_missing',
            missing_fields=missing,legacy_estimate_fallback_permitted=False,
            reason='Current brush/paper/ink width and dense-patch verification are absent')
    widths=[current_profile[k] for k in ['width_p95_mm','start_diameter_p95_mm','end_diameter_p95_mm']]
    if not all(isinstance(v,(float,int)) and np.isfinite(v) and v>0 for v in widths):
        raise ValueError('Positive finite measured footprint widths required')
    if current_profile['dense_patch_passed'] is not True:
        return dict(ready_for_robot=False,status='dense_patch_not_passed',missing_fields=[],
            legacy_estimate_fallback_permitted=False)
    return dict(ready_for_robot=False,status='requires_width_spacing_and_contact_plan_review',
                missing_fields=[],legacy_estimate_fallback_permitted=False,
                reason='A measured profile is necessary but does not validate the full artwork')

def sensitivity(paths,canvas_mm,widths=(.3,.6,1.,2.04,3.),start_factors=(1.,2.5),ppm=6.):
    rows=[];fields={}
    for width in widths:
        for factor in start_factors:
            summary,per_curve,counts,starts=footprint_audit(paths,float(width),canvas_mm=canvas_mm,
                                                          ppm=ppm,start_factor=factor)
            lengths=np.array([length(p) for p in paths])
            summary.update(curves_shorter_than_assumed_width=int((lengths<width).sum()),
                assumption_only=True,ink_wetting_or_drying_model=False)
            key=f'w{width:g}_s{factor:g}'
            rows.append(dict(key=key,summary=summary,per_curve=per_curve))
            fields[key]=(counts,starts)
    return rows,fields
