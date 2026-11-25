"""Offline source-width-aware ink planning; no robot connection."""
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import sys, json, shutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import vision, imgio, shuimo, ink_control, job, watercolor
from original_d_tracer import LandscapeBrushGenerator as Trace
from unified_painting import build_plan
from painting_workflow import save_outputs
from scripts.preview_sparse_creek import spread
from config import ShuimoConfig


def join_source_paths(paths):
    """Join collinear branches at shared source skeleton nodes only."""
    paths = [np.asarray(p, float) for p in paths if len(p) >= 2]
    for _ in range(20):
        nodes = {}
        for i, p in enumerate(paths):
            for end in (0, -1):
                nodes.setdefault(tuple(p[end]), []).append((i, end))
        used, joined = set(), []
        for entries in nodes.values():
            best = None
            for i, ei in entries:
                if i in used: continue
                a = paths[i] if ei == -1 else paths[i][::-1]
                for j, ej in entries:
                    if j <= i or j in used: continue
                    b = paths[j] if ej == 0 else paths[j][::-1]
                    u, v = a[-1]-a[max(0,len(a)-4)], b[min(3,len(b)-1)]-b[0]
                    score = np.dot(u,v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-8)
                    if score > .8 and (best is None or score > best[0]):
                        best = (score, i, j, a, b)
            if best:
                _, i, j, a, b = best
                used.update((i,j)); joined.append(np.concatenate((a,b[1:])))
        if not used: break
        paths = [p for i,p in enumerate(paths) if i not in used] + joined
    return paths


def width_strokes(work, scale, pad=None, source_image=None):
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    local = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = (((gray < 100) & (hsv[:, :, 1] < 120)) |
            ((local > 24) & (gray < 195) & (hsv[:, :, 1] < 150))).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 5
    mask = keep[labels]
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    paths = join_source_paths(Trace._trace_skeleton_paths(Trace._thin_binary(mask * 255)))
    strokes = []

    def add(points, presses, role):
        if len(points) < 2: return
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))] * scale
        if arc[-1] < (1.0 if role == 'source_center' else 2.0): return
        ramp = np.minimum(np.clip(arc / .6, 0, 1), np.clip((arc[-1]-arc) / .8, 0, 1))
        presses = np.maximum(.06, presses * (.55 + .45 * ramp))
        s = dict(points=points.tolist(), press=presses.tolist(), speed=[18.] * len(points),
                 layer=0 if np.mean(points[:, 1]) < work.shape[0] * .28 else 1,
                 pal_index=shuimo.ink_palette_index(), tail_mm=0., lift_mm=4.,
                 role=role, name='原图墨线宽度分层', design_origin='full-resolution-source-width')
        strokes.append(ink_control.refresh_stats(s, scale))

    for raw in paths:
        points = np.asarray(raw, float)
        if len(points) < 3: continue
        # Keep the full source path but reduce samples to about 0.35 mm.
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))] * scale
        if arc[-1] < 1: continue
        positions = np.linspace(0, arc[-1], max(3, int(np.ceil(arc[-1] / .35)) + 1))
        points = np.column_stack([np.interp(positions, arc, points[:, i]) for i in (0, 1)])
        ij = np.rint(points).astype(int)
        widths = 2 * distance[ij[:, 1], ij[:, 0]] * scale
        pressures = np.interp(widths, [.25, .55, 1.0, 1.6], [.06, .10, .20, .30])
        add(points, pressures, 'source_center')
        tangent = np.gradient(points, axis=0)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-8)
        # Supplement only broad source ink. Each rail remains inside the dark mask
        # with clearance for the estimated brush radius; never thicken fine lines.
        for offset in (-2.25, -1.5, -.75, .75, 1.5, 2.25):
            rail = points + normal * offset / scale
            q = np.rint(rail).astype(int)
            valid = (q[:, 0] >= 0) & (q[:, 0] < mask.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < mask.shape[0])
            q[:, 0] = np.clip(q[:, 0], 0, mask.shape[1]-1)
            q[:, 1] = np.clip(q[:, 1], 0, mask.shape[0]-1)
            valid &= (widths > 2 * abs(offset) + .9) & (distance[q[:, 1], q[:, 0]] * scale >= .60)
            changes = np.diff(np.r_[False, valid, False].astype(int))
            for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                add(rail[a:b], np.full(b-a, .24), 'source_broad_rail')
    assert strokes and max(max(s['press']) for s in strokes) <= ShuimoConfig.PRESS_MAX_MM
    return strokes, dict(algorithm='full-resolution-width-and-in-mask-rails', strokes=len(strokes),
                         brush_width_calibrated_for_this_job=False,
                         broad_rails=sum(s['role']=='source_broad_rail' for s in strokes))


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'输出'/('层峦松水_粗细分明_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    reference = out/'参考原图.png'
    shutil.copy2(sys.argv[1], reference)
    with patch('original_d_adapter.build_strokes', width_strokes):
        plan, work, lineart, waypoints, meta = build_plan(reference, '山水')
    m = plan['modules']['color']
    old = m['strokes']
    m['strokes'] = spread([s for s in old if s['color']=='blue'], 180) + spread([s for s in old if s['color']=='green'], 50)
    m['stroke_count'] = len(m['strokes'])
    m['regions'] = [r for r in m['regions'] if r['color'] in ('blue', 'green')]
    for r in m['regions']:
        r['stroke_count'] = sum(s['region_id']==r['id'] for s in m['strokes'])
        r.pop('planned_coverage_fraction', None)
        r['status'] = 'planned_sparse' if r['stroke_count'] else 'sparse_skip'
    m['motion_candidate']['draw_velocity_percent'] = 25
    m['reload_strokes_per_load'] = 8
    plan['note'] = 'New source image. Source-width-aware black paths and broad-ink rails. Blue/green only. Per-point pressure preview uses an unverified brush-width model, not actual diffusion.'
    prefix = out/'层峦松水_粗细分明'
    save_outputs(plan, work, str(prefix)+'_计划.json')
    job.save_job(str(prefix)+'_黑线任务.json', waypoints, meta)
    canvas, _ = watercolor.render_image(work, m, include_ink=False)
    for s in plan['modules']['outline']['strokes']:
        pts = np.rint(s['points']).astype(int)
        for i, (a,b) in enumerate(zip(pts, pts[1:])):
            width = (shuimo.press_to_width_mm(s['press'][i])+shuimo.press_to_width_mm(s['press'][i+1]))/2
            cv2.line(canvas, tuple(a), tuple(b), (35,35,35), max(1,round(width/plan['mm_per_px'])), cv2.LINE_AA)
    imgio.imwrite(str(prefix)+'_预览.png', canvas)
    imgio.imwrite(str(prefix)+'_提取线稿.png', lineart)
    print('OUTPUT',out)
    print('BLACK',len(plan['modules']['outline']['strokes']),'COLOR',m['stroke_count'],'STATS',meta['quality_stats'])


if __name__ == '__main__': main()
