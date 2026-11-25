# -*- coding: utf-8 -*-
"""Offline three-line reload-frequency comparison after baseline photo."""
import sys,json,hashlib
from pathlib import Path
from dataclasses import asdict,replace
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from run_brush_pilot_contact_20261103 import contact_scope,CONTACT,ROOT as BASE
from config import ArmConfig,PALETTE
from geometry import CanvasMap,check_waypoints
from ink_control import InkPolicy,build_plan,refresh_stats
from run_mountain_v4_robot import safe_transfers
from run_mountain_balanced import load_reviewed
import job

OUT=BASE.parent/'reload_each_stroke_comparison'
PHOTO=LAB.parent/'CLIPascene_20261021'/'temp'/'codex-clipboard-5fcedf92-592e-449f-bd51-2932c537f108.jpg'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    if list(OUT.glob('*_job_runs/*execution_log.json')):raise ValueError('Comparison already has a run record; do not overwrite')
    baseline=read(BASE/'first_group_completed.json');log=read(baseline['execution_log'])
    if not log['success'] or log['completed_strokes']!=3:raise ValueError('Baseline incomplete')
    # Store visual observations separately from expert annotation/training targets.
    save(BASE/'photo_observations.json',dict(photo=str(PHOTO),photo_sha256=sha(PHOTO),
        baseline_execution_log=baseline['execution_log'],observer='assistant_visual_review',
        photo_top='darkest, most continuous line',photo_middle='lighter with small interruptions',
        photo_bottom='faint and visibly interrupted',all_lines='locally wider at image-right end',
        photo_rows_to_stroke_numbers=None,ruler_present=False,absolute_width_mm=None,
        endpoint_start_or_end_not_verified=True,ink_depletion_hypothesis_unconfirmed=True,
        controlled_pressure_and_speed_constant=True,paper_plane_and_ink_state_not_measured=True,
        expert_ground_truth=False,ready_for_training=False))
    data=read(BASE/'A_baseline_new_contacts_job.json');meta=data['meta']
    base=deepcopy(read(BASE.parent/'B_lighter_job.json')['meta']['ink_execution']['base_strokes'])
    for i,s in enumerate(base):
        s.update(id='reload-each-line-%d'%(i+1),name='reload-each-line-%d'%(i+1),
            press=[.6]*len(s['points']),speed=[20.]*len(s['points']))
    with contact_scope():
        mapper=CanvasMap(meta['image_size'])
        base=[refresh_stats(s,mapper.mm_per_px) for s in base]
        policy=replace(InkPolicy(**meta['ink_execution']['policy']),max_strokes_per_load=1)
        steps,stats=build_plan(base,0,mapper.mm_per_px,policy)
        if stats['dips']!=3:raise ValueError('Exactly three separate loads expected')
        waypoints,_=job.steps_to_waypoints(steps,mapper);waypoints=safe_transfers(waypoints)
        if not check_waypoints(waypoints)[0]:raise ValueError('Static reach failed')
        meta.update(mode='reload-frequency-comparison-only',group='fresh-load-per-line',strokes=3,
            actually_drawn=False,current_pose_checked=False,operator_execution_authorized=False,
            trial_changes=['reload before every line instead of before each three-line group'],
            confounds=['different paper location; paper-plane flatness not measured'])
        meta['ink_execution'].update(base_strokes=base,policy=asdict(policy),
            calibration=dict(user=2,tool=0,rpy=CONTACT['rpy'],speed_factor=20,palette=deepcopy(PALETTE)))
        target=OUT/'three_lines_reload_each_job.json';job.save_job(str(target),waypoints,meta)
        load_reviewed(str(target))
    save(OUT/'proposal.json',dict(job=str(target),job_sha256=sha(target),strokes=3,length_each_mm=20,
        press_mm=.6,draw_velocity_percent=20,global_speed_factor=20,dip_mm=1.2,dwell_s=1.,planned_dips=3,
        rpy=CONTACT['rpy'],paper_z=-23.2686,draw_z=-22.6686,dip_z=-20.9935,
        group_xy='User2/Tool0 X160 to180, Y75/85/95; blank area must be confirmed',
        approved_to_execute=False,robot_connected=False,actually_drawn=False,
        baseline_photo_order_pending=True,purpose='compare repeatable fresh-load lines with three lines on one load; not proof of cause'))
    print('Baseline photo recorded, order/scale pending. Comparison prepared: 3 x 20mm, press=.60, v20/global20, load each line, dip1.20/dwell1s. NO MOTION.')

if __name__=='__main__':main()
