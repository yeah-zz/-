# -*- coding: utf-8 -*-
"""Existing guarded executor, only the operator-approved three-line comparison."""
import argparse,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from run_brush_pilot_contact_20261103 import contact_scope,ROOT as BASE,CONTACT
from run_mountain_balanced import load_reviewed,check_robot_readonly,execute_reviewed_job
ROOT=BASE.parent/'reload_each_stroke_comparison'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['check','check-robot','execute'],default='check');a=parser.parse_args()
    proposal=json.loads((ROOT/'proposal.json').read_text('utf-8'));target=Path(proposal['job'])
    baseline=json.loads((BASE/'manifest.json').read_text('utf-8'))
    if target.resolve()!=(ROOT/'three_lines_reload_each_job.json').resolve() or sha(target)!=proposal['job_sha256']:raise ValueError('Task path/hash changed')
    if any(sha(r['path'])!=r['sha256'] for r in baseline['shared_files']):raise ValueError('Shared settings changed')
    if sha(Path(__file__).with_name('run_brush_pilot_contact_20261103.py'))!=baseline['script_sha256']:raise ValueError('Contact scope changed')
    if CONTACT!=baseline['contacts']:raise ValueError('Contacts changed')
    auth=json.loads((ROOT/'operator_authorization.json').read_text('utf-8'))
    if auth['job_sha256']!=proposal['job_sha256'] or not auth['allow_only_this_group'] or auth['strokes']!=3:raise ValueError('Group-specific authorization required')
    with contact_scope():
        strokes,mapper,policy,meta=load_reviewed(str(target))
        if len(strokes)!=3 or len({s['id'] for s in strokes})!=3 or policy.max_strokes_per_load!=1:raise ValueError('Only 3 distinct lines with per-line refill allowed')
        if policy.initial_dip_s!=1 or policy.refresh_dip_s!=1 or policy.fresh_press_factor!=1:raise ValueError('Ink policy changed')
        for s in strokes:
            if set(s['press'])!={.6} or set(s['speed'])!={20.} or s['tail_mm']!=0:raise ValueError('Pressure/speed changed')
            xy=mapper.to_robot(s['points'])
            if abs(xy[0][0]-160)>.001 or abs(xy[-1][0]-180)>.001:raise ValueError('Approved line location changed')
        if a.phase=='check':print('PASS: offline job/contacts/authorization checks; no connection.');return 0
        if a.phase=='check-robot':return check_robot_readonly(str(target))
        if list(ROOT.glob('*_job_runs/*execution_log.json')):raise ValueError('Execution record exists; no automatic redraw')
        if input('Only approved comparison: 3 lines, fresh ink before each. Type RUN: ').strip()!='RUN':return 0
        return execute_reviewed_job(str(target))

if __name__=='__main__':raise SystemExit(main())
