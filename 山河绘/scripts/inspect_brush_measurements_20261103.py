"""Read-only checks before using physical brush measurements for training."""
import argparse,hashlib,json,math
from pathlib import Path


def inspect(root):
    root=Path(root);plan=json.loads((root/'capture_plan.json').read_text('utf-8'))
    expected={r['sample_id']:r for r in plan['samples']};records={};issues=[]
    for p in sorted((root/'measured').glob('records_*.json')):
        records.update(json.loads(p.read_text('utf-8')).get('records',{}))
    ready=[];photos={};sessions={}
    for key,r in records.items():
        bad=[];ref=expected.get(key)
        if ref is None:issues.append(dict(sample_id=key,issues=['unknown sample']));continue
        if any(r.get(k)!=ref[k] for k in ['sample_id','session_id','split','load_batch_id','shape','stroke_since_load']):
            bad.append('plan/session/split mismatch')
        for k in ['brush_id','paper_id','ink_id','touch_reference_id','lighting_id']:
            if not r.get(k):bad.append('missing '+k)
        for k in ['actual_press_mm','actual_speed_mm_s','actual_dip_depth_mm','actual_dwell_s','actual_length_mm','cumulative_since_load_mm']:
            v=r.get(k)
            if not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<0 or (k in ['actual_speed_mm_s','actual_length_mm'] and v==0):bad.append('missing/invalid '+k)
        name=r.get('photo_name');photo=(root/'measured'/str(name)).resolve()
        if not name or photo.parent!=(root/'measured').resolve() or not photo.is_file():bad.append('missing local original photo')
        elif hashlib.sha256(photo.read_bytes()).hexdigest()!=r.get('photo_sha256'):bad.append('photo hash mismatch')
        else:photos.setdefault(r['photo_sha256'],set()).add(ref['split'])
        if r.get('quality') in [None,'unreviewed','uncertain']:bad.append('quality unknown')
        if r.get('quality')!='no_mark':
            for k in ['start_width_mm','middle_width_mm','end_width_mm']:
                v=r.get(k)
                if not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:bad.append('missing/invalid '+k)
        sessions.setdefault(ref['session_id'],set()).add(ref['split'])
        if bad:issues.append(dict(sample_id=key,issues=bad))
        else:ready.append(key)
    leakage=[h for h,s in photos.items() if len(s)>1]
    by_split={s:sum(expected[k]['split']==s for k in ready) for s in ['train','calibration','test']}
    return dict(recorded=len(records),valid_width_or_no_mark_records=len(ready),valid_by_split=by_split,
        missing_samples=len(expected.keys()-records.keys()),issues=issues,photo_cross_split_leakage=leakage,
        pilot_complete_for_width_training=bool(len(ready)==len(expected) and not issues and not leakage),
        ink_density_training_ready=False,ink_density_reason='No normalized photometric measurements in this protocol yet',
        training_performed=False,ready_for_robot=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture_directory');args=p.parse_args()
    print(json.dumps(inspect(args.capture_directory),ensure_ascii=False,indent=2))
