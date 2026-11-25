"""Read executor evidence from disk only; never connects to the robot."""
import argparse
import json
from pathlib import Path


def summarize(path):
    info = dict(path=str(Path(path).resolve()), completed_strokes=0,
                submitted_moves=0, acknowledged_moves=0, last_mode=None,
                last_alarms=None, last_confirmed=None, fault=None, fault_snapshot=None)
    moves = {}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A live writer can still be completing its last line.
            info['last_host_time'] = row['host_time']
            info['active_stroke'] = row.get('stroke_number')
            event = row['event']
            if event == 'command_submit' and row['command'].startswith('MovL('):
                info['submitted_moves'] += 1
            if event == 'command_reply':
                if row['command'].startswith('MovL('):
                    if row['response'].startswith('0,'):
                        info['acknowledged_moves'] += 1
                        moves[row.get('controller_queue_id')] = row
                elif row['command'] == 'RobotMode()':
                    info['last_mode'] = row['response']
                elif row['command'] == 'GetErrorID()':
                    info['last_alarms'] = row['response']
            elif event == 'action_complete' and row.get('action') == 'stroke':
                info['completed_strokes'] = row['stroke_number']
            elif event == 'arrival_confirmed':
                info['last_confirmed'] = {k:row.get(k) for k in
                    ('host_time','target','observed_pose','context')}
            elif event == 'fault':
                info['fault'] = row
            elif event == 'fault_snapshot':
                info['fault_snapshot'] = row
    if info['fault_snapshot']:
        queue_id = info['fault_snapshot'].get('controller_queue_id')
        if queue_id is not None:
            info['current_id_matched_submission'] = moves.get(queue_id)
        info['interpretation'] = 'Controller queue ID identifies a command; neither ACK nor ID proves physical completion or hardware cause.'
    return info


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=summarize(args.path)
    payload=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:args.output.write_text(payload,encoding='utf-8')
    print(payload)
