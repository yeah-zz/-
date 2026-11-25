"""Append-only evidence from the executor's own serialized Dashboard socket.

No polling thread, movement, recovery or controller parameter changes here.
Targets and acknowledgements are intent; only arrival events contain verified
feedback. A controller command ID identifies a queue item, not physical arrival.
"""
from datetime import datetime
import json
import re
import time


def attached(robot):
    # Avoid synthesizing attributes on mocked drivers or legacy adapters.
    return vars(robot).get('_diagnostic')


def queue_id(reply):
    match = re.match(r'^0,\{(\d+)\},', reply or '')
    return int(match.group(1)) if match else None


class MotionJournal:
    poll_interval = .5  # At most two arrival polls/s, on the executor thread.

    def __init__(self, path):
        self.handle = open(path, 'x', encoding='utf-8', buffering=1)
        self.context = {}
        self.event_number = 0
        self.command_number = 0
        self.batch_number = 0
        self.last_confirmed = None
        self.failed = False
        self.record('journal_start', meaning='submissions are not actual trajectory',
                    controller_wall_clock='unknown; host timestamps are not controller time')

    def record(self, event, **fields):
        self.event_number += 1
        row = dict(event_number=self.event_number,
                   host_time=datetime.now().astimezone().isoformat(),
                   monotonic_s=time.monotonic(), event=event, **self.context)
        row.update(fields)
        self.handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
        self.handle.flush()

    def before_command(self, command):
        self.command_number += 1
        number = self.command_number
        self.record('command_submit', command_number=number, command=command)
        return number

    def reply(self, number, command, response):
        self.record('command_reply', command_number=number, command=command,
                    response=response, controller_queue_id=queue_id(response))

    def arrival(self, target, observed_pose):
        self.last_confirmed = dict(target=list(target), observed_pose=list(observed_pose),
                                   host_time=datetime.now().astimezone().isoformat(),
                                   context=dict(self.context))
        self.record('arrival_confirmed', **self.last_confirmed)

    def fault(self, robot, error, mode=None, alarms=None):
        # One snapshot, no automatic retry or recovery. In-flight batches may
        # contain accepted moves; ceasing submissions is not a confirmed stop.
        if self.failed:
            return
        self.failed = True
        self.record('fault', error=str(error), mode=mode, alarms=alarms,
                    last_confirmed=self.last_confirmed, stop_confirmed=False)
        if not vars(robot).get('connected', False):
            self.record('fault_snapshot_unavailable', reason='Dashboard disconnected')
            return
        try:
            if mode is None:
                mode = robot.get_mode()
            if alarms is None:
                alarms = robot.get_error_id()
            pose = robot.get_pose()
            current_id = robot.send('GetCurrentCommandID()')
            self.record('fault_snapshot', mode=mode, alarms=alarms,
                        observed_pose=pose, current_command_reply=current_id,
                        controller_queue_id=queue_id(current_id), stop_confirmed=False)
        except Exception as exc:
            self.record('fault_snapshot_error', error=str(exc), mode=mode, alarms=alarms)

    def close(self):
        self.handle.close()
