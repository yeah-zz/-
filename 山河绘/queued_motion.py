"""Bounded motion queues; drawing can refill without artificial in-stroke stops.

Dobot V4: GetCurrentCommandID reports the executing queue command, NOT arrival.
Keep that command outstanding and only reclaim slots for strictly older IDs.
Final physical arrival still uses the caller's mode/alarm/pose verification.
"""
from collections import deque
import re
import time
from config import ArmConfig
from robot import RobotError

BATCH_SIZE = 32  # Below the documented 64-command queue.
REFILL_AT = 16
PROGRESS_TIMEOUT = 120.0
POLL_INTERVAL = .02


def send_batch(robot, motions, wait_finished, *, metadata=None, sequence_offset=0,
               continuous=False):
    if type(sequence_offset) is not int or sequence_offset < 0:
        raise ValueError('Invalid motion sequence offset')
    if metadata is not None and len(metadata) != len(motions):
        raise ValueError('Motion metadata length mismatch')
    # Short moves already fit without an intermediate forced stop. Station
    # motion and callers not opting in retain the existing arrival barriers.
    if continuous and len(motions) > BATCH_SIZE - sequence_offset % BATCH_SIZE:
        return _send_continuous(robot, motions, wait_finished, metadata, sequence_offset)
    from motion_diagnostics import attached
    journal = attached(robot)
    start = 0
    while start < len(motions):
        # A pre-contact arrival check may split a stroke. Keep the original
        # 32-point boundaries so the split does not relocate stops within ink.
        count = BATCH_SIZE - (sequence_offset + start) % BATCH_SIZE
        batch = motions[start:start+count]
        if journal:
            journal.batch_number += 1
            journal.context.update(batch_number=journal.batch_number,
                                   batch_offset=sequence_offset+start, batch_size=len(batch))
            journal.record('batch_start', target=list(batch[-1][0]))
        for i, (target, velocity, cp) in enumerate(batch):
            if i == len(batch)-1:
                cp = 0
            command = ('MovL(pose={%.4f,%.4f,%.4f,%.4f,%.4f,%.4f},user=%d,tool=%d,v=%d,cp=%d)' %
                       (*target, ArmConfig.USER_INDEX, ArmConfig.TOOL_INDEX, velocity, cp))
            if journal:
                journal.context.update(motion_offset=sequence_offset+start+i, batch_command=i+1,
                                       waypoint=(metadata[start+i] if metadata else None))
            reply = robot.send(command)
            if not robot._ok(reply):
                try:
                    state = 'mode=%r alarms=%r' % (robot.get_mode(), robot.get_error_id())
                except Exception as exc:
                    state = 'State read failed: %s' % exc
                error = RobotError('Queued move rejected at batch offset %d; command=%s; reply=%r; %s. '
                                 'Stop and clear controller queue before restarting' %
                                 (start+i, command, reply, state))
                if journal: journal.fault(robot, error)
                raise error
        wait_finished(robot, batch[-1][0])
        if journal:
            current_id = robot.send('GetCurrentCommandID()')
            journal.record('batch_complete', current_command_reply=current_id)
        start += len(batch)


def _command_id(reply, name):
    pattern = r'\s*0,\{(\d+)\},' + re.escape(name) + r'\(.*\);?\s*'
    match = re.fullmatch(pattern, reply) if isinstance(reply, str) else None
    if match is None:
        raise RobotError('Cannot verify controller queue ID: %r; no further moves sent' % (reply,))
    return int(match.group(1))


def _queue_state(robot):
    from run_mountain_v4_robot import no_errors
    mode = robot.get_mode()
    alarms = robot.get_error_id()
    try:
        clear = no_errors(alarms)
    except (ValueError, TypeError, IndexError, AttributeError):
        clear = False
    if mode not in (5, 7, 8) or not clear:
        raise RobotError('Drawing queue interrupted; mode=%s; alarms=%r; no further moves sent'
                         % (mode, alarms))
    return mode, _command_id(robot.send('GetCurrentCommandID()'), 'GetCurrentCommandID')


def _send_continuous(robot, motions, wait_finished, metadata, sequence_offset):
    from motion_diagnostics import attached
    journal = attached(robot)
    try:
        mode, previous = _queue_state(robot)
        if mode != 5:
            raise RobotError('Drawing queue must start idle; no further moves sent')
        pending = deque()
        last_sent = previous
        deadline = time.monotonic() + PROGRESS_TIMEOUT
        start = 0
        if journal:
            journal.record('stream_start', queue_limit=BATCH_SIZE, refill_at=REFILL_AT,
                           initial_command_id=previous, target=list(motions[-1][0]))
        while start < len(motions):
            if pending:
                while True:
                    mode, current = _queue_state(robot)
                    if current < previous or current > last_sent:
                        raise RobotError('Controller queue ID moved outside submitted sequence; no further moves sent')
                    if current != previous and current not in pending:
                        raise RobotError('Controller reported an unowned queue ID; no further moves sent')
                    if current > previous:
                        previous = current
                        deadline = time.monotonic() + PROGRESS_TIMEOUT
                    # The currently executing command remains counted, even
                    # when mode=5. No command ID is treated as verified arrival.
                    while pending and pending[0] < current:
                        pending.popleft()
                    if journal:
                        journal.record('queue_progress', controller_queue_id=current,
                                       outstanding=len(pending), mode=mode)
                    if time.monotonic() >= deadline:
                        raise RobotError('Controller queue made no progress within %g s; no further moves sent'
                                         % PROGRESS_TIMEOUT)
                    if len(pending) <= REFILL_AT:
                        break
                    time.sleep(POLL_INTERVAL)
            count = min(BATCH_SIZE - len(pending), len(motions) - start)
            if journal:
                journal.batch_number += 1
                journal.context.update(batch_number=journal.batch_number,
                                       batch_offset=sequence_offset+start, batch_size=count)
                journal.record('queue_refill', outstanding_before=len(pending), count=count)
            for index in range(start, start + count):
                target, velocity, cp = motions[index]
                # Keep each caller-defined contact/lift corner. Only the final
                # endpoint is forced exact; refill boundaries retain blending.
                if index == len(motions)-1:
                    cp = 0
                command = ('MovL(pose={%.4f,%.4f,%.4f,%.4f,%.4f,%.4f},user=%d,tool=%d,v=%d,cp=%d)' %
                           (*target, ArmConfig.USER_INDEX, ArmConfig.TOOL_INDEX, velocity, cp))
                if journal:
                    journal.context.update(motion_offset=sequence_offset+index, batch_command=index-start+1,
                                           waypoint=(metadata[index] if metadata else None))
                reply = robot.send(command)
                if not robot._ok(reply):
                    raise RobotError('Queued draw rejected at offset %d: %r; no further moves sent'
                                     % (sequence_offset+index, reply))
                command_id = _command_id(reply, 'MovL')
                if command_id <= last_sent:
                    raise RobotError('Controller returned non-increasing queue IDs; no further moves sent')
                pending.append(command_id)
                last_sent = command_id
            start += count
        # Preserve both controller progress and the physical arrival check.
        # The latter alone could match an earlier repeated position in a path.
        while True:
            mode, current = _queue_state(robot)
            if current < previous or current > last_sent or (current != previous and current not in pending):
                raise RobotError('Invalid final controller queue ID; no further moves sent')
            if current > previous:
                previous = current
                deadline = time.monotonic() + PROGRESS_TIMEOUT
            if mode == 5 and current == last_sent:
                break
            if time.monotonic() >= deadline:
                raise RobotError('Final drawing queue completion timed out; no further moves sent')
            time.sleep(POLL_INTERVAL)
        wait_finished(robot, motions[-1][0])
        if journal:
            journal.record('stream_complete', controller_queue_id=last_sent)
    except Exception as exc:
        if journal:
            journal.fault(robot, exc)
        raise
