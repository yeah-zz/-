"""人工核查后解除中断记录，让下一次开画不被挡住。

两种处置，都必须人工确认（`--confirm-reviewed`）：

  A. 【续画】——只处理"确定性"情形，不猜、不放大：
     删掉那条未确认的【空走】指令；或者某个落笔点的【行笔】和【抬笔】都在
     记录里，按证据补标完成。之后用 --resume-record 接着画同一张纸。

  B. 【作废】`--abandon` —— 这张纸不要了，改完参数换一张纸重画。
     中断记录会永远挡住下一次开画（run_color_experiment 见到未解决记录就拒绝），
     而"续画"要求轨迹逐字节一致 —— 改了参数就永远续不了，于是形成死结。
     本模式把记录标成 abandoned（保留全部原始证据，只加一个作废标记），
     从此它不再计入"未解决记录"，也不再可续画。

        python scripts/resolve_interrupt.py --abandon                     # 只打印会发生什么
        python scripts/resolve_interrupt.py --abandon --confirm-reviewed  # 写入作废

用法（A 续画）：
    python scripts/resolve_interrupt.py                      # 只检查，打印结论（不动文件）
    python scripts/resolve_interrupt.py --confirm-reviewed    # 写入解除（保留审计字段）

★ 任何模式都只动"记录里的确认状态"，不修改任何轨迹、坐标、笔压或运动参数。
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_paths import ROBOT_LOG_DIR


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review_abandon(record):
    """作废前的只读盘点：告诉人"你正要放弃哪些东西"。"""
    notes = []
    pending = record.get('pending_move')
    contacts = record.get('contacts') or []
    bad = [c['id'] for c in contacts if not c.get('completed')]
    done = [c['id'] for c in contacts if c.get('completed')]
    if pending:
        notes.append('未确认的指令：name=%s target=%s'
                     % (pending.get('name'), pending.get('target')))
    else:
        notes.append('记录里没有未确认的指令')
    notes.append('落笔点 %d 个：完成 %d 个%s'
                 % (len(contacts), len(done), ('，未完成 %s' % bad) if bad else ''))
    notes.append('蘸色 %d 次；纸面 Z=%s' % (len(record.get('dips') or []), record.get('paper_z')))
    if record.get('success'):
        return False, notes + ['这条记录 success=True，不是中断记录，不需要作废']
    if not pending and not bad:
        return False, notes + ['这条记录既没有未确认指令也没有未完成落笔，本来就不挡路，不需要作废']
    return True, notes


def review(record):
    """返回 (能不能解除, 说明列表, 需要补标完成的落笔点 id 列表)。

    只处理两类"确定性"情形（都由记录本身证明）：
      A. 未确认的指令是【空走】(抬笔/横移/去杯口)，不是行笔；
      B. 某个落笔点没标 completed，但它的【行笔】和【抬笔】都在 steps 里 ——
         说明这一笔已经画完、笔也离开纸面了，只是没来得及写回完成标记。
    """
    notes = []
    ok = True
    mark_completed = []
    pending = record.get('pending_move')
    if pending:
        name = str(pending.get('name', ''))
        if name.endswith(':draw-path') or pending.get('targets'):
            return False, ['未确认的指令是【行笔】(name=%s)，纸面可能多画/断在半路，必须人工看纸后单独处理' % name], []
        notes.append('未确认的指令是空走：name=%s target=%s' % (name, pending.get('target')))
    contacts = record.get('contacts') or []
    steps = record.get('steps') or []
    step_names = [s.get('name', '') for s in steps]
    if not contacts:
        # 连一个落笔点都没记录 = 那一笔还没碰到纸（通常停在第一次蘸墨/横移）
        notes.append('记录里没有任何落笔点（touch 都没发生）-> 纸面一定没被画到')
    else:
        bad = [c['id'] for c in contacts if not c.get('completed')]
        if bad:
            finished = all(('%s:draw-path' % i) in step_names and ('%s:lift' % i) in step_names
                           for i in bad)
            if finished and len(bad) == 1:
                mark_completed = list(bad)
                notes.append('落笔点 %s 未标记完成，但它的【行笔】和【抬笔】都在记录里 -> '
                             '这一笔已画完、笔已离纸，按证据补标完成' % bad[0])
            else:
                ok = False
                notes.append('有未完成的落笔点（且无法证明已画完/已抬笔）：%s' % bad)
        else:
            notes.append('落笔点 %d 个，全部 completed' % len(contacts))
    if not pending and not mark_completed:
        ok = False
        notes.append('这条记录既没有未确认的指令，也没有可补标的落笔点，不需要用本脚本处理')
    if steps:
        notes.append('最后完成的动作：%s' % steps[-1].get('name'))
    if record.get('success'):
        ok = False
        notes.append('这条记录标记为 success=True，不该按中断处理')
    return ok, notes, mark_completed


def main(argv=None):
    parser = argparse.ArgumentParser(description='人工核查后解除"未确认的空走指令"造成的中断')
    parser.add_argument('--record', help='指定记录文件；不给就用唯一那条未解决记录')
    parser.add_argument('--confirm-reviewed', action='store_true',
                        help='已人工确认：那条未确认的移动没有落笔、机械臂已回到纸外高位、原纸没动')
    parser.add_argument('--abandon', action='store_true',
                        help='作废这张纸：把记录标成 abandoned，从此不再挡路也不再可续画。'
                             '（改完参数换一张新纸重画时用）')
    args = parser.parse_args(argv)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.record:
        path = Path(args.record)
    else:
        from experiment_resume import unresolved_records
        blocked = unresolved_records(ROBOT_LOG_DIR)
        if len(blocked) != 1:
            print('未解决记录有 %d 条，必须指定 --record（或先逐条核查）' % len(blocked))
            for p, _ in blocked:
                print('   ', p.name)
            return 2
        path = blocked[0][0]

    record = json.loads(path.read_text(encoding='utf-8'))

    if args.abandon:
        ok, notes = review_abandon(record)
        print('记录:', path.name)
        for line in notes:
            print('  -', line)
        print('结论:', '可以作废' if ok else '不需要作废')
        if not ok:
            return 2
        if not args.confirm_reviewed:
            print('加上 --abandon --confirm-reviewed 才会写入。前提：')
            print('  1) 笔尖已人工抬离纸面，机械臂已回到纸外高位（或你即将用示教器把它移开）；')
            print('  2) 这张纸你不打算再接着画了 —— 改完参数会换一张新纸重画；')
            print('  3) 已经看过纸面并拍过照，知道这次断在哪里。')
            print('作废后这条记录不再计入"未解决记录"，但也永远不能续画。')
            return 0
        pending = record.get('pending_move')
        record['abandoned'] = dict(
            abandoned_at=datetime.now().astimezone().isoformat(),
            reviewed_by='operator_with_agent',
            reason='operator abandons this sheet: parameters changed, will restart on a fresh sheet',
            pending_move_at_abandon=pending,
            incomplete_contacts=[c['id'] for c in (record.get('contacts') or [])
                                 if not c.get('completed')],
            previous_sha256=digest(path))
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print('已写入作废。新 sha256:', digest(path))
        print('现在可以改参数、换新纸，直接重跑（不会再被这条记录挡住）。')
        return 0

    ok, notes, mark_completed = review(record)
    print('记录:', path.name)
    for line in notes:
        print('  -', line)
    print('结论:', '可以解除' if ok else '不能自动解除')
    if not (ok and args.confirm_reviewed):
        if ok:
            print('加上 --confirm-reviewed 才会写入。前提：')
            print('  1) 那条未确认的移动路径上什么都没碰到（笔在纸外高位空走）；')
            print('  2) 笔尖已抬离纸面、机械臂已回到纸外高位停车点（go_to_start.py）；')
            print('  3) 原纸 / 笔 / 垫板都没有移位。')
        return 0 if ok else 2

    removal = record.pop('pending_move', None)
    if mark_completed:
        for c in record.get('contacts') or []:
            if c['id'] in mark_completed and not c.get('completed'):
                c['completed'] = True
    record['resolution'] = dict(
        resolved_at=datetime.now().astimezone().isoformat(),
        reviewed_by='operator_with_agent',
        reason='unconfirmed move was a non-drawing travel move; stroke draw-path+lift both recorded',
        cleared_pending_move=removal, marked_completed=mark_completed, previous_sha256=digest(path))
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已写入解除。新 sha256:', digest(path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())