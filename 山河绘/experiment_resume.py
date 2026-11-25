"""Audited continuation only after a fully acknowledged stroke endpoint."""
import hashlib
import json
from pathlib import Path
from config import ArmConfig, CanvasConfig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resume_suffix(path, lines):
    old=json.loads(Path(path).read_text(encoding='utf8'))
    normalize=lambda value: json.loads(json.dumps(value))
    expected=dict(paper_z=ArmConfig.PAPER_Z,user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,
                  canvas_corners=normalize(CanvasConfig.ROBOT_CORNERS),rpy=list(ArmConfig.RPY))
    if any(old.get(k)!=v for k,v in expected.items()): raise ValueError('续画标定与中断记录不一致')
    if old.get('color_stage_closed'):
        raise ValueError('This color stage was explicitly ended; do not resume it automatically')
    if old.get('abandoned'):
        raise ValueError('这条中断记录已被人工作废（换纸重画），不能续画；'
                         '如果只是想接着画，请找回那张纸对应的记录')
    if old.get('lines')!=normalize(lines): raise ValueError('续画轨迹与中断记录不一致，不能猜测续画位置')
    if old.get('pending_move'): raise ValueError('存在未确认到位的指令，需要单独核查，不能自动续画')
    contacts=old.get('contacts',[])
    if not contacts or len(contacts)>=len(lines): raise ValueError('没有可自动续画的剩余笔触')
    for i,c in enumerate(contacts):
        if c['id']!=lines[i]['id'] or c['xy']!=normalize(lines[i]['xy']):
            raise ValueError('接触记录不是当前任务的连续前缀')
        if not c.get('completed'):
            if i!=len(contacts)-1: raise ValueError('较早的接触记录未完成')
            draws=[p for name,p in lines[i]['sequence'] if name=='draw']
            last=old.get('steps',[])[-1] if old.get('steps') else {}
            if last.get('name')!=c['id']+':draw-path' or last.get('targets')!=normalize(draws):
                raise ValueError('中断笔的行笔终点未确认，需要检查纸面后单独处理')
    return lines[len(contacts):],dict(source=str(Path(path).resolve()),sha256=digest(path),
         skipped_ids=[c['id'] for c in contacts],operator_recovered_to_high=True)


def unresolved_records(folder):
    """还没了结、会挡住下一次开画的中断记录。

    了结方式只有两种（都由记录本身证明）：
      1. 被后续一次【成功的续画】引用（success + resume_audit 对得上 sha256）；
      2. 被人工显式【作废】（record['abandoned']，见 scripts/resolve_interrupt.py --abandon）。
    作废意味着这张纸不要了、重画会另起一张 —— 所以它不再算"需要续画的悬案"。
    """
    records=[(p,json.loads(p.read_text(encoding='utf8'))) for p in Path(folder).glob('color_experiment_*.json')]
    resolved=set()
    for _,r in records:
        audit=r.get('resume_audit')
        if r.get('success') and audit and Path(audit['source']).is_file() and digest(audit['source'])==audit['sha256']:
            resolved.add(str(Path(audit['source']).resolve()))
    return [(p,r) for p,r in records if str(p.resolve()) not in resolved
            and not r.get('abandoned')
            and not r.get('color_stage_closed')
            and (r.get('pending_move') or any(not c.get('completed') for c in r.get('contacts',[])))]


def abandoned_records(folder):
    """已人工作废的记录（保留证据，不再挡路）。"""
    out=[]
    for p in sorted(Path(folder).glob('color_experiment_*.json'),key=lambda q:q.stat().st_mtime):
        try: r=json.loads(p.read_text(encoding='utf8'))
        except Exception: continue
        if r.get('abandoned'): out.append((p,r))
    return out
