# -*- coding: utf-8 -*-
"""Loopback-only platform. Reads explicit plan-bound journals; sends no motion."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import argparse,hashlib,json,os,time,re
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT=LAB/'hierarchical_painting_20261105'
SCENES=('G03_01','G05_03','T02')
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def execution_state(root=ROOT,now=None):
    now=time.time() if now is None else now
    base=dict(mode='execution_journal',status='not_bound',camera_connected=False,
              source='explicit local journal only; no robot/controller queries',confirmed_sequences=[],active_sequence=None,
              note='尚未绑定本轮实机任务；离线回放不代表机械臂正在绘画。')
    bind=read(root/'execution_binding.json')
    if not bind.get('journal'):return base
    scene=bind.get('scene')
    if scene not in SCENES:return {**base,'status':'invalid_binding','note':'任务名称无效'}
    planfile=root/scene/'plan.json'
    if sha(planfile)!=bind.get('plan_sha256') or not bind.get('run_id'):
        return {**base,'status':'plan_mismatch','note':'执行绑定与本轮计划不一致，拒绝套用旧进度'}
    plan=read(planfile);path=Path(bind['journal'])
    if not path.exists():return {**base,'status':'waiting_for_journal','scene':scene}
    if path.stat().st_size>16*1024*1024:return {**base,'status':'journal_too_large'}
    data=path.read_bytes();lines=data.splitlines(keepends=True);confirmed=set();active=None;last=0;errors=[];events=0
    terminal=None
    for number,line in enumerate(lines):
        if not line.endswith(b'\n') and number==len(lines)-1:continue # incomplete writer flush
        try:
            event=json.loads(line);seq=event.get('sequence');kind=event['event']
            if event.get('plan_sha256')!=bind['plan_sha256'] or event.get('run_id')!=bind['run_id']:raise ValueError('event binding mismatch')
            stamp=datetime.fromisoformat(event['time'].replace('Z','+00:00')).timestamp()
            if stamp>now+5:raise ValueError('future event')
            if stamp<last:raise ValueError('nonmonotone event time')
            if kind in ('stroke_started','stroke_completed'):
                if not isinstance(seq,int) or isinstance(seq,bool) or not 1<=seq<=len(plan['actions']):raise ValueError('invalid sequence')
                if event.get('stroke_id')!=plan['actions'][seq-1]['stroke_id']:raise ValueError('stroke identity mismatch')
                if terminal:raise ValueError('stroke after terminal event')
                if kind=='stroke_started':
                    if active is not None or seq in confirmed or seq!=len(confirmed)+1:raise ValueError('out of order stroke')
                    active=seq
                else:
                    if seq!=active:raise ValueError('completion without matching start')
                    confirmed.add(seq);active=None
            elif kind in ('paused','stopped'):
                terminal=kind
            elif kind=='resumed':terminal=None
            elif kind=='run_completed':
                if len(confirmed)!=len(plan['actions']) or active is not None:raise ValueError('premature full completion')
                terminal='complete'
            else:raise ValueError('unknown event')
            last=stamp;events+=1
        except (ValueError,KeyError,TypeError) as exc:
            errors.append(dict(line=number+1,reason=str(exc)));break
    age=max(0,now-last) if last else None
    status='invalid_journal' if errors else terminal or ('waiting' if not events else 'stale' if age>10 else 'active')
    return {**base,'scene':scene,'status':status,'confirmed_sequences':sorted(confirmed),
            'active_sequence':active,'total':len(plan['actions']),'last_event_age_seconds':age,
            'events_read':events,'errors':errors,'plan_sha256':bind['plan_sha256'],
            'note':'仅显示有日志依据的到笔状态；非相机画面。日志超时不视为正在运行。'}

class ExecutionJournal:
    """Optional executor hook. Call completed ONLY after arrival confirmation.

    This utility never executes a stroke. Integration must retain stroke IDs and
    plan digest; current offline plans are NOT directly executable robot jobs.
    """
    def __init__(self,path,plan_sha256,run_id):self.path=Path(path);self.digest=plan_sha256;self.run_id=run_id
    def append(self,event,sequence=None,stroke_id=None):
        row=dict(event=event,sequence=sequence,stroke_id=stroke_id,plan_sha256=self.digest,
                 run_id=self.run_id,time=datetime.now(timezone.utc).isoformat())
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        p=urlsplit(self.path).path;parts=p.strip('/').split('/')
        try:
            if p=='/api/state':body=json.dumps(execution_state(ROOT),ensure_ascii=False,allow_nan=False).encode();mime='application/json; charset=utf-8'
            else:
                allowed={'/':ROOT/'index.html','/index.html':ROOT/'index.html','/catalog.json':ROOT/'catalog.json','/report.html':ROOT/'report.html','/reference_readout.jpg':ROOT/'reference_readout.jpg'}
                if len(parts)==2 and parts[0] in SCENES and parts[1] in ('plan.json','source.png','modules.png','comparison.jpg','sequence_comparison.jpg','pavilion.jpg','rock.jpg','tree.jpg','join_gallery.jpg'):
                    allowed[p]=ROOT/parts[0]/parts[1]
                if p not in allowed:self.send_error(404);return
                file=allowed[p];body=file.read_bytes()
                mime={'.html':'text/html; charset=utf-8','.json':'application/json; charset=utf-8','.png':'image/png','.jpg':'image/jpeg'}[file.suffix]
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
        except (OSError,ValueError,KeyError,TypeError):self.send_error(503,'Local plan or journal temporarily unavailable')
    def do_POST(self):self.send_error(405,'Read-only viewer; no robot commands')

def main():
    global ROOT,SCENES
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=0);p.add_argument('--root',type=Path);args=p.parse_args()
    if args.root is not None:
        ROOT=args.root.resolve()
        entries=read(ROOT/'catalog.json')['scenes']
        if not entries or any(not re.fullmatch(r'[A-Za-z0-9_-]+',r['id']) for r in entries):raise ValueError('Invalid catalog scene IDs')
        SCENES=tuple(r['id'] for r in entries)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);url=f'http://127.0.0.1:{server.server_port}/'
    (ROOT/'server.json').write_text(json.dumps(dict(url=url,pid=os.getpid(),port=server.server_port,
        loopback_only=True,read_only=True,robot_connected=False),indent=2),encoding='utf8')
    print('PLATFORM '+url+' | NO ROBOT CONNECTION',flush=True);server.serve_forever(.5)
if __name__=='__main__':main()
