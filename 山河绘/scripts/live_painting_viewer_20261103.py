# -*- coding: utf-8 -*-
"""Loopback-only, read-only execution viewer. Never imports robot drivers."""
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit
from datetime import datetime,timezone
import json,threading,time,argparse,os
import numpy as np
import cv2

ROOT=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\complete_G03_01_trial_20261103')
OUT=ROOT/'live_viewer'

def read(path):return json.loads(Path(path).read_text('utf-8'))

class Feed:
    def __init__(self):
        self.lock=threading.Lock();self.last=None;self.last_refresh=0;self.key=None
        self.geometry=None;self.inverse=None

    def select(self):
        active=read(ROOT/'active_task.json');folder=Path(active['task'])
        checkpoint=read(folder/'checkpoint.json');path=Path(checkpoint['job'])
        if self.key!=str(path):
            data=read(path);meta=data['meta'];ss=meta['ink_execution']['base_strokes'];w,h=meta['image_size']
            src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
            dst=np.float32(meta['canvas_corners']);self.inverse=np.linalg.inv(cv2.getPerspectiveTransform(src,dst))
            self.geometry=dict(name='G03_01 · 松树、亭子与临水山石',width=w,height=h,
                strokes=[dict(id=s['id'],points=s['points'],length_mm=s['length_mm'],
                    source_curve_id=s.get('source_curve_id'),press_mm=s['press'][0],v_percent=s['speed'][0]) for s in ss],
                total=len(ss),paper_z=meta['paper_z'],policy=meta['ink_execution']['policy'],
                declaration='轨迹与执行日志可视化；不是相机画面或墨水仿真。')
            self.key=str(path)
        return active

    def state(self):
        with self.lock:
            now=time.monotonic()
            if self.last and now-self.last_refresh<.7:return self.last
            try:
                a=self.select();log=Path(a['log']);d=read(log);active=d.get('active_action') or {}
                diag=Path(d['diagnostic_path']);arrival=None;last_event=None
                with diag.open('rb') as f:
                    f.seek(max(0,diag.stat().st_size-131072));lines=f.read().decode('utf-8',errors='replace').splitlines()
                for line in reversed(lines):
                    try:r=json.loads(line)
                    except (ValueError,TypeError):continue
                    if last_event is None:last_event=r
                    if r.get('event')=='arrival_confirmed' and r.get('observed_pose'):
                        arrival=r;break
                point=None
                if arrival:
                    pose=arrival['observed_pose'];q=self.inverse@np.array([pose[0],pose[1],1.])
                    point=dict(x=float(q[0]/q[2]),y=float(q[1]/q[2]),z=pose[2],
                        time=arrival.get('host_time'),action=arrival.get('action'),stroke_number=arrival.get('stroke_number'),
                        tag=(arrival.get('waypoint') or {}).get('tag'),evidence='arrival_confirmed')
                stale=max(0,time.time()-diag.stat().st_mtime)
                status='running'
                if d.get('paused'):status='paused'
                elif d.get('finished_at'):status='complete' if d.get('success') else 'stopped'
                elif stale>90:status='stale'
                completed=int(d['completed_strokes']);current=active.get('stroke_number') if active.get('kind')=='stroke' else None
                if current is not None and not 1<=current<=d['total_strokes']:current=None
                self.last=dict(status=status,completed=completed,total=d['total_strokes'],
                    action=active.get('kind'),current_stroke=current,dips=d.get('dips',0),
                    elapsed_seconds=d.get('elapsed_seconds',0),started_at=d.get('started_at'),
                    observed_at=datetime.now(timezone.utc).isoformat(),last_event_time=(last_event or {}).get('host_time'),
                    last_event_age_seconds=round(stale,1),confirmed_point=point,error=d.get('error'),
                    source='local execution journal only; no controller queries',camera_connected=False,
                    last_read_ok=True)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                # Writers may momentarily be replacing the progress JSON.
                if self.last:self.last=dict(self.last,last_read_ok=False,read_note='日志写入中，暂用上次成功读数')
                else:raise
            self.last_refresh=now;return self.last

    def get_geometry(self):
        with self.lock:self.select();return self.geometry

FEED=Feed()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        path=urlsplit(self.path).path
        try:
            if path in ['/','/index.html']:body=Path(__file__).with_name('live_painting_viewer_20261103.html').read_bytes();mime='text/html; charset=utf-8'
            elif path=='/api/state':body=json.dumps(FEED.state(),ensure_ascii=False,allow_nan=False).encode('utf-8');mime='application/json; charset=utf-8'
            elif path=='/api/geometry':body=json.dumps(FEED.get_geometry(),ensure_ascii=False,allow_nan=False).encode('utf-8');mime='application/json; charset=utf-8'
            elif path=='/reference.png':body=(ROOT/'source.png').read_bytes();mime='image/png'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception:self.send_error(503,'Execution log temporarily unavailable')

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=0);a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);port=server.server_address[1]
    (OUT/'server.json').write_text(json.dumps(dict(pid=os.getpid(),url=f'http://127.0.0.1:{port}/',
        port=port,loopback_only=True,read_only=True,robot_connected=False),indent=2),encoding='utf-8')
    print(f'VIEWER http://127.0.0.1:{port}/ | READ ONLY | NO ROBOT CONNECTION',flush=True)
    server.serve_forever(poll_interval=.5)

if __name__=='__main__':main()
