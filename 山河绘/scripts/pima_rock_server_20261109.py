# -*- coding: utf-8 -*-
"""Loopback-only, fixed allowlist, ranged video; no controller endpoints."""
import sys,json,os,mimetypes,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
ROOT=LAB/'pima_one_rock_cycle_20261109'
ALLOWED={'index.html','cycle_preview.jpg','tutorial_excerpt.gif','reference_review.jpg','final_transferred_reference.jpg','finished_frame.jpg',
 'finished_old_model_raw.jpg','finished_new_model_raw.jpg','liupengkai_other_work_new_model.jpg',
 'references/group_before.png','reference_manifest.json','training_report.json','final_discovery_report.json','verification.json','decision.json'}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_HEAD(self):self.serve(False)
 def do_GET(self):self.serve(True)
 def do_POST(self):self.send_error(405,'Read only')
 def serve(self,body):
  name=urlsplit(self.path).path.lstrip('/') or 'index.html'
  if name=='tutorial.mp4':p=LAB/'pima_tutorial_search_20261109'/'short_pima.mp4'
  elif name in ALLOWED:p=ROOT/name
  else:self.send_error(404);return
  if not p.is_file():self.send_error(404);return
  size=p.stat().st_size;start,end=0,size-1;status=200
  if self.headers.get('Range'):
   m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers['Range'])
   if not m:self.send_error(416);return
   start=int(m[1]);end=min(size-1,int(m[2]) if m[2] else size-1)
   if start>=size or end<start:self.send_error(416);return
   status=206
  self.send_response(status);self.send_header('Content-Type',(mimetypes.guess_type(str(p))[0] or 'application/octet-stream')+('; charset=utf-8' if p.suffix in ['.html','.json'] else ''))
  self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1));self.send_header('Cache-Control','no-store')
  if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  if body:
   try:
    with p.open('rb') as f:
     f.seek(start);left=end-start+1
     while left:
      data=f.read(min(left,65536))
      if not data:break
      self.wfile.write(data);left-=len(data)
   except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
if __name__=='__main__':
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);url=f'http://127.0.0.1:{server.server_port}/';(ROOT/'server.json').write_text(json.dumps(dict(url=url,pid=os.getpid(),robot_connected=False)),encoding='utf8');print(url,flush=True);server.serve_forever()
