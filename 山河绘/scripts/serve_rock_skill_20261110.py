# -*- coding: utf-8 -*-
"""Loopback read-only viewer. Fixed filenames; no robot or mutation API."""
import sys,json,os,mimetypes
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
ROOT=LAB/'rock_skill_offline_20261110'
ALLOWED={'index.html','comparison.png','source_crop.png','reference_overlay.png','reference_roles.png','model_direct.png','width_target.png','layer_replay.gif','visible_reference.json','model_direct.json','plan.json','verification.json','brush_forward/pipeline_result.json','brush_forward/measurement_template.csv'}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):self.serve(True)
 def do_HEAD(self):self.serve(False)
 def do_POST(self):self.send_error(405)
 def serve(self,body):
  name=urlsplit(self.path).path.lstrip('/') or 'index.html'
  if name not in ALLOWED:self.send_error(404);return
  p=ROOT/name
  if not p.is_file():self.send_error(404);return
  data=p.read_bytes();self.send_response(200);self.send_header('Content-Type',(mimetypes.guess_type(str(p))[0] or 'application/octet-stream')+('; charset=utf-8' if p.suffix in {'.html','.json'} else ''));self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers()
  if body:
   try:self.wfile.write(data)
   except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
if __name__=='__main__':
 html=Path(__file__).with_name('rock_skill_view_20261110.html').read_text('utf-8');(ROOT/'index.html').write_text(html,encoding='utf-8')
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);url=f'http://127.0.0.1:{server.server_port}/';(ROOT/'server.json').write_text(json.dumps(dict(url=url,pid=os.getpid(),robot_connected=False)),encoding='utf-8');print(url,flush=True);server.serve_forever()
