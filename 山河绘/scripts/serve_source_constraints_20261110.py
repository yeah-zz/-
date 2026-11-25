# -*- coding: utf-8 -*-
import sys,json,os,mimetypes
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
ROOT=LAB/'source_constrained_strokes_20261110'
ALLOWED={'index.html','comparison.png','evaluation.json','decision.json','counterexample_checks.json'}
for key in ['right_rocks','real_training_module','liupengkai_other_work','new_howto_frame_01']:
 for tail in ['source.png','paths.png','overlay.png']:ALLOWED.add(key+'_'+tail)
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_POST(self):self.send_error(405)
 def do_GET(self):self.serve(True)
 def do_HEAD(self):self.serve(False)
 def serve(self,body):
  name=urlsplit(self.path).path.lstrip('/') or 'index.html'
  if name not in ALLOWED:self.send_error(404);return
  p=ROOT/name
  if not p.is_file():self.send_error(404);return
  data=p.read_bytes();self.send_response(200);self.send_header('Content-Type',(mimetypes.guess_type(name)[0] or 'application/octet-stream')+('; charset=utf-8' if p.suffix in {'.html','.json'} else ''));self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers()
  if body:
   try:self.wfile.write(data)
   except (BrokenPipeError,ConnectionResetError):pass
if __name__=='__main__':
 s=ThreadingHTTPServer(('127.0.0.1',0),Handler);url=f'http://127.0.0.1:{s.server_port}/';(ROOT/'server.json').write_text(json.dumps(dict(url=url,pid=os.getpid(),robot_connected=False)),encoding='utf-8');print(url,flush=True);s.serve_forever()
