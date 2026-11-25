# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,os
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
ROOT=LAB/'cunfa_recognition_reproduction_20261107'
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):
  allowed={'/':'index.html','/index.html':'index.html','/classification_test.jpg':'classification_test.jpg','/reproduction_test.jpg':'reproduction_test.jpg','/current_painting_probes.jpg':'current_painting_probes.jpg','/pilot_preview.jpg':'pilot_preview.jpg'}
  p=urlsplit(self.path).path
  if p not in allowed:self.send_error(404);return
  f=ROOT/allowed[p];b=f.read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8' if f.suffix=='.html' else 'image/jpeg');self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(b)
 def do_POST(self):self.send_error(405,'Read-only experiment report')
if __name__=='__main__':
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);url=f'http://127.0.0.1:{server.server_port}/';(ROOT/'server.json').write_text(json.dumps(dict(url=url,pid=os.getpid(),robot_connected=False)),encoding='utf8');print(url,flush=True);server.serve_forever()
