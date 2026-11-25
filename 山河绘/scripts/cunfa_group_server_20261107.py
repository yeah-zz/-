# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,os
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
ROOT=LAB/'cunfa_group_inverse_20261107'
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):
  files={'/':'index.html','/index.html':'index.html','/real_group_test.jpg':'real_group_test.jpg','/synthetic_group_test.jpg':'synthetic_group_test.jpg','/real_group_preview.jpg':'real_group_preview.jpg'}
  p=urlsplit(self.path).path
  if p not in files:self.send_error(404);return
  f=ROOT/files[p];body=f.read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8' if f.suffix=='.html' else 'image/jpeg');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def do_POST(self):self.send_error(405,'Read only; no robot control')
if __name__=='__main__':
 s=ThreadingHTTPServer(('127.0.0.1',0),Handler);url=f'http://127.0.0.1:{s.server_port}/';(ROOT/'server.json').write_text(json.dumps(dict(url=url,pid=os.getpid(),robot_connected=False)),encoding='utf8');print(url,flush=True);s.serve_forever()
