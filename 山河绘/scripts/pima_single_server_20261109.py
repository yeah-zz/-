# -*- coding: utf-8 -*-
"""Loopback-only report: no model download, arbitrary files or robot endpoints."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,os,mimetypes
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=LAB/'pima_single_stroke_20261109'
ALLOWED={'index.html','reference_curves.jpg','single_curve_preview.jpg','compact_preview.jpg','single_observations.jpg','single_reference_manifest.json','single_curve_training_report.json','verification.json','decision.json','inspection/37_212.6_213.15.jpg'}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  name=urlsplit(self.path).path.lstrip('/') or 'index.html'
  if name not in ALLOWED:self.send_error(404);return
  p=ROOT/name;body=p.read_bytes();self.send_response(200);self.send_header('Content-Type',(mimetypes.guess_type(str(p))[0] or 'application/octet-stream')+('; charset=utf-8' if p.suffix in ['.html','.json'] else ''));self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def do_POST(self):self.send_error(405,'Read only')
if __name__=='__main__':
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);url=f'http://127.0.0.1:{server.server_port}/';(ROOT/'server.json').write_text(json.dumps(dict(url=url,pid=os.getpid(),robot_connected=False)),encoding='utf8');print(url,flush=True);server.serve_forever()
