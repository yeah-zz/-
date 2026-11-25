"""Loopback-only read-only report, no robot code, arbitrary files or weights."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,os,mimetypes
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
ROOT=LAB/'pima_demonstration_20261109'
ALLOWED={'index.html','real_demonstration_preview.jpg','next_group_preview.jpg','prediction_validity.jpg','next_group_report.json','source_curation.json','verification.json','sequence_manifest.json'}
ALLOWED.update('sequence/'+p.name for p in (ROOT/'sequence').glob('*.png'))
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):
  name=urlsplit(self.path).path.lstrip('/') or 'index.html'
  if name not in ALLOWED:self.send_error(404);return
  p=ROOT/name;data=p.read_bytes();self.send_response(200);self.send_header('Content-Type',(mimetypes.guess_type(str(p))[0] or 'application/octet-stream')+('; charset=utf-8' if p.suffix in ['.html','.json'] else ''));self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def do_POST(self):self.send_error(405,'Read only')
if __name__=='__main__':
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);url=f'http://127.0.0.1:{server.server_port}/'
 (ROOT/'server.json').write_text(json.dumps(dict(url=url,pid=os.getpid(),robot_connected=False)),encoding='utf8');print(url,flush=True);server.serve_forever()
