# -*- coding: utf-8 -*-
import sys,json,os,mimetypes
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
ROOT=LAB/'clear_rock_sources_20261117'
ALLOWED={'index.html','local_reference_review.jpg','fuchun_overview.jpg','local_references.json',
         'source_preflight.json','source_access_audit.json','review_status.json','verification.json'}
ALLOWED.update('views/'+p.name for p in (ROOT/'views').iterdir() if p.suffix in {'.png','.jpg'})
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):self.send_error(405)
    def do_GET(self):
        name=urlsplit(self.path).path.lstrip('/') or 'index.html'
        if name not in ALLOWED or not (ROOT/name).is_file():self.send_error(404);return
        data=(ROOT/name).read_bytes();self.send_response(200)
        self.send_header('Content-Type',(mimetypes.guess_type(name)[0] or 'application/octet-stream')+('; charset=utf-8' if name.endswith(('.html','.json')) else ''))
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers()
        try:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass
if __name__=='__main__':
    s=ThreadingHTTPServer(('127.0.0.1',0),Handler);url=f'http://127.0.0.1:{s.server_port}/'
    (ROOT/'server.json').write_text(json.dumps(dict(url=url,pid=os.getpid(),robot_connected=False)),encoding='utf8')
    print(url,flush=True);s.serve_forever()
