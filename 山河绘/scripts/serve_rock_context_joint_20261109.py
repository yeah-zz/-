# -*- coding: utf-8 -*-
"""Read-only loopback report. Images/selected JSON only; no robot or weights."""
import sys,json,os,mimetypes
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.rock_context_data import ROOT
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit
JSON={'data_manifest.json','training_report.json','threshold_calibration.json','evaluation_report.json','protocol_checks.json','verification.json','decision.json'}
ALLOWED={'index.html','joint_preview.jpg','real_partial_reference.png','procedural_examples.jpg'}|JSON
SCENES=['real_training_module','liupengkai_other_work','same_work_right_rock','new_howto_frame_00','new_howto_frame_01','blank_control','local_context_ablation']
ALLOWED|={f'{s}_{kind}.png' for s in SCENES for kind in ['input','curves','roles','masks','anchors','endpoints']}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  name=urlsplit(self.path).path.lstrip('/') or 'index.html'
  if name not in ALLOWED:self.send_error(404);return
  p=ROOT/name
  if not p.is_file():self.send_error(404);return
  data=p.read_bytes();self.send_response(200);self.send_header('Content-Type',(mimetypes.guess_type(str(p))[0] or 'application/octet-stream')+('; charset=utf-8' if p.suffix in ['.html','.json'] else ''));self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
 def do_POST(self):self.send_error(405,'Read only')
if __name__=='__main__':
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);url=f'http://127.0.0.1:{server.server_port}/';(ROOT/'server.json').write_text(json.dumps(dict(url=url,pid=os.getpid(),robot_connected=False)),encoding='utf8');print(url,flush=True);server.serve_forever()
