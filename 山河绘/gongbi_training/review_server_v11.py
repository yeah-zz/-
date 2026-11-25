"""Loopback review utility. Saves explicit user labels only, exclusively on E."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse, hashlib, json, math, os, threading, time, uuid

SCHEMA='shanhehui.partial_strokes.v11'
ROLES={'contour','cun','tree','architecture','water','dot','unknown'}
CUNFA={'unknown','pima','fupi','zhedai','heye','yudian','niuma','juanyun','midian','mixed','other'}
WRITE_LOCK=threading.Lock()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

def empty_document(record, package_id):
    return dict(schema=SCHEMA,package_id=package_id,sample_id=record['id'],image_sha256=record['image_sha256'],
                width=record['width'],height=record['height'],source_group=record['source_group'],
                review_state='draft',annotation_scope='partial',unlabeled='unknown',strokes=[],regions=[])

def validate_document(d,r,package_id):
    if not isinstance(d,dict):raise ValueError('document must be an object')
    expected=empty_document(r,package_id)
    for key in ('schema','package_id','sample_id','image_sha256','width','height','source_group','annotation_scope','unlabeled'):
        if d.get(key)!=expected[key]:raise ValueError('document metadata mismatch: '+key)
    if d.get('review_state') not in {'draft','checked_partial'}:raise ValueError('invalid review state')
    seen=set()
    for kind in ('strokes','regions'):
        if not isinstance(d.get(kind),list) or len(d[kind])>2000:raise ValueError('invalid annotation list')
        for a in d[kind]:
            if not isinstance(a,dict) or not isinstance(a.get('id'),str) or len(a['id'])>100 or a['id'] in seen:raise ValueError('invalid or duplicate annotation id')
            seen.add(a['id']);points=a.get('points');minimum=2 if kind=='strokes' else 3
            if not isinstance(points,list) or not minimum<=len(points)<=10000:raise ValueError('invalid points')
            for p in points:
                if not isinstance(p,list) or len(p)!=2:raise ValueError('invalid point')
                for value,limit in zip(p,(r['width'],r['height'])):
                    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<limit:raise ValueError('point outside image')
            if kind=='strokes':
                if a.get('kind')!='stroke' or a.get('role') not in ROLES:raise ValueError('invalid stroke role')
                allowed=CUNFA if a['role']=='cun' else {'not_applicable'}
                if a.get('cunfa') not in allowed:raise ValueError('invalid cunfa')
                for key,options in [('width_profile',{'unknown','thin','medium','thick'}),('tone_profile',{'unknown','light','medium','dark'})]:
                    if not isinstance(a.get(key),list) or len(a[key])!=3 or any(v not in options for v in a[key]):raise ValueError('invalid appearance profile')
                if not isinstance(a.get('direction_known'),bool):raise ValueError('invalid direction flag')
            else:
                if a.get('kind') not in {'background','uncertain'}:raise ValueError('invalid region kind')
                if a.get('reason') not in {'paper_texture','damage','color_boundary','inscription','blank','other'}:raise ValueError('invalid region reason')
            if not isinstance(a.get('note',''),str) or len(a.get('note',''))>500:raise ValueError('invalid note')
    if d['review_state']=='checked_partial' and not (d['strokes'] or d['regions']):raise ValueError('cannot check empty document')
    # Do not trust client-supplied provenance or arbitrary additional fields.
    out={key:d[key] for key in expected};out['review_state']=d['review_state']
    out['strokes']=[{k:a[k] for k in ('id','kind','points','role','cunfa','width_profile','tone_profile','direction_known','note')} for a in d['strokes']]
    out['regions']=[{k:a[k] for k in ('id','kind','points','reason','note')} for a in d['regions']]
    out.update(label_provenance='user_reviewed' if d['review_state']=='checked_partial' else 'user_draft',
               dense_gold=False,independent_test=False,physical_pressure=None,robot_ready=False,edited_unix=time.time())
    return out

def atomic_json(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(temporary,path)

def make_server(root,port):
    root=Path(root).resolve();manifest=json.loads((root/'package_manifest.json').read_text(encoding='utf-8'))
    records={r['id']:r for r in manifest['samples']};labels=root/'labels';labels.mkdir(exist_ok=True)
    backups=root/'backups';backups.mkdir(exist_ok=True);exports=root/'exports';exports.mkdir(exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def send(self,value,kind='application/json; charset=utf-8',status=200):
            if not isinstance(value,bytes):value=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(value)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(value)
        def valid_host(self):
            return self.headers.get('Host')=='127.0.0.1:'+str(self.server.server_port)
        def do_GET(self):
            if not self.valid_host():return self.send({'error':'invalid host'},status=403)
            if self.path in ('/','/index.html'):return self.send((root/'review.html').read_bytes(),'text/html; charset=utf-8')
            if self.path=='/api/manifest':return self.send(manifest)
            if self.path=='/api/health':return self.send({'package_id':manifest['package_id'],'status':'ready'})
            if self.path.startswith('/api/labels/'):
                key=self.path.removeprefix('/api/labels/')
                if key not in records:return self.send({'error':'unknown sample'},status=404)
                path=labels/(key+'.json');doc=json.loads(path.read_text(encoding='utf-8')) if path.exists() else empty_document(records[key],manifest['package_id'])
                return self.send({'document':doc,'revision':digest(path)})
            for prefix in ('samples','overlays'):
                if self.path.startswith('/'+prefix+'/'):
                    name=self.path.split('/')[-1];key=name.removesuffix('.png')
                    if name!=key+'.png' or key not in records:return self.send({'error':'unknown image'},status=404)
                    path=root/prefix/name
                    if path.exists():return self.send(path.read_bytes(),'image/png')
            return self.send({'error':'not found'},status=404)
        def do_POST(self):
            if not self.valid_host() or self.headers.get('Origin')!='http://127.0.0.1:'+str(self.server.server_port):return self.send({'error':'origin rejected'},status=403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=8_000_000:raise ValueError('invalid body size')
                data=json.loads(self.rfile.read(size))
                if self.path=='/api/save':
                    r=records[data['id']];out=validate_document(data['document'],r,manifest['package_id']);path=labels/(r['id']+'.json')
                    with WRITE_LOCK:
                        if data.get('revision')!=digest(path):return self.send({'error':'另一个页面已修改本张；请先刷新，避免覆盖。'},status=409)
                        if path.exists():(backups/(r['id']+'_'+str(time.time_ns())+'.json')).write_bytes(path.read_bytes())
                        atomic_json(path,out);revision=digest(path)
                    return self.send({'saved':str(path),'revision':revision})
                if self.path=='/api/export':
                    with WRITE_LOCK:
                        docs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(labels.glob('*.json'))]
                        path=exports/('review_backup_'+str(time.time_ns())+'.json');atomic_json(path,dict(schema=SCHEMA,package_id=manifest['package_id'],documents=docs))
                    return self.send({'saved':str(path)})
                if self.path=='/api/stop':
                    self.send({'stopping':True});threading.Thread(target=self.server.shutdown,daemon=True).start();return
                return self.send({'error':'not found'},status=404)
            except (ValueError,KeyError,TypeError) as e:return self.send({'error':str(e)},status=400)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--port',type=int,default=8779);a=p.parse_args()
    server=make_server(a.root,a.port);print('REVIEW_READY http://127.0.0.1:'+str(a.port),flush=True)
    try:server.serve_forever()
    finally:server.server_close()
if __name__=='__main__':main()
