"""Local review editor. Saves labels directly to E, never browser Downloads."""
from http.server import BaseHTTPRequestHandler,HTTPServer
from io import BytesIO
import argparse
import base64
import json
import time
from PIL import Image
import numpy as np
from .runtime import LAB

HTML='''<!doctype html><meta charset="utf-8"><title>工笔结构线标注</title>
<style>body{font:16px sans-serif;background:#eee9de;color:#24362b}canvas{max-width:95vw;cursor:crosshair}button,select,input{font:inherit;margin:5px}#stack{position:relative;display:inline-block}#labels{position:absolute;left:0;top:0;opacity:.75}</style>
<h2>结构线复核 / 标注</h2><p>只标你能确认的位置。未知区域不作为负样本；标注尺寸与处理后的原图一致。保存直接写入 E 盘。</p>
<select id="record"></select><select id="mode"><option value="2">结构线（红）</option><option value="1">明确背景（绿）</option><option value="3">细节线（蓝）</option><option value="0">擦除为未知</option></select>
笔宽<input id="width" type="number" min="1" max="100" value="4"><button id="save">保存到 E 盘</button><span id="status"></span><br>
<div id="stack"><canvas id="photo"></canvas><canvas id="labels"></canvas></div>
<script>
const photo=document.querySelector('#photo'),canvas=document.querySelector('#labels'),ctx=canvas.getContext('2d',{willReadFrequently:true}),sel=document.querySelector('#record');let drawing=false,last=null,revision=null;
const colors=['rgba(0,0,0,0)','#23bf70','#ed2939','#3368ff'];let records=[];
async function openRecord(){let r=records.find(x=>x.id===sel.value),im=new Image();im.src='/image/'+r.id;await im.decode();photo.width=canvas.width=im.width;photo.height=canvas.height=im.height;photo.getContext('2d').drawImage(im,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);let res=await fetch('/labels/'+r.id);let d=await res.json();revision=d.revision;if(d.rgba){let img=new Image();img.src='data:image/png;base64,'+d.rgba;await img.decode();ctx.drawImage(img,0,0);}document.querySelector('#status').textContent=r.split==='train'?'训练组':'独立验证组：保持留出，不加入训练';}
function point(e){let b=canvas.getBoundingClientRect();return[(e.clientX-b.left)*canvas.width/b.width,(e.clientY-b.top)*canvas.height/b.height]}
function stroke(e){let p=point(e),v=+document.querySelector('#mode').value;ctx.globalCompositeOperation=v===0?'destination-out':'source-over';ctx.strokeStyle=colors[v];ctx.fillStyle=colors[v];ctx.lineWidth=+document.querySelector('#width').value;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(...(last||p));ctx.lineTo(...p);ctx.stroke();last=p;}
canvas.onpointerdown=e=>{drawing=true;last=point(e);canvas.setPointerCapture(e.pointerId);stroke(e)};canvas.onpointermove=e=>{if(drawing)stroke(e)};canvas.onpointerup=()=>{drawing=false;last=null};
document.querySelector('#save').onclick=async()=>{let d=ctx.getImageData(0,0,canvas.width,canvas.height),labels=new Uint8Array(canvas.width*canvas.height);for(let i=0;i<labels.length;i++){let j=i*4;if(d.data[j+3]<100)continue;labels[i]=d.data[j]>150?2:(d.data[j+1]>120?1:3)}let text='';for(let i=0;i<labels.length;i+=10000)text+=String.fromCharCode(...labels.subarray(i,i+10000));let response=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:sel.value,width:canvas.width,height:canvas.height,data:btoa(text),revision})});let result=await response.json();document.querySelector('#status').textContent=JSON.stringify(result);if(response.ok)revision=result.revision;};
fetch('/records').then(x=>x.json()).then(x=>{records=x;sel.innerHTML=x.map(r=>`<option>${r.id}</option>`).join('');sel.onchange=openRecord;openRecord()});
</script>'''


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766);args=parser.parse_args(argv)
    manifest=json.loads((LAB/'datasets'/'manifest.json').read_text(encoding='utf-8'))
    records={r['id']:r for r in manifest['real']}
    folder=LAB/'data'/'manual';folder.mkdir(exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def send(self,data,kind='application/json',status=200):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/':return self.send(HTML.encode('utf-8'),'text/html; charset=utf-8')
            if self.path=='/records':return self.send([dict(id=r['id'],split=r['split']) for r in records.values()])
            kind,_,key=self.path.lstrip('/').partition('/')
            if key not in records:return self.send({'error':'unknown record'},status=404)
            if kind=='image':return self.send(PathImage(records[key]['image']),'image/png')
            if kind=='labels':
                path=folder/(key+'.png');revision=None
                if path.exists():
                    a=np.asarray(Image.open(path));lut=np.array([[0,0,0,0],[35,191,112,255],[237,41,57,255],[51,104,255,255]],np.uint8)
                    buffer=BytesIO();Image.fromarray(lut[a]).save(buffer,format='PNG');rgba=base64.b64encode(buffer.getvalue()).decode();revision=str(path.stat().st_mtime_ns)
                else:rgba=None
                return self.send(dict(rgba=rgba,revision=revision))
            self.send({'error':'not found'},status=404)
        def do_POST(self):
            # Loopback binding plus exact Origin/Host stops cross-site writes.
            allowed=f'127.0.0.1:{args.port}'
            if self.path!='/save' or self.headers.get('Host')!=allowed or self.headers.get('Origin')!='http://'+allowed:return self.send({'error':'origin rejected'},status=403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if size<=0 or size>12_000_000:raise ValueError('invalid body size')
                d=json.loads(self.rfile.read(size));r=records[d['id']];w,h=r['size']
                if [d['width'],d['height']]!=[w,h]:raise ValueError('image shape mismatch')
                data=base64.b64decode(d['data'],validate=True)
                labels=np.frombuffer(data,np.uint8).reshape(h,w)
                if labels.max()>3:raise ValueError('invalid class')
                path=folder/(r['id']+'.png');revision=str(path.stat().st_mtime_ns) if path.exists() else None
                if d.get('revision')!=revision:return self.send({'error':'标签已在别处更新，请重新载入'},status=409)
                if path.exists():path.replace(folder/(r['id']+f'.backup_{time.time_ns()}.png'))
                Image.fromarray(labels).save(path)
                meta=dict(id=r['id'],split=r['split'],group=r['group'],classes={'0':'unknown','1':'background','2':'structure','3':'detail'},
                          label_type='user_reviewed_sparse_pixels',dense_gold=False,edited_unix=time.time())
                (folder/(r['id']+'.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
                self.send(dict(saved=str(path),revision=str(path.stat().st_mtime_ns)))
            except (ValueError,KeyError,TypeError) as error:self.send({'error':str(error)},status=400)
    print(f'ANNOTATION http://127.0.0.1:{args.port} | E-drive save only',flush=True)
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


def PathImage(path):
    from pathlib import Path
    return Path(path).read_bytes()
