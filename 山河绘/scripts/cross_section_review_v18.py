# -*- coding: utf-8 -*-
"""Local-only cross-section annotation UI; saves directly to E, no browser downloads."""
from rock_v15_common import LAB,read,jsave
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import json,time,threading,io
from PIL import Image,ImageDraw
ROOT=LAB/'cross_sections_v18_20261025'
STORE=ROOT/'expert_cross_sections'
STORE.mkdir(parents=True,exist_ok=True)
CANDIDATES=read(ROOT/'review_candidates.json')
CANDIDATES=[r for r in CANDIDATES if r['sample_id'] in ['R01','R02','N04','N05','N06','N09']]
ALLOWED={r['id']:r for r in CANDIDATES}
SOURCES={r['id']:r for r in read(ROOT/'manifest.json')['records']}
HTML=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>笔触横截面复核</title><style>
body{font:16px system-ui;margin:0;background:#efeee8;color:#242722}header{position:sticky;top:0;z-index:5;background:#faf9f4;border-bottom:1px solid #ccc;padding:14px 22px}h1{font-size:21px;margin:0 0 8px}p{margin:5px 0;line-height:1.6}.tools{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,select{font:inherit;padding:7px 12px;border:1px solid #aaa;border-radius:6px;background:white;cursor:pointer}button.save{background:#244d41;color:white}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(410px,1fr));gap:18px;padding:20px}.card{background:#fff;padding:14px;border-radius:10px;border:2px solid transparent}.card.checked{border-color:#3d866b}.card h2{font-size:17px;margin:0 0 7px}canvas{display:block;width:384px;max-width:100%;height:auto;touch-action:none;border:1px solid #ddd;cursor:crosshair;margin:10px 0}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.meta{font-size:13px;color:#666}.legend{font-size:14px}#status{margin-left:auto}input[type=checkbox]{width:20px;height:20px;vertical-align:middle}.badge{color:#805d13}
</style><header><h1>笔触横截面复核</h1><p>每张小图把笔触转正并放大。红线标中心，橙线标两侧边缘。先选要调整的线，再点击或拖动到正确位置；边缘以连续可见的主墨带为准；飞白小孔不另分笔，零散纸纹不纳入。灰淡但可见的笔触请选择“淡墨”；无法判断时选“不确定”。</p><div class="tools"><label>当前调整 <select id="guide"><option value="center">中心（红）</option><option value="left">左边缘（橙）</option><option value="right">右边缘（橙）</option></select></label><button class="save" id="save">保存到 E 盘</button><span id="status">正在读取…</span></div><p class="meta">仅“已检查”的项目作为你的复核意见。其余是程序派生草稿。保存到 E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\cross_sections_v18_20261025\expert_cross_sections，修改不会加入正在运行的训练。</p></header><main id="cards"></main><script>
let rows=[],state={},images={},dirty=false;const status=document.querySelector('#status');
function report(){let n=Object.values(state).filter(s=>s.reviewed).length;status.textContent=`已检查 ${n}/${rows.length} 项${dirty?' · 尚未保存':''}`}
function draw(r,canvas){const s=state[r.id],c=canvas.getContext('2d');c.clearRect(0,0,384,192);c.drawImage(images[r.id],0,0,384,192);for(const key of ['left','right','center']){let x=(95.5+s[key]*3)*2;c.strokeStyle=key==='center'?'#e33042':'#d99908';c.lineWidth=key==='center'?2.5:2;c.beginPath();c.moveTo(x,0);c.lineTo(x,192);c.stroke()}c.strokeStyle='#1f567799';c.setLineDash([5,5]);c.beginPath();c.moveTo(0,96);c.lineTo(384,96);c.stroke();c.setLineDash([])}
function render(){document.querySelector('#cards').replaceChildren();for(const r of rows){const s=state[r.id];const card=document.createElement('article');card.className='card'+(s.reviewed?' checked':'');const title=document.createElement('h2');title.textContent=`${r.sample_id} · ${r.kind==='faint'?'淡笔候选':r.kind==='background'?'纸纹／边框候选':'粗笔候选'} · ${r.id}`;card.append(title);const canvas=document.createElement('canvas');canvas.width=384;canvas.height=192;card.append(canvas);const row=document.createElement('div');row.className='row';const select=document.createElement('select');for(const [v,t] of [['ink','清楚墨线'],['faint','淡墨'],['background','背景／无笔画'],['uncertain','不确定']]){let o=new Option(t,v);select.append(o)}select.value=s.label;select.onchange=()=>{s.label=select.value;dirty=true;report()};row.append(select);const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=s.reviewed;check.onchange=()=>{s.reviewed=check.checked;card.classList.toggle('checked',check.checked);dirty=true;report()};label.append(check,document.createTextNode(' 已检查'));row.append(label);const reset=document.createElement('button');reset.textContent='恢复草稿';reset.onclick=()=>{Object.assign(s,{left:r.left,center:r.center,right:r.right,reviewed:false,label:r.kind==='faint'?'faint':r.kind==='background'?'uncertain':'ink'});dirty=true;draw(r,canvas);report();check.checked=false;card.classList.remove('checked')};row.append(reset);card.append(row);const meta=document.createElement('p');meta.className='meta';meta.textContent='横向为笔画宽度，虚线处为该横截面。连续看不清或与别的笔画重叠可标“不确定”。';card.append(meta);const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='查看原图位置';const context=document.createElement('img');context.src='/context/'+r.id;context.loading='lazy';context.style.maxWidth='100%';details.append(summary,context);card.append(details);let drag=false;const change=e=>{let box=canvas.getBoundingClientRect(),px=(e.clientX-box.left)/box.width*384,value=Math.max(-23.5,Math.min(23.5,(px/2-95.5)/3));s[document.querySelector('#guide').value]=Math.round(value*100)/100;dirty=true;draw(r,canvas);report()};canvas.onpointerdown=e=>{drag=true;canvas.setPointerCapture(e.pointerId);change(e)};canvas.onpointermove=e=>{if(drag)change(e)};canvas.onpointerup=()=>drag=false;const img=new Image;images[r.id]=img;img.onload=()=>draw(r,canvas);img.src='data:image/png;base64,'+r.clip;document.querySelector('#cards').append(card)}}
async function save(){const bad=rows.filter(r=>state[r.id].reviewed&&['ink','faint'].includes(state[r.id].label)&&!(state[r.id].left<state[r.id].center&&state[r.id].center<state[r.id].right));if(bad.length){alert('中心必须在两侧边缘之间，请检查：'+bad.map(r=>r.id).join('、'));return}status.textContent='保存中…';try{let resp=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({annotations:Object.values(state)})});let x=await resp.json();if(!resp.ok)throw Error(x.error);dirty=false;report();status.textContent+=' · 已保存至 E 盘'}catch(e){status.textContent='保存失败：'+e.message}}
document.querySelector('#save').onclick=save;window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue=''}});(async()=>{try{let data=await(await fetch('/data')).json();rows=data.rows;for(const r of rows)state[r.id]={id:r.id,left:r.left,center:r.center,right:r.right,label:r.kind==='faint'?'faint':r.kind==='background'?'uncertain':'ink',reviewed:false,...data.saved[r.id]};render();report()}catch(e){status.textContent=e.message}})();
</script></html>'''
LOCK=threading.Lock()
class Handler(BaseHTTPRequestHandler):
    def respond(self,code,payload,kind='application/json; charset=utf-8'):
        b=payload.encode('utf-8') if isinstance(payload,str) else json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8');self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if self.path=='/':return self.respond(200,HTML,'text/html; charset=utf-8')
        if self.path.startswith('/context/'):
            key=self.path.split('/')[-1]
            if key not in ALLOWED:return self.respond(404,dict(error='Unknown section'))
            row=ALLOWED[key];rgb=Image.open(SOURCES[row['sample_id']]['image']).convert('RGB');d=ImageDraw.Draw(rgb);p=row['p'];n=row['normal'];a=(p[0]-24*n[0],p[1]-24*n[1]);b=(p[0]+24*n[0],p[1]+24*n[1]);d.line([a,b],fill='#d52d39',width=2);d.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),outline='#d52d39',width=2);rgb.thumbnail((520,420));buf=io.BytesIO();rgb.save(buf,format='PNG');payload=buf.getvalue();self.send_response(200);self.send_header('Content-Type','image/png');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload);return
        if self.path=='/data':
            saved=read(STORE/'reviewed.json')['annotations'] if (STORE/'reviewed.json').exists() else []
            return self.respond(200,dict(rows=CANDIDATES,saved={r['id']:r for r in saved}))
        self.respond(404,dict(error='Not found'))
    def do_POST(self):
        if self.path!='/save':return self.respond(404,dict(error='Not found'))
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<1000000:raise ValueError('Payload size')
            rows=json.loads(self.rfile.read(length))['annotations'];clean=[]
            for r in rows:
                if r['id'] not in ALLOWED or r['label'] not in ['ink','faint','background','uncertain']:raise ValueError('Unknown annotation')
                vals={k:float(r[k]) for k in ['left','center','right']}
                if any(not -24<v<24 for v in vals.values()):raise ValueError('Guide outside cross section')
                reviewed=bool(r.get('reviewed',False))
                if reviewed and r['label'] in ['ink','faint'] and not vals['left']<vals['center']<vals['right']:raise ValueError('Center must lie within edges')
                clean.append(dict(id=r['id'],sample_id=ALLOWED[r['id']]['sample_id'],label=r['label'],reviewed=reviewed,**vals))
            obj=dict(annotations=clean,reviewed_count=sum(r['reviewed'] for r in clean),source='user_cross_section_review',not_training_until_explicitly_ingested=True,saved_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            with LOCK:
                jsave(STORE/('revision_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns())+'.json'),obj);jsave(STORE/'reviewed.json',obj)
            self.respond(200,dict(ok=True,reviewed_count=obj['reviewed_count']))
        except Exception as e:self.respond(400,dict(error=str(e)))
    def log_message(self,fmt,*args):print(fmt%args,flush=True)
if __name__=='__main__':
    print('CROSS_SECTION_REVIEW http://127.0.0.1:8781/; E-disk storage; %d candidates'%len(CANDIDATES),flush=True)
    ThreadingHTTPServer(('127.0.0.1',8781),Handler).serve_forever()


