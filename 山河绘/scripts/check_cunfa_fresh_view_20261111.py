import sys,json,time,subprocess,base64,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests,websocket
OUT=LAB/'cunfa_fresh_validation_20261111';profile=OUT/'browser_profile';url=json.loads((OUT/'server.json').read_text('utf8'))['url'];freeze=json.loads((OUT/'freeze_manifest.json').read_text('utf8'))
for f in freeze['files']:assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
proc=subprocess.Popen([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe','--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--remote-allow-origins=http://localhost','--user-data-dir='+str(profile),'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
try:
 for _ in range(100):
  f=profile/'DevToolsActivePort'
  if f.exists():break
  time.sleep(.1)
 port=f.read_text().splitlines()[0];pages=requests.get(f'http://127.0.0.1:{port}/json',timeout=5).json();ws=websocket.create_connection(next(x['webSocketDebuggerUrl'] for x in pages if x['type']=='page'),origin='http://localhost',timeout=8);counter=0
 def call(method,params=None):
  global counter
  counter+=1;idx=counter;ws.send(json.dumps(dict(id=idx,method=method,params=params or {})))
  while True:
   r=json.loads(ws.recv())
   if r.get('id')==idx:
    if 'error' in r:raise RuntimeError(r['error'])
    return r.get('result',{})
 def js(s):
  r=call('Runtime.evaluate',dict(expression=s,returnByValue=True,awaitPromise=True))
  if 'exceptionDetails' in r:raise RuntimeError(r['exceptionDetails'])
  return r.get('result',{}).get('value')
 def select(key,value):js(f'document.getElementById({json.dumps(key)}).value={json.dumps(value)};document.getElementById({json.dumps(key)}).dispatchEvent(new Event("change"))')
 call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False));call('Page.navigate',dict(url=url))
 for _ in range(100):
  if js('window.reportReady===true'):break
  time.sleep(.1)
 assert js('window.reportReady===true');assert js("document.querySelectorAll('#grid .card').length")==13
 for _ in range(50):
  if js('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)'):break
  time.sleep(.1)
 assert js('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
 select('status','wrong');assert js("document.querySelectorAll('#grid .card').length")==6
 select('method','local_only');assert js("document.querySelectorAll('#grid .card').length")==5
 select('status','');select('artist','Xia_Gui');assert js("document.querySelectorAll('#grid .card').length")==4
 select('artist','');select('method','local_plus_context');assert js("document.querySelectorAll('#metrics tr').length")==3;assert js("document.querySelectorAll('#excluded .card').length")==3
 assert js("document.querySelectorAll('#artists tr').length")==4
 call('Emulation.setDeviceMetricsOverride',dict(width=390,height=1000,deviceScaleFactor=1,mobile=False));assert js('document.documentElement.scrollWidth<=window.innerWidth')
 call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False));(OUT/'browser_view.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
 assert requests.post(url,timeout=5).status_code==405;assert requests.get(url+'frozen/context_head_exploratory.joblib',timeout=5).status_code==404
 result=dict(url=url,all13_inputs_contexts_neighbours_loaded=True,context_errors=6,local_errors=5,Xia_Gui_filter=4,excluded_preserved=3,authors_table=4,mobile_no_overflow=True,freeze_hashes_unchanged=True,post_disabled=True,model_not_exposed=True,robot_connected=False)
 (OUT/'browser_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False),flush=True)
 try:call('Browser.close')
 except Exception:pass
finally:
 if proc.poll() is None:proc.terminate()
