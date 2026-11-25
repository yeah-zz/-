import sys,json,time,subprocess,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests,websocket
OUT=LAB/'guohua_objects_v2_20261113';profile=OUT/('browser_profile_'+str(time.time_ns()));url=json.loads((OUT/'server.json').read_text('utf8'))['url']
proc=subprocess.Popen([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe','--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--remote-allow-origins=http://localhost','--user-data-dir='+str(profile),'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
try:
 for _ in range(150):
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
 def loaded():
  for _ in range(100):
   if js('Array.from(document.images).filter(i=>i.hasAttribute("src")).every(i=>i.complete&&i.naturalWidth>0)'):return
   time.sleep(.05)
  raise AssertionError('Image not loaded')
 call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1150,deviceScaleFactor=1,mobile=False));call('Page.navigate',dict(url=url))
 for _ in range(100):
  if js('window.reportReady===true'):break
  time.sleep(.1)
 assert js('window.reportReady===true');assert js("document.querySelectorAll('#case option').length")==27;assert js("document.querySelectorAll('#metrics tr').length")==6;assert js("document.querySelectorAll('#works tr').length")==4
 for i in range(27):
  js(f'document.getElementById("case").selectedIndex={i};document.getElementById("case").dispatchEvent(new Event("change"))');loaded()
 js('document.getElementById("case").selectedIndex=0;document.getElementById("case").dispatchEvent(new Event("change"))');loaded();assert js('document.getElementById("refnote").textContent.includes("无完整对象边界")');assert js('document.body.textContent.includes("树木画法训练保持关闭")')
 call('Emulation.setDeviceMetricsOverride',dict(width=390,height=1000,deviceScaleFactor=1,mobile=False));assert js('document.documentElement.scrollWidth<=window.innerWidth')
 call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1150,deviceScaleFactor=1,mobile=False));(OUT/'browser_view.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
 assert requests.post(url,timeout=5).status_code==405
 for name in ['expanded_control.pt','line_aware_aug.pt','../expanded_control.pt','predictions/H_14_expanded_control.npz']:assert requests.get(url+name,timeout=5).status_code==404
 for name in ['protocol.json','results.json','selection_before_download.json','source_checked_cunfa_concepts.json','verification.json']:assert requests.get(url+name,timeout=5).ok
 result=dict(url=url,all27_observations_switch_and_load=True,metrics_six_rows=True,work_metrics_four_rows=True,user_regression_disclosed=True,tree_skill_stage_still_closed=True,mobile_no_overflow=True,post_disabled=True,weights_not_served=True,robot_connected=False)
 (OUT/'browser_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False),flush=True)
 try:call('Browser.close')
 except Exception:pass
finally:
 if proc.poll() is None:proc.terminate()
