import sys,json,time,subprocess,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests,websocket
OUT=LAB/'cunfa_band_checked_20261112';profile=OUT/'browser_profile';url=json.loads((OUT/'server.json').read_text('utf8'))['url']
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
 def loaded():
  for _ in range(80):
   if js('Array.from(document.images).filter(i=>i.hasAttribute("src")).every(i=>i.complete&&i.naturalWidth>0)'):return
   time.sleep(.1)
  raise AssertionError('Image not loaded')
 call('Emulation.setDeviceMetricsOverride',dict(width=1520,height=1160,deviceScaleFactor=1,mobile=False));call('Page.navigate',dict(url=url))
 for _ in range(100):
  if js('window.reportReady===true'):break
  time.sleep(.1)
 assert js('window.reportReady===true');assert js("document.querySelectorAll('#case option').length")==5;assert js("document.querySelectorAll('#metrics tr').length")==2
 for i in range(5):
  js(f'document.getElementById("case").selectedIndex={i};document.getElementById("case").dispatchEvent(new Event("change"))');loaded()
 assert js('document.getElementById("split").textContent.includes("复核")');assert js('document.getElementById("conclusion").textContent.includes("新增遗漏")');assert js('document.body.textContent.includes("不是新作者盲测")')
 call('Emulation.setDeviceMetricsOverride',dict(width=390,height=1000,deviceScaleFactor=1,mobile=False));assert js('document.documentElement.scrollWidth<=window.innerWidth')
 call('Emulation.setDeviceMetricsOverride',dict(width=1520,height=1160,deviceScaleFactor=1,mobile=False));(OUT/'browser_view.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
 assert requests.post(url,timeout=5).status_code==405
 for name in ['flat_roles.pt','ink_then_role.pt','../flat_roles.pt']:assert requests.get(url+name,timeout=5).status_code==404
 for name in ['protocol.json','results.json','decision.json','verification.json','width_comparison.jpg']:assert requests.get(url+name,timeout=5).ok
 v=json.loads((OUT/'verification.json').read_text('utf8'));assert v['unknown_edge_gradient_zero'] and v['old_v2_model_unchanged']
 result=dict(url=url,all_five_reference_cases_load=True,all_raw_prediction_images_load=True,comparison_table_two_methods=True,regression_disclosed=True,limited_heldout_not_claimed_blind=True,mobile_no_overflow=True,post_disabled=True,weights_not_served=True,robot_connected=False)
 (OUT/'browser_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False),flush=True)
 try:call('Browser.close')
 except Exception:pass
finally:
 if proc.poll() is None:proc.terminate()
