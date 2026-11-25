import sys,json,time,subprocess,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests,websocket
OUT=LAB/'cunfa_gabor_baseline_20261110';profile=OUT/'browser_profile'
url=json.loads((OUT/'server.json').read_text('utf8'))['url']
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
 call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False));call('Page.navigate',dict(url=url))
 for _ in range(100):
  if js('window.reportReady===true'):break
  time.sleep(.1)
 assert js('window.reportReady===true')
 assert js("document.querySelectorAll('#grid .card').length")==40
 for _ in range(30):
  if js('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)'):break
  time.sleep(.1)
 assert js('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
 js("document.getElementById('status').value='wrong';document.getElementById('status').dispatchEvent(new Event('change'))")
 assert js("document.querySelectorAll('#grid .card').length")==23
 js("document.getElementById('status').value='';document.getElementById('cl').selectedIndex=1;document.getElementById('cl').dispatchEvent(new Event('change'))")
 assert js("document.querySelectorAll('#grid .card').length")==20
 js("document.getElementById('cl').selectedIndex=0;document.getElementById('artist').value='Li_Tang';document.getElementById('artist').dispatchEvent(new Event('change'))")
 assert js("document.querySelectorAll('#grid .card').length")==6
 js("document.getElementById('artist').value='';document.getElementById('artist').dispatchEvent(new Event('change'))")
 assert js("document.querySelectorAll('#probes .card').length")==5
 call('Emulation.setDeviceMetricsOverride',dict(width=390,height=1000,deviceScaleFactor=1,mobile=False))
 assert js('document.documentElement.scrollWidth<=window.innerWidth')
 call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False))
 (OUT/'browser_view.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
 assert requests.get(url+'gabor_pca_svm_pilot.joblib',timeout=5).status_code==404
 assert requests.post(url,timeout=5).status_code==405
 result=dict(url=url,known_images=40,other_style_images=5,images_loaded=True,error_filter_count=23,class_filter_count=20,artist_filter_count=6,mobile_no_horizontal_overflow=True,weight_not_exposed=True,post_disabled=True,robot_connected=False)
 (OUT/'browser_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False),flush=True)
 try:call('Browser.close')
 except Exception:pass
finally:
 if proc.poll() is None:proc.terminate()
