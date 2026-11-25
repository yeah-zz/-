# -*- coding: utf-8 -*-
"""Headless browser verifies replay transitions and read-only presentation."""
import sys,json,time,subprocess,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests,websocket
OUT=LAB/'rock_skill_offline_20261110';profile=OUT/'browser_profile'
url=json.loads((OUT/'server.json').read_text('utf-8'))['url']
proc=subprocess.Popen([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe','--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--remote-allow-origins=http://localhost','--user-data-dir='+str(profile),'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
try:
 for _ in range(100):
  f=profile/'DevToolsActivePort'
  if f.exists():break
  time.sleep(.1)
 port=f.read_text().splitlines()[0];pages=requests.get(f'http://127.0.0.1:{port}/json',timeout=5).json();ws=websocket.create_connection(next(x['webSocketDebuggerUrl'] for x in pages if x['type']=='page'),origin='http://localhost',timeout=8)
 counter=0
 def call(method,params=None):
  global counter
  counter+=1;idx=counter;ws.send(json.dumps(dict(id=idx,method=method,params=params or {})))
  while True:
   r=json.loads(ws.recv())
   if r.get('id')==idx:
    if 'error' in r:raise RuntimeError(r['error'])
    return r.get('result',{})
 def js(s):return call('Runtime.evaluate',dict(expression=s,returnByValue=True,awaitPromise=True)).get('result',{}).get('value')
 call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1400,deviceScaleFactor=1,mobile=False));call('Page.navigate',dict(url=url))
 for _ in range(80):
  if js("document.querySelectorAll('.item').length") == 27:break
  time.sleep(.1)
 assert js("document.querySelectorAll('.item').length")==27
 assert js("Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)")
 js("document.getElementById('moduleButtons').children[1].click()")
 assert 'B' in js("document.getElementById('state').textContent")
 js("document.getElementById('reset').click();document.getElementById('play').click()")
 time.sleep(.8)
 assert js("Number(document.getElementById('scrub').value)")>=2
 js("document.getElementById('play').click()")
 js("document.getElementById('scrub').value=27;document.getElementById('scrub').dispatchEvent(new Event('input'))")
 assert 'C' in js("document.getElementById('state').textContent")
 state=js("document.getElementById('state').textContent")
 assert '训练新提取权重' in js("document.body.innerText")
 call('Emulation.setDeviceMetricsOverride',dict(width=390,height=1000,deviceScaleFactor=1,mobile=False))
 assert js("document.documentElement.scrollWidth<=window.innerWidth")
 call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1400,deviceScaleFactor=1,mobile=False))
 image=call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=True))['data'];(OUT/'browser_view.png').write_bytes(base64.b64decode(image))
 result=dict(page=url,images_loaded=True,reference_segments=27,playback_advances=True,module_jump=True,final_state=state,mobile_no_horizontal_overflow=True,not_model_success_notice=True,robot_connected=False)
 (OUT/'browser_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
 try:call('Browser.close')
 except Exception:pass
finally:
 if proc.poll() is None:proc.terminate()
