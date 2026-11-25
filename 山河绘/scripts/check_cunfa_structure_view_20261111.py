# -*- coding: utf-8 -*-
import sys,json,time,subprocess,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests,websocket,joblib
OUT=LAB/'cunfa_structure_20261111';profile=OUT/'browser_profile';url=json.loads((OUT/'server.json').read_text('utf8'))['url']
data=json.loads((OUT/'dataset_manifest.json').read_text('utf8'));assert len(data['external_holdout'])==23
for kind in ['structure_statistics','structure_plus_gabor','colour_control']:
 model=joblib.load(OUT/(kind+'_pilot.joblib'));assert set(model['trained_ids'])=={r['id'] for r in data['development_records']};assert set(model['trained_ids']).isdisjoint({r['id'] for r in data['external_holdout']})
queue=json.loads((OUT/'expert_review_queue.json').read_text('utf8'));assert len(queue)==63 and all(r['reviewed_local_label'] is None and not r['eligible_for_formal_test'] for r in queue)
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
 def setselect(key,value):js(f'document.getElementById({json.dumps(key)}).value={json.dumps(value)};document.getElementById({json.dumps(key)}).dispatchEvent(new Event("change"))')
 call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False));call('Page.navigate',dict(url=url))
 for _ in range(100):
  if js('window.reportReady===true'):break
  time.sleep(.1)
 assert js('window.reportReady===true');assert js("document.querySelectorAll('#grid .card').length")==23
 for _ in range(50):
  if js('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)'):break
  time.sleep(.1)
 assert js('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
 setselect('status','wrong');assert js("document.querySelectorAll('#grid .card').length")==10
 setselect('method','frozen_gabor');assert js("document.querySelectorAll('#grid .card').length")==7
 setselect('status','');setselect('artist','Liu_Songnian');assert js("document.querySelectorAll('#grid .card').length")==2
 setselect('artist','');setselect('method','structure_plus_gabor');assert js("document.querySelectorAll('#grid .card').length")==23
 assert js("document.querySelectorAll('#metrics tr').length")==4
 call('Emulation.setDeviceMetricsOverride',dict(width=390,height=1000,deviceScaleFactor=1,mobile=False));assert js('document.documentElement.scrollWidth<=window.innerWidth')
 call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False));(OUT/'browser_view.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
 assert requests.post(url,timeout=5).status_code==405;assert requests.get(url+'structure_statistics_pilot.joblib',timeout=5).status_code==404
 result=dict(url=url,known_test_cards=23,all_crop_context_diagnostic_images_loaded=True,primary_errors=10,frozen_errors=7,artist_filter_count=2,metric_rows=4,mobile_no_horizontal_overflow=True,all_models_trained_only_on_old40=True,expert_review_63_answers_unfilled=True,post_disabled=True,model_weights_not_exposed=True,robot_connected=False)
 (OUT/'browser_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False),flush=True)
 try:call('Browser.close')
 except Exception:pass
finally:
 if proc.poll() is None:proc.terminate()
