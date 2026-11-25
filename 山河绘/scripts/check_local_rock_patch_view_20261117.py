import sys,json,time,subprocess,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests,websocket
OUT=LAB/'local_rock_patch_20261117';profile=OUT/('browser_profile_'+str(time.time_ns()));url=json.loads((OUT/'server.json').read_text('utf8'))['url']
def main():
 proc=subprocess.Popen([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe','--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--remote-allow-origins=http://localhost','--user-data-dir='+str(profile),'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  for _ in range(150):
   f=profile/'DevToolsActivePort'
   if f.exists():break
   time.sleep(.1)
  port=f.read_text().splitlines()[0];pages=requests.get(f'http://127.0.0.1:{port}/json',timeout=5).json();ws=websocket.create_connection(next(x['webSocketDebuggerUrl'] for x in pages if x['type']=='page'),origin='http://localhost',timeout=8);counter=0
  def call(method,params=None):
   nonlocal counter
   counter+=1;ws.send(json.dumps(dict(id=counter,method=method,params=params or {})))
   while True:
    r=json.loads(ws.recv())
    if r.get('id')==counter:
     if 'error' in r:raise RuntimeError(r['error'])
     return r.get('result',{})
  def js(s):
   r=call('Runtime.evaluate',dict(expression=s,returnByValue=True,awaitPromise=True))
   if 'exceptionDetails' in r:raise RuntimeError(r['exceptionDetails'])
   return r.get('result',{}).get('value')
  def loaded():
   for _ in range(120):
    if js('plan!==null&&sourceImage.complete&&sourceImage.naturalWidth>0'):return
    time.sleep(.1)
   raise AssertionError('Plan/source not loaded')
  call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1150,deviceScaleFactor=1,mobile=False));call('Page.navigate',dict(url=url));loaded()
  assert js('document.querySelectorAll("#scene option").length')==3;assert js('scene')=='G03_01_local_candidate';assert js('document.getElementById("scene").value')=='G03_01_local_candidate';assert js('document.getElementById("failure").textContent')==''
  assert js('plan.curves.length')==489;assert js('localRange()[1]-localRange()[0]')==21
  for off,word in [(0,'主轮廓'),(4,'结构转折'),(8,'纹理候选')]:
   js(f'cursor=localRange()[0]+{off};render()');assert js('document.querySelector("#phases .on").textContent.includes('+json.dumps(word)+')')
  js('cursor=localRange()[1]-.01;playing=true;document.getElementById("speed").value="650"')
  for _ in range(30):
   if not js('playing'):break
   time.sleep(.05)
  assert js('cursor===localRange()[1]&&!playing')
  for scene,count in [('G03_01_approved500',500),('G03_01_baseline489',489),('G03_01_local_candidate',489)]:
   js('load('+json.dumps(scene)+').then(focusRock)');loaded();assert js('plan.curves.length')==count
  js('document.getElementById("scene").value="G03_01_local_candidate";focusRock();document.getElementById("ghost").checked=false;render()')
  call('Emulation.setDeviceMetricsOverride',dict(width=390,height=1000,deviceScaleFactor=1,mobile=False));assert js('document.documentElement.scrollWidth<=innerWidth')
  call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1150,deviceScaleFactor=1,mobile=False));(OUT/'platform_preview.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  state=requests.get(url+'api/state',timeout=5).json();assert state['status']=='not_bound' and not state['camera_connected']
  assert requests.post(url,timeout=5).status_code==405;assert requests.get(url+'baseline_lock.json',timeout=5).status_code==404;assert requests.get(url+'../baseline489.json',timeout=5).status_code==404
  call('Page.navigate',dict(url=url+'report.html'))
  for _ in range(100):
   if js('document.images.length>0&&Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)'):break
   time.sleep(.1)
  assert js('document.images.length')==1;assert js('document.body.textContent.includes("暂不采用")');assert js('document.body.textContent.includes("没有换算成下压")')
  call('Emulation.setDeviceMetricsOverride',dict(width=390,height=1000,deviceScaleFactor=1,mobile=False));assert js('document.documentElement.scrollWidth<=innerWidth')
  checks=dict(three_baselines_switch=True,local21_action_block=True,local_three_phases_display=True,local_play_stops_at_block_end=True,no_frontend_errors=True,mobile_no_overflow=True,report_image_loaded=True,read_only=True,no_robot_binding=True,robot_connected=False,url=url)
  (OUT/'browser_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(checks,ensure_ascii=False),flush=True)
  try:call('Browser.close')
  except Exception:pass
 finally:
  if proc.poll() is None:proc.terminate()
if __name__=='__main__':main()
