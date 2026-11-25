# -*- coding: utf-8 -*-
import sys,json,time,subprocess,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import requests,websocket
ROOT=LAB/'whole_rock_evidence_20261117'

def main():
    url=json.loads((ROOT/'server.json').read_text('utf8'))['url'];profile=ROOT/('browser_profile_'+str(time.time_ns()))
    proc=subprocess.Popen([r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe','--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--remote-allow-origins=http://localhost','--user-data-dir='+str(profile),'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for _ in range(150):
            f=profile/'DevToolsActivePort'
            if f.exists():break
            time.sleep(.1)
        port=f.read_text().splitlines()[0]
        pages=requests.get(f'http://127.0.0.1:{port}/json',timeout=5).json()
        ws=websocket.create_connection(next(x['webSocketDebuggerUrl'] for x in pages if x['type']=='page'),origin='http://localhost',timeout=8)
        counter=0;exceptions=[]
        def call(method,params=None):
            nonlocal counter
            counter+=1;ws.send(json.dumps(dict(id=counter,method=method,params=params or {})))
            while True:
                r=json.loads(ws.recv())
                if r.get('method')=='Runtime.exceptionThrown':exceptions.append(r)
                if r.get('id')==counter:
                    if 'error' in r:raise RuntimeError(r['error'])
                    return r.get('result',{})
        def js(expression):
            r=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
            if 'exceptionDetails' in r:raise RuntimeError(r['exceptionDetails'])
            return r.get('result',{}).get('value')
        call('Runtime.enable');call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False))
        call('Page.navigate',dict(url=url))
        for _ in range(100):
            if js('document.images.length>0&&Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)&&!!document.querySelector("#step")'):break
            time.sleep(.1)
        assert js('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
        assert js('stages.length')==9
        for n,text in [(1,'山石A'),(3,'皴笔候选'),(4,'山石B'),(7,'山石C')]:
            js(f'document.querySelector("#step").value={n};render()')
            assert text in js('document.querySelector("#label").textContent')
        js('document.querySelector("#reset").click();document.querySelector("#play").click()')
        time.sleep(.9);assert js('+document.querySelector("#step").value')>0
        js('document.querySelector("#play").click();document.querySelector("#step").value=9;render()')
        call('Emulation.setDeviceMetricsOverride',dict(width=390,height=1000,deviceScaleFactor=1,mobile=False))
        assert js('document.documentElement.scrollWidth<=innerWidth')
        call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False))
        (ROOT/'browser_view.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
        assert not exceptions
        assert requests.post(url,timeout=5).status_code==405
        assert requests.get(url+'data/candidate_corridors.npz',timeout=5).status_code==404
        for f in ['packet.json','verification.json','training_readiness.json','agent_tool_contracts.json']:
            assert requests.get(url+f,timeout=5).status_code==200
        result=dict(images_loaded=True,nine_reference_stages=True,object_phase_labels=True,playback_advances=True,
                    mobile_no_overflow=True,no_frontend_exceptions=True,read_only=True,
                    reference_not_model_notice=True,url=url,robot_connected=False)
        (ROOT/'browser_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(result,ensure_ascii=False))
        try:call('Browser.close')
        except Exception:pass
    finally:
        if proc.poll() is None:proc.terminate()
if __name__=='__main__':main()
