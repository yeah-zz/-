# -*- coding: utf-8 -*-
"""Exercise real browser interactions and annotation semantics in an isolated copy."""
import sys,os,json,time,shutil,threading,subprocess,base64,urllib.request,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.review_server_v11 import make_server,empty_document,validate_document
from prepare_user_review_v11 import ROOT
from compile_user_review_v11 import compile_labels,rasterize
import websocket
import numpy as np

def run():
    sandbox=ROOT/'selftest'/('run_'+str(time.time_ns()));sandbox.mkdir()
    for name in ('package_manifest.json','review_app.py','review.html','compile_labels.py'):shutil.copy2(ROOT/name,sandbox/name)
    for name in ('samples','overlays'):shutil.copytree(ROOT/name,sandbox/name)
    manifest=json.loads((sandbox/'package_manifest.json').read_text(encoding='utf-8'))
    record=manifest['samples'][0];fixture=empty_document(record,manifest['package_id']);fixture['strokes']=[dict(id='test_line',kind='stroke',points=[[10,10],[30,30]],role='cun',cunfa='unknown',width_profile=['unknown']*3,tone_profile=['unknown']*3,direction_known=False,note='test fixture only')]
    fixture['regions']=[dict(id='test_bg',kind='background',points=[[50,50],[80,50],[80,80],[50,80]],reason='paper_texture',note='test fixture only')]
    maps=rasterize(fixture);assert maps['centerline_confidence'][200,200]==0;assert maps['centerline_confidence'][20,20]==1;assert maps['explicit_background'][60,60]==1
    fixture['regions'].append(dict(id='test_unknown',kind='uncertain',points=[[15,15],[25,15],[25,25],[15,25]],reason='other',note='test fixture only'))
    assert rasterize(fixture)['centerline_confidence'][20,20]==0
    bad=json.loads(json.dumps(fixture));bad['strokes'][0]['points'][0][0]=-1
    try:validate_document(bad,record,manifest['package_id']);raise AssertionError('invalid coordinates accepted')
    except ValueError:pass
    bad=json.loads(json.dumps(fixture));bad['regions'][0]['points']=[[5,5],[35,5],[35,35],[5,35]]
    try:rasterize(bad);raise AssertionError('positive/negative overlap accepted')
    except ValueError:pass
    server=make_server(sandbox,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();origin='http://127.0.0.1:'+str(server.server_port)
    # New E-drive browser profile; never reuse the user's ordinary browser.
    edge=Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe');log=(sandbox/'edge.log').open('wb');proc=None;ws=None
    try:
        proc=subprocess.Popen([str(edge),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-extensions',
          '--user-data-dir='+str(sandbox/'edge-profile'),'--disk-cache-dir='+str(sandbox/'edge-cache'),'--crash-dumps-dir='+str(sandbox/'crash-dumps'),'--remote-debugging-port=0','--window-size=1440,1050',origin],stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        portfile=sandbox/'edge-profile'/'DevToolsActivePort'
        for _ in range(100):
            if portfile.exists():break
            if proc.poll() is not None:raise RuntimeError('headless Edge exited')
            time.sleep(.15)
        debug_port=int(portfile.read_text().splitlines()[0]);tabs=json.load(urllib.request.urlopen('http://127.0.0.1:'+str(debug_port)+'/json'))
        tab=next(t for t in tabs if t['type']=='page');ws=websocket.create_connection(tab['webSocketDebuggerUrl'],timeout=20,suppress_origin=True);counter=0
        def cdp(method,params=None):
            nonlocal counter
            counter+=1;key=counter;ws.send(json.dumps(dict(id=key,method=method,params=params or {})))
            while True:
                reply=json.loads(ws.recv())
                if reply.get('id')==key:
                    if 'error' in reply:raise RuntimeError(reply['error'])
                    return reply.get('result',{})
        def js(expression):
            r=cdp('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
            if 'exceptionDetails' in r:raise RuntimeError(r['exceptionDetails'])
            return r['result'].get('value')
        cdp('Runtime.enable')
        for _ in range(100):
            if js("typeof doc!=='undefined' && !!doc && !busy"):break
            time.sleep(.1)
        assert js('current.id')=='A01'
        # Use the actual UI event handlers, not an invented successful API save.
        assert js("(()=>{const b=$('canvas').getBoundingClientRect();for(const p of [[30,40],[80,70],[120,100]])$('canvas').dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:b.left+p[0],clientY:b.top+p[1]}));$('finish').click();return doc.strokes.length})()") == 1
        assert js("(async()=>{await $('save').onclick();return !dirty})()") is True
        assert (sandbox/'labels'/'A01.json').exists()
        assert js("(async()=>{await load('A02');await load('A01');return doc.strokes.length})()") == 1
        # Add an explicit background polygon using the UI mode selector.
        assert js("(()=>{$('mode').value='background';$('mode').onchange();const b=$('canvas').getBoundingClientRect();for(const p of [[230,230],[280,230],[280,280],[230,280]])$('canvas').dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:b.left+p[0],clientY:b.top+p[1]}));$('finish').click();return doc.regions.length})()") == 1
        assert js("(async()=>{await $('reviewed').onclick();return !dirty && doc.review_state==='checked_partial'})()") is True
        saved=json.loads((sandbox/'labels'/'A01.json').read_text(encoding='utf-8'));assert saved['label_provenance']=='user_reviewed';assert len(list((sandbox/'backups').glob('A01*.json')))==1
        report=compile_labels(sandbox);assert len(report['records'])==1;assert len(report['pending'])==5
        data=np.load(sandbox/'compiled'/'A01.npz');assert data['centerline_confidence'][400,400]==0
        # A stale browser tab must not overwrite newer labels.
        request=urllib.request.Request(origin+'/api/save',data=json.dumps(dict(id='A01',revision=None,document=saved)).encode(),headers={'Content-Type':'application/json','Origin':origin},method='POST')
        try:urllib.request.urlopen(request);raise AssertionError('stale save accepted')
        except urllib.error.HTTPError as e:assert e.code==409
        request=urllib.request.Request(origin+'/api/export',data=b'{}',headers={'Content-Type':'application/json'},method='POST')
        try:urllib.request.urlopen(request);raise AssertionError('foreign origin accepted')
        except urllib.error.HTTPError as e:assert e.code==403
        # Return to the untouched first sample for a neutral UI screenshot.
        js("(async()=>{await load('A02');$('mode').value='stroke';$('mode').onchange();return true})()")
        shot=cdp('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False));(sandbox/'review_ui.png').write_bytes(base64.b64decode(shot['data']))
        result=dict(status='passed',isolated_test_root=str(sandbox),browser_interactions=['polyline click','finish','save','reload persisted labels','background polygon','checked state','revision conflict','export origin restriction'],
          sparse_semantics=['unmarked unknown','uncertain mask excluded','positive/background conflict rejected','coordinates checked'],
          real_package_user_labels=len(list((ROOT/'labels').glob('*.json'))),actual_training=False,screenshot=str(sandbox/'review_ui.png'))
        (ROOT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
        # Use CDP to close only the isolated headless browser.
        try:cdp('Browser.close')
        except Exception:pass
    finally:
        if ws:ws.close()
        if proc:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
        log.close();server.shutdown();server.server_close();thread.join(timeout=5)
if __name__=='__main__':run()
