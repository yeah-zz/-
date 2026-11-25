# -*- coding: utf-8 -*-
"""Explicit external DirectShow camera preview, localhost only; no robot imports.

Capture stays in RAM. One startup diagnostic JPEG is saved to the E: session.
No microphone, recording, controller connection, or automatic camera scanning.
"""
import argparse
import ctypes as ct
from ctypes import wintypes as wt
import json
import secrets
from pathlib import Path
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import cv2

ROOT=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\camera_preview_20261104')

def frame_quality(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    sample=gray[::8,::8]
    mean=float(sample.mean());std=float(sample.std())
    # A low-light warning, not a diagnosis of optical obstruction.
    return dict(brightness_mean=round(mean,2),brightness_std=round(std,2),
                black_frame=bool(mean<3. and std<2.))

class GUID(ct.Structure):
    _fields_=[('data',ct.c_ubyte*16)]
    def __init__(self,value):
        super().__init__();ct.memmove(ct.addressof(self),uuid.UUID(value.strip('{}')).bytes_le,16)

def method(ptr,index,result,args):
    table=ct.cast(ptr,ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    return ct.WINFUNCTYPE(result,ct.c_void_p,*args)(table[index])

def release(ptr):
    if ptr:method(ptr,2,wt.ULONG,[])(ptr)

def dshow_devices():
    """Read FriendlyName/DevicePath via native COM, matching OpenCV's DShow list.

    No generated COM wrappers or caches and no camera is opened here.
    """
    ole=ct.OleDLL('ole32');auto=ct.OleDLL('oleaut32')
    ole.CoInitialize(None)
    ole.CoCreateInstance.argtypes=[ct.POINTER(GUID),ct.c_void_p,wt.DWORD,ct.POINTER(GUID),ct.POINTER(ct.c_void_p)]
    auto.VariantClear.argtypes=[ct.c_void_p]
    dev=ct.c_void_p();enum=ct.c_void_p();records=[]
    try:
        hr=ole.CoCreateInstance(ct.byref(GUID('62BE5D10-60EB-11D0-BD3B-00A0C911CE86')),None,1,
            ct.byref(GUID('29840822-5B84-11D0-BD3B-00A0C911CE86')),ct.byref(dev))
        if hr!=0:raise RuntimeError('Cannot enumerate DirectShow video sources: %s'%hr)
        hr=method(dev,3,wt.LONG,[ct.POINTER(GUID),ct.POINTER(ct.c_void_p),wt.DWORD])(
            dev,ct.byref(GUID('860BB310-5D01-11D0-BD3B-00A0C911CE86')),ct.byref(enum),0)
        if hr!=0:return []
        while True:
            mon=ct.c_void_p();fetched=wt.ULONG()
            hr=method(enum,3,wt.LONG,[wt.ULONG,ct.POINTER(ct.c_void_p),ct.POINTER(wt.ULONG)])(enum,1,ct.byref(mon),ct.byref(fetched))
            if hr!=0 or not fetched.value:break
            bag=ct.c_void_p()
            try:
                hr=method(mon,9,wt.LONG,[ct.c_void_p,ct.c_void_p,ct.POINTER(GUID),ct.POINTER(ct.c_void_p)])(
                    mon,None,None,ct.byref(GUID('55272A00-42CB-11CE-8135-00AA004BB851')),ct.byref(bag))
                if hr!=0:raise RuntimeError('Camera property bag unavailable')
                values={}
                for name in ['FriendlyName','DevicePath']:
                    variant=ct.create_string_buffer(24 if ct.sizeof(ct.c_void_p)==8 else 16)
                    hr=method(bag,3,wt.LONG,[wt.LPCWSTR,ct.c_void_p,ct.c_void_p])(bag,name,ct.byref(variant),None)
                    if hr==0 and ct.c_ushort.from_buffer(variant).value==8:
                        bstr=ct.c_void_p.from_buffer(variant,8).value
                        values[name]=ct.wstring_at(bstr) if bstr else ''
                    auto.VariantClear(ct.byref(variant))
                records.append(dict(index=len(records),**values))
            finally:
                release(bag);release(mon)
    finally:
        release(enum);release(dev);ole.CoUninitialize()
    return records

PAGE='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>绘画相机 · 实时预览</title>
<style>body{margin:0;background:#13251f;color:#f5f3e9;font:17px/1.6 system-ui}main{max-width:1320px;margin:auto;padding:22px}header{display:flex;justify-content:space-between;align-items:center;gap:15px}h1{font-size:25px;margin:0}img{display:block;max-width:100%;max-height:78vh;margin:14px auto;background:#060c09;object-fit:contain}button{font:inherit;padding:8px 15px;border:0;border-radius:7px;background:#e8e7d3;color:#172d25;cursor:pointer}.info{color:#bfccc3}#state{color:#b7deb7}</style>
<main><header><div><h1>外接USB摄像头 · 实时预览</h1><div id="state">正在连接画面…</div></div><button id="stop">关闭摄像头</button></header><img id="frame" alt="外接相机实时画面"><p class="info">仅本机预览，不录制视频或采集麦克风，不控制机械臂。画面用途：检查纸面覆盖、笔尖遮挡和清晰度。</p></main>
<script>let stopped=false,lastDisplayed=0;const frame=document.getElementById('frame'),state=document.getElementById('state');function next(){if(!stopped)frame.src='/frame.jpg?t='+Date.now()}frame.onload=()=>{lastDisplayed=Date.now();if(!stopped)setTimeout(next,100)};frame.onerror=()=>{state.textContent='画面读取失败，正在重试';if(!stopped)setTimeout(next,700)};async function status(){try{const response=await fetch('/status',{cache:'no-store',signal:AbortSignal.timeout(3000)});if(!response.ok)throw Error('status failed');const s=await response.json();if(s.error||s.frame_age_s>3){state.textContent='相机画面已中断，请检查USB连接';}else if(!lastDisplayed||Date.now()-lastDisplayed>4000){state.textContent='已检测到相机，浏览器画面尚未更新';}else if(s.black_frame){state.textContent='收到黑色画面：请检查镜头遮挡和照明';}else{state.textContent=`画面正常 · ${s.width}×${s.height} · ${s.capture_fps.toFixed(1)}帧/秒`;}}catch(e){state.textContent='预览服务已断开，请重新打开预览'}}document.getElementById('stop').onclick=async()=>{stopped=true;try{await fetch('/stop',{method:'POST',headers:{'X-Camera-Token':'__TOKEN__'},signal:AbortSignal.timeout(5000)});state.textContent='摄像头已关闭';document.getElementById('stop').disabled=true}catch(e){state.textContent='关闭请求未确认，请检查预览服务'}};next();status();setInterval(()=>{if(!stopped)status()},1500);</script></html>'''

def run():
    parser=argparse.ArgumentParser();parser.add_argument('--enumerate',action='store_true')
    parser.add_argument('--port',type=int,default=0)
    parser.add_argument('--hd',action='store_true',help='Explicit optional MJPG 1280x720 request')
    args=parser.parse_args()
    devices=dshow_devices()
    if args.enumerate:
        print(json.dumps(devices,ensure_ascii=False),flush=True);return
    chosen=[r for r in devices if 'vid_1bcf&pid_c001' in r.get('DevicePath','').lower()]
    if len(chosen)!=1:raise RuntimeError('Expected exactly one external VID_1BCF/PID_C001; refusing other cameras')
    ROOT.mkdir(parents=True,exist_ok=True)
    folder=ROOT/time.strftime('%Y%m%d_%H%M%S');folder.mkdir(exist_ok=False)
    (folder/'devices.json').write_text(json.dumps(devices,ensure_ascii=False,indent=2),encoding='utf-8')
    lock=threading.Lock();stop=threading.Event();ready=threading.Event()
    state=dict(jpeg=None,error=None,count=0,stamp=0.,width=0,height=0,capture_fps=0.,black_frame=True)
    def capture():
        cap=None
        try:
            cap=cv2.VideoCapture(chosen[0]['index'],cv2.CAP_DSHOW)
            if not cap.isOpened():raise RuntimeError('External DirectShow source could not be opened')
            if args.hd:
                cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
                cap.set(cv2.CAP_PROP_FPS,15)
            start=time.monotonic();bad=0;saved=False
            while not stop.is_set():
                ok,frame=cap.read()
                if not ok:
                    bad+=1
                    if bad>30:raise RuntimeError('External camera stopped delivering frames')
                    time.sleep(.03);continue
                bad=0
                ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
                if not ok:continue
                now=time.monotonic()
                quality=frame_quality(frame)
                with lock:
                    state['jpeg']=encoded.tobytes();state['count']+=1;state['stamp']=now
                    state['width']=frame.shape[1];state['height']=frame.shape[0]
                    state['capture_fps']=state['count']/max(now-start,.001)
                    state.update(quality)
                if now-start>1 and not saved:
                    (folder/'startup_frame.jpg').write_bytes(encoded.tobytes());saved=True;ready.set()
            ready.set()
        except Exception as exc:
            with lock:state['error']=str(exc)
            ready.set()
        finally:
            if cap is not None:cap.release()
    worker=threading.Thread(target=capture,daemon=True);worker.start()
    if not ready.wait(20):stop.set();raise RuntimeError('Camera preview startup timed out')
    if state['error']:raise RuntimeError(state['error'])
    token=secrets.token_urlsafe(24)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_data(self,status,mime,data):
            self.send_response(status);self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/':return self.send_data(200,'text/html; charset=utf-8',PAGE.replace('__TOKEN__',token).encode('utf-8'))
            with lock:
                snapshot={k:v for k,v in state.items() if k!='jpeg'};jpeg=state['jpeg']
            if path=='/status':
                snapshot['frame_age_s']=round(time.monotonic()-snapshot.pop('stamp'),3)
                snapshot.update(device='USB Camera',robot_connected=False,recording_video=False)
                return self.send_data(200,'application/json',json.dumps(snapshot).encode())
            if path=='/frame.jpg':
                valid=jpeg is not None and snapshot['error'] is None and time.monotonic()-snapshot['stamp']<3
                return self.send_data(200 if valid else 503,'image/jpeg',jpeg if valid else b'')
            self.send_data(404,'text/plain',b'Not found')
        def do_POST(self):
            if self.path!='/stop' or self.headers.get('X-Camera-Token')!=token:
                return self.send_data(403,'text/plain',b'Forbidden')
            stop.set();worker.join(timeout=3)
            self.send_data(200,'application/json',b'{"camera_stopped":true}')
            threading.Thread(target=self.server.shutdown,daemon=True).start()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    info=dict(pid=__import__('os').getpid(),url=f'http://127.0.0.1:{server.server_port}/',
              folder=str(folder),device=chosen[0],startup_snapshot=str(folder/'startup_frame.jpg'),
              robot_connected=False,recording_video=False,saved_only_one_startup_snapshot=True,
              video_mode='requested_1280x720_MJPG' if args.hd else 'device_native_default')
    (ROOT/'active_preview.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'session.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False),flush=True)
    try:server.serve_forever(poll_interval=.2)
    finally:
        stop.set();worker.join(timeout=4);server.server_close()
        (folder/'closed.json').write_text(json.dumps(dict(camera_released=not worker.is_alive(),robot_connected=False)),encoding='utf-8')

if __name__=='__main__':run()
