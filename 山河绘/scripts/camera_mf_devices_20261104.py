# -*- coding: utf-8 -*-
"""Native Media Foundation source order, without opening any camera."""
import ctypes as ct
from ctypes import wintypes as wt
import json
from usb_camera_preview_20261104 import GUID,method,release

def mf_devices():
    ole=ct.OleDLL('ole32');mfplat=ct.OleDLL('mfplat');mf=ct.OleDLL('mf')
    ole.CoInitialize(None);ole.CoTaskMemFree.argtypes=[ct.c_void_p]
    attributes=ct.c_void_p();sources=ct.POINTER(ct.c_void_p)();count=wt.UINT()
    started=False;rows=[]
    try:
        if mfplat.MFStartup(0x20070,0)!=0:raise RuntimeError('Media Foundation startup failed')
        started=True
        mfplat.MFCreateAttributes.argtypes=[ct.POINTER(ct.c_void_p),wt.UINT]
        hr=mfplat.MFCreateAttributes(ct.byref(attributes),1)
        if hr!=0:raise RuntimeError('Cannot create Media Foundation attributes')
        hr=method(attributes,24,wt.LONG,[ct.POINTER(GUID),ct.POINTER(GUID)])(
            attributes,ct.byref(GUID('c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3')),
            ct.byref(GUID('8ac3587a-4ae7-42d8-99e0-0a6013eef90f')))
        if hr!=0:raise RuntimeError('Cannot select video capture sources')
        mf.MFEnumDeviceSources.argtypes=[ct.c_void_p,ct.POINTER(ct.POINTER(ct.c_void_p)),ct.POINTER(wt.UINT)]
        hr=mf.MFEnumDeviceSources(attributes,ct.byref(sources),ct.byref(count))
        if hr!=0:raise RuntimeError('Media Foundation enumeration failed: %s'%hr)
        for i in range(count.value):
            values=dict(index=i)
            for name,guid in [('FriendlyName','60d0e559-52f8-4fa2-bbce-acdb34a8ec01'),
                              ('DevicePath','58f0aad8-22bf-4f8a-bb3d-d2c4978c6e2f')]:
                value=ct.c_void_p();size=wt.UINT()
                hr=method(sources[i],13,wt.LONG,[ct.POINTER(GUID),ct.POINTER(ct.c_void_p),ct.POINTER(wt.UINT)])(
                    sources[i],ct.byref(GUID(guid)),ct.byref(value),ct.byref(size))
                if hr==0 and value:
                    values[name]=ct.wstring_at(value,size.value);ole.CoTaskMemFree(value)
            rows.append(values)
    finally:
        for i in range(count.value):release(sources[i])
        if sources:ole.CoTaskMemFree(sources)
        release(attributes)
        if started:mfplat.MFShutdown()
        ole.CoUninitialize()
    return rows

if __name__=='__main__':print(json.dumps(mf_devices(),ensure_ascii=False),flush=True)
