# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from freeze_stability_brush_20261101 import ROOT,read,jsave,digest
from pathlib import Path
from extract_complete_lines_20261101 import extract
from stroke_geometry_brush_20261101 import process
import argparse,json,time,traceback

def verify_lock():
    for row in read(ROOT/'baseline_lock.json')['files']:assert digest(row['path'])==row['sha256'],'Accepted baseline changed: '+row['path']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ids',default='ready');parser.add_argument('--native',action='store_true');a=parser.parse_args();verify_lock();records=[read(p) for p in sorted((ROOT/'input_records').glob('T*.json'))];ids=[r['id'] for r in records] if a.ids in ['ready','all'] else a.ids.split(',');summary=[]
    if a.native:ids=[i for i in ids if i in read(ROOT/'evaluation_protocol.json')['native_resolution_substudy']]
    for row in records:
        ident=row['id']
        if ident not in ids:continue
        folder=ROOT/('native_outputs' if a.native else 'outputs')/ident;out=folder/'complete';source=row['original'] if a.native else row['input'];result=folder/'run_result.json'
        if result.exists():summary.append(read(result));continue
        try:
            started=time.time();stats=extract(source,folder);geometry=process(source,out/'complete_paths.json',folder/'brush');entry=dict(id=ident,native=a.native,success=True,input_sha256=digest(source),extraction=stats,brush=geometry,seconds=time.time()-started,baseline_lock_sha256=digest(ROOT/'baseline_lock.json'),robot_connected=False);jsave(result,entry);summary.append(entry);print('STABILITY_DONE',ident,'native='+str(a.native),json.dumps(geometry),flush=True)
        except Exception as e:
            error=dict(id=ident,native=a.native,success=False,error=str(e),traceback=traceback.format_exc(),input_sha256=digest(source),robot_connected=False);jsave(result,error);summary.append(error);print('STABILITY_FAILED',ident,str(e),flush=True)
        verify_lock()
    jsave(ROOT/('native_results.json' if a.native else 'results_available.json'),dict(records=summary,selection='All available generated inputs retained',used_for_training=False));verify_lock()
if __name__=='__main__':main()
