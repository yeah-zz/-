# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from run_stability_brush_20261101 import ROOT,read,jsave,verify_lock
from stroke_geometry_brush_20261101 import process,CONFIG
import argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',default='ready');p.add_argument('--native',action='store_true');a=p.parse_args();verify_lock();records=[read(f) for f in sorted((ROOT/'input_records').glob('T*.json'))];ids=[r['id'] for r in records] if a.ids=='ready' else a.ids.split(',')
    for r in records:
        if r['id'] not in ids:continue
        folder=ROOT/('native_outputs' if a.native else 'outputs')/r['id'];source=r['original'] if a.native else r['input'];paths=folder/'complete/complete_paths.json';stats=folder/'brush/statistics.json'
        if not paths.exists():continue
        if stats.exists() and read(stats).get('config_version')==CONFIG['version']:continue
        result=process(source,paths,folder/'brush');jsave(folder/'brush_refresh.json',dict(reason='Image-only width regularization; extraction and accepted baseline unchanged',statistics=result));print('BRUSH_REFRESHED',r['id'],result['valid_profile_fraction'],flush=True)
    jsave(ROOT/'evaluation_scope_amendment.json',dict(extraction=dict(frozen=True,training_on_test_images=False,tuning_on_test_images=False),brush_preview=dict(development_case='T01',r1_preview='brush_iterations/raw_profile_T01',change='Retain raw profiles, reject isolated patch-like widths, regularize along each curve',blind_validation=False),original_protocol_retained=True,notes='The no-tuning flag in the original evaluation protocol applies to line extraction. Brush preview is a new development feature, not a blind benchmark.'))
    verify_lock()
if __name__=='__main__':main()
