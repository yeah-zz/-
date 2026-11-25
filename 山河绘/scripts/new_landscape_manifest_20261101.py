# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from freeze_stability_brush_20261101 import ROOT,read,jsave,digest
from pathlib import Path
from PIL import Image
import argparse,shutil,json
GENERATED=Path(r'E:\Codex_Workspace_Data\CLIPascene_20261021\generated_images\01a15e88-26fa-7a12-9622-3be075148c5d')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','register','manifest']);p.add_argument('--id');p.add_argument('--filename');a=p.parse_args();prompts=read(Path(__file__).with_name('stability_brush_prompts_20261101.json'))
    if a.action=='init':
        path=ROOT/'generation_prompts.json'
        if path.exists():assert read(path)==prompts
        else:jsave(path,prompts)
        print('NEW_INPUT_PROMPTS_FROZEN',digest(path));return
    assert read(ROOT/'generation_prompts.json')==prompts
    if a.action=='register':
        assert a.id in [r['id'] for r in prompts];source=GENERATED/a.filename;assert source.parent==GENERATED and source.is_file();record=ROOT/'input_records'/(a.id+'.json')
        if record.exists():
            assert read(record)['generated_file_sha256']==digest(source);print('ALREADY_REGISTERED',a.id);return
        original=ROOT/'originals'/(a.id+'.png');original.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,original);assert digest(original)==digest(source);im=Image.open(original).convert('RGB');native_size=list(im.size);im.thumbnail((496,496),Image.Resampling.LANCZOS);input_path=ROOT/'inputs'/(a.id+'.png');input_path.parent.mkdir(exist_ok=True);im.save(input_path);prompt=next(r for r in prompts if r['id']==a.id)
        jsave(record,dict(id=a.id,group=a.id,level=prompt['level'],prompt=prompt['prompt'],original=str(original),input=str(input_path),native_size=native_size,input_size=list(im.size),generated_file=str(source),generated_file_sha256=digest(source),original_sha256=digest(original),input_sha256=digest(input_path),independent_generation_call=True,crop=False,used_for_training=False,selective_regeneration=False));print('NEW_INPUT_REGISTERED',a.id,native_size);return
    records=[read(f) for f in sorted((ROOT/'input_records').glob('T*.json'))];assert len(records)==20;assert len({r['original_sha256'] for r in records})==20;doc=dict(records=records,prompt_sha256=digest(ROOT/'generation_prompts.json'),frozen_baseline_sha256=digest(ROOT/'baseline_lock.json'),used_for_training=False,regenerated_after_failure=False)
    path=ROOT/'new_input_manifest.json'
    if path.exists():assert read(path)==doc
    else:jsave(path,doc)
    print('NEW_INPUT_MANIFEST_COMPLETE',len(records),digest(path))
if __name__=='__main__':main()
