"""Meaningful offline checks for tiling, exported geometry and review server."""
import json
import argparse
import subprocess
import sys
import time
import urllib.request
import urllib.error
import numpy as np
import torch
from .runtime import LAB,ROOT
from .infer import predict,paths_from_probability


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='mobilenet_unet_v01');args=parser.parse_args(argv)
    torch.set_num_threads(2)
    class Constant(torch.nn.Module):
        def forward(self,x):return torch.zeros((len(x),2,*x.shape[-2:]),device=x.device)+.7
    for shape in [(89,71),(453,659)]:
        p=predict(Constant(),np.zeros((*shape,3),np.uint8),torch.device('cpu'))
        assert p.shape==(2,*shape) and np.max(np.abs(p-1/(1+np.exp(-.7))))<2e-6,'tiling seam or padding error'
    probability=np.zeros((120,100),np.float32)
    probability[20:23,10:85]=.8;probability[80:82,40:43]=.9
    paths,_=paths_from_probability(probability)
    assert paths and all(0<=x<100 and 0<=y<120 for p in paths for x,y in p)
    assert all(y<40 for p in paths for _,y in p),'small isolated noise survived'
    run=LAB/'runs'/args.run
    manifest_file=run/'dataset_manifest_snapshot.json'
    manifest=json.loads((manifest_file if manifest_file.exists() else LAB/'datasets'/'manifest.json').read_text(encoding='utf-8'))
    assert not(set(manifest['training_groups'])&set(manifest['validation_groups']))
    assert len({r['group'] for r in manifest['real'] if r['id'].startswith('user_')})==1
    summary=json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    assert summary['completed_steps']==config['requested_steps'] and summary['head_mean_absolute_parameter_update']>1e-6
    for r in json.loads((run/'inference'/'summary.json').read_text(encoding='utf-8')):
        d=json.loads((run/'inference'/r['id']/'paths.json').read_text(encoding='utf-8'))
        assert all(0<=x<d['width'] and 0<=y<d['height'] for p in d['paths'] for x,y in p)
    child=subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts'/'gongbi_lab.py'),'annotate','--port','8769'],cwd=str(ROOT),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        for attempt in range(40):
            try:
                with urllib.request.urlopen('http://127.0.0.1:8769/records',timeout=1) as response:records=json.load(response)
                break
            except OSError:time.sleep(.2)
        else:raise RuntimeError('annotation server did not start')
        assert len(records)==23
        with urllib.request.urlopen('http://127.0.0.1:8769/labels/'+records[0]['id']) as response:
            assert 'revision' in json.load(response)
        rejected=False
        try:urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8769/save',data=b'{}',method='POST'))
        except urllib.error.HTTPError as error:rejected=error.code==403
        assert rejected,'cross-origin save must be rejected'
    finally:child.terminate();child.wait(timeout=10)
    result=dict(overlap_inference_no_seams=True,small_image_padding_correct=True,path_bounds_valid=True,
                isolated_noise_removed=True,group_split_disjoint=True,weights_really_updated=True,
                annotation_server_read_and_origin_check=True,robot_connected=False)
    (run/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
