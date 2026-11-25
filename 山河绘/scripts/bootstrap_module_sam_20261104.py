# -*- coding: utf-8 -*-
"""Download official SAM weights to the isolated E: experiment, never run hardware."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\module_stroke_planner_20261104')
URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'

if __name__ == '__main__':
    target = ROOT/'weights'/'sam_vit_b_01ec64.pth'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix('.part')
        with urllib.request.urlopen(URL, timeout=45) as source, partial.open('wb') as dest:
            expected = int(source.headers.get('Content-Length', 0))
            size = 0
            while True:
                block = source.read(1024*1024)
                if not block:
                    break
                dest.write(block)
                size += len(block)
                if size % (32*1024*1024) == 0:
                    print('downloaded_MiB', size//1024//1024, flush=True)
        if expected and size != expected:
            raise ValueError('Incomplete download; partial file retained')
        if size < 300_000_000:
            raise ValueError('Unexpected official ViT-B checkpoint size')
        partial.replace(target)
    h = hashlib.sha256()
    with target.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    manifest = dict(source=URL, file=str(target), bytes=target.stat().st_size,
                    sha256=h.hexdigest(), role='pretrained automatic object-mask proposals',
                    fine_tuning_performed=False, robot_connected=False)
    (ROOT/'weights'/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)
