"""Fetch new museum open-access test inputs to E; never use project references."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

STORAGE = Path(r'E:\Codex_Workspace_Data\Landscape_Web_Test_20261022')
for key, suffix in {'TEMP': 'temp', 'TMP': 'temp', 'PIP_CACHE_DIR': 'cache/pip',
                    'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface',
                    'XDG_CACHE_HOME': 'cache', 'CUDA_CACHE_PATH': 'cache/cuda'}.items():
    os.environ[key] = str(STORAGE/suffix)
sys.dont_write_bytecode = True

SAMPLES = [
    ('01_zha_shibiao', '1955.37.1', '查士标：仿倪瓒山水', '疏朗水墨'),
    ('02_wang_yuanqi', '1954.583', '王原祁：仿倪瓒设色山水图', '淡设色与浅墨'),
    ('03_wang_meng', '1997.94', '王蒙：松下著书图', '密集皴纹与树木'),
    ('04_chen_ruyan', '1997.95', '陈汝言：仙山图', '青绿设色'),
]


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'LandscapeInkResearch/1.0 (museum open access image testing)'})
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read()


def fetch(sample):
    key, accession, label, kind = sample
    folder = STORAGE/'inputs'/key
    folder.mkdir(parents=True, exist_ok=True)
    endpoint = 'https://openaccess-api.clevelandart.org/api/artworks/?accession_number=' + accession
    metadata_file = folder/'museum_metadata.json'
    if metadata_file.exists():
        obj = json.loads(metadata_file.read_text(encoding='utf-8'))
    else:
        raw = get(endpoint)
        response = json.loads(raw)
        matches = [x for x in response.get('data', []) if x.get('accession_number') == accession]
        if len(matches) != 1:
            raise ValueError(f'Expected accession {accession}; found {len(matches)} matches')
        obj = matches[0]
        metadata_file.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    if obj.get('share_license_status') != 'CC0':
        raise ValueError(f'Not confirmed CC0: {accession}: {obj.get("share_license_status")}')
    variant = 'print'
    image_info = obj['images'][variant]
    url = image_info['url']
    image_file = folder/('museum_' + variant + '.jpg')
    if not image_file.exists():
        image_file.write_bytes(get(url))
    from PIL import Image
    with Image.open(image_file) as im:
        dimensions = list(im.size)
        im.verify()
    record = dict(key=key, accession=accession, label=label, test_category=kind,
                  title=obj['title'], creators=obj.get('creators'),
                  page_url=obj.get('url', 'https://www.clevelandart.org/art/'+accession),
                  metadata_url=endpoint, image_url=url, image_variant=variant, license='CC0',
                  downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                  downloaded_file=str(image_file), dimensions=dimensions,
                  bytes=image_file.stat().st_size,
                  sha256=hashlib.sha256(image_file.read_bytes()).hexdigest())
    (folder/'source.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: record[k] for k in ('key', 'title', 'dimensions', 'bytes', 'image_url')}, ensure_ascii=True), flush=True)
    return record


def main():
    (STORAGE/'temp').mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, SAMPLES))
    (STORAGE/'sources.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SOURCES_READY', str(STORAGE), flush=True)


if __name__ == '__main__':
    main()
