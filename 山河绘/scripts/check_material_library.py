"""Check five canonical jobs; optional isolated check without old artwork folders."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web.catalog import ARTWORKS


def check(root):
    for folder, title, _, source in ARTWORKS:
        material = root/'待画作品'/folder
        task = root/source
        if not task.resolve().is_relative_to((root/'待画作品').resolve()):
            raise ValueError('Artwork is outside the material library')
        bundle = json.loads(task.read_text(encoding='utf-8-sig'))
        reference = bundle['ink_job']['meta']['image']
        if Path(reference).is_absolute() or not (task.parent/reference).resolve().is_relative_to(material.resolve()):
            raise ValueError('Reference image is outside this artwork package')
        if not (task.parent/reference).is_file():
            raise ValueError('Missing reference material')
        for path in material.rglob('*'):
            if path.suffix.lower() in ('.py', '.pyc', '.bat', '.exe'):
                raise ValueError('Program file found among painting materials: ' + str(path))
        ink = bundle['ink_job']['meta']['ink_execution']['base_strokes']
        colors = bundle['workflow']['modules']['color']['strokes']
        stroke_hash = hashlib.sha256(json.dumps({'black':ink,'color':colors}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        index = json.loads((task.parent/'绘画顺序索引.json').read_text(encoding='utf-8-sig'))
        if index['stroke_data_sha256'] != stroke_hash:
            raise ValueError('Stroke order index is stale')
        command = [sys.executable, str(root/'web'/'worker.py'), str(task), '--check']
        settings = root/'web'/'data'/'site_settings.json'
        if settings.exists():
            command += ['--settings', str(settings)]
        result = subprocess.run(command, cwd=root, capture_output=True, encoding='utf-8', errors='replace', timeout=90)
        if result.returncode:
            raise ValueError(title + ': ' + result.stdout + result.stderr)
        print('%s: black %d, color %d, offline check passed.' % (title, len(ink), len(colors)), flush=True)
    # The web must also find its complete gallery with no old folders present.
    result = subprocess.run([sys.executable, '-c',
        'from web.server import Workbench; a=Workbench(); assert len(a.items)==5; '
        'assert all(i["path"].is_file() and i["image"].is_file() for i in a.items.values()); '
        'print("Five gallery entries and all reference pictures readable.")'],
        cwd=root, capture_output=True, encoding='utf-8', errors='replace', timeout=30)
    if result.returncode:
        raise ValueError(result.stdout + result.stderr)
    print(result.stdout.strip(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isolated', action='store_true')
    args = parser.parse_args()
    if not args.isolated:
        check(ROOT)
        return
    with tempfile.TemporaryDirectory(prefix='shanhe-material-check-') as folder:
        isolated = Path(folder)
        for path in ROOT.iterdir():
            if path.is_file() and path.suffix in ('.py', '.json', '.toml'):
                shutil.copy2(path, isolated/path.name)
        for name in ('web', 'scripts'):
            destination = isolated/name
            destination.mkdir()
            for path in (ROOT/name).rglob('*.py'):
                relative = path.relative_to(ROOT/name)
                (destination/relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination/relative)
        saved = ROOT/'web'/'data'/'site_settings.json'
        if saved.exists():
            (isolated/'web'/'data').mkdir()
            shutil.copy2(saved, isolated/'web'/'data'/'site_settings.json')
        shutil.copytree(ROOT/'待画作品', isolated/'待画作品')
        assert all(not (isolated/name).exists() for name in ('输出','已确认_含山河绘三字_20261004','展示选画_20260927'))
        check(isolated)
        print('Isolated check passed with project programs and 待画作品 only; no old material directories.')


if __name__ == '__main__':
    main()
