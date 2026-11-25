"""Consolidate five material packages; do not draw or reorder any strokes."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from painting_session import save, session_lock
from web.catalog import ARTWORKS

LEGACY = [
    '已确认_含山河绘三字_20261004/painting.json',
    '已确认_含山河绘三字_20261004/竖题山河绘_松岸曲水_绘字修复_20261004/painting.json',
    '已确认_含山河绘三字_20261004/横题山河绘_云山亭阶_20261004/painting.json',
    '输出/横题山河绘_柳岸渔舟_净山岛新版_20261004/painting.json',
    '输出/横题山河绘_青绿梯田云山_20261005/painting.json',
]


def scoped(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Material operation is outside the project: ' + str(path))
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def stroke_data(bundle):
    return {'black': bundle['ink_job']['meta']['ink_execution']['base_strokes'],
            'color': bundle['workflow']['modules']['color']['strokes']}


def index_for(bundle):
    from artwork_order import ink_stages
    parts = ink_stages(bundle['ink_job'],bundle['workflow'])
    stages = {'black': parts['black']}
    for stroke in bundle['workflow']['modules']['color']['strokes']:
        stages.setdefault(stroke['color'], []).append(stroke)
    if 'title' in parts:
        stages['title'] = parts['title']
    return {stage: [dict(order=i, id=s.get('id'), role=s.get('role'),
                        region=s.get('drawing_region', s.get('semantic_region', s.get('region_id'))),
                        parent_stroke_id=s.get('parent_stroke_id'), length_mm=s.get('length_mm'))
                   for i, s in enumerate(strokes, 1)] for stage, strokes in stages.items()}


def consolidate():
    if (ROOT/'待画作品'/'画稿清单.json').exists():
        raise ValueError('画稿库已完成统一；后续请修改当前材料库，不再用旧目录覆盖新版画稿。')
    if (ROOT/'robot_runs'/'blue_wash_motion.lock').exists():
        raise ValueError('A painting task is active; keep its materials unchanged.')
    report = []
    with session_lock(ROOT/'robot_runs'):
        for (folder, title, description, canonical), legacy in zip(ARTWORKS, LEGACY):
            package = scoped(ROOT/'待画作品'/folder)
            task, preview = package/'任务资料', package/'预览与下压标注'
            task.mkdir(parents=True, exist_ok=True)
            preview.mkdir(exist_ok=True)
            original = ROOT/legacy
            source = original if original.exists() else ROOT/canonical
            baseline = json.loads(source.read_text(encoding='utf-8-sig'))
            before = digest(stroke_data(baseline))
            # Refresh only this artwork's material files, not sibling paintings.
            for path in source.parent.iterdir():
                if not path.is_file() or path == ROOT/canonical:
                    continue
                if path.suffix.lower() in ('.json', '.csv', '.md'):
                    target = task/path.name
                elif path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
                    target = package/path.name if path.name in ('参考原图.png', '用户原图.png') else preview/path.name
                else:
                    continue
                if path.resolve() != target.resolve():
                    shutil.copy2(path, target)
            bundle = copy.deepcopy(baseline)
            reference_name = Path(bundle['ink_job']['meta']['image'].replace('\\', '/')).name
            reference = '../参考原图.png' if reference_name == '参考原图.png' else '../预览与下压标注/' + reference_name
            if not (task/reference).resolve().is_file():
                raise ValueError('Reference material missing: ' + reference)
            bundle['ink_job']['meta']['image'] = reference
            bundle['workflow']['reference_image'] = reference
            assert digest(stroke_data(bundle)) == before
            save(ROOT/canonical, bundle)
            save(task/'black_only.json', bundle['ink_job'])
            for companion in task.glob('*_黑线任务.json'):
                save(companion, bundle['ink_job'])
            for companion in task.glob('*_计划.json'):
                save(companion, bundle['workflow'])
            # Keep recipe data in the material library, code snapshots outside it.
            code = package/'源码备查'
            if code.exists():
                for recipe in code.rglob('*.json'):
                    target = task/'生成模板'/recipe.relative_to(code)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(recipe, target)
                destination = scoped(ROOT/'作品生成工具'/folder/'历史生成脚本')
                scoped(code)
                if destination.exists():
                    raise ValueError('Code archive destination already exists: ' + str(destination))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(code), str(destination))
            order = index_for(bundle)
            counts = {stage: len(rows) for stage, rows in order.items()}
            save(task/'绘画顺序索引.json', {
                'format': 'shanhe-stroke-order-index-v1',
                'note': '只索引当前笔触，不改变顺序。先勾勒；颜色之间由网页选色，各颜色内部按数组顺序执行。',
                'stroke_data_sha256': before, 'counts': counts, 'stages': order})
            table = io.StringIO(newline='')
            writer = csv.writer(table)
            writer.writerow(['阶段', '当前顺序', '笔触ID', '部位或区域', '笔触类别', '原笔触ID', '长度mm'])
            for stage, rows in order.items():
                for row in rows:
                    writer.writerow([stage, row['order'], row['id'], row['region'], row['role'], row['parent_stroke_id'], row['length_mm']])
            (task/'逐笔顺序索引.csv').write_text(table.getvalue(), encoding='utf-8-sig')
            (package/'阅读说明.md').write_text(
                '# %s\n\n%s。参考图含“山河绘”题字。\n\n'
                '本目录是此画唯一的日常材料来源。双击项目根目录“启动绘画工作台.bat”，在网页选择本幅。\n\n'
                '- `参考原图.png`：选画参考。\n'
                '- `任务资料/painting.json`：网页实际执行的完整画稿，后续修改以它为准。\n'
                '- `任务资料/绘画顺序索引.json`、`逐笔顺序索引.csv`：当前每笔顺序和部位索引。\n'
                '- `预览与下压标注`：线稿、路径、配色及下压示意。\n'
                '- `任务资料`中的分拆 JSON 和生成模板：修改、追溯用；日常只需完整画稿。\n\n'
                '笔数：%s。\n\n本次整理保留所有笔触、下压量、配色和现有顺序。'
                '后续勾勒顺序、色内笔触顺序统一在本目录的完整画稿上修改，并同步分拆任务与预览。'
                '色间顺序目前由网页选色，执行器仍先勾勒再上色。\n\n'
                '程序和现场配置仍在项目根目录。历史生成脚本移到 `作品生成工具/%s/历史生成脚本`，'
                '仅备查，日常执行不调用它们。原“输出”“已确认”“展示选画”不再是本幅的执行来源。\n'
                % (title, description, '，'.join('%s %d笔' % (stage, count) for stage, count in counts.items()), folder), encoding='utf-8')
            # Preserve any existing abnormal-run guard under the new identity.
            old_key = hashlib.sha256(str(original.resolve()).casefold().encode()).hexdigest()[:24]
            new_key = hashlib.sha256(str((ROOT/canonical).resolve()).casefold().encode()).hexdigest()[:24]
            prior = ROOT/'robot_runs'/('painting_session_'+old_key+'.json')
            migrated = ROOT/'robot_runs'/('painting_session_'+new_key+'.json')
            if prior.exists() and not migrated.exists():
                record = json.loads(prior.read_text(encoding='utf-8-sig'))
                record['source_migration'] = {'old_source': record['source'], 'date': '2026-10-06'}
                record['source'] = str((ROOT/canonical).resolve())
                save(migrated, record)
                retired = scoped(ROOT/'robot_runs'/'source_migration_backup'/prior.name)
                retired.parent.mkdir(parents=True, exist_ok=True)
                os_replace = __import__('os').replace
                os_replace(scoped(prior), retired)
            report.append(dict(folder=folder, title=title, painting_json=canonical,
                               reference_image=(package/'参考原图.png').relative_to(ROOT).as_posix(),
                               counts=counts, stroke_data_sha256=before,
                               preserved_order=True, legacy_source=legacy))
        save(ROOT/'待画作品'/'画稿清单.json', {'format': 'shanhe-material-library-v1', 'date': '2026-10-06', 'artworks': report})
    print('Consolidated five material packages without changing strokes or order.')


if __name__ == '__main__':
    consolidate()
