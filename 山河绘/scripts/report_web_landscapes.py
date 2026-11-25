"""Write provenance and a local switchable viewer for the completed web tests."""
from pathlib import Path
import hashlib
import html
import json
import os
import sys
import shutil

STORAGE = Path(r'E:\Codex_Workspace_Data\Landscape_Web_Test_20261022')
for key, suffix in {'TEMP': 'temp', 'TMP': 'temp', 'PIP_CACHE_DIR': 'cache/pip',
                    'TORCH_HOME': 'cache/torch', 'HF_HOME': 'cache/huggingface',
                    'XDG_CACHE_HOME': 'cache', 'CUDA_CACHE_PATH': 'cache/cuda'}.items():
    os.environ[key] = str(STORAGE/suffix)
sys.dont_write_bytecode = True


def main():
    out = STORAGE/'results'
    records = json.loads((out/'experiments.json').read_text(encoding='utf-8'))
    assert len(records) == 4
    assert len({r['source']['page_url'] for r in records}) == 4
    assert len({json.dumps(r['config'], sort_keys=True) for r in records}) == 1
    assert len({r['source_ink_sha256'] for r in records}) == 1
    root = Path(__file__).resolve().parents[1]
    assert hashlib.sha256((root/'source_ink.py').read_bytes()).hexdigest() == records[0]['source_ink_sha256']
    lines = ['# 网上新山水画 / 固定参数 v2 测试', '',
             '四张输入均从 Cleveland Museum of Art 官网 API 新下载，授权元数据为 CC0。未使用项目已有画稿。', '',
             '统一裁掉装裱边，保持比例，将长边上限设为 1800 像素。画内题跋和印章保留；未逐图调参、增强墨线、修改算法或人工修线。', '',
             '首先查看 `overview.png`，再查看各目录的 `comparison.png`（含全图与预先指定的局部）。`compare.html` 可在原图、中心线和红线叠加之间切换，并放大查看。', '',
             '| 图片 / 官网来源 | 类型 | 处理尺寸 | 几何路径数 | 提取耗时 |',
             '|---|---|---|---:|---:|']
    cards = []
    for r in records:
        s = r['source']
        key = s['key']
        w, h = r['work_size']
        failed = r['stats'] is None
        names = ('input.png', 'failure.json') if failed else ('input.png', 'centerlines.png', 'overlay.png', 'comparison.png', 'paths.json', 'centerlines.svg')
        for name in names:
            assert (out/key/name).is_file()
        count = '未完成' if failed else str(r['stats']['paths'])
        duration = '>600 秒，已停止' if failed else f"{r['seconds']:.1f} 秒"
        lines.append(f"| [{s['label']}]({s['page_url']}) | {s['test_category']} | {w}×{h} | {count} | {duration} |")
        title = html.escape(s['label'])
        source_url = html.escape(s['page_url'], quote=True)
        if failed:
            cards.append(f'''<section><h2>{title}</h2><p>{html.escape(s['test_category'])} · {w}×{h} · <a href="{source_url}">博物馆来源</a></p>
<p>此图运行超过 10 分钟仍没有完整输出，已停止。不能评价其提线质量；超时不是成功样本。</p>
<div class="viewport"><img src="{key}/input.png" alt="超时测试的原图"></div></section>''')
            continue
        cards.append(f'''<section><h2>{title}</h2><p>{html.escape(s['test_category'])} · {w}×{h} · {r['stats']['paths']} 条几何路径 · <a href="{source_url}">博物馆来源</a></p>
<div class="controls"><button onclick="show('{key}','input')">原图</button><button onclick="show('{key}','centerlines')">中心线</button><button onclick="show('{key}','overlay')">红线叠加</button>
<label>放大 <input type="range" min="100" max="300" value="100" oninput="zoom('{key}',this.value)"></label>
<a href="{key}/comparison.png">全图与局部对照</a> · <a href="{key}/centerlines.svg">SVG</a></div>
<div class="viewport"><img id="image-{key}" src="{key}/overlay.png" alt="{title} 原图轨迹叠加"></div></section>''')
    lines += ['', '路径数量不是准确率，也不等于真实画家的落笔次数。输出含题跋；紧凑点候选单独保存在 dots 字段，未渲染为执行点按。', '',
              '全部中间掩膜、SVG、路径 JSON、原始下载图、图片 SHA256、裁切坐标、缩放比例、参数和算法 SHA256 随实验保存。', '',
              '王蒙密集皴纹样本运行超过 10 分钟仍无完整输出，停止后保留 failure.json。此样本不能评价提线质量，不计为成功结果。后续单张测试设置 600 秒运行上限。', '',
              '本轮仅检验可见几何，无人工完整真值，不报告准确率；未连接机械臂、未生成执行任务。不同作品的风格和数字化条件不同，尚不能据此计算算法泛化成功率。', '',
              '复现：先在项目目录用已有 E:\\Anaconda\\python.exe -B 运行 scripts/fetch_web_landscapes.py，再运行 scripts/test_web_landscapes.py 和 scripts/report_web_landscapes.py。三个脚本均在导入图像库前定向 E 盘缓存和临时目录。test_web_landscapes.py --resume 会复用已完成结果，并保留已有超时记录，不重跑这几张。']
    (out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>网上新图 · v2 提线对照</title><style>
body{margin:0;background:#f4f3ee;color:#243b34;font:17px/1.65 "Microsoft YaHei",sans-serif}main{max-width:1350px;margin:auto;padding:30px}h1{font-size:30px}h2{font-size:23px;margin-bottom:0}section{margin:26px 0;padding:24px;background:white;border:1px solid #d9dfd9;border-radius:12px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:14px 0}button{font:inherit;color:inherit;padding:7px 17px;border:1px solid #98aaa0;border-radius:6px;background:#eef4ee;cursor:pointer}a{color:#2f6955}.viewport{overflow:auto;max-height:850px;background:#e8e8e2;border:1px solid #d4d8d1}.viewport img{display:block;width:100%;max-width:none;height:auto}.note{color:#53675d}
</style><main><h1>网上新找四张山水画 · 固定 v2 参数测试</h1><p class="note">原图、中心线、叠加可切换；放大后可滚动检查。图内题跋与印章保留。画面数据全部来自博物馆，文件保存在本地，没有远程图片请求。</p>'''
    page += '\n'.join(cards)
    page += '''<p class="note">没有逐图调参或人工修线；路径数不是准确率。仅为离线几何检查，毛笔运笔与墨色尚未标定。</p></main><script>
function show(key,variant){document.getElementById('image-'+key).src=key+'/'+variant+'.png'}
function zoom(key,value){document.getElementById('image-'+key).style.width=value+'%'}
</script></html>'''
    (out/'compare.html').write_text(page, encoding='utf-8')
    shutil.copy2(Path(__file__).resolve(), out/'code_snapshot'/Path(__file__).name)
    manifest = []
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name != 'artifact_manifest.json':
            manifest.append(dict(path=str(path.relative_to(out)), bytes=path.stat().st_size,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (out/'artifact_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(verified_samples=len(records), completed=sum(r['stats'] is not None for r in records),
                         timeouts=sum(r['stats'] is None for r in records), same_algorithm_and_config=True,
                         result_files=len(manifest), result_bytes=sum(x['bytes'] for x in manifest),
                         viewer=str(out/'compare.html')), ensure_ascii=True))


if __name__ == '__main__':
    main()
