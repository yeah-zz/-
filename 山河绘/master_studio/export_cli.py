# -*- coding: utf-8 -*-
"""导出命令行：由父进程选择含 OpenCV 的 Python 运行。"""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from master_studio.export import export_bundle  # noqa: E402


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: export_cli.py image plan out title")
    image, plan_path, out_path, title = sys.argv[1:5]
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    result = export_bundle(image, plan, out_path, title=title)
    Path(out_path).with_suffix(".result.json").write_text(
        json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()