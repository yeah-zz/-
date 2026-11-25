# -*- coding: utf-8 -*-
"""批量风格训练：把一个文件夹（或每个子文件夹）的上千幅原作蒸馏成笔法模型。

用法：
  python -m master_studio.train_corpus --folder D:\名画库\黄公望 --name 黄公望 --base huang_gongwang
  python -m master_studio.train_corpus --folder D:\名画库 --per-folder --base ni_zan

说明：v1 学习的是可解释统计特征（墨色覆盖、边缘密度、笔向熵、留白、色度、
冷暖、线粗估计），不是神经风格迁移。特征向量已单独保存，后续可替换为
CNN/CLIP 编码器，渲染与导出层无需改动。
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from . import learning

PROJECT = Path(__file__).resolve().parents[1]
STORE = Path(__file__).resolve().parent / "data" / "custom_profiles.json"
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def images_in(folder, limit=5000):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS and p.is_file())
    if len(files) > limit:
        random.Random(7).shuffle(files)
        files = files[:limit]
    out = []
    for p in files:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((900, 900), Image.LANCZOS)
            out.append(im)
        except Exception:
            continue
    return out


def save_profiles(profiles):
    STORE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if STORE.exists():
        try:
            data = json.loads(STORE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    for p in profiles:
        data[p["id"]] = p
    STORE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--base", default="huang_gongwang")
    parser.add_argument("--per-folder", action="store_true")
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    root = Path(args.folder)
    if not root.is_dir():
        raise SystemExit("文件夹不存在：" + str(root))
    jobs = []
    if args.per_folder:
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            jobs.append((d.name, images_in(d, args.limit)))
    else:
        jobs.append((args.name or root.name, images_in(root, args.limit)))
    made = []
    for name, imgs in jobs:
        if not imgs:
            print("跳过（没有可读图片）：" + name)
            continue
        prof = learning.train_from_images(name, args.base, imgs)
        made.append(prof)
        print("训练完成：%s  %d 幅  id=%s" % (name, len(imgs), prof["id"]))
        print("  指纹：" + json.dumps(prof["fingerprint"], ensure_ascii=False))
    if not made:
        raise SystemExit("没有训练出模型。")
    total = save_profiles(made)
    print("已写入 %s，共 %d 个自定义模型。" % (STORE, total))


if __name__ == "__main__":
    main()