# -*- coding: utf-8 -*-
"""从已下载的公共领域图库提取山石 patch，训练四类皴法模型。"""
import argparse
import json
from pathlib import Path

from master_studio import corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="重新提取 patch")
    ap.add_argument("--per-image", type=int, default=8)
    ap.add_argument("--patch", type=int, default=240)
    args = ap.parse_args()
    m = corpus.manifest()
    if args.extract:
        total = 0
        for aid, info in m["artists"].items():
            if not corpus.images_for(aid):
                print("skip (no images):", info["name"])
                continue
            n = corpus.extract_patches(aid, args.per_image, args.patch)
            total += n
            print("patches", info["name"], n)
        print("total patches", total)
    model = corpus.train_model()
    print(json.dumps({k: model[k] for k in ("classes", "counts", "loo_accuracy", "samples")}, ensure_ascii=False, indent=2))
    print("model:", corpus.MODEL)


if __name__ == "__main__":
    main()