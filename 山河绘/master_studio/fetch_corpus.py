# -*- coding: utf-8 -*-
"""下载 Met Open Access 公共领域图库（需在可访问外网的机器上运行）。"""
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from master_studio import corpus


def download(artist_id, limit=None, force=False):
    m = corpus.manifest()
    info = m["artists"][artist_id]
    d = corpus.artist_dir(artist_id)
    d.mkdir(parents=True, exist_ok=True)
    got = []
    works = info["works"][:limit] if limit else info["works"]
    for w in works:
        target = d / (str(w["objectID"]) + ".jpg")
        if target.exists() and not force:
            got.append(target)
            continue
        url = w["image"].replace("/original/", "/web-large/")
        req = Request(url, headers={"User-Agent": "ShanheHui-MasterStudio/1.0 (research; public domain)"})
        try:
            with urlopen(req, timeout=25) as r:
                data = r.read()
            if len(data) < 8000:
                raise ValueError("下载内容过小")
            target.write_bytes(data)
            got.append(target)
            print("saved", target, len(data))
        except Exception as exc:
            print("failed", w["objectID"], exc)
        time.sleep(0.4)
    meta = [dict(w, file=str((d / (str(w["objectID"]) + ".jpg")).relative_to(corpus.CORPUS))) for w in works if (d / (str(w["objectID"]) + ".jpg")).exists()]
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    m = corpus.manifest()
    ids = list(m["artists"]) if args.artist == "all" else [args.artist]
    for aid in ids:
        print("==", m["artists"][aid]["name"])
        download(aid, args.limit, args.force)


if __name__ == "__main__":
    main()