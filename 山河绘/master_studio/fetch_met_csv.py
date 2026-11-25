# -*- coding: utf-8 -*-
"""从 Met Open Access CSV 过滤出的中国画 ID 取详情并下载图片。"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "corpus" / "met"
API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
UA = {"User-Agent": "ShanheHui-Crawler/1.0 (public domain research)"}


def get_json(oid, timeout=20):
    req = urllib.request.Request(API + str(oid), headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_detail(row):
    for attempt in range(4):
        try:
            d = get_json(row["objectID"])
            if d.get("primaryImageSmall"):
                return dict(row, image=d["primaryImageSmall"], title=d.get("title") or row.get("title"),
                            artist=d.get("artistDisplayName") or row.get("artist"), source=d.get("objectURL"))
            return None
        except Exception as e:
            if "429" in str(e):
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
    return None


def download(row, i):
    target = OUT / "images" / ("%04d.jpg" % i)
    if target.exists():
        return True
    for attempt in range(4):
        try:
            req = urllib.request.Request(row["image"], headers=UA)
            with urllib.request.urlopen(req, timeout=35) as r:
                data = r.read()
            if len(data) < 10000:
                return False
            target.write_bytes(data)
            row["file"] = target.name
            return True
        except Exception as e:
            if "429" in str(e):
                time.sleep(2.0 * (attempt + 1))
                continue
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = json.loads((ROOT / "data" / "met_chinese_paintings.json").read_text(encoding="utf-8"))
    # 优先有作者/标题的
    rows.sort(key=lambda r: (r.get("artist", "") == "Unidentified artist", not r.get("title")))
    rows = rows[:args.limit]
    details = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(get_detail, r): r for r in rows}
        for i, f in enumerate(as_completed(futs), 1):
            d = f.result()
            if d:
                details.append(d)
            if i % 100 == 0:
                print("checked", i, "valid", len(details), flush=True)
            if len(details) >= args.limit:
                break
    details = details[:args.limit]
    (OUT / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print("valid details", len(details), flush=True)
    kept = []
    for i, row in enumerate(details):
        if download(row, i):
            kept.append(row)
        if (i + 1) % 25 == 0:
            print("downloaded", i + 1, len(kept), flush=True)
        time.sleep(.15)
    (OUT / "manifest.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", len(kept))


if __name__ == "__main__":
    main()