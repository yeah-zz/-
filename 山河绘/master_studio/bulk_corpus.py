# -*- coding: utf-8 -*-
"""从 Met Open Access 批量建立中国山水画语料（自监督编码器预训练用）。"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

BULK = Path(__file__).resolve().parent / "corpus" / "bulk"
API = "https://collectionapi.metmuseum.org/public/collection/v1"
QUERIES = ["Chinese landscape", "landscape", "mountains", "ink landscape", "river landscape", "scholar landscape", "Chinese painting", "ink painting", "pine tree", "bamboo", "album", "handscroll"]


def get_json(url, timeout=30):
    req = Request(url, headers={"User-Agent": "ShanheHui-Encoder/1.0 (public domain research)"})
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def search_all(limit=100, pages=3):
    ids = []
    for q in QUERIES:
        for page in range(pages):
            try:
                d = get_json("%s.1/search?q=%s&hasImages=true&isPublicDomain=true&departmentId=6&limit=%d&offset=%d" %
                             (API, quote(q), limit, page * limit))
                ids += d.get("objectIDs", [])
                print("search", q, page, len(d.get("objectIDs", [])), flush=True)
            except Exception as e:
                print("search failed", q, page, e, flush=True)
    return list(dict.fromkeys(ids))


def object_info(oid):
    try:
        d = get_json("%s/objects/%d" % (API, oid), timeout=20)
        medium = (d.get("medium") or "").lower()
        if (d.get("isPublicDomain") and d.get("department") == "Asian Art"
                and d.get("classification") == "Paintings"
                and d.get("primaryImageSmall")
                and ("ink" in medium or "color" in medium or "silk" in medium or "paper" in medium)):
            return dict(objectID=oid, title=d.get("title"), artist=d.get("artistDisplayName"),
                        culture=d.get("culture"), date=d.get("objectDate"), medium=d.get("medium"),
                        image=d["primaryImageSmall"], source=d.get("objectURL"))
    except Exception:
        pass
    return None


def download(row, force=False):
    target = BULK / "images" / (str(row["objectID"]) + ".jpg")
    if target.exists() and not force:
        return True
    try:
        req = Request(row["image"], headers={"User-Agent": "ShanheHui-Encoder/1.0"})
        with urlopen(req, timeout=25) as r:
            data = r.read()
        if len(data) < 8000:
            return False
        target.write_bytes(data)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--search-limit", type=int, default=100)
    ap.add_argument("--pages", type=int, default=3)
    args = ap.parse_args()
    BULK.mkdir(parents=True, exist_ok=True)
    (BULK / "images").mkdir(exist_ok=True)
    ids = search_all(args.search_limit, args.pages)
    print("unique ids", len(ids), flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(object_info, oid): oid for oid in ids}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                rows.append(r)
            if i % 50 == 0:
                print("checked", i, "kept", len(rows), flush=True)
            if len(rows) >= args.limit:
                break
    rows = rows[:args.limit]
    old = []
    mp = BULK / "manifest.json"
    if mp.exists():
        try:
            old = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            old = []
    merged = {int(r["objectID"]): r for r in old if r.get("objectID")}
    for r in rows:
        merged[int(r["objectID"])] = r
    rows = list(merged.values())
    mp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = 0
    for i, row in enumerate(rows, 1):
        if download(row):
            ok += 1
        if i % 25 == 0:
            print("downloaded", i, ok, flush=True)
        time.sleep(0.05)
    print("done", len(rows), ok)


if __name__ == "__main__":
    main()