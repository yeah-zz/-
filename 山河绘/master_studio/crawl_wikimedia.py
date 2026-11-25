# -*- coding: utf-8 -*-
"""从 Wikimedia Commons 爬取公共领域中国山水画（无需手工给 URL）。"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "corpus" / "web"
API = "https://commons.wikimedia.org/w/api.php"
QUERIES = [
    "Chinese landscape painting", "Chinese ink painting", "shan shui painting",
    "Song dynasty painting", "Yuan dynasty painting", "Ming dynasty painting",
    "Qing dynasty painting", "Chinese painting scroll", "Chinese painting album",
    "landscape painting China", "mountains Chinese painting", "Chinese literati painting",
]
SKIP = re.compile(r"map|timeline|diagram|photo|portrait|coin|seal|calligraphy|book|page|cover", re.I)


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "ShanheHui-Crawler/1.0 (public domain research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def search(query, pages=4):
    out = []
    for page in range(pages):
        params = dict(action="query", format="json", generator="search",
                      gsrsearch="filetype:bitmap " + query, gsrnamespace=6,
                      gsrlimit=50, gsroffset=page * 50,
                      prop="imageinfo", iiprop="url|size|mime|extmetadata", iiurlwidth=1400)
        try:
            d = get(API + "?" + urllib.parse.urlencode(params))
        except Exception as e:
            print("search fail", query, page, e, flush=True)
            continue
        for p in d.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            title = p.get("title", "")
            if not ii.get("thumburl") or ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if ii.get("width", 0) < 800 or ii.get("height", 0) < 800:
                continue
            if SKIP.search(title):
                continue
            meta = ii.get("extmetadata", {})
            out.append(dict(title=title, url=ii["thumburl"], width=ii["width"], height=ii["height"],
                            artist=(meta.get("Artist", {}) or {}).get("value", ""),
                            license=(meta.get("LicenseShortName", {}) or {}).get("value", ""),
                            description=(meta.get("ImageDescription", {}) or {}).get("value", "")))
        print("search", query, page, len(out), flush=True)
        time.sleep(.25)
    return out


def download(row, i):
    target = OUT / "images" / ("%04d.jpg" % i)
    if target.exists():
        return True
    for attempt in range(4):
        try:
            req = urllib.request.Request(row["url"], headers={"User-Agent": "ShanheHui-Crawler/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                data = r.read()
            if len(data) < 20000:
                return False
            target.write_bytes(data)
            row["file"] = target.name
            return True
        except HTTPError as e:
            if e.code == 429:
                time.sleep(2.5 * (attempt + 1))
                continue
            print("download fail", i, e, flush=True)
            return False
        except Exception as e:
            print("download fail", i, e, flush=True)
            time.sleep(1.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--one", action="store_true")
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    queries = QUERIES[:1] if args.one else QUERIES
    for q in queries:
        rows += search(q, pages=4)
        if len(rows) >= args.limit * 2:
            break
    # 去重
    seen = set(); uniq = []
    for r in rows:
        if r["url"] in seen: continue
        seen.add(r["url"]); uniq.append(r)
    print("unique candidates", len(uniq), flush=True)
    kept = []
    for i, r in enumerate(uniq[:args.limit * 2]):
        if len(kept) >= args.limit:
            break
        if download(r, i):
            kept.append(r)
        if len(kept) % 25 == 0:
            print("downloaded", len(kept), flush=True)
        time.sleep(.8)
    (OUT / "manifest.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", len(kept))


if __name__ == "__main__":
    main()