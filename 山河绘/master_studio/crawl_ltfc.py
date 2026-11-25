# -*- coding: utf-8 -*-
"""中华珍宝馆（g2.ltfc.net）公开缩略图爬取：按你给的画家清单抓作品。

站点详情页的真实路由是 /view/<src>/<id>；清单里的 /view/100001 是无效示例。
这里直接调用其公开 listResourcePage 接口，拿真实 id 和 987×1600 缩略图。
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "corpus" / "ltfc"
API = "https://api.quanku.art/cag2.ResourceService/listResourcePage"
UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json", "Origin": "https://g2.ltfc.net"}
ARTISTS = ["王希孟", "黄公望", "倪瓒", "王蒙", "吴镇", "沈周", "文徵明", "唐寅", "仇英",
           "徐渭", "八大山人", "朱耷", "石涛", "郑板桥", "郑燮", "吴昌硕", "齐白石", "张大千"]


def fetch_list():
    body = {"page": 1, "pageSize": 5000, "src": "SUHA"}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def works(d):
    out = []
    seen = set()
    for g in d.get("data", []):
        for x in g.get("data", []):
            h = x.get("huia") or {}
            wid = h.get("Id") or x.get("id")
            if not wid or wid in seen:
                continue
            seen.add(wid)
            url = h.get("thumbTileUrl") or h.get("thumbTileUrlx2")
            if not url:
                continue
            out.append(dict(id=wid, src=x.get("src") or "HUIA", author=h.get("author") or "",
                            name=h.get("name") or "", url=url, tags=h.get("tags") or [],
                            size=h.get("thumbTileSize"), snap=h.get("snapUrl") or ""))
    return out


def download(row, i):
    target = OUT / row["artist_dir"] / (row["id"] + ".jpg")
    if target.exists():
        return True
    for attempt in range(3):
        try:
            req = urllib.request.Request(row["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                data = r.read()
            if len(data) < 10000:
                return False
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            row["file"] = str(target.relative_to(ROOT))
            return True
        except Exception as e:
            print("download fail", i, e, flush=True)
            time.sleep(1.5 * (attempt + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--artist", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = fetch_list()
    allw = works(d)
    print("works", len(allw), flush=True)
    picked = []
    for w in allw:
        if args.artist and args.artist not in w["author"]:
            continue
        au = w["author"]
        name = w["name"]
        tags = "".join(w["tags"])
        hit = next((a for a in ARTISTS if a in au), None)
        if hit is None:
            # 作者字段缺失时，用标题+标签兜底（只收山水/绘画）
            if any(k in name for k in ("山水", "溪山", "秋山", "春山", "云山", "江", "泉")) and ("绘画" in tags or "书画" in tags):
                hit = "佚名山水"
        if hit or (args.artist and args.artist in w["author"]):
            hit = hit or w["author"]
            w["artist"] = hit
            w["artist_dir"] = hit
            picked.append(w)
        if len(picked) >= args.limit:
            break
    print("picked", len(picked), flush=True)
    kept = []
    for i, w in enumerate(picked):
        if download(w, i):
            kept.append(w)
        if (i + 1) % 25 == 0:
            print("downloaded", i + 1, len(kept), flush=True)
        time.sleep(.12)
    (OUT / "manifest.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", len(kept))


if __name__ == "__main__":
    main()