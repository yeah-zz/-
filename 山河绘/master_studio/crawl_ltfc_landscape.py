# -*- coding: utf-8 -*-
"""从 ltfc 公开接口按 tags/名称抓山水作品（不限于指定画家）。"""
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "corpus" / "ltfc_landscape"
API = "https://api.quanku.art/cag2.ResourceService/listResourcePage"
UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json", "Origin": "https://g2.ltfc.net"}
GOOD = ("山水", "溪山", "秋山", "春山", "云山", "夏山", "冬山", "江", "泉", "林", "峰", "壑", "亭", "舟")
BAD = ("书", "诗", "花", "鸟", "竹", "兰", "人物", "仕女", "佛", "像", "马", "牛", "猫", "虾", "蟹", "菜", "果", "石谱", "画谱", "临", "仿")


def safe(s):
    s = re.sub(r'[\\/:*?"<>|]', "", s or "佚名")
    return s[:24] or "佚名"


def main():
    body = {"page": 1, "pageSize": 5000, "src": "SUHA"}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers=UA)
    d = json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))
    rows = []
    seen = set()
    for g in d.get("data", []):
        for x in g.get("data", []):
            h = x.get("huia") or {}
            wid = h.get("Id") or x.get("id")
            if not wid or wid in seen: continue
            name = h.get("name") or ""; tags = "".join(h.get("tags") or []); author = h.get("author") or "佚名"
            if any(b in name for b in BAD): continue
            if any(b in tags for b in ("书法", "花鸟", "人物")): continue
            if not (any(g in name for g in GOOD) or "山水" in tags): continue
            url = h.get("thumbTileUrl") or h.get("thumbTileUrlx2")
            if not url: continue
            seen.add(wid)
            rows.append(dict(id=wid, author=author, name=name, tags=h.get("tags") or [], url=url))
    print("landscape candidates", len(rows), flush=True)
    # 优先指定画家
    pri = ("王希孟", "黄公望", "倪瓒", "王蒙", "吴镇", "沈周", "文徵明", "唐寅", "仇英", "徐渭",
           "八大山人", "朱耷", "石涛", "郑板桥", "郑燮", "吴昌硕", "齐白石", "张大千", "龚贤")
    rows.sort(key=lambda r: (not any(a in r["author"] for a in pri), r["author"]))
    OUT.mkdir(parents=True, exist_ok=True)

    def dl(r):
        target = OUT / safe(r["author"]) / (r["id"] + ".jpg")
        if target.exists():
            r["file"] = str(target.relative_to(ROOT)); return r
        for attempt in range(3):
            try:
                data = urllib.request.urlopen(urllib.request.Request(r["url"], headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()
                if len(data) < 10000: return None
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                r["file"] = str(target.relative_to(ROOT)); return r
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        return None
    with ThreadPoolExecutor(max_workers=6) as ex:
        kept = [r for r in ex.map(dl, rows[:300]) if r]
    (OUT / "manifest.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", len(kept))


if __name__ == "__main__":
    main()