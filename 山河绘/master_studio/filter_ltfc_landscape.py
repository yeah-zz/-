# -*- coding: utf-8 -*-
"""用 ltfc 的 tags/名称筛出山水作品，避免把花鸟、书法、人物裁进山石 patch。"""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "corpus" / "ltfc"
API = "https://api.quanku.art/cag2.ResourceService/listResourcePage"
UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json", "Origin": "https://g2.ltfc.net"}
GOOD = ("山水", "溪山", "秋山", "春山", "云山", "夏山", "冬山", "江", "泉", "林", "峰", "壑", "亭", "舟")
BAD = ("书", "诗", "花", "鸟", "竹", "兰", "人物", "仕女", "佛", "像", "马", "牛", "猫", "虾", "蟹", "菜", "果", "石谱", "画谱")


def main():
    body = {"page": 1, "pageSize": 5000, "src": "SUHA"}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers=UA)
    d = json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))
    meta = {}
    for g in d.get("data", []):
        for x in g.get("data", []):
            h = x.get("huia") or {}
            wid = h.get("Id") or x.get("id")
            if wid:
                meta[wid] = dict(author=h.get("author") or "", name=h.get("name") or "",
                                 tags=h.get("tags") or [])
    rows = []
    for p in sorted(ROOT.glob("*/*.jpg")):
        m = meta.get(p.stem)
        if not m: continue
        name = m["name"]; tags = "".join(m["tags"])
        if any(b in name for b in BAD): continue
        if any(b in tags for b in ("书法", "花鸟", "人物")): continue
        if not (any(g in name for g in GOOD) or "山水" in tags): continue
        rows.append(dict(id=p.stem, artist=p.parent.name, file=str(p.relative_to(ROOT.parent.parent)),
                         name=name, tags=m["tags"]))
    (ROOT / "landscape.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    print("landscape", len(rows), Counter(r["artist"] for r in rows))


if __name__ == "__main__":
    main()