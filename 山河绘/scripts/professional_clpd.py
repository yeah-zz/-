"""Acquire a pinned CLPD pilot and build a whole-work split before training."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


ROOT = Path("E:/Codex_Workspace_Data/CLIPascene_20261021/pro_data_20261125")
REPO = "Rane7/CLP_dataset"
REV = "54cba6131985352b643d11394663a214dc008c51"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/"
FILES = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/"
LICENSE = "CC BY-NC 4.0 (repository README)"
MAP = {1: 0, 2: 1, 3: 2, 4: 0, 5: 3, 6: 3, 7: 3, 8: 3,
       9: 3, 10: 0, 11: 3, 12: 0, 13: 1, 14: 3}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def request(url):
    return urlopen(Request(url, headers={"User-Agent": "ShanHeHui academic data audit"}), timeout=22)


def tree(path):
    url = API + path + "?limit=500"
    while url:
        with request(url) as response:
            yield from json.load(response)
            link = re.search(r'<([^>]+)>; rel="next"', response.headers.get("Link", ""))
        url = link[1] if link else None


def fetch(root, item, cache=None):
    remote = item["path"]
    target = root / remote
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get("lfs", {}).get("oid")
    for attempt in range(3):
        try:
            cached = target if target.is_file() else (cache / remote if cache else None)
            content = cached.read_bytes() if cached is not None and cached.is_file() else None
            stale=content is None or len(content)!=item["size"] or (expected and sha256(content).hexdigest()!=expected)
            if stale:
                with request(FILES + quote(remote)) as response:
                    content = response.read()
            if len(content)!=item["size"] or (expected and sha256(content).hexdigest()!=expected):
                raise ValueError(f"Publisher digest or size mismatch for {remote}")
            if stale or not target.is_file():
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(content)
                temporary.replace(target)
            return str(target), sha256(content).hexdigest()
        except OSError:
            if attempt==2:
                raise


def work_key(name):
    base = Path(name).stem.lower()
    if re.fullmatch(r"(harvard|met|princeton)[_-]?\d+", base):
        return None
    key = re.sub(r"_\d+$", "", base)
    return key if key!=base else None


def fingerprint(image):
    import cv2
    tile = np.asarray(image.convert("L").resize((32,32), Image.Resampling.LANCZOS), np.float32)
    digest=[]
    for array in (tile,tile[:,::-1]):
        values=cv2.dct(array.copy())[:8,:8].ravel()[1:]
        digest.append(int("".join("1" if x>np.median(values) else "0" for x in values),2))
    return digest


def near_distance(a,b):
    return min(bin(x^y).count("1") for x in a for y in b)


def split_groups(rows):
    parent={row["work_key"]:row["work_key"] for row in rows}

    def root(key):
        while parent[key]!=key:
            key=parent[key]
        return key

    def union(a,b):
        a,b=sorted((root(a),root(b)))
        parent[b]=a

    near=[]
    for i,a in enumerate(rows):
        for b in rows[:i]:
            exact=a["image_sha256"]==b["image_sha256"]
            distance=near_distance(a["phash"],b["phash"])
            if exact or distance<=6:
                union(a["work_key"],b["work_key"])
                if a["work_key"]!=b["work_key"]:
                    near.append([a["id"],b["id"],distance])
    keys=sorted({root(k) for k in parent},key=lambda k:sha256(("grouped pilot 20261125:"+k).encode()).hexdigest())
    n=max(1,len(keys)//5)
    if len(keys)<6:
        raise ValueError("Too few distinct work groups for a train / development / evaluation split")
    for row in rows:
        group=root(row["work_key"])
        row["group"]=group
        row["split"]="evaluation" if group in keys[:n] else "development" if group in keys[n:2*n] else "train"
    return near


def acquire(output, per_work, work_limit, cache=None):
    output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    inventory={}
    for source in ("training","validation"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            images,masks=pool.map(lambda key:list(tree(f"clp2k/{key}/{source}")),("images","annotations"))
        mask_map={PurePosixPath(x["path"]).stem:x for x in masks if x["path"].endswith(".png")}
        images=[x for x in images if x["path"].lower().endswith((".jpg",".png",".jpeg"))]
        inventory[source]={"images":len(images),"masks":len(mask_map),"paired":0,"unresolved_work":0}
        for image in images:
            name=PurePosixPath(image["path"]).stem
            if name not in mask_map:
                continue
            inventory[source]["paired"]+=1
            work=work_key(name)
            if work:
                groups[work].append((image,mask_map[name],source,work))
            else:
                inventory[source]["unresolved_work"]+=1
    chosen=sorted(groups,key=lambda k:(-len(groups[k]),k))[:work_limit]
    pairs=[]
    for key in chosen:
        group=sorted(groups[key],key=lambda v:sha256(v[0]["path"].encode()).hexdigest())
        pairs.extend(group[:per_work])
    write(output/"inventory.json",{"revision":REV,"publisher_splits":inventory,"named_work_groups":len(groups),
                                  "selected_groups":chosen,"license":LICENSE})
    print(f"Indexed {sum(v['paired'] for v in inventory.values())} pairs. Acquiring {len(pairs)}.",flush=True)

    def read_pair(t):
        image,mask,split,work=t
        image_path,img_sha=fetch(output,image,cache)
        mask_path,mask_sha=fetch(output,mask,cache)
        with Image.open(image_path) as im, Image.open(mask_path) as annotation:
            array=np.asarray(annotation)
            ids=np.unique(array).tolist()
            if im.size!=annotation.size or array.ndim!=2 or set(ids)-{0,255,*MAP}:
                raise ValueError(f"Unknown mask encoding or dimensions for {mask_path}: {ids[:30]}")
            return {"id":Path(image_path).stem,"dataset":REPO,"revision":REV,
                    "image":image_path,"mask":mask_path,"image_sha256":img_sha,"mask_sha256":mask_sha,
                    "image_url":FILES+image["path"],"mask_url":FILES+mask["path"],
                    "publisher_split":split,"work_key":work,"work_identity":"filename title proxy",
                    "author":None,"width":im.width,"height":im.height,"roi":[0,0,im.width,im.height],
                    "license":LICENSE,"annotation":"publisher semantic mask, local precision unverified",
                    "raw_ids":ids,"phash":fingerprint(im)}
    rows=[]
    failed=[]
    with ThreadPoolExecutor(max_workers=9) as pool:
        for i,future in enumerate([pool.submit(read_pair,v) for v in pairs],1):
            try:
                rows.append(future.result())
            except (OSError,ValueError) as error:
                failed.append({"path":pairs[i-1][0]["path"],"reason":str(error)})
            if i%16==0:
                print(f"Processed {i}/{len(pairs)} pairs",flush=True)
    near=split_groups(rows)
    selected=[]
    seen=set()
    for row in rows:
        if row["image_sha256"] not in seen:
            selected.append(row)
            seen.add(row["image_sha256"])
    write(output/"manifest.json",{"revision":REV,"inventory":inventory,"rows":selected,
                "near_duplicate_cross_titles":near,"exact_duplicates_removed":len(rows)-len(selected),
                "failures":failed,"heldout_scope":"publisher image masks; filename title groups; not independent art acceptance"})
    print(f"Ready: {len(selected)} unique pairs, {len(failed)} acquisition failures.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-work",type=int,default=5)
    parser.add_argument("--works",type=int,default=24)
    parser.add_argument("--output", type=Path, default=ROOT / "clpd")
    parser.add_argument("--cache", type=Path)
    options=parser.parse_args()
    if options.per_work<1 or options.works<6:
        parser.error("positive per-work cap and at least six work groups required")
    acquire(options.output, options.per_work, options.works, options.cache)
