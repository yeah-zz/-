# -*- coding: utf-8 -*-
"""Evaluate image-supported structure proposals with frozen tree/rock probabilities."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure
from gongbi_training.visual_structure import detect, sections
from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines
from recognition_scene_topology_20261124 import write


OUT = LAB / "visual_structure_round_20261124"
HOLDOUT = LAB / "trunk_rock_context_holdout_20261124"
MODES = ("dark", "light", "paired", "fused")
COLORS = ("#156b9c", "#934bd0", "#009e84", "#127da1")


def hits(mask, points):
    yy, xx = np.ogrid[-3:4, -3:4]
    disk = yy**2+xx**2 <= 9
    padded = np.pad(mask, 3)
    return [bool(np.any(padded[y:y+7, x:x+7] & disk)) for y,x in points]


def main():
    configure()
    import cv2
    import torch
    from skimage.morphology import skeletonize
    from gongbi_training.scene_objects import SceneObjects
    from gongbi_training.source_scene_mapping import source_to_scene
    from recognition_trunk_rock_context_20261124 import scene_probabilities
    import gongbi_training.visual_structure as module

    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    baselines = verify_baselines()
    previous = read(HOLDOUT/"protocol.json")
    checkpoint = SCENE/"mapping_fixed.pt"
    assert sha(checkpoint) == previous["checkpoint_sha256"]
    refs = {name: HOLDOUT/name for name in ("reference_freeze.json","scan_reference_freeze.json")}
    write(OUT/"protocol.json", {"checkpoint": sha(checkpoint), "candidate_code": sha(module.__file__),
                                "runner": sha(__file__), "references": {k:sha(p) for k,p in refs.items()},
                                "uses_annotations_for_prediction": False, "stage": "archive_development"})
    model = SceneObjects()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    model.eval()
    cases = defaultdict(list)
    for case in read(refs["reference_freeze.json"])["cases"]:
        assert sha(case["image"]) == case["image_sha256"]
        cases[case["image"]].append(case)
    windows = {row["image"]:row for row in read(refs["scan_reference_freeze.json"])["windows"]}
    panel = Image.new("RGB", (1040, 68+11*145), "#f4f1e8")
    board = ImageDraw.Draw(panel)
    label(board, (12,8), "暗墨与多源结构候选对照", 24)
    label(board, (12,40), "红圈为树身核查点。亮脊、配对轴线和暗墨候选均在原图坐标下生成。", 15)
    panel_row = 0
    reference_scores, negatives, by_image = [], [], []
    for path in dict.fromkeys([*cases, *windows]):
        with Image.open(path) as opened:
            source = opened.convert("RGB")
        rgb = np.asarray(source)
        cues = detect(rgb,cv2,skeletonize)
        masks = {mode:cues[mode] for mode in MODES[:-1]}
        masks["fused"] = np.logical_or.reduce(list(masks.values()))
        probs = scene_probabilities(rgb,model,torch,cv2)
        mean = (probs[256]+probs[512]) / 2
        tree = (mean[2]-mean[1] >= .18) & (mean[2] >= mean[[0,3]].max(axis=0))
        rock = (mean[1]-mean[2] >= .18) & (mean[1] >= mean[[0,3]].max(axis=0))
        paired_body = np.zeros_like(cues["half_width"])
        for sample in sections(cues, masks["paired"]):
            a,b = [tuple(np.rint(p).astype(int)) for p in sample["sides_xy"]]
            cv2.line(paired_body,a,b,1,1)
        for case in cases[path]:
            x0,y0,x1,y1 = case["crop"]
            region = np.s_[y0:y1,x0:x1]
            points = case["points"]
            entry = {"key":case["key"], "title":case["title"], "role":case["role"], "crop":case["crop"],
                     "points_yx":points, "candidate_hits":{m:hits(v,points) for m,v in masks.items()},
                     "tree_hits":{m:hits(v&tree,points) for m,v in masks.items()},
                     "paired_body_exact_hits":[bool(paired_body[y,x]) for y,x in points],
                     "candidate_counts":{m:int(v[region].sum()) for m,v in masks.items()},
                     "tree_counts":{m:int((v&tree)[region].sum()) for m,v in masks.items()}}
            reference_scores.append(entry)
            raw=source.crop(case["crop"])
            zoom=min(3.0,96/raw.height,178/raw.width)
            size=(round(raw.width*zoom),round(raw.height*zoom))
            yy=68+145*panel_row
            label(board,(12,yy),case["title"],15)
            for k,mode in enumerate((None,*MODES)):
                image=raw.resize(size, Image.Resampling.NEAREST)
                draw=ImageDraw.Draw(image)
                if mode:
                    for py,px in np.argwhere(masks[mode][region]):
                        tx,ty=(px+.5)*zoom,(py+.5)*zoom
                        color=COLORS[k-1] if mode!="fused" or tree[py+y0,px+x0] else "#aa7caf"
                        draw.ellipse((tx-1,ty-1,tx+1,ty+1),fill=color)
                for py,px in points:
                    tx,ty=(px-x0+.5)*zoom,(py-y0+.5)*zoom
                    draw.ellipse((tx-3,ty-3,tx+3,ty+3),outline="#ff4b38")
                xx=12+204*k
                panel.paste(image,(xx,yy+20))
                label(board,(xx,yy+118),("原图","暗墨","亮脊","双侧与色带","融合 · 紫色未获树木证据")[k],13)
            panel_row+=1
            if points:
                keep = np.zeros_like(masks["paired"])
                keep[region] = masks["paired"][region]
                records = sections(cues,keep)
                for item in records:
                    px,py=item["center_xy"]
                    item["tree_probability"]=float(mean[2,py,px])
                    item["rock_probability"]=float(mean[1,py,px])
                    item["association"]="tree" if tree[py,px] else "rock" if rock[py,px] else "uncertain"
                write(OUT/f"{case['key']}_sections.json", {"source":path,"source_sha256":sha(path),
                    "geometry_kind":"inferred_body_sections_not_drawing_paths","sections":records,
                    "source_size_wh":source.size,
                    "observed_dark_candidate_yx":(np.argwhere(masks["dark"][region])+[y0,x0]).tolist(),
                    "inferred_light_center_yx":(np.argwhere(masks["light"][region])+[y0,x0]).tolist()})
                Image.fromarray(paired_body[region]*255).save(OUT/f"{case['key']}_inferred_body.png")
        if path in windows:
            row=windows[path]
            assert sha(path)==row["sha256"] and sha(row["mask"])==row["mask_sha256"]
            with Image.open(row["mask"]) as target:
                truth=source_to_scene(np.asarray(target))
            for box in row["boxes"]:
                x0,y0,x1,y1=box
                select = np.s_[y0:y1,x0:x1]
                eligible = truth[select]==1
                negatives.append({"id":row["id"],"group":row["work_group"],"crop":box,"rough_rock_support":int(eligible.sum()),
                    "counts":{m:{"geometric":int((v[select]&eligible).sum()),
                                 "tree":int((v[select]&tree[select]&eligible).sum()),
                                 "rock":int((v[select]&rock[select]&eligible).sum()),
                                 "uncertain":int((v[select]&~tree[select]&~rock[select]&eligible).sum())} for m,v in masks.items()}})
            by_image.append(row["id"])
        print(Path(path).name, {m:int(v.sum()) for m,v in masks.items()},flush=True)
    assert baselines == verify_baselines()
    panel.save(OUT/"structure_comparison.png")
    write(OUT/"results.json", {"manual_regions":reference_scores,"negative_windows":negatives,
                              "archive_images":by_image,"painting_baselines":baselines})


if __name__=="__main__":
    main()
