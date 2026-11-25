# -*- coding: utf-8 -*-
"""Offline object veto and short, source-supported branch-gap trials."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from gongbi_training.runtime import LAB, configure
from recognition_groups_20261123 import SCENE, label, read, sha, summarize, verify_baselines
from recognition_rejection_fusion_20261123 import object_inputs
from recognition_structure_20261123 import geodesic_votes


OUT = LAB / "recognition_role_gap_20261123"
PREVIOUS = LAB / "recognition_structure_20261123"
OBJECTS = LAB / "object_prospective_20261118"
SETTINGS = dict(tree_ink_fraction=.5, minimum_ink=12, cross_contrast=.012,
                anchor_contrast=.025, minimum_tree_probability=.50, max_gap=3,
                maximum_context_side=1280)
ARM_NAMES = ("形态单独", "256 对象否决", "连通树区否决")


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")


def shifted(array, dx, dy):
    """Sample at (x + dx, y + dy), without wrapping across the image edge."""
    height, width = array.shape
    result = np.zeros_like(array)
    x0, x1 = max(0, -dx), min(width, width - dx)
    y0, y1 = max(0, -dy), min(height, height - dy)
    result[y0:y1, x0:x1] = array[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return result


def short_tree_gaps(gray, low_tree, base, proposal, cv2):
    """Admit a short opposing-anchor gap when every pixel follows a source ridge."""
    smooth = cv2.GaussianBlur(gray, (0, 0), .7)
    broad = cv2.GaussianBlur(smooth, (0, 0), 7) - smooth
    has_tree = (base == 2) & (low_tree >= SETTINGS["minimum_tree_probability"])
    accept = np.zeros_like(has_tree)
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        normal_a, normal_b = shifted(smooth, dy, -dx), shifted(smooth, -dy, dx)
        cross = np.minimum(normal_a, normal_b) - smooth
        center = (cross >= SETTINGS["cross_contrast"]) & (broad > 0)
        anchors = has_tree & (broad >= SETTINGS["anchor_contrast"])
        candidates = center & proposal & (low_tree >= SETTINGS["minimum_tree_probability"])
        continuous = np.ones_like(has_tree)
        for span in range(1, SETTINGS["max_gap"] + 1):
            continuous &= shifted(candidates, span * dx, span * dy)
            pair = anchors & continuous & shifted(anchors, (span + 1) * dx, (span + 1) * dy)
            for step in range(1, span + 1):
                accept |= shifted(pair, -step * dx, -step * dy)
    return accept


def scene_evidence(rgb, net, cv2, torch, solver):
    from guohua_objects_train_20261112 import pack

    probs = []
    for size in (256, 512):
        x, _, (l, t, w, h, ow, oh) = pack(rgb, size=size)
        with torch.inference_mode():
            prob = net(torch.from_numpy(x)[None]).softmax(1)[0].numpy()
        probs.append(np.stack([cv2.resize(p[t:t + h, l:l + w], (ow, oh)) for p in prob]))
    low = probs[0].argmax(0)
    mean = (probs[0] + probs[1]).argmax(0)
    (connected, _), _ = geodesic_votes(rgb, probs, mean, cv2, solver)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    fill = short_tree_gaps(gray, probs[0][2], connected, (connected == 1) & (low == 2), cv2)
    return low.astype(np.uint8), connected.astype(np.uint8), mean.astype(np.uint8), fill


def style_sources(baseline):
    works = {row["id"]: row for row in read(OBJECTS / "dataset_manifest.json")["records"]}
    witness = {r["id"]: r for r in read(OBJECTS / "local_references.json")["records"]}
    museum = {r["id"]: r for r in read(LAB / "cunfa_fresh_validation_20261111/test_manifest.json")["records"]}
    rows = []
    for row in baseline:
        if row["admitted"] is None:
            continue
        assert row["sha256"] == sha(row["image"])
        source, box = row["image"], row["box"]
        if row["id"] in witness:
            source = works[witness[row["id"]]["image_id"]]["image"]
        if row["id"] in museum:
            source, box = museum[row["id"]]["source_image"], museum[row["id"]]["source_box"]
        rows.append(dict(id=row["id"], source=source, box=box, source_sha256=sha(source),
                         expected=row["expected"], admitted=row["admitted"],
                         reference_kind=row["reference_kind"], shape_score=row["scores"]["shape"]))
    return rows


def role_experiment(rows, infer, cv2):
    gathered = []
    sheet = Image.new("RGB", (1210, 74 + 136 * ((len(rows) + 3) // 4)), "#f4f1e8")
    draw = ImageDraw.Draw(sheet)
    label(draw, (12, 14), "皴法候选和树木归属对照", 25)
    label(draw, (12, 45), "从左到右依次看形态、256 对象否决、连通树区否决。", 15)
    for i, row in enumerate(rows):
        assert sha(row["source"]) == row["source_sha256"]
        im, (low, connected, _, _) = infer(row["source"])
        gray = np.asarray(im.convert("L"), dtype=np.float32) / 255
        ink = cv2.GaussianBlur(gray, (0, 0), 7) - gray >= .025
        x0, y0, x1, y1 = row["box"] or (0, 0, *im.size)
        patch = np.s_[y0:y1, x0:x1]
        is_ink = ink[patch]
        support = int(is_ink.sum())
        fractions = [float(m[patch][is_ink].mean()) if support else None
                     for m in (low == 2, (low == 2) & (connected == 2))]
        veto = [support >= SETTINGS["minimum_ink"] and f >= SETTINGS["tree_ink_fraction"]
                for f in fractions]
        shape_pass = row["shape_score"] >= .5
        decisions = [shape_pass, shape_pass and not veto[0], shape_pass and not veto[1]]
        gathered.append(dict(**row, ink_samples=support, tree_fractions=fractions,
                             tree_veto=veto, accepted=dict(zip(ARM_NAMES, decisions))))
        x, y = i % 4 * 302 + 10, i // 4 * 136 + 72
        thumb = ImageOps.contain(im.crop((x0, y0, x1, y1)), (111, 100))
        sheet.paste(thumb, (x, y))
        label(draw, (x + 120, y + 3), row["id"], 14)
        label(draw, (x + 120, y + 25), row["expected"] or "类外参考", 14)
        label(draw, (x + 120, y + 49), " / ".join("放行" if d else "未知" for d in decisions), 13)
        label(draw, (x + 120, y + 72), "树占比 " + " / ".join(
            f"{f:.0%}" if f is not None else "—" for f in fractions), 13)
    sheet.save(OUT / "cunfa_object_veto.png")
    return gathered


def gap_experiment(rows, infer, cv2):
    from gongbi_training.source_scene_mapping import source_to_scene

    results = []
    for row in rows:
        assert row["image_sha256"] == sha(row["image"]) and row["mask_sha256"] == sha(row["mask"])
        _, (_, original, _, fill) = infer(row["image"])
        revised = np.where(fill, 2, original)
        with Image.open(row["mask"]) as mask:
            truth = source_to_scene(np.asarray(mask))
        valid = truth != 255
        contact = np.zeros_like(valid)
        for own, other in ((1, 2), (2, 1)):
            if np.any(truth == other):
                distance = cv2.distanceTransform((truth != other).astype(np.uint8), cv2.DIST_L2, 5)
                contact |= (truth == own) & (distance <= 8)
        results.append(dict(id=row["id"], batch=row["batch"], filled=int(fill.sum()),
                            filled_truth={str(k): int(np.count_nonzero(fill & (truth == k)))
                                          for k in (0, 1, 2, 3, 255)},
                            boundary=[np.bincount(truth[contact].astype(int)*4 + m[contact],
                                                 minlength=16).reshape(4,4).tolist()
                                      for m in (original, revised)]))
    batch = {}
    for name in sorted({r["batch"] for r in results}):
        cases = [r for r in results if r["batch"] == name]
        batch[name] = [summarize(sum(np.array(r["boundary"][i]) for r in cases)) for i in (0, 1)]
    return dict(rows=results, boundary=batch)


def branch_review(infer, references):
    notes = []
    sheet = Image.new("RGB", (1030, 365), "#f4f1e8")
    draw = ImageDraw.Draw(sheet)
    label(draw, (12, 12), "同一分枝的旧标点与原图复核位置", 24)
    label(draw, (12, 47), "橙圈为旧标点，绿圈为同一枝条上可见墨芯的新核查点。", 16)
    for k, ref in enumerate(references):
        im, (low, connected, mean, fill) = infer(ref["image"])
        x, y = ref["xy"]
        status = dict(low=int(low[y,x]), mean=int(mean[y,x]), connected=int(connected[y,x]),
                      after=int(2 if fill[y,x] else connected[y,x]))
        notes.append(dict(**ref, prediction=status))
    im, _ = infer(references[0]["image"])
    x0, y0, x1, y1 = (118, 403, 156, 441)
    tile = im.crop((x0, y0, x1, y1)).resize((266,266), Image.Resampling.NEAREST)
    sheet.paste(tile, (12,79))
    for k, ref in enumerate(notes):
        x, y = ref["xy"]
        px, py = 12 + 7 * (x - x0 + .5), 79 + 7 * (y - y0 + .5)
        color = "#e48631" if k == 0 else "#0c9e70"
        draw.ellipse((px-4, py-4, px+4, py+4), outline=color, width=2)
        object_names = {0: "空景", 1: "山石", 2: "树木", 3: "其他"}
        label(draw, (298, 86 + 48 * k), f"{ref['id']}  ({x}, {y})", 16, color)
        status = ref["prediction"]
        label(draw, (298, 108 + 48 * k),
              "256、平均、连通、补缝　" + " ／ ".join(object_names[v] for v in status.values()), 15)
    sheet.save(OUT / "branch_reference_review.png")
    return notes


def main():
    configure()
    import cv2
    import torch
    from skimage.graph import MCP_Geometric
    from gongbi_training.scene_objects import SceneObjects

    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True, parents=True)
    locked = verify_baselines()
    style = style_sources(read(PREVIOUS / "results.json")["cunfa"]["records"])
    objects = object_inputs()
    source = next(r["image"] for r in objects if r["id"] == "zouzhe_xiangjiaoqiuwutu_43")
    branch = [dict(id=ident, xy=xy, image=source, scope=scope)
              for ident, xy, scope in [
                  ("旧参考", [135,421], "off_ink_between_limbs; retained for audit"),
                  ("复核墨芯 1", [134,418], "same branch source visual"),
                  ("复核墨芯 2", [138,416], "same branch source visual"),
                  ("复核墨芯 3", [142,413], "same branch source visual")]]
    save("protocol.json", dict(script_sha256=sha(__file__), parameters=SETTINGS, style_refs=style,
                               object_refs=objects, branch_refs=branch,
                               source_only_reference_review=True, historical_regression=True))
    weight = SCENE / "mapping_fixed.pt"
    assert sha(weight) == read(SCENE / "checkpoint_freeze.json")["mapping_fixed"]
    net = SceneObjects()
    net.load_state_dict(torch.load(weight, map_location="cpu", weights_only=False)["model"])
    net.eval()
    cache = {}

    def infer(path):
        path = str(Path(path))
        if path not in cache:
            with Image.open(path) as f:
                im = f.convert("RGB")
            limit = SETTINGS["maximum_context_side"]
            work = ImageOps.contain(im, (limit, limit)) if max(im.size) > limit else im
            maps = scene_evidence(np.asarray(work), net, cv2, torch, MCP_Geometric)
            if work.size != im.size:
                maps = tuple(cv2.resize(mask.astype(np.uint8), im.size, interpolation=cv2.INTER_NEAREST)
                             for mask in maps)
                maps = (*maps[:3], maps[3].astype(bool))
            cache[path] = im, maps
        return cache[path]

    gates = role_experiment(style, infer, cv2)
    print("CUNFA", len(gates), flush=True)
    gaps = gap_experiment(objects, infer, cv2)
    review = branch_review(infer, branch)
    assert locked == verify_baselines()
    save("results.json", dict(role_checks=gates, gap_checks=gaps, branch_review=review,
                              painting_baseline=locked, adopted=False, independent_validation=False,
                              robot_connected=False))
    print("GAPS", sum(r["filled"] for r in gaps["rows"]), flush=True)


if __name__ == "__main__":
    main()
