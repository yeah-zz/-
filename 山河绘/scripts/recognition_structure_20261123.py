# -*- coding: utf-8 -*-
"""Offline trials for grouped ink morphology and source-connected tree ownership."""

import importlib.util
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from gongbi_training.runtime import LAB, configure
from recognition_groups_20261123 import FROZEN, SCENE, label, read, sha, summarize, verify_baselines
from recognition_rejection_fusion_20261123 import object_inputs, style_references


OUT = LAB / "recognition_structure_20261123"
PREV = LAB / "recognition_rejection_fusion_20261123"
STYLE_MODES = ("paired_before", "shape", "appearance_shape")
OBJECT_MODES = ("256", "mean", "protect", "connected_ink", "connected_elongated")
CONNECTIVITY_POLICY = dict(weak_ink_contrast=0.018, strong_ink_contrast=0.04,
                           tree_anchor_probability=0.65, rock_anchor_lead=0.20,
                           maximum_path_cost=28, maximum_tree_to_rock_cost_ratio=0.72,
                           elongated_anchor_minimum_extent=14)


def save(name, data):
    import json

    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def derivatives(gray, sigma, cv2):
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigma)
    xx = cv2.Sobel(smoothed, cv2.CV_32F, 2, 0) * (sigma ** 2 / 4)
    yy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 2) * (sigma ** 2 / 4)
    xy = cv2.Sobel(smoothed, cv2.CV_32F, 1, 1) * (sigma ** 2 / 4)
    delta = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2)
    major, minor = (xx + yy + delta) * .5, (xx + yy - delta) * .5
    orientation = np.arctan2(2 * xy, xx - yy)
    return major, minor, orientation


def shape_features(image, cv2):
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    scale = 128 / max(gray.shape)
    gray = cv2.resize(gray, tuple(max(8, round(n * scale)) for n in gray.shape[::-1]),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    features = []
    by_scale = []
    for sigma in (1.0, 2.4, 4.5):
        major, minor, angle = derivatives(gray, sigma, cv2)
        response = np.clip(major, 0, None)
        roundness = np.clip(minor, 0, None) / np.maximum(response, 1e-6)
        blob = response * roundness
        ridge = response * (1 - roundness)
        norm = float(ridge.sum()) + 1e-6
        coherent = np.hypot(cv2.GaussianBlur(ridge * np.cos(angle), (0, 0), 5),
                            cv2.GaussianBlur(ridge * np.sin(angle), (0, 0), 5))
        valid = response > .008
        compact = response * ((roundness >= .5) & valid)
        axial = np.hypot(float((ridge * np.cos(angle)).sum()), float((ridge * np.sin(angle)).sum()))
        features.extend((float(valid.mean()), float(blob.mean()), float(ridge.mean()),
                         float(compact.sum() / (response.sum() + 1e-6)),
                         float(coherent.sum() / norm), float(axial / norm)))
        by_scale.append(dict(ridge=ridge, blob=blob))
    return np.array(features), by_scale


def open_view(row):
    assert sha(row["image"]) == row["sha256"]
    with Image.open(row["image"]) as source:
        im = source.convert("RGB")
    return im.crop(row["box"]) if row["box"] else im


def dot_probes():
    specs = [
        ("米点局部参考", LAB / "cunfa_cross_source_20261117/units/ntm_mi.png",
         [525, 265, 779, 438], "visible_region_of_named_mi_example"),
        ("圆钝墨点，归属待定", LAB / "cunfa_fresh_validation_20261111/crops/fresh_230757_0.png",
         [30, 70, 64, 105], "morphology_only; dot moss not confirmed"),
    ]
    return [dict(id=ident, image=str(path), sha256=sha(path), box=box, reference_kind=kind,
                 admitted=None, expected=None, source_group="visual_diagnostic")
            for ident, path, box, kind in specs]


def compare_style(torch, cv2, train, test, probes):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, normalize
    from torchvision.models import mobilenet_v3_small

    freeze = read(FROZEN / "freeze_manifest.json")
    for item in freeze["files"]:
        assert sha(item["path"]) == item["sha256"]
    processor = FROZEN / "frozen/encoder_and_preprocessing.py"
    spec = importlib.util.spec_from_file_location("frozen_preprocess", processor)
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    weights = next(item["path"] for item in freeze["files"] if item["path"].endswith(".pth"))
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model = model.features.eval()
    variants = train + test + probes
    views = [open_view(row) for row in variants]
    shapes, maps = zip(*(shape_features(im, cv2) for im in views))
    n = len(train)
    scaler = StandardScaler().fit(shapes[:n])
    shape = normalize(np.clip(scaler.transform(shapes), -4, 4))
    appearance = []
    with torch.inference_mode():
        for pos in range(0, len(views), 8):
            x = np.stack([prep.preprocess(im) for im in views[pos:pos + 8]])
            vec = model(torch.from_numpy(x)).mean((2, 3))
            appearance.append(torch.nn.functional.normalize(vec, dim=1).numpy())
    appearance = np.concatenate(appearance)
    baseline = joblib.load(PREV / "paired_gate.joblib")
    assert set(baseline["train_ids"]) == {r["id"] for r in train}
    scores = {"paired_before": baseline["model"].predict_proba(appearance[n:])[:, 1]}
    target = np.array([r["admitted"] for r in train])
    for key, features in (("shape", shape), ("appearance_shape", normalize(np.c_[appearance, shape]))):
        fit = LogisticRegression(C=1, class_weight="balanced", max_iter=2000)
        fit.fit(features[:n], target)
        scores[key] = fit.predict_proba(features[n:])[:, 1]
        joblib.dump(dict(model=fit, scaler=scaler, threshold=0.5, train_ids=[r["id"] for r in train],
                         production_use=False), OUT / f"{key}.joblib")

    rows = [dict(**row, scores={key: float(scores[key][i]) for key in STYLE_MODES})
            for i, row in enumerate(test + probes)]
    metrics = {}
    for mode in STYLE_MODES:
        positive = np.array([r["scores"][mode] for r in rows if r["admitted"] is True])
        negative = np.array([r["scores"][mode] for r in rows if r["admitted"] is False])
        win = positive[:, None] - negative
        mi = next(r["scores"][mode] for r in rows if r["id"] == "ntm_mi")
        metrics[mode] = dict(accepted=[int((positive >= .5).sum()), len(positive)],
                             rejected=[int((negative < .5).sum()), len(negative)],
                             mi_score=mi, positive_above_mi=int((positive > mi).sum()),
                             auc=float(np.mean((win > 0) + .5 * (win == 0))))

    board = Image.new("RGB", (1170, 84 + 123 * len(rows)), "#f4f1e8")
    draw = ImageDraw.Draw(board)
    label(draw, (12, 13), "皴法笔组形态对照", 25)
    label(draw, (12, 47), "右侧为目标皴法支持分数。形态图以蓝色显示细长脊线，橙色显示近圆响应。", 16)
    for i, (row, im, scales) in enumerate(zip(rows, views[n:], maps[n:])):
        y = 83 + i * 123
        board.paste(ImageOps.contain(im, (177, 108)), (15, y))
        for j, field in enumerate(scales):
            ridge = np.clip(field["ridge"] * 38, 0, 1)
            blob = np.clip(field["blob"] * 38, 0, 1)
            color = np.stack((1 - .73 * ridge, 1 - .52 * ridge - .39 * blob,
                              1 - .16 * ridge - .87 * blob), axis=-1)
            overlay = Image.fromarray(np.uint8(np.clip(color, 0, 1) * 255))
            board.paste(ImageOps.contain(overlay, (153, 108)), (217 + 166 * j, y))
        label(draw, (738, y + 2), row["id"], 15)
        label(draw, (738, y + 23), row["expected"] or ("类外参考" if row["admitted"] is False else "观察参考"), 15)
        for j, (tag, mode) in enumerate(zip(("上轮对照", "笔组形态", "外观与笔组"), STYLE_MODES)):
            label(draw, (738, y + 43 + 20 * j), f"{tag}  {row['scores'][mode]:.3f}", 14)
    board.save(OUT / "cunfa_morphology.png")
    return dict(metrics=metrics, records=rows)


def geodesic_votes(im, probs, mean, cv2, solver):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    smooth = cv2.GaussianBlur(gray, (0, 0), .7)
    contrast = cv2.GaussianBlur(smooth, (0, 0), 7) - smooth
    config = CONNECTIVITY_POLICY
    ink = contrast >= config["weak_ink_contrast"]
    core = contrast >= config["strong_ink_contrast"]
    low, high = probs
    protect = (low.argmax(0) == 2) & (mean == 1)
    tree_anchors = core & (mean == 2) & (low[2] >= config["tree_anchor_probability"])
    rock_anchors = core & (low.argmax(0) == 1) & (high.argmax(0) == 1) & (
        (np.minimum(low[1] - low[2], high[1] - high[2])) >= config["rock_anchor_lead"])
    count, ids, stats, _ = cv2.connectedComponentsWithStats(tree_anchors.astype(np.uint8), connectivity=8)
    elongated = np.zeros_like(tree_anchors)
    for idx in range(1, count):
        width, height, area = stats[idx, 2:5]
        if area >= 8 and max(width, height) >= config["elongated_anchor_minimum_extent"] and max(width, height) >= 1.7 * min(width, height):
            elongated |= ids == idx

    def distances(anchors, propensity):
        points = np.argwhere(anchors)
        if not len(points):
            return np.full(gray.shape, np.inf)
        support = np.where(ink, 1.0 + 2.5 * (1.0 - propensity)
                           + np.maximum(.06 - contrast, 0) * 10, np.inf)
        route = solver(support, fully_connected=True)
        distance, _ = route.find_costs(points.tolist(), find_all_ends=False)
        return distance

    rock_distance = distances(rock_anchors, (low[1] + high[1]) * .5)
    predictions = []
    detail = []
    for anchors in (tree_anchors, elongated):
        tree_distance = distances(anchors, low[2])
        accepted = protect & (tree_distance <= config["maximum_path_cost"]) & (
            tree_distance <= config["maximum_tree_to_rock_cost_ratio"] * rock_distance)
        predictions.append(np.where(accepted, 2, mean))
        detail.append((anchors, accepted))
    return predictions, detail


def evaluate_objects(torch, cv2, rows):
    from skimage.graph import MCP_Geometric
    from gongbi_training.scene_objects import SceneObjects
    from gongbi_training.source_scene_mapping import source_to_scene
    from guohua_objects_train_20261112 import pack

    checkpoint = SCENE / "mapping_fixed.pt"
    assert sha(checkpoint) == read(SCENE / "checkpoint_freeze.json")["mapping_fixed"]
    model = SceneObjects()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    model.eval()
    totals = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))
    results = []
    reviewed_runs = read(LAB / "ink_run_review_20261118/results.json")["records"]
    line_scores = []
    board = Image.new("RGB", (1200, 86 + 139 * len(rows)), "#f4f1e8")
    draw = ImageDraw.Draw(board)
    label(draw, (10, 11), "沿原图墨迹判断树石归属", 25)
    label(draw, (10, 44), "橙色为树误作石，紫色为石误作树。青色为细长树区起点和从起点补回的墨迹。", 16)
    for i, row in enumerate(rows):
        assert sha(row["image"]) == row["image_sha256"] and sha(row["mask"]) == row["mask_sha256"]
        with Image.open(row["image"]) as file:
            im = file.convert("RGB")
        raw = np.asarray(im)
        probs = []
        for size in (256, 512):
            x, _, (l, t, w, h, ow, oh) = pack(raw, size=size)
            with torch.inference_mode():
                prob = model(torch.from_numpy(x)[None]).softmax(1)[0].numpy()
            probs.append(np.stack([cv2.resize(p[t:t + h, l:l + w], (ow, oh)) for p in prob]))
        low = probs[0].argmax(0)
        mean = (probs[0] + probs[1]).argmax(0)
        protect = np.where((mean == 1) & (low == 2), 2, mean)
        connected, evidence = geodesic_votes(raw, probs, mean, cv2, MCP_Geometric)
        with Image.open(row["mask"]) as file:
            truth = source_to_scene(np.asarray(file))
        valid = truth != 255
        boundary = np.zeros_like(valid)
        for own, opposite in ((1, 2), (2, 1)):
            if np.any(truth == opposite):
                d = cv2.distanceTransform((truth != opposite).astype(np.uint8), cv2.DIST_L2, 5)
                boundary |= (truth == own) & (d <= 8)
        measures = {}
        branches = []
        for j, (mode, pred) in enumerate(zip(OBJECT_MODES, (low, mean, protect, *connected))):
            measures[mode] = {}
            for region, keep in (("all", valid), ("boundary", boundary)):
                c = np.bincount(truth[keep].astype(int) * 4 + pred[keep], minlength=16).reshape(4, 4)
                totals[row["batch"], region, mode] += c
                measures[mode][region] = c.tolist()
            for ref in reviewed_runs:
                if ref["image_id"] != row["id"] or ref["object_type"] == "unknown":
                    continue
                x, y = np.concatenate(ref["runs"]).T
                cls = 2 if ref["object_type"] == "tree" else 1
                branches.append(dict(id=ref["id"], mode=mode, role=ref["object_type"],
                                     correct=int(np.count_nonzero(pred[y, x] == cls)), points=len(x)))
            tile = np.asarray(ImageOps.grayscale(im).convert("RGB")).copy()
            tile[(truth == 2) & (pred == 1)] = (237, 121, 43)
            tile[(truth == 1) & (pred == 2)] = (159, 62, 191)
            board.paste(ImageOps.contain(Image.fromarray(tile), (164, 101)),
                        ((j + 1) * 171 + 8, 103 + i * 139))
        anchors, restored = evidence[-1]
        tile = np.asarray(ImageOps.grayscale(im).convert("RGB")).copy()
        tile[anchors | restored] = (2, 182, 177)
        board.paste(ImageOps.contain(im, (164, 101)), (8, 103 + i * 139))
        board.paste(ImageOps.contain(Image.fromarray(tile), (164, 101)), (6 * 171 + 8, 103 + i * 139))
        label(draw, (8, 84 + i * 139), row["id"], 14)
        for j, tag in enumerate(("原图", "256", "平均融合", "全保护", "沿墨迹连通", "从细长墨迹出发", "细长起点及补回墨迹")):
            label(draw, (j * 171 + 8, 208 + i * 139), tag, 13)
        results.append(dict(**row, scores=measures,
                            anchors=[int(np.count_nonzero(a)) for a, _ in evidence],
                            restored=[int(np.count_nonzero(r)) for _, r in evidence]))
        line_scores.extend(branches)
    board.save(OUT / "tree_ink_connectivity.png")
    aggregates = {batch: {region: {mode: summarize(totals[batch, region, mode])
                                   for mode in OBJECT_MODES}
                          for region in ("all", "boundary")}
                  for batch in sorted({r["batch"] for r in rows})}
    return dict(aggregates=aggregates, records=results, reviewed_run_knot_checks=line_scores)


def main():
    configure()
    import cv2
    import torch

    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True, parents=True)
    baselines = verify_baselines()
    train, test = style_references()
    probes, rows = dot_probes(), object_inputs()
    save("protocol.json", dict(script_sha256=sha(__file__), style_fitting=train, style_check=test,
                               morphology_only=probes, object_check=rows,
                               morphology_scales=[1, 2.4, 4.5], cutoff=0.5,
                               connectivity=CONNECTIVITY_POLICY, masks_for_evaluation_only=True,
                               robot_connected=False, independent_validation=False))
    style = compare_style(torch, cv2, train, test, probes)
    print("CUNFA", style["metrics"], flush=True)
    objects = evaluate_objects(torch, cv2, rows)
    assert baselines == verify_baselines()
    save("results.json", dict(cunfa=style, objects=objects, baselines=baselines, adopted=False,
                              independent_validation=False, robot_connected=False))
    print("OBJECTS", len(objects["records"]), "archived crops", flush=True)


if __name__ == "__main__":
    main()
