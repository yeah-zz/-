# -*- coding: utf-8 -*-
"""Offline stroke-group evidence and two-way object confusion regression."""

import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from gongbi_training.runtime import LAB, configure
from recognition_main_tracks_20261122 import FROZEN, SCENE, read, sha


OUT = LAB / "recognition_groups_20261123"
PRIOR = LAB / "guohua_priority_check_20261116"
HKMA = LAB / "recognition_main_tracks_20261122"
AAM = LAB / "role_style_evidence_20261118"
LABELS = ("披麻皴", "斧劈皴")
MODES = ("256", "512", "mean")
MODE_NAMES = ("256 输入", "512 输入", "固定平均")
FONT = Path("C:/Windows/Fonts/msyh.ttc")

# Boxes are in percent of the original crop. Each contrast shows visible
# non-group material in the same example. Neither box acts as pixel truth.
GROUPS = [
    ("香港艺术馆·斧劈", HKMA / "axe_rock_crop.png", "斧劈皴",
     (55, 55, 91, 91), (13, 1, 49, 37), "岩面宽短斜墨带", "岩顶勾边"),
    ("香港艺术馆·披麻", HKMA / "hemp_rock_crop.png", "披麻皴",
     (0, 28, 30, 85), (35, 30, 55, 83), "贴左坡的成组长线", "坡面留白"),
    ("亚洲艺术博物馆·小斧劈", AAM / "views/small_axe.png", "斧劈皴",
     (11, 19, 56, 79), (65, 1, 97, 24), "宽头短削的密集墨块", "上方浅底"),
    ("亚洲艺术博物馆·大斧劈", AAM / "views/large_axe.png", "斧劈皴",
     (23, 12, 66, 74), (67, 45, 94, 71), "下行宽墨带与斜边", "坡面浅底"),
    ("亚洲艺术博物馆·短披麻", AAM / "views/short_hemp.png", "披麻皴",
     (44, 30, 92, 84), (0, 8, 14, 40), "绕岩面转向的短弧组", "轮廓外浅底"),
    ("亚洲艺术博物馆·长披麻", AAM / "views/long_hemp.png", "披麻皴",
     (24, 11, 70, 87), (0, 7, 12, 47), "沿坡势弯转的长线组", "坡外浅底"),
    ("亚洲艺术博物馆·米点", AAM / "views/mi_dot.png", "米点皴",
     (48, 6, 81, 66), (1, 5, 33, 28), "山团上部叠墨密集区", "山团外浅底"),
]


def write(name, value):
    import json

    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")


def label(draw, xy, value, size=17, color="#20392f"):
    draw.text(xy, value, font=ImageFont.truetype(str(FONT), size), fill=color)


def to_pixels(image, box):
    return [round(part * side / 100) for part, side in zip(box, image.size * 2)]


def verify_baselines():
    lock = read(LAB / "local_rock_patch_20261117/baseline_lock.json")
    for name in ("approved500", "working489"):
        assert sha(lock[name]) == lock[name + "_sha256"], name
    return {name: lock[name + "_sha256"] for name in ("approved500", "working489")}


def group_probe(torch):
    import joblib
    from torchvision.models import mobilenet_v3_small

    frozen = read(FROZEN / "freeze_manifest.json")
    for file in frozen["files"]:
        assert sha(file["path"]) == file["sha256"], file["path"]
    processor = FROZEN / "frozen/encoder_and_preprocessing.py"
    spec = importlib.util.spec_from_file_location("locked_cunfa_preprocessing", processor)
    preprocessing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocessing)
    head = joblib.load(FROZEN / "frozen/local_only_head.joblib")
    assert set(head["trained_ids"]) == set(frozen["train_ids"])
    assert list(head["model"].classes_) == [0, 1]
    weight = next(file["path"] for file in frozen["files"] if file["path"].endswith(".pth"))
    encoder = mobilenet_v3_small(weights=None)
    encoder.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    encoder = encoder.features.eval()

    records = []
    for name, path, kind, group, control, evidence, contrast in GROUPS:
        with Image.open(path) as source:
            im = source.convert("RGB")
        records.append(dict(name=name, image=str(path), image_sha256=sha(path),
                            named_example=kind, group_box=to_pixels(im, group),
                            comparison_box=to_pixels(im, control), evidence=evidence,
                            contrast=contrast, reviewer="assistant_visual",
                            scope="known_teaching_example; unverified group evidence; no pixel label",
                            illustrator_verified=False, eligible_for_independent_test=False))
    write("group_reference_freeze.json", records)
    board = Image.new("RGB", (1130, 87 + 168 * len(records)), "#f4f1e8")
    draw = ImageDraw.Draw(board)
    label(draw, (18, 12), "皴法笔组位置与冻结分类器对照", 26)
    label(draw, (18, 49), "依据原图选框。绿色为可见笔组参考，橙色为同图对照。", 17)
    for index, row in enumerate(records):
        with Image.open(row["image"]) as source:
            im = source.convert("RGB")
        views = [im, im.crop(row["group_box"]), im.crop(row["comparison_box"])]
        samples = np.stack([preprocessing.preprocess(view) for view in views])
        with torch.inference_mode():
            features = encoder(torch.from_numpy(samples)).mean((2, 3))
            features = torch.nn.functional.normalize(features, dim=1).numpy()
        raw = head["model"].decision_function(features)
        pred = head["model"].predict(features)
        row["predictions"] = {
            mode: dict(label=LABELS[int(k)], axe_signed_margin=float(margin))
            for mode, k, margin in zip(("full", "group", "comparison"), pred, raw)
        }
        if row["named_example"] in LABELS:
            direction = 1 if row["named_example"] == LABELS[1] else -1
            row["group_margin_gain"] = float(direction * (raw[1] - raw[2]))
        marked = im.copy()
        pen = ImageDraw.Draw(marked)
        for key, color in (("group_box", "#13bd84"), ("comparison_box", "#f28f45")):
            pen.rectangle(row[key], outline=color, width=max(1, im.width // 100))
        y = 85 + 168 * index
        label(draw, (18, y + 1), row["name"], 18)
        label(draw, (18, y + 32), row["evidence"], 16)
        label(draw, (18, y + 59), row["contrast"], 16)
        for j, (view, tag, mode) in enumerate(zip(
                [marked, *views[1:]], ("原图", "笔组框", "同图对照"),
                ("full", "group", "comparison"))):
            x = 322 + 263 * j
            tile = ImageOps.contain(view, (228, 129))
            board.paste(tile, (x, y))
            label(draw, (x, y + 133), f"{tag}  {row['predictions'][mode]['label']}", 16)
    board.save(OUT / "cunfa_groups.png")
    return records


def summarize(confusion):
    return {
        "tree_to_rock": [int(confusion[2, 1]), int(confusion[2].sum())],
        "rock_to_tree": [int(confusion[1, 2]), int(confusion[1].sum())],
        "tree_recall": [int(confusion[2, 2]), int(confusion[2].sum())],
        "rock_recall": [int(confusion[1, 1]), int(confusion[1].sum())],
    }


def object_probe(torch):
    import cv2
    from gongbi_training.scene_objects import SceneObjects
    from gongbi_training.source_scene_mapping import source_to_scene
    from guohua_objects_train_20261112 import pack

    checkpoint = SCENE / "mapping_fixed.pt"
    assert sha(checkpoint) == read(SCENE / "checkpoint_freeze.json")["mapping_fixed"]
    model = SceneObjects()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    model.eval()
    rows = read(PRIOR / "test_manifest.json")["records"]
    totals = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))
    results = []
    board = Image.new("RGB", (1250, 82 + 195 * len(rows)), "#f4f1e8")
    draw = ImageDraw.Draw(board)
    label(draw, (18, 11), "树石双向错误 · 两个历史作品组的四张裁片", 26)
    label(draw, (18, 48), "原图粗标：米黄＝山石，绿＝树木。  错误图：橙＝树→石，紫＝石→树。", 17)
    for i, row in enumerate(rows):
        im = Image.open(row["image"]).convert("RGB")
        data = np.asarray(im)
        target = source_to_scene(np.asarray(Image.open(row["source_mask"])))
        archived = np.asarray(Image.open(row["target"]))
        assert target.shape == data.shape[:2] and np.array_equal(target, archived)
        scores = {}
        probs = []
        for size in (256, 512):
            x, _, (left, top, width, height, ow, oh) = pack(data, size=size)
            with torch.inference_mode():
                p = model(torch.from_numpy(x)[None]).softmax(dim=1)[0].numpy()
            p = p[:, top:top + height, left:left + width]
            probs.append(np.stack([cv2.resize(channel, (ow, oh)) for channel in p]))
        preds = [p.argmax(0) for p in (*probs, (probs[0] + probs[1]) * 0.5)]
        reference = np.full(data.shape, 231, dtype=np.uint8)
        reference[target == 1] = (213, 166, 102)
        reference[target == 2] = (53, 109, 82)
        images = [im, Image.fromarray(reference)]
        valid = target != 255
        for mode, pred in zip(MODES, preds):
            confusion = np.bincount(
                target[valid].astype(int) * 4 + pred[valid], minlength=16).reshape(4, 4)
            scores[mode] = summarize(confusion)
            scores[mode]["confusion"] = confusion.tolist()
            totals[row["work_group"], mode] += confusion
            totals["all", mode] += confusion
            diagnostic = np.asarray(ImageOps.grayscale(im).convert("RGB")).copy()
            diagnostic[(target == 2) & (pred == 1)] = (237, 121, 43)
            diagnostic[(target == 1) & (pred == 2)] = (159, 62, 191)
            images.append(Image.fromarray(diagnostic))
        y = 84 + i * 195
        label(draw, (10, y), row["id"], 16)
        for j, (image, tag) in enumerate(zip(images, ("原图", "源粗标", *MODE_NAMES))):
            x = 12 + j * 249
            board.paste(ImageOps.contain(image, (230, 136)), (x, y + 23))
            label(draw, (x, y + 162), tag, 15)
        results.append(dict(id=row["id"], work_group=row["work_group"],
                            image_sha256=sha(row["image"]), mask_sha256=sha(row["source_mask"]),
                            scores=scores))
    board.save(OUT / "tree_rock_errors.png")
    return dict(crops=results, known_work_groups=2,
                aggregates={group: {mode: summarize(totals[group, mode]) for mode in MODES}
                            for group in ("all", *sorted({row["work_group"] for row in rows}))},
                first_exposure="2026-11-16", reference_scope="coarse source object masks; known regression",
                object_checkpoint_sha256=sha(checkpoint))


def main():
    configure()
    import torch

    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True, parents=True)
    baselines = verify_baselines()
    write("results.json", dict(cunfa=group_probe(torch), tree_rock=object_probe(torch),
                               painting_baselines=baselines, new_weights=False,
                               independent_validation=False, adopted=False, robot_connected=False))
    assert verify_baselines() == baselines


if __name__ == "__main__":
    main()
