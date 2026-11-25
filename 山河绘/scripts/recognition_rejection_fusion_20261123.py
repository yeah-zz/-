# -*- coding: utf-8 -*-
"""Fit a localized evidence gate and evaluate asymmetric tree/rock fusion offline."""

import importlib.util
import json
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from gongbi_training.runtime import LAB, configure
from recognition_groups_20261123 import FROZEN, LABELS, PRIOR, SCENE, label, read, sha, summarize, verify_baselines


OUT = LAB / "recognition_rejection_fusion_20261123"
GROUPS = LAB / "recognition_groups_20261123/group_reference_freeze.json"
OBJECT_ARCHIVE = LAB / "object_prospective_20261118"
OUT_OF_SCOPE = {"aam_outside_raveled_rope", "aam_outside_bands_dragged_in_mud", "aam_outside_nail_head"}
OBJECT_ARMS = ("256", "512", "mean", "protect_tree", "protect_tree_margin")
STYLE_ARMS = ("forced_binary", "unit_gate", "paired_gate")
BOUNDARY_DISTANCE = 8


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def style_references():
    train, evaluate = [], []

    def add(rows, ident, path, group, kind, known, expected=None, box=None):
        rows.append(dict(id=ident, image=str(path), sha256=sha(path), source_group=group,
                         reference_kind=kind, admitted=known, expected=expected, box=box))

    units = read(LAB / "cunfa_photometry_20261117/rejection_units.json")["units"]
    museum_units = {row["id"]: row for row in read(
        LAB / "cunfa_cross_source_20261117/new_test_units.json")}
    named_labels = {"pima": LABELS[0], "fupi": LABELS[1]}
    for row in units:
        if row["id"] in OUT_OF_SCOPE or row["id"].startswith("hkma_"):
            continue
        known = row["known"] == 1
        assert row["sha256"] == sha(row["image"])
        is_train = row["split"] == "train"
        source = museum_units.get(row["id"])
        if source:
            assert row["sha256"] == source["sha256"]
        expected = named_labels.get(source["method_caption"]) if source else None
        group = source["source_group"] if source else row["source_group"]
        add(train if is_train else evaluate, row["id"], row["image"], group,
            "named_teaching_unit" if is_train else ("archive_museum" if known else "other_cunfa"),
            known, expected)
    for index, row in enumerate(read(GROUPS)):
        if row["named_example"] not in LABELS:
            continue
        assert row["image_sha256"] == sha(row["image"])
        dev = row["name"].startswith("亚洲")
        group = "Lampo_Leong_all_examples" if dev else "HKMA_illustrator_unverified"
        rows = train if dev else evaluate
        add(rows, f"pair_{index}_ink", row["image"], group,
            "localized_group", True, row["named_example"], row["group_box"])
        add(rows, f"pair_{index}_control", row["image"], group,
            "contour_or_light_ground", False, box=row["comparison_box"])
    for row in read(FROZEN / "test_manifest.json")["records"]:
        if row["admission"] != "source_label_test":
            continue
        assert sha(row["image"]) == row["crop_sha256"]
        add(evaluate, row["id"], row["image"], row["artist_group"],
            "archive_museum", True, row["label"])
    images = {r["id"]: r for r in read(OBJECT_ARCHIVE / "dataset_manifest.json")["records"]}
    for ref in read(OBJECT_ARCHIVE / "local_references.json")["records"]:
        if ref["object_type"] != "tree":
            continue
        source = images[ref["image_id"]]
        assert sha(source["image"]) == source["image_sha256"]
        add(evaluate, ref["id"], source["image"], source["work_group"],
            "tree_core", False, box=ref["box"])
    assert not ({r["source_group"] for r in train} & {r["source_group"] for r in evaluate})
    assert not ({r["sha256"] for r in train} & {r["sha256"] for r in evaluate})
    return train, evaluate


def style_probe(torch, train, evaluate):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from torchvision.models import mobilenet_v3_small

    frozen = read(FROZEN / "freeze_manifest.json")
    for file in frozen["files"]:
        assert sha(file["path"]) == file["sha256"]
    spec = importlib.util.spec_from_file_location(
        "frozen_preprocessing", FROZEN / "frozen/encoder_and_preprocessing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    head = joblib.load(FROZEN / "frozen/local_only_head.joblib")
    assert set(head["trained_ids"]) == set(frozen["train_ids"])
    weights = next(row["path"] for row in frozen["files"] if row["path"].endswith(".pth"))
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    encoder = model.features.eval()

    views = []
    for row in train + evaluate:
        assert sha(row["image"]) == row["sha256"]
        with Image.open(row["image"]) as source:
            im = source.convert("RGB")
        if row["box"] is not None:
            im = im.crop(row["box"])
        views.append(im)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(views), 8):
            batch = np.stack([module.preprocess(im) for im in views[start:start + 8]])
            vectors.append(torch.nn.functional.normalize(
                encoder(torch.from_numpy(batch)).mean((2, 3)), dim=1).numpy())
    vectors = np.concatenate(vectors)
    indices = np.arange(len(train))
    unit_indices = indices[[r["reference_kind"] == "named_teaching_unit" for r in train]]
    scores = {}
    known = np.array([r["admitted"] for r in train])
    for arm, fit in (("unit_gate", unit_indices), ("paired_gate", indices)):
        gate = LogisticRegression(C=1, class_weight="balanced", max_iter=2000)
        gate.fit(vectors[fit], known[fit])
        scores[arm] = gate.predict_proba(vectors[len(train):])[:, 1]
        joblib.dump(dict(model=gate, threshold=0.5, train_ids=[train[i]["id"] for i in fit],
                         production_use=False), OUT / f"{arm}.joblib")

    classifications = head["model"].predict(vectors[len(train):])
    result = []
    for index, (row, prediction) in enumerate(zip(evaluate, classifications)):
        kind = LABELS[int(prediction)]
        decisions = {"forced_binary": kind}
        for arm, values in scores.items():
            decisions[arm] = kind if values[index] >= 0.5 else "未知"
        result.append(dict(**row, predictions=decisions,
                           known_scores={k: float(v[index]) for k, v in scores.items()}))

    metrics = {}
    for arm in STYLE_ARMS:
        target = [r for r in result if r["admitted"]]
        excluded = [r for r in result if not r["admitted"]]
        metrics[arm] = dict(
            accepted_target=[sum(r["predictions"][arm] != "未知" for r in target), len(target)],
            correct_target=[sum(r["predictions"][arm] == r["expected"] for r in target), len(target)],
            rejected_outside=[sum(r["predictions"][arm] == "未知" for r in excluded), len(excluded)],
            outside_by_kind={kind: [
                sum(r["reference_kind"] == kind and r["predictions"][arm] == "未知" for r in excluded),
                sum(r["reference_kind"] == kind for r in excluded)]
                for kind in sorted({r["reference_kind"] for r in excluded})})
        if arm != "forced_binary":
            positive = np.array([r["known_scores"][arm] for r in target])
            negative = np.array([r["known_scores"][arm] for r in excluded])
            win = positive[:, None] - negative
            mi_dot = next(r["known_scores"][arm] for r in excluded if r["id"] == "ntm_mi")
            metrics[arm]["ranking_auc"] = float(np.mean((win > 0) + 0.5 * (win == 0)))
            metrics[arm]["target_ranked_above_mi_dot"] = [
                int((positive > mi_dot).sum()), len(positive)]

    board = Image.new("RGB", (1280, 82 + 130 * ((len(result) + 3) // 4)), "#f4f1e8")
    draw = ImageDraw.Draw(board)
    label(draw, (12, 14), "皴法判定与拒识 · 历史归档复测", 25)
    label(draw, (12, 48), "每格依次显示参考类别、旧二分类、仅示例训练、加入同图对照训练", 16)
    for index, (row, im) in enumerate(zip(result, views[len(train):])):
        x, y = (index % 4) * 320 + 8, (index // 4) * 130 + 82
        board.paste(ImageOps.contain(im, (132, 103)), (x, y))
        label(draw, (x + 137, y + 5), row["id"], 14)
        label(draw, (x + 137, y + 28), f"参考  {row['expected'] or '范围外'}", 14)
        for j, arm in enumerate(STYLE_ARMS):
            label(draw, (x + 137, y + 51 + j * 20), row["predictions"][arm], 14)
    board.save(OUT / "cunfa_rejection.png")
    return dict(metrics=metrics, records=result, fit_count=len(train),
                unit_fit_count=len(unit_indices), test_already_opened=True)


def object_inputs():
    previous = [dict(id=r["id"], image=r["image"], mask=r["source_mask"],
                     work_group=r["work_group"], batch="development_regression")
                for r in read(PRIOR / "test_manifest.json")["records"]]
    transfer = [dict(id=r["id"], image=r["image"], mask=r["mask"],
                     work_group=r["work_group"], batch="other_archive")
                for r in read(OBJECT_ARCHIVE / "dataset_manifest.json")["records"]]
    for row in previous + transfer:
        row.update(image_sha256=sha(row["image"]), mask_sha256=sha(row["mask"]))
    return previous + transfer


def object_probe(torch, rows):
    import cv2
    from gongbi_training.scene_objects import SceneObjects
    from gongbi_training.source_scene_mapping import source_to_scene
    from guohua_objects_train_20261112 import pack

    checkpoint = SCENE / "mapping_fixed.pt"
    assert sha(checkpoint) == read(SCENE / "checkpoint_freeze.json")["mapping_fixed"]
    model = SceneObjects()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    model.eval()
    board = Image.new("RGB", (1230, 88 + len(rows) * 136), "#f4f1e8")
    draw = ImageDraw.Draw(board)
    label(draw, (10, 11), "树石错分对照 · 保护低分辨率树区", 25)
    label(draw, (10, 43), "橙色为树误作石，紫色为石误作树。", 16)
    for j, name in enumerate(("原图", "256 输入", "512 输入", "固定平均", "保护全部树区", "仅保护优势树区")):
        label(draw, (j * 202 + 10, 65), name, 14)
    totals = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))
    boundary_totals = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))
    results = []
    for i, row in enumerate(rows):
        assert row["image_sha256"] == sha(row["image"]) and row["mask_sha256"] == sha(row["mask"])
        with Image.open(row["image"]) as im:
            im = im.convert("RGB")
        with Image.open(row["mask"]) as source:
            truth = source_to_scene(np.asarray(source))
        valid = truth != 255
        boundary = np.zeros_like(valid)
        for own, opposite in ((1, 2), (2, 1)):
            if np.any(truth == opposite):
                distance = cv2.distanceTransform((truth != opposite).astype(np.uint8), cv2.DIST_L2, 5)
                boundary |= (truth == own) & (distance <= BOUNDARY_DISTANCE)
        probs = []
        for size in (256, 512):
            x, _, (left, top, width, height, ow, oh) = pack(np.asarray(im), size=size)
            with torch.inference_mode():
                p = model(torch.from_numpy(x)[None]).softmax(1)[0].numpy()
            probs.append(np.stack([cv2.resize(m[top:top+height, left:left+width], (ow, oh)) for m in p]))
        low, high = [p.argmax(0) for p in probs]
        mean = (probs[0] + probs[1]).argmax(0)
        tree_to_rock = (low == 2) & (mean == 1)
        lead = probs[0][2] - probs[0][1]
        protected = np.where(tree_to_rock, 2, mean)
        confident = np.where(tree_to_rock & (lead >= 0.15), 2, mean)
        masks = {}
        for j, (arm, pred) in enumerate(zip(OBJECT_ARMS, (low, high, mean, protected, confident)), 1):
            confusion = np.bincount(truth[valid].astype(int) * 4 + pred[valid], minlength=16).reshape(4, 4)
            local = np.bincount(
                truth[boundary].astype(int) * 4 + pred[boundary], minlength=16).reshape(4, 4)
            for key in (row["batch"], row["work_group"]):
                totals[key, arm] += confusion
                boundary_totals[key, arm] += local
            masks[arm] = dict(**summarize(confusion), confusion=confusion.tolist(), boundary=summarize(local))
            colored = np.asarray(ImageOps.grayscale(im).convert("RGB")).copy()
            colored[(truth == 2) & (pred == 1)] = (237, 121, 43)
            colored[(truth == 1) & (pred == 2)] = (159, 62, 191)
            board.paste(ImageOps.contain(Image.fromarray(colored), (190, 101)), (j * 202 + 10, 111 + i * 136))
        board.paste(ImageOps.contain(im, (190, 101)), (10, 111 + i * 136))
        label(draw, (10, 92 + i * 136), row["id"], 13)
        results.append(dict(**row, scores=masks))
    board.save(OUT / "tree_protected_fusion.png")
    keys = sorted({k for k, _ in totals})
    return dict(records=results,
                aggregates={key: {arm: summarize(totals[key, arm]) for arm in OBJECT_ARMS}
                            for key in keys},
                boundary_aggregates={key: {arm: summarize(boundary_totals[key, arm]) for arm in OBJECT_ARMS}
                                     for key in keys})


def main():
    configure()
    import torch

    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True, parents=True)
    baselines = verify_baselines()
    train, evaluate = style_references()
    images = object_inputs()
    write("protocol.json", dict(
        script_sha256=sha(__file__), reference_scope="previously opened archives",
        cunfa_train=train, cunfa_evaluate=evaluate, tree_rock=images,
        style_classifier=dict(kind="balanced logistic on frozen normalized embeddings", C=1, threshold=0.5),
        fusion=dict(base="equal-probability average", change="restore 256 tree where mean becomes rock",
                    variants=["protect all", "protect where 256 tree exceeds rock by at least 0.15"]),
        boundary=dict(coarse_reference_distance_pixels=BOUNDARY_DISTANCE,
                      post_hoc_diagnostic=True, uses_predictions_to_define_membership=False),
        intended_screen="Preserve target detections while reducing known negative admissions and bidirectional confusion",
        independent_validation=False, robot_connected=False))
    style = style_probe(torch, train, evaluate)
    print("CUNFA", style["metrics"], flush=True)
    objects = object_probe(torch, images)
    assert baselines == verify_baselines()
    write("results.json", dict(cunfa=style, objects=objects, painting_baselines=baselines,
                               adopted=False, independent_validation=False, robot_connected=False))
    print("OBJECTS", len(objects["records"]), "archived crops", flush=True)


if __name__ == "__main__":
    main()
