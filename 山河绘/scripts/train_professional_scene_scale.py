"""Work-group-controlled CLPD size, resolution and partial-unfreeze comparisons."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
from PIL import Image
from professional_clpd import MAP, ROOT, near_distance, write

OUT = ROOT.with_name("pro_scale_20261125")
SEED = 20261125
CLASSES = {"mountain_rock": 1, "trees": 2}


def read(path):
    return json.loads(path.read_text("utf-8"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, destination):
    pilot = read(ROOT / "clpd/manifest.json")["rows"]
    expanded = read(source)["rows"]
    fixed = [row for row in pilot if row["split"] != "train"]
    train = [row for row in pilot if row["split"] == "train"]
    test_groups = {row["work_key"] for row in fixed}
    seen = {row["image_sha256"] for row in pilot}
    expansion = defaultdict(list)
    excluded = []
    for row in expanded:
        cohort = fixed + train
        digest_key = row["image_sha256"]
        if digest_key in seen:
            continue
        if row["work_key"] in test_groups or any(
            near_distance(row["phash"], previous["phash"]) <= 6 for previous in fixed
        ):
            excluded.append({"id": row["id"], "reason": "fixed cohort title or perceptual match"})
            continue
        if any(near_distance(row["phash"], previous["phash"]) <= 6 for previous in cohort):
            excluded.append({"id": row["id"], "reason": "near existing training image"})
            continue
        expansion[row["work_key"]].append({**row, "group": row["work_key"], "split": "train"})
        seen.add(digest_key)
    train += [row for group in sorted(expansion) for row in expansion[group]]
    assert not {r["group"] for r in train} & {r["group"] for r in fixed}
    for row in train + fixed:
        for kind in ("image", "mask"):
            if digest(row[kind]) != row[f"{kind}_sha256"]:
                raise ValueError(f"{kind} integrity mismatch: {row['id']}")
    plan = dict(
        rows=train + fixed, pilot_train_size=sum(r["split"] == "train" for r in pilot),
        excluded=excluded, new_training_images=sum(map(len, expansion.values())),
        work_identity="filename title proxy; partial/alternate-view overlap unverified",
        original_manifest_sha256=digest(ROOT / "clpd/manifest.json"),
        expanded_manifest_sha256=digest(source), sealed_independent_test=False,
    )
    write(destination, plan)
    print("Fixed cohorts and expanded training:", Counter(r["split"] for r in plan["rows"]), flush=True)


def score(matrix):
    result = {}
    for name, label in CLASSES.items():
        tp = int(matrix[label, label])
        pred = int(matrix[:, label].sum())
        actual = int(matrix[label].sum())

        def ratio(a, b):
            return a / b if b else None

        other = 3 - label
        result[name] = dict(iou=ratio(tp, actual + pred - tp), precision=ratio(tp, pred),
                            recall=ratio(tp, actual), f1=ratio(2 * tp, actual + pred),
                            wrong_tree_rock=ratio(int(matrix[label, other]), actual),
                            support=actual)
    return result


def confusion(target, pred, valid):
    return np.bincount(4 * target[valid].astype(int) + pred[valid], minlength=16).reshape(4, 4)


def interface_band(target):
    rock, tree = target == 1, target == 2
    if not rock.any() or not tree.any():
        return np.zeros_like(rock)
    return (rock & (cv2.distanceTransform((~tree).astype("uint8"), cv2.DIST_L2, 5) <= 8)
            | tree & (cv2.distanceTransform((~rock).astype("uint8"), cv2.DIST_L2, 5) <= 8))


def load_rows(rows):
    lookup = np.full(256, 255, np.uint8)
    for key, label in MAP.items():
        lookup[key] = label
    for row in rows:
        with Image.open(row["image"]) as im, Image.open(row["mask"]) as target:
            yield row, np.asarray(im.convert("RGB")), lookup[np.asarray(target)]


def run(args):
    from gongbi_training.runtime import configure
    configure()
    import torch
    from gongbi_training.scene_objects import SceneObjects
    from guohua_objects_train_20261112 import pack, WEIGHTS
    from recognition_groups_20261123 import SCENE, verify_baselines

    plan = read(args.plan)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    devices = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    baseline_hashes = verify_baselines()
    checkpoint = SCENE / "mapping_fixed.pt"
    frozen_sha = read(SCENE / "checkpoint_freeze.json")["mapping_fixed"]
    assert digest(checkpoint) == frozen_sha

    training = [r for r in plan["rows"] if r["split"] == "train"]
    cohorts = {name: list(load_rows(r for r in plan["rows"] if r["split"] == name))
               for name in ("development", "evaluation")}
    boundaries = {r["image_sha256"]: interface_band(t)
                  for group in cohorts.values() for r, _, t in group}

    def predict(net, rgb, size):
        x, _, (l, t, w, h, ow, oh) = pack(rgb, size=size)
        with torch.inference_mode():
            values = net(torch.from_numpy(x)[None].to(devices))[0].softmax(0).cpu().numpy()
        return np.argmax([cv2.resize(a[t:t+h, l:l+w], (ow, oh)) for a in values], axis=0).astype("uint8")

    def observe(net, group, size):
        total, edge = np.zeros((2, 4, 4), dtype=np.int64)
        per_group = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))
        for row, rgb, target in group:
            pred = predict(net, rgb, size)
            current = confusion(target, pred, target != 255)
            total += current
            edge += confusion(target, pred, boundaries[row["image_sha256"]])
            per_group[row["group"]] += current
        return dict(metrics=score(total), interface=score(edge), confusion=total.tolist(),
                    per_work={name: score(cf) for name, cf in per_group.items()})

    s0 = SceneObjects().to(devices).eval()
    s0.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    s2 = SceneObjects().to(devices).eval()
    s2_file = ROOT / "scene_pilot/s2_clpd_head.pt"
    assert digest(s2_file) == read(ROOT / "scene_pilot/results.json")["checkpoint_sha256"]
    s2.load_state_dict(torch.load(s2_file, map_location="cpu", weights_only=True)["model"])
    results = {"S0": {k: observe(s0, v, 256) for k, v in cohorts.items()},
               "S2": {k: observe(s2, v, 256) for k, v in cohorts.items()}}
    # Use one class weighting policy from the 62-image control for all arms.
    reference_masks = [pack(rgb, target, size=256)[1]
                       for _, rgb, target in load_rows(training[:plan["pilot_train_size"]])]
    count = sum(np.bincount(m[m != 255], minlength=4) for m in reference_masks)
    class_weights = torch.tensor(1 / np.sqrt(count / count.sum()), device=devices, dtype=torch.float32)
    arms = [("pilot256", plan["pilot_train_size"], 256, 0),
            ("expanded256", len(training), 256, 0),
            ("expanded512", len(training), 512, 0),
            ("unfreeze256", len(training), 256, 3)]
    for name, number, size, unfreeze in arms:
        if args.arm and name not in args.arm:
            continue
        start = time.perf_counter()
        torch.manual_seed(SEED)
        if devices == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = SceneObjects().to(devices)
        for layer in list(model.head) + list(model.lateral):
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for block in list(model.encoder)[len(model.encoder) - unfreeze:]:
            block.requires_grad_(True)
        optimizer = torch.optim.AdamW([
            {"params": [p for module in (model.lateral, model.head) for p in module.parameters()], "lr": .002},
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": .0001},
        ])
        tensors = [pack(rgb, target, size=size)[:2] for _, rgb, target in load_rows(training[:number])]
        rng, augmentation = random.Random(SEED), random.Random(SEED + 1)
        order = []
        config = dict(train_images=number, work_proxy_count=len({r["group"] for r in training[:number]}),
                      input_size=size, encoder="ImageNet MobileNetV3-Small", unfrozen_tail=unfreeze,
                      batch=6, microbatch=3 if size == 512 else 6, epochs=args.epochs, steps=args.steps,
                      optimizer="AdamW", head_lr=.002, backbone_lr=.0001 if unfreeze else None,
                      seed=SEED, loss="fixed weighted cross entropy", device=devices,
                      encoder_source_sha256=digest(WEIGHTS), s0_checkpoint_sha256=frozen_sha,
                      s2_checkpoint_sha256=digest(s2_file), plan_sha256=digest(args.plan),
                      torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                      checkpoint_selection="fixed end of budget, equal optimizer steps per arm")
        curve = []
        for epoch in range(args.epochs):
            losses = []
            model.train()
            for _ in range(args.steps):
                while len(order) < 6:
                    permutation = list(range(number))
                    rng.shuffle(permutation)
                    order.extend(permutation)
                ids, order = order[:6], order[6:]
                samples = [(x[:, :, ::-1], y[:, ::-1]) if augmentation.random() < .5 else (x, y)
                           for x, y in (tensors[i] for i in ids)]
                x, y = (torch.from_numpy(np.stack(seq)).to(devices) for seq in zip(*samples))
                y = y.long()
                divisor = (class_weights[y.clamp_max(3)] * y.ne(255)).sum().clamp_min(1)
                optimizer.zero_grad(set_to_none=True)
                loss_value = 0.
                for xs, ys in zip(x.split(config["microbatch"]), y.split(config["microbatch"])):
                    loss = torch.nn.functional.cross_entropy(
                        model(xs), ys, weight=class_weights, ignore_index=255, reduction="sum") / divisor
                    loss.backward()
                    loss_value += float(loss.detach())
                optimizer.step()
                losses.append(loss_value)
            model.eval()
            development = observe(model, cohorts["development"], size)
            entry = dict(epoch=epoch + 1, loss=sum(losses) / len(losses),
                         dev_iou={k: v["iou"] for k, v in development["metrics"].items()})
            curve.append(entry)
            write(root / f"{name}_progress.json", dict(config=config, curve=curve))
            print(name, entry, flush=True)
        assert not unfreeze or all(p.grad is not None for p in model.encoder.parameters() if p.requires_grad)
        path = root / f"{name}.pt"
        torch.save(dict(model=model.state_dict(), config=config), path)
        results[name] = dict(config=config, curve=curve, checkpoint=path.name,
                             checkpoint_sha256=digest(path), runtime_s=time.perf_counter()-start,
                             peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if devices=="cuda" else None,
                             **{k: observe(model, v, size) for k, v in cohorts.items()})
        write(root / "results.json", dict(results=results, baselines=baseline_hashes,
                                         independent_artwork_acceptance=False))
        del tensors, optimizer, model
    assert baseline_hashes == verify_baselines() and frozen_sha == digest(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=OUT / "fixed_plan.json")
    parser.add_argument("--prepare", type=Path, help="Expanded CLPD manifest to prepare before training")
    parser.add_argument("--output", type=Path, default=OUT / "comparison")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=26, help="Optimizer steps per epoch")
    parser.add_argument("--arm", action="append", choices=["pilot256","expanded256","expanded512","unfreeze256"])
    args = parser.parse_args()
    if min(args.epochs, args.steps) < 1:
        parser.error("Training budget must be positive")
    if args.prepare:
        prepare(args.prepare, args.plan)
    else:
        if (args.output / "results.json").exists():
            parser.error("Choose a new output folder to preserve previous results")
        run(args)
