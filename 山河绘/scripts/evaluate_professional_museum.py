"""Apply the released B5 wrappers and compare frozen models on museum references."""

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_shanshuiseg_research import forward
from professional_clpd import MAP, ROOT, write
from train_professional_scene_scale import OUT, confusion, digest, read, score

PALETTE = np.array([[230, 225, 213], [173, 111, 67], [52, 121, 86], [82, 109, 151]], np.uint8)


def review_models():
    from gongbi_training.runtime import configure
    configure()
    import torch
    from torch.nn import functional as f
    from gongbi_training.scene_objects import SceneObjects
    from guohua_objects_train_20261112 import pack
    from recognition_groups_20261123 import SCENE

    torch.set_num_threads(4)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = ROOT / "source_audit/s1_weights.pth"
    expected = "3c8781f9a2d0b620c84a0d10455a1fbfe5f61d754497abecb044d8e9027dcd3d"
    matches = sha256()
    with weight_path.open("rb") as stream:
        for block in iter(lambda:stream.read(1 << 20), b""):
            matches.update(block)
    assert matches.hexdigest() == expected
    state = torch.load(weight_path, map_location=torch_device, weights_only=True)
    official_path = ROOT / "source_audit/s1_infer.py"
    expected_files = {
        "s1_infer.py":"1f85b5d6969e7998539d5812f2a8257ab27ac41368837ac25109aa84eb3dcd11",
        "s1_labels.json":"7af441712cd1ae8a7d0610834dc9a3ce2c4255290ada671383b089160d8cc38b",
    }
    for name, expected_hash in expected_files.items():
        assert digest(official_path.with_name(name)) == expected_hash
    spec = spec_from_file_location("shanshuiseg_v1_resize", official_path)
    official = module_from_spec(spec)
    spec.loader.exec_module(official)
    official.load_labels(ROOT / "source_audit/s1_labels.json")
    mapping = np.full(256, 255, np.uint8)
    for key, value in MAP.items():
        mapping[key] = value

    def logits(crop):
        normalized = (np.asarray(crop, np.float32) / 255 - official.MEAN) / official.STD
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)[None].to(torch_device)
        with torch.inference_mode():
            return forward(tensor, state, torch).float()

    def native_tiles(crops):
        return np.concatenate([f.interpolate(logits(crop), size=(512, 512), mode="bilinear",
                                             align_corners=False).cpu().numpy() for crop in crops])

    def predict_public(im):
        raw = official.resize_predict(im, native_tiles)
        with im.resize((512, 512), Image.Resampling.BILINEAR) as resized:
            old = f.interpolate(logits(resized), size=(im.height, im.width), mode="bilinear",
                                align_corners=False)[0].argmax(0).cpu().numpy() + 1
        return mapping[raw], float((raw==old).mean())

    baseline = SCENE / "mapping_fixed.pt"
    s2_weight = ROOT / "scene_pilot/s2_clpd_head.pt"
    assert digest(baseline) == read(SCENE / "checkpoint_freeze.json")["mapping_fixed"]
    assert digest(s2_weight) == read(ROOT / "scene_pilot/results.json")["checkpoint_sha256"]
    models = {}
    for method, path in (("S0", baseline), ("S2", s2_weight)):
        model = SceneObjects().to(torch_device).eval()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        models[method] = model

    def predict_local(model, im):
        x, _, (left, top, width, height, ow, oh) = pack(np.asarray(im), size=256)
        with torch.inference_mode():
            probs = model(torch.from_numpy(x)[None].to(torch_device))[0].softmax(0).cpu().numpy()
        return np.argmax([cv2.resize(p[top:top+height,left:left+width], (ow, oh))
                          for p in probs], axis=0).astype("uint8")

    fixed = read(OUT / "object_review.json")["rows"]
    intake = read(OUT / "museum_intake.json")["rows"]
    assert {r["id"] for r in fixed}.isdisjoint(r["id"] for r in intake)
    rows = []
    totals = {k: np.zeros((4, 4), dtype=np.int64) for k in ("S0", "S2", "B5")}
    for index, subset in enumerate((fixed, intake)):
        board = Image.new("RGB", (1070, 38+226*len(subset)), "#f6f2e8")
        pen = ImageDraw.Draw(board)
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
        pen.text((12, 10), "馆藏对照   原图 / S0 / S2 / B5 发布版 resize 流程",
                 font=font, fill="#243d31")
        for i, row in enumerate(subset):
            with Image.open(row["image"]) as orig:
                im = orig.convert("RGB")
            ow, oh = im.size
            im.thumbnail((1152, 1152))
            if digest(row["image"]) != row["image_sha256"]:
                raise ValueError(f"Museum original changed: {row['id']}")
            predictions = {method: predict_local(net, im) for method, net in models.items()}
            predictions["B5"], agreement = predict_public(im)
            metrics = {}
            if index == 0:
                mask = Image.new("L", im.size, 255)
                draw = ImageDraw.Draw(mask)
                for mark in row["marks"]:
                    if mark["shape"] == "core":
                        xy = [(round(x*im.width/ow), round(y*im.height/oh)) for x,y in mark["xy"]]
                        draw.polygon(xy, fill=mark["scene_label"])
                target = np.asarray(mask)
                for method, prediction in predictions.items():
                    cf = confusion(target, prediction, target != 255)
                    totals[method] += cf
                    metrics[method] = score(cf)
            rows.append(dict(id=row["id"], accession=row["accession"], official_resize_vs_old_port=agreement,
                             distinct_predictions={k: np.bincount(p.ravel(), minlength=4).tolist()
                                                   for k,p in predictions.items()}, reviewed_core_metrics=metrics))
            pen.text((10, 37+226*i), str(row["id"]) + "   " + row["accession"], font=font, fill="#243d31")
            for j, item in enumerate((im, *(Image.fromarray(PALETTE[p]) for p in predictions.values()))):
                item.thumbnail((250, 195))
                board.paste(item, (10+264*j, 59+226*i))
        board.save(OUT / ("reviewed_museum_models.png" if index==0 else "prospective_museum_models.png"))
    write(OUT / "museum_model_comparison.json", dict(rows=rows, reviewed_core_totals={k:score(v) for k,v in totals.items()},
          b5_wrapper_sha256=digest(official_path), b5_weights_sha256=expected,
          b5_protocol="official resize_predict with project native FP32 B5 backend, no tiling",
          mmcv_mmseg_forward_parity=False, whole_work_independent_acceptance=False,
          note="Only five assistant-reviewed core polygons have labels. Prospective works are qualitative."))


if __name__ == "__main__":
    review_models()
