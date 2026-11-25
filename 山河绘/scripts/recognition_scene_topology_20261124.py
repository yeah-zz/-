# -*- coding: utf-8 -*-
"""Compare semantic-only, path and junction ownership on the same source ink."""

import json

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure


OUT = LAB / "scene_topology_20261124"
PREV = LAB / "trunk_rock_context_holdout_20261124"
MODES = ("semantic", "path", "junction")
COLORS = {1: "#e19147", 2: "#00bfa9", 255: "#ad7ee2"}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def score(truth, skeleton, maps, cv2):
    distance = {
        kind: cv2.distanceTransform((truth != kind).astype(np.uint8), cv2.DIST_L2, 5)
        if np.any(truth == kind) else np.full(truth.shape, np.inf)
        for kind in (1, 2)
    }
    result = {}
    for mode, prediction in maps.items():
        result[mode] = {}
        for kind, other in ((1, 2), (2, 1)):
            selected = (truth == kind) & skeleton
            for region, mask in (("ink", selected), ("boundary", selected & (distance[other] <= 8))):
                result[mode][f"{region}_{kind}"] = [
                    int(np.count_nonzero(mask & (prediction == value))) for value in (1, 2, 255)
                ]
    return result


def point_hits(maps, skeleton, points, radius=3):
    result = {name: [] for name in ("ink", *MODES)}
    h, w = skeleton.shape
    for y, x in points:
        yy, xx = np.ogrid[max(0,y-radius):min(h,y+radius+1), max(0,x-radius):min(w,x+radius+1)]
        near = (yy-y)**2 + (xx-x)**2 <= radius**2
        result["ink"].append(bool(np.any(near & skeleton[yy, xx])))
        for mode, pred in maps.items():
            result[mode].append(bool(np.any(near & (pred[yy, xx] == 2))))
    return result


def image_panel(examples, label):
    names = ("原图", "仅语义分割", "沿墨迹融合", "加入交汇接续")
    width = 1020
    sheet = Image.new("RGB", (width, 86 + 238*len(examples)), "#f4f1e8")
    draw = ImageDraw.Draw(sheet)
    label(draw, (12, 10), "语义分割与墨迹拓扑融合", 25)
    label(draw, (12, 46), "青色为树木，橙色为岩面，紫色为待核墨迹，红点为树身参考。", 16)
    for index, (title, image, crop, points, skeleton, maps) in enumerate(examples):
        y = 86 + 238*index
        label(draw, (12, y), title, 17)
        x0, y0, x1, y1 = crop
        raw = image.crop(crop)
        scale = min(4.0, 183/raw.height, 226/raw.width)
        size = (round(raw.width*scale), round(raw.height*scale))
        sy, sx = np.where(skeleton[y0:y1, x0:x1])
        for column, mode in enumerate((None, *MODES)):
            x = 12 + 252*column
            tile = raw.resize(size, Image.Resampling.NEAREST)
            pen = ImageDraw.Draw(tile)
            if mode is not None:
                own = maps[mode][sy+y0, sx+x0]
                for py, px, kind in zip(sy, sx, own):
                    tx, ty = (px+.5)*scale, (py+.5)*scale
                    pen.ellipse((tx-1, ty-1, tx+1, ty+1), fill=COLORS[kind])
            for py, px in points:
                tx, ty = (px-x0+.5)*scale, (py-y0+.5)*scale
                pen.ellipse((tx-3, ty-3, tx+3, ty+3), outline="#fa4339", width=1)
            sheet.paste(tile, (x, y+26))
            label(draw, (x, y+213), names[column], 15)
    return sheet


def main():
    configure()
    import cv2
    import torch
    from skimage.morphology import skeletonize
    from gongbi_training import scene_topology
    from gongbi_training.scene_objects import SceneObjects
    from gongbi_training.source_scene_mapping import source_to_scene
    from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines
    from recognition_trunk_rock_context_20261124 import scene_probabilities

    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    baselines = verify_baselines()
    checkpoint = SCENE / "mapping_fixed.pt"
    expected = read(PREV / "protocol.json")["checkpoint_sha256"]
    assert sha(checkpoint) == expected
    reference_files = (PREV/"reference_freeze.json", PREV/"scan_reference_freeze.json")
    cases = read(reference_files[0])["cases"]
    sources = read(reference_files[1])["windows"]
    prior = read(PREV / "scan_results.json")["records"]
    focus = {(r["id"], tuple(r["crop"])) for r in prior if r["generated_candidate"]}
    write(OUT/"protocol.json", {
        "checkpoint_sha256": expected, "script_sha256": sha(__file__),
        "fusion_sha256": sha(scene_topology.__file__), "reference_sha256": [sha(p) for p in reference_files],
        "same_ink_support_for_all_modes": True, "reference_used_in_prediction": False,
        "development_regression": True
    })
    net = SceneObjects()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    net.eval()
    examples, point_results, archive = [], [], []

    def infer(path):
        with Image.open(path) as source:
            image = source.convert("RGB")
        rgb = np.asarray(image)
        probabilities = scene_probabilities(rgb, net, torch, cv2)
        mean = (probabilities[256] + probabilities[512]) / 2
        skeleton, maps, segments = scene_topology.infer_ownership(rgb, mean, cv2, skeletonize)
        return image, skeleton, maps, segments

    def graph_file(name, segments, maps):
        def run_spans(points, prediction):
            values = prediction[tuple(np.asarray(points).T)]
            breaks = np.flatnonzero(np.diff(values)) + 1
            return [[int(a), int(b), int(values[a])] for a,b in
                    zip(np.r_[0,breaks], np.r_[breaks,len(values)])]

        for segment in segments:
            segment["ownership_spans"] = run_spans(segment["yx"], maps["junction"])
        write(OUT/f"{name}.json", {"coordinate_order": "yx", "role_codes": {"rock": 1, "tree": 2, "unknown": 255},
                                  "segments": segments})

    by_image = {}
    for case in cases:
        if case["points"]:
            by_image.setdefault(case["image"], []).append(case)
    for image_path, references in by_image.items():
        assert sha(image_path) == references[0]["image_sha256"]
        image, skeleton, maps, segments = infer(image_path)
        graph_file(references[0]["key"], segments, maps)
        for case in references:
            hits = point_hits(maps, skeleton, case["points"])
            point_results.append({"title": case["title"], "points_yx": case["points"], "hits": hits})
            examples.append((case["title"], image, case["crop"], case["points"], skeleton, maps))
    for row in sources:
        assert sha(row["image"]) == row["sha256"] and sha(row["mask"]) == row["mask_sha256"]
        image, skeleton, maps, segments = infer(row["image"])
        with Image.open(row["mask"]) as source:
            truth = source_to_scene(np.asarray(source))
        windows = []
        for crop in row["boxes"]:
            x0,y0,x1,y1 = crop
            if (row["id"], tuple(crop)) in focus:
                examples.append(("树根与岩面交界", image, crop, [], skeleton, maps))
                graph_file("root_boundary", segments, maps)
            region = np.s_[y0:y1, x0:x1]
            windows.append({"crop": crop, "counts": {
                name: [int(np.count_nonzero(skeleton[region] & (pred[region] == k))) for k in (1,2,255)]
                for name,pred in maps.items()
            }})
        archive.append({"id": row["id"], "group": row["work_group"], "windows": windows,
                        "counts": score(truth, skeleton, maps, cv2), "ink_pixels": int(skeleton.sum()),
                        "chains": len(segments), "paired_junctions": sum(
                            link is not None for segment in segments for link in segment["links"])//2})
        print(row["id"], archive[-1]["ink_pixels"], archive[-1]["paired_junctions"], flush=True)
    assert verify_baselines() == baselines
    image_panel(examples,label).save(OUT/"topology_comparison.png")
    write(OUT/"results.json", {"point_checks": point_results, "archive_checks": archive,
                               "painting_baselines": baselines, "adopted": False, "robot_connected": False})


if __name__ == "__main__":
    main()
