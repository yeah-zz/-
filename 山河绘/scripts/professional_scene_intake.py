"""Acquire accessioned museum candidates, and screen for exact and crop overlaps."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from gongbi_training.runtime import LAB
from professional_clpd import ROOT, fingerprint, near_distance, request, write

OUT = ROOT.with_name("pro_scale_20261125")
MET_URL = "https://www.metmuseum.org/art/collection/search/"
MET_API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
OBJECTS = (49132, 51483, 51552, 45650, 737778, 41487, 737762, 36100,
           49185, 36163, 48953, 49179)


def read(path):
    return json.loads(path.read_text("utf-8"))


def museum_object(identifier):
    with request(MET_API + str(identifier)) as response:
        entry = json.load(response)
    if not entry.get("isPublicDomain") or not entry.get("primaryImage"):
        raise ValueError("No verified public-domain primary image")
    if entry["objectID"] != identifier or not entry.get("accessionNumber"):
        raise ValueError("Object identifier/accession mismatch")
    source = entry["primaryImage"]
    target = OUT / "museum" / f"met_{identifier}.jpg"
    if target.is_file():
        content = target.read_bytes()
    else:
        with request(source) as response:
            content = response.read()
    with Image.open(BytesIO(content)) as original:
        original.load()
        size = original.size
    if not target.is_file():
        target.write_bytes(content)
    return dict(id=identifier, work_group=f"met:{entry['accessionNumber']}",
                accession=entry["accessionNumber"], title=entry["title"],
                artist=entry["artistDisplayName"], medium=entry["medium"],
                source=MET_URL + str(identifier), image_url=source, image=str(target),
                image_sha256=sha256(content).hexdigest(), size=size,
                license="Met public domain", partition="candidate only, no evaluation labels")


def acquire():
    OUT.mkdir(exist_ok=True)
    (OUT / "museum").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {i: pool.submit(museum_object, i) for i in OBJECTS}
        rows, failures = [], []
        for i, future in futures.items():
            try:
                rows.append(future.result())
            except (OSError, ValueError, KeyError) as error:
                failures.append({"id": i, "reason": str(error)})
    write(OUT / "museum_intake.json", dict(rows=rows, failures=failures))
    print(f"Met review candidates: {len(rows)}; fetch failures: {len(failures)}", flush=True)


def signatures(path, mirror=False):
    with Image.open(path) as source:
        im = source.convert("RGB")
        original = np.asarray(im)
        pixel_digest = sha256(str(original.shape).encode() + original.tobytes()).hexdigest()
        hashes = fingerprint(im)
        im.thumbnail((960, 960))
        gray = np.asarray(im.convert("L"))
    detector = cv2.SIFT_create(nfeatures=384)
    def feature_map(array):
        points, vectors = detector.detectAndCompute(array, None)
        xy = np.asarray([k.pt for k in points], dtype="float32")
        return xy, vectors
    return pixel_digest, hashes, gray.shape, feature_map(gray), feature_map(cv2.flip(gray, 1)) if mirror else None


def crop_match(query, reference, source_shape, target_shape):
    left, features = query
    right, target_features = reference
    if features is None or target_features is None or min(len(features), len(target_features)) < 12:
        return None
    pairs = cv2.BFMatcher().knnMatch(features, target_features, k=2)
    accepted = sorted((a for a, b in pairs if a.distance < .72 * b.distance),
                      key=lambda match: match.distance)
    unique = {}
    for match in accepted:
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    if len(good) < 12:
        return None
    src = left[[m.queryIdx for m in good]]
    dst = right[[m.trainIdx for m in good]]
    affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3., maxIters=600)
    if affine is None or inliers is None or not .075 <= np.linalg.norm(affine[:, 0]) <= 13.3:
        return None
    keep = inliers.ravel().astype(bool)
    count = int(keep.sum())
    if count < 12 or count / len(good) < .40:
        return None
    span = np.ptp(src[keep], axis=0)
    target_span = np.ptp(dst[keep], axis=0)
    if (float(span.prod()) / np.prod(source_shape) < .012 or min(span) < 24
            or float(target_span.prod()) / np.prod(target_shape) < .003 or min(target_span) < 22):
        return None
    return count


def overlap(left, right):
    match = {}
    if left[0] == right[0]:
        match["decoded_rgb_sha256_match"] = True
    distance = near_distance(left[1], right[1])
    if distance <= 6:
        match["whole_view_distance"] = distance
    if match:
        return match
    for name, variant in (("original", left[3]), ("mirror", left[4])):
        if variant is not None:
            count = crop_match(variant, right[3], left[2], right[2])
            if count is not None:
                return dict(sift_affine_inliers=count, orientation=name)
    return None


def audit():
    cv2.setNumThreads(2)
    reference = {}
    for path in (OUT / "clpd_300/manifest.json", ROOT / "clpd/manifest.json"):
        for row in read(path)["rows"]:
            reference[row["image"]] = {"source": "clpd", "id": row["id"]}
    for info in read(ROOT / "scene_pilot/overlap_audit.json")["rows"]:
        for hit in info["matches"]:
            reference[hit["image"]] = {"source": "historical confirmed comparison", "id": Path(hit["image"]).stem}
    for folder in ("guohua_objects_pilot_20261112","guohua_objects_v2_20261113",
                   "guohua_objects_v3_20261116","guohua_objects_v4_sparse_20261116",
                   "guohua_scene4_mapping_fix_20261116","object_prospective_20261118"):
        for manifest in [*(LAB/folder).glob("*manifest.json"), *(LAB/folder).glob("*split*.json")]:
            doc = read(manifest)
            if not isinstance(doc, dict):
                continue
            for row in doc.get("records", []):
                path = row.get("image") or row.get("image_path")
                if isinstance(path, str) and Path(path).is_file():
                    reference.setdefault(str(Path(path)), {"source":"historical project manifest",
                                                          "id":Path(path).stem})
    for path in (ROOT / "source_audit").glob("met_*.jpg"):
        reference[str(path)] = {"source": "prior museum/manual", "id": path.stem}
    coco = ROOT / "source_audit/coco_image_0000.png"
    reference[str(coco)] = {"source": "instance segmentation sample", "id": coco.name}
    candidates = read(OUT / "museum_intake.json")["rows"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {path: pool.submit(signatures, path) for path in reference}
        inventory = {p: f.result() for p, f in futures.items()}
    report = []
    for row in candidates:
        signature = signatures(row["image"], mirror=True)
        hits = []
        for path, known in inventory.items():
            evidence = overlap(signature, known)
            if evidence:
                hits.append({**reference[path], **evidence})
        report.append(dict(id=row["id"], title=row["title"], work_group=row["work_group"],
                           accession=row["accession"], matches=hits,
                           status="overlap review" if hits else "unlabelled prospective candidate"))
    # Anchor the known positive control so an audit without overlap is interpretable.
    probe = OUT / "crop_method_control.png"
    with Image.open(candidates[0]["image"]) as image:
        w,h = image.size
        transformed = image.crop((w//7, h//5, 6*w//7, 4*h//5)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        transformed.thumbnail((680,680))
        ImageEnhance.Contrast(transformed).enhance(1.13).save(probe)
    crop_control = overlap(signatures(probe, mirror=True), signatures(candidates[0]["image"]))
    assert crop_control and "sift_affine_inliers" in crop_control, crop_control
    coco_hits = []
    for path, known in inventory.items():
        if path != str(coco):
            evidence = overlap(known, inventory[str(coco)])
            if evidence:
                coco_hits.append({**reference[path], **evidence})
    write(OUT / "cross_source_audit.json", dict(rows=report, coco_reference_matches=coco_hits, crop_method_positive_control=crop_control,
          reference_by_source={src: sum(v["source"] == src for v in reference.values())
                               for src in sorted({v["source"] for v in reference.values()})},
          methods=["decoded RGB SHA-256", "mirrored perceptual hash", "SIFT + affine RANSAC on reachable images"],
          scope="Nonoverlapping views and unaudited training images may still share a work.",
          public_model_training_inventory="unavailable", independent_test_frozen=False))
    print(f"Checked {len(report)} museum candidates against {len(reference)} reachable references.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["acquire", "audit"])
    options = parser.parse_args()
    {"acquire": acquire, "audit": audit}[options.action]()
