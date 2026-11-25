"""Build compact training plots, equal-stroke replays and a portable experiment manifest."""

from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from master_studio.digital_strokes import DigitalStroke, render, replay
from professional_clpd import ROOT, write
from professional_scene_drawing import render_pair
from train_professional_scene_scale import OUT, digest, load_rows, read

ARMS = ("pilot256", "expanded256", "expanded512", "unfreeze256")
INK = ("#596d81", "#bb703b", "#36856b", "#805475")


def chart(results, path):
    board = Image.new("RGB", (930, 348), "#f8f4ea")
    pen = ImageDraw.Draw(board)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    pen.text((12, 9), "固定开发集   CLPD 训练曲线", fill="#273b34", font=font)
    for j, title in enumerate(("训练交叉熵", "山石 IoU", "树木 IoU")):
        x0, y0 = 38 + 305*j, 268
        width, height = 246, 200
        scales = (0., 1.) if j==0 else (.5, .85)
        pen.text((x0, 40), title, fill="#273b34", font=font)
        pen.rectangle((x0, y0-height, x0+width, y0), outline="#aaa99d")
        for v in np.linspace(*scales, 5):
            y = y0 - (v-scales[0])/(scales[1]-scales[0])*height
            pen.line((x0, y, x0+width, y), fill="#dbd7c9")
            pen.text((x0-35, y-9), f"{v:.2f}", fill="#68695d", font=font)
        for arm, color in zip(ARMS, INK):
            epochs = results[arm]["curve"]
            values = [r["loss"] if j==0 else r["dev_iou"]["mountain_rock" if j==1 else "trees"]
                      for r in epochs]
            coords = [(x0+i*width/max(1,len(values)-1),
                       y0-(v-scales[0])/(scales[1]-scales[0])*height) for i,v in enumerate(values)]
            pen.line(coords, fill=color, width=3)
            for i,(x,y) in enumerate(coords):
                pen.ellipse((x-2,y-2,x+2,y+2), fill=color)
                if arm==ARMS[0]:
                    pen.text((x-5,y0+4),str(i+1),fill="#68695d",font=font)
    for i, (arm, color) in enumerate(zip(ARMS, INK)):
        x = 28+223*i
        pen.line((x, 318, x+18, 318), fill=color, width=4)
        pen.text((x+23, 308), arm, fill="#273b34", font=font)
    board.save(path)


def historical_regression():
    from gongbi_training.runtime import LAB
    folder = LAB / "digital_specialty_round_20261124"
    prior = LAB / "crosswork_drawing_20261124"
    pilot = ROOT / "scene_pilot"
    groups = (sorted(p for p in folder.glob("*.json") if p.with_suffix(".png").is_file()),
              sorted(p for p in prior.glob("*.json") if p.with_suffix(".png").is_file()),
              [pilot / "s0_drawing.json", pilot / "s2_drawing.json"])
    assert [len(group) for group in groups] == [6, 14, 2]
    checked = []
    for sequence in groups:
        for p in sequence:
            canvas, _ = replay(p)
            with Image.open(p.with_suffix(".png")) as original:
                assert canvas.tobytes() == original.convert(canvas.mode).tobytes(), p
            checked.append({"archive": p.parent.name, "json": p.name, "sha256":digest(p)})
    return checked


def render_tree_reference(output):
    reviewed = read(OUT / "object_review.json")["rows"][0]
    tree_marks = [r for r in reviewed["marks"] if r["shape"] in {"centerline", "branch"}]
    origin = np.array(reviewed["crop"][:2])
    strokes = []
    ids = {r["name"]: f"cp28-{i}" for i,r in enumerate(tree_marks)}
    for i, mark in enumerate(tree_marks):
        xy = ((np.array(mark["xy"])-origin)*.43+30).tolist()
        widths = np.linspace(10.5 if i==0 else 3.2, 2.8 if i==0 else .8, len(xy)).tolist()
        parents = () if mark["parent"] is None else (ids[mark["parent"]],)
        strokes.append(DigitalStroke(ids[mark["name"]], "cp28-tree", mark["name"], "visible structure study",
            xy, widths, (53,47,39), .83, .22, .29, i, parents,
            {"geometry":"assistant-traced CP28 structure; branch ownership inferred",
             "width_ink":"fixed digital rules", "source":reviewed["source"]}))
    size = (330, 450)
    board = Image.new("RGB", (690, 493), "#f4eee1")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    labels = {"legacy": "旧数字毛笔", "ribbon": "带状数字毛笔"}
    for i, mode in enumerate(labels):
        dst = output / f"cp28_{mode}"
        path = dst.with_suffix(".json")
        write(path, dict(canvas_wh=size, coordinate_system="xy pixels", seed=24, renderer=mode, strokes=[s.record() for s in strokes]))
        img, frames = render(strokes, size, frames=True, renderer=mode)
        assert replay(path)[0].tobytes() == img.tobytes()
        img.save(dst.with_suffix(".png"))
        frames[0].save(dst.with_suffix(".gif"), save_all=True, append_images=frames[1:],
                       duration=240, loop=0)
        board.paste(img, (10+340*i, 35))
        ImageDraw.Draw(board).text((10+340*i, 9), f"同 {len(strokes)} 笔  {labels[mode]}",
                                   fill="#414d42", font=font)
    board.save(output / "cp28_brush_comparison.png")
    return len(strokes)


def drawing(results, plan, folder, output):
    from gongbi_training.runtime import configure
    configure()
    import torch
    from gongbi_training.scene_objects import SceneObjects
    from guohua_objects_train_20261112 import pack
    from recognition_groups_20261123 import SCENE

    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    winner = max(ARMS, key=lambda arm: sum(v["iou"] for v in results[arm]["development"]["metrics"].values()))
    row_result = results[winner]
    candidate_path = folder / row_result["checkpoint"]
    assert digest(candidate_path) == row_result["checkpoint_sha256"]
    models = {}
    for name, path, size in (("s0", SCENE / "mapping_fixed.pt", 256),
                             (winner, candidate_path, row_result["config"]["input_size"])):
        model = SceneObjects().to(device).eval()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        models[name] = model,size

    def predict(model, rgb, size):
        x, _, (l,t,w,h,ow,oh) = pack(rgb, size=size)
        with torch.inference_mode():
            probabilities = model(torch.from_numpy(x)[None].to(device))[0].softmax(0).cpu().numpy()
        return np.stack([cv2.resize(a[t:t+h,l:l+w],(ow,oh)) for a in probabilities]).argmax(0)

    rows = sorted((r for r in plan["rows"] if r["split"] == "development"),key=lambda r:r["id"])
    for row,rgb,_ in load_rows(rows):
        old,new = [predict(model,rgb,size) for model,size in models.values()]
        protocol = render_pair(output, row, rgb, old, new, candidate_name=winner)
        if protocol is not None:
            return winner, protocol
    raise RuntimeError("No development picture has paired usable silhouettes")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=OUT / "comparison")
    parser.add_argument("--delivery", type=Path, default=OUT / "delivery")
    parser.add_argument("--publish-manifest", type=Path)
    args = parser.parse_args()
    plan = read(OUT/"fixed_plan.json")
    comparison = args.comparison
    report = read(comparison/"results.json")
    results = report["results"]
    assert all(name in results for name in ARMS)
    delivery = args.delivery
    delivery.mkdir(parents=True, exist_ok=True)
    chart(results,delivery/"training_curves.png")
    checked = historical_regression()
    winner, protocol = drawing(results,plan,comparison,delivery)
    tree_count = render_tree_reference(delivery)
    write(delivery/"replay_checks.json",dict(historical=checked, historical_n=len(checked),
          new_predicted_silhouette=protocol, new_brush_comparison_strokes=tree_count,
          all_new_replay_pixel_match=True, selected_by_development=winner))
    if args.publish_manifest:
        fields = ("id","image_url","mask_url","image_sha256","mask_sha256","split","group","work_identity")
        rows = [{k:r[k] for k in fields} for r in plan["rows"]]
        intake = read(OUT/"museum_intake.json")
        museums = [{k:v for k,v in row.items() if k!="image"} for row in intake["rows"]]
        manifest = dict(clpd_rows=rows,clpd_counts=dict(Counter(r["split"] for r in rows)),
              work_identity=plan["work_identity"],fixed_plan_sha256=digest(OUT/"fixed_plan.json"),
              museums=museums, model_results=results,baseline_hashes=report["baselines"],
              reviewed_museum=[{k:v for k,v in row.items() if k!="image"}
                               for row in read(OUT/"object_review.json")["rows"]],
              cross_source_audit=read(OUT/"cross_source_audit.json"),
              museum_model_comparison=read(OUT/"museum_model_comparison.json"),
              replay_summary=read(delivery/"replay_checks.json"),independent_artwork_acceptance=False)
        with args.publish_manifest.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
