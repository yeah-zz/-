"""Visible ink and topology references traced on Met original images before inference."""

from hashlib import sha256
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from professional_clpd import ROOT, write

OUT = ROOT.with_name("pro_scale_20261125")
FIGURES = {
    63225: {
        "crop": [475, 385, 1055, 1260],
        "accession": "CP28",
        "marks": [
            ("core", "main trunk", [[492, 787], [518, 804], [517, 831], [497, 848],
                                   [499, 810], [478, 793]], 2, None),
            ("core", "main trunk", [[350, 645], [372, 645], [359, 681], [348, 708],
                                   [335, 704], [347, 671]], 2, None),
            ("centerline", "main trunk", [[500, 844], [502, 810], [462, 782], [410, 768],
                                           [371, 749], [346, 716], [352, 675], [359, 641],
                                           [338, 593], [310, 551], [290, 517],
                                           [296, 476], [291, 424]], None, None),
            ("branch", "lower left branch", [[379, 751], [343, 747], [307, 744], [272, 735]], None, "main trunk"),
            ("branch", "right branch", [[361, 638], [409, 638], [458, 625], [498, 601]], None, "main trunk"),
            ("branch", "right secondary", [[416, 636], [426, 610], [443, 584]], None, "right branch"),
            ("branch", "middle left branch", [[291, 522], [259, 526], [232, 518], [185, 508]], None, "main trunk"),
            ("unknown", "branch overlap", [[191, 479], [244, 479], [244, 541], [191, 541]], None, None),
            ("unknown", "base/root behind ink and crop", [[465, 840], [548, 840], [548, 872], [465, 872]], None, None),
        ],
    },
    41462: {
        "crop": [105, 1370, 870, 1930],
        "accession": "1985.120.1",
        "marks": [
            ("core", "trunk interior", [[153, 112], [199, 108], [178, 160], [150, 220],
                                         [129, 234], [136, 179]], 2, None),
            ("core", "upper exposed rock", [[508, 112], [559, 109], [592, 130],
                                           [617, 157], [577, 172], [523, 166]], 1, None),
            ("core", "exposed rock facing right", [[603, 224], [657, 198], [662, 231],
                                                   [646, 264], [609, 268]], 1, None),
            ("unknown", "root grass and pigment overlap", [[20, 359], [260, 350], [272, 499],
                                                           [30, 514]], None, None),
            ("unknown", "dark contact between the tree and rock", [[260, 263], [367, 212], [397, 355],
                                                                  [311, 398]], None, None),
        ],
    },
}


def main():
    queue = json.loads((ROOT / "annotation_queue.json").read_text("utf-8"))["tasks"]
    sources = {r["id"]: r for r in queue}
    docs = []
    board = Image.new("RGB", (1380, 930), "#f2eee4")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    for column, (identifier, record) in enumerate(FIGURES.items()):
        x, y, x1, y1 = record["crop"]
        source = sources[identifier]
        path = Path(source["file"])
        assert sha256(path.read_bytes()).hexdigest() == source["sha256"]
        with Image.open(path) as im:
            orig = im.convert("RGB").crop((x, y, x1, y1))
        overlay = orig.copy()
        pen = ImageDraw.Draw(overlay, "RGBA")
        marks = []
        for shape, name, xy, label, parent in record["marks"]:
            color = (43, 175, 114) if label == 2 else (234, 120, 47) if label == 1 else (
                (71, 148, 251) if shape != "unknown" else (247, 208, 60))
            if shape in {"core", "unknown"}:
                pen.polygon([tuple(p) for p in xy], fill=(*color, 62), outline=(*color, 255), width=3)
            else:
                pen.line([tuple(p) for p in xy], fill=(*color, 230), width=3)
            marks.append(dict(shape=shape, name=name, xy=[[a + x, b + y] for a, b in xy],
                              scene_label=label, parent=parent))
        overlay.thumbnail((660, 870))
        pos_x = 10 + 687 * column
        board.paste(overlay, (pos_x, 45))
        draw = ImageDraw.Draw(board)
        draw.text((pos_x, 12), record["accession"] + "  可见墨迹与结构参考", fill="#26352c", font=font)
        docs.append(dict(id=identifier, image=str(path), image_sha256=source["sha256"],
                         accession=record["accession"], source=source["url"], crop=record["crop"],
                         marks=marks, observation="assistant visual review before model inference",
                         scope="core polygons only for object scoring; inferred centerline and parents separate",
                         partition="development references, not a formal independent test"))
    write(OUT / "object_review.json", dict(rows=docs, inference_used=False, expert_reviewed=False))
    board.save(OUT / "object_review.png")


if __name__ == "__main__":
    main()
