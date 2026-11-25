# -*- coding: utf-8 -*-
"""Trace tree-owned source ink past a semantic boundary and test its use for cunfa."""

import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from gongbi_training.runtime import LAB, configure
from recognition_groups_20261123 import SCENE, label, read, sha, summarize, verify_baselines
from recognition_role_gap_20261123 import scene_evidence


OUT = LAB / "recognition_attached_ink_20261123"
PRIOR = LAB / "recognition_role_gap_20261123"
SETTINGS = dict(source_contrast=.023, maximum_length=12, minimum_forward_cosine=.65,
                initial_alignment=.65, turn_cosine=.66, trunk_span_fraction=.05,
                minimum_trunk_span=14, trunk_aspect=1.35, root_ink_fraction=.20,
                maximum_context_side=1280)
STEPS = [(dy, dx, float(np.hypot(dy, dx)))
         for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def trace_attached(support, stable, max_length):
    """Walk source skeleton beyond known nodes with forward and corner constraints."""
    from scipy.ndimage import binary_dilation

    stable = stable & support
    height, width = support.shape
    recovered = np.zeros_like(support)
    if not stable.any():
        return recovered

    def neighbors(y, x):
        for dy, dx, distance in STEPS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and support[ny, nx]:
                yield ny, nx, dy, dx, distance

    frontier = set()
    exits = support & ~stable & binary_dilation(stable, structure=np.ones((3,3),dtype=bool))
    for y, x in np.argwhere(exits):
        for sy, sx, dy, dx, length in neighbors(y, x):
            if stable[sy, sx]:
                frontier.add((sy, sx, -dy, -dx, length))
    for sy, sx, dy, dx, first_length in sorted(frontier):
        # Require two successive upstream tree samples with directional agreement.
        upstream = False
        for ay, ax, ady, adx, length in neighbors(sy, sx):
            if not stable[ay, ax] or -(ady*dy + adx*dx) < SETTINGS["initial_alignment"]*length*first_length:
                continue
            if any(stable[by,bx] and -(bdy*dy + bdx*dx) >= SETTINGS["initial_alignment"]*blen*first_length
                   for by,bx,bdy,bdx,blen in neighbors(ay,ax)):
                upstream = True
                break
        if not upstream:
            continue
        y, x = sy + dy, sx + dx
        queue = deque([(y, x, dy, dx, first_length)])
        seen = {(y, x, dy, dx): first_length}
        while queue:
            cy, cx, pdy, pdx, travelled = queue.popleft()
            recovered[cy, cx] = True
            prev_length = float(np.hypot(pdy, pdx))
            for ny,nx,ndy,ndx,step in neighbors(cy,cx):
                next_distance = travelled + step
                if stable[ny,nx] or next_distance > max_length:
                    continue
                if (ndy*pdy + ndx*pdx) < SETTINGS["turn_cosine"]*prev_length*step:
                    continue
                if (ndy*dy + ndx*dx) < SETTINGS["minimum_forward_cosine"]*first_length*step:
                    continue
                state = (ny,nx,ndy,ndx)
                if next_distance < seen.get(state, np.inf):
                    seen[state] = next_distance
                    queue.append((*state,next_distance))
    return recovered


def attachment_maps(rgb, low, connected, cv2, skeletonize):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    smooth = cv2.GaussianBlur(gray, (0, 0), .7)
    ink = cv2.GaussianBlur(smooth, (0, 0), 7) - smooth >= SETTINGS["source_contrast"]
    skeleton = skeletonize(np.ascontiguousarray(ink))
    stable = skeleton & (low == 2) & (connected == 2)
    count, ids, stats, _ = cv2.connectedComponentsWithStats(stable.astype(np.uint8), connectivity=8)
    minimum_height = max(SETTINGS["minimum_trunk_span"], max(rgb.shape[:2])*SETTINGS["trunk_span_fraction"])
    rooted = np.zeros_like(stable)
    for k in range(1,count):
        width,height,area = stats[k,2:5]
        if height >= minimum_height and height >= SETTINGS["trunk_aspect"]*width and area >= 12:
            rooted |= ids == k
    extensions = [trace_attached(skeleton, a, SETTINGS["maximum_length"]) for a in (stable,rooted)]
    kernel = np.ones((3,3),np.uint8)
    fills = [cv2.dilate(x.astype(np.uint8),kernel).astype(bool) & ink & (connected == 1)
             for x in extensions]
    root_support = cv2.dilate((rooted | extensions[1]).astype(np.uint8),kernel).astype(bool) & ink
    return ink, root_support, fills, extensions


def make_comparison(rows, infer, cv2, refs):
    from gongbi_training.source_scene_mapping import source_to_scene

    output = []
    spans = []
    modes = ("连通旧版", "枝条墨迹延伸", "从细长树区出发")
    board = Image.new("RGB",(1200,82+137*len(rows)),"#f4f1e8")
    draw = ImageDraw.Draw(board)
    label(draw,(12,12),"沿可见枝条越过树石分区边缘",24)
    label(draw,(12,46),"橙色为树误作石，紫色为石误作树。青色为原图上候选延伸的墨迹。",16)
    for i,row in enumerate(rows):
        assert sha(row["image"]) == row["image_sha256"] and sha(row["mask"]) == row["mask_sha256"]
        im, connected, _, _, fills, extensions = infer(row["image"])
        with Image.open(row["mask"]) as mask:
            truth = source_to_scene(np.asarray(mask))
        y = 80 + 137*i
        label(draw,(12,y),row["id"],13)
        board.paste(ImageOps.contain(im,(206,95)),(12,y+20))
        boundary = np.zeros_like(truth,dtype=bool)
        for own,other in ((1,2),(2,1)):
            if (truth==other).any():
                distance = cv2.distanceTransform((truth!=other).astype(np.uint8),cv2.DIST_L2,5)
                boundary |= (truth==own) & (distance<=8)
        summaries = {}
        preds = [connected, *(np.where(mask,2,connected) for mask in fills)]
        valid = truth!=255
        for j,(name,pred) in enumerate(zip(modes,preds),1):
            x = j*237 + 12
            errors = np.asarray(ImageOps.grayscale(im).convert("RGB")).copy()
            errors[(truth==2)&(pred==1)] = (237,121,43)
            errors[(truth==1)&(pred==2)] = (159,62,191)
            board.paste(ImageOps.contain(Image.fromarray(errors),(206,95)),(x,y+20))
            label(draw,(x,y+118),name,13)
            summaries[name] = dict()
            for key,test in (("all",valid),("boundary",boundary)):
                c = np.bincount(truth[test].astype(int)*4 + pred[test],minlength=16).reshape(4,4)
                summaries[name][key] = c.tolist()
        tile = np.asarray(ImageOps.grayscale(im).convert("RGB")).copy()
        tile[fills[0] | extensions[0]] = (2,182,177)
        board.paste(ImageOps.contain(Image.fromarray(tile),(206,95)),(960,y+20))
        label(draw,(960,y+118),"枝条延伸位置",13)
        for ref in refs:
            if Path(ref["image"]) != Path(row["image"]):
                continue
            x0,y0 = ref["xy"]
            spans.append(dict(**ref, predictions={mode:int(pred[y0,x0]) for mode,pred in zip(modes,preds)}))
        output.append(dict(id=row["id"],batch=row["batch"],scores=summaries,
                           changed=[dict(total=int(fill.sum()), tree=int(np.count_nonzero(fill&(truth==2))),
                                         rock=int(np.count_nonzero(fill&(truth==1)))) for fill in fills]))
    board.save(OUT/"attached_tree_errors.png")
    aggregate = {}
    for batch in sorted({r["batch"] for r in output}):
        aggregate[batch] = {mode:{key:summarize(sum(np.array(r["scores"][mode][key]) for r in output if r["batch"]==batch))
                                  for key in ("all","boundary")} for mode in modes}
    return dict(records=output, aggregate=aggregate, known_branch_checks=spans)


def role_checks(rows, infer):
    output = []
    board = Image.new("RGB",(1190,79+132*((len(rows)+3)//4)),"#f4f1e8")
    draw = ImageDraw.Draw(board)
    label(draw,(12,10),"皴法候选与连续树状墨迹",25)
    label(draw,(12,44),"青色显示从细长树区起点连出的墨迹。依次查看形态、旧对象检查、新检查。",16)
    for i,row in enumerate(rows):
        im, _, ink, rooted, _, _ = infer(row["source"])
        left,top,right,bottom = row["box"] or (0,0,*im.size)
        region = np.s_[top:bottom,left:right]
        support = max(1,int(np.count_nonzero(ink[region])))
        fraction = float(np.count_nonzero(rooted[region])/support)
        before = row["accepted"]["连通树区否决"]
        appearance = row["accepted"]["形态单独"]
        after = bool(appearance and not (row["tree_veto"][1] and fraction >= SETTINGS["root_ink_fraction"]))
        output.append(dict(id=row["id"], admitted=row["admitted"], reference_kind=row["reference_kind"],
                           rooted_fraction=fraction, previous=before, proposed=after))
        x,y = i%4*296+10,i//4*132+78
        tile = np.asarray(im.crop((left,top,right,bottom))).copy()
        tile[rooted[region]] = (2,182,177)
        board.paste(ImageOps.contain(Image.fromarray(tile),(115,102)),(x,y))
        label(draw,(x+120,y+2),row["id"],13)
        label(draw,(x+120,y+24),row["expected"] or "类外参考",14)
        label(draw,(x+120,y+47)," / ".join("放行" if d else "未知" for d in (appearance,before,after)),13)
        label(draw,(x+120,y+67),f"连到细长树区 {fraction:.0%}",13)
    board.save(OUT/"cunfa_attached_evidence.png")
    return output


def main():
    configure()
    import cv2
    import torch
    from skimage.graph import MCP_Geometric
    from skimage.morphology import skeletonize
    from gongbi_training.scene_objects import SceneObjects

    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True,parents=True)
    lock = verify_baselines()
    old = read(PRIOR/"results.json")
    rows = read(PRIOR/"protocol.json")["object_refs"]
    save("protocol.json",dict(script_sha256=sha(__file__), previous_sha256=sha(PRIOR/"results.json"),
                              settings=SETTINGS, source_gates_use_labels=False,
                              known_regression=True, independent_validation=False))
    checkpoint = SCENE/"mapping_fixed.pt"
    assert sha(checkpoint)==read(SCENE/"checkpoint_freeze.json")["mapping_fixed"]
    net = SceneObjects()
    net.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    net.eval()
    cache = {}

    def infer(path):
        path = str(Path(path))
        if path not in cache:
            with Image.open(path) as src:
                im = src.convert("RGB")
            cap = SETTINGS["maximum_context_side"]
            image = ImageOps.contain(im,(cap,cap)) if max(im.size)>cap else im
            rgb = np.asarray(image)
            low, connected, _, _ = scene_evidence(rgb,net,cv2,torch,MCP_Geometric)
            ink, rooted, fills, extensions = attachment_maps(rgb,low,connected,cv2,skeletonize)
            if image.size!=im.size:
                ink, rooted = (cv2.resize(m.astype(np.uint8),im.size,interpolation=cv2.INTER_NEAREST)>0
                               for m in (ink,rooted))
            cache[path] = im, connected, ink, rooted, fills, extensions
        return cache[path]

    objects = make_comparison(rows,infer,cv2,old["branch_review"])
    print("OBJECTS",len(objects["records"]),flush=True)
    role = role_checks(old["role_checks"],infer)
    assert lock == verify_baselines()
    save("results.json",dict(objects=objects, cunfa=role, painting_baselines=lock, adopted=False,
                              independent_validation=False, robot_connected=False))
    print("CUNFA",len(role),flush=True)


if __name__=="__main__":
    main()
