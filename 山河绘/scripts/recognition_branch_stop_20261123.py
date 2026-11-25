# -*- coding: utf-8 -*-
"""Test when an attached source-ink trace should stop at a rock face."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from gongbi_training.runtime import LAB, configure
from recognition_attached_ink_20261123 import SETTINGS, attachment_maps, trace_attached
from recognition_groups_20261123 import SCENE, label, read, sha, summarize, verify_baselines
from recognition_role_gap_20261123 import scene_evidence
from recognition_structure_20261123 import derivatives


OUT = LAB / "recognition_branch_stop_20261123"
ATTACHED = LAB / "recognition_attached_ink_20261123"
GAP = LAB / "recognition_role_gap_20261123"
LIMITS = dict(near_px=2.0, far_px=4.5, weak_side_limit=.008, near_imbalance=.035,
              far_imbalance=.075, paired_near=.014, paired_far=.020, sustained_threshold=.6)
MODES = ("connected", "extended", "rock_face_stop", "paired_sides")
NAMES = ("连通旧版", "上轮延伸", "遇岩面侧差停止", "要求双侧墨脊")


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def section_tests(rgb, cv2):
    """Distinguish a dark line with light flanks from a one-sided tonal edge."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    image = cv2.GaussianBlur(gray, (0, 0), .65)
    strength, _, angle = derivatives(gray, 1.1, cv2)
    nx, ny = np.cos(angle*.5), np.sin(angle*.5)
    yy, xx = np.indices(image.shape, dtype=np.float32)

    def flanks(offset):
        return [cv2.remap(image, xx + nx*offset*s, yy + ny*offset*s, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE) - image for s in (-1, 1)]

    left, right = flanks(LIMITS["near_px"])
    wide_left, wide_right = flanks(LIMITS["far_px"])
    near_diff = left - right
    far_diff = wide_left - wide_right
    one_sided = ((np.minimum(left,right) <= LIMITS["weak_side_limit"])
                 & (np.abs(near_diff) >= LIMITS["near_imbalance"])
                 & (np.abs(far_diff) >= LIMITS["far_imbalance"])
                 & (np.sign(near_diff) == np.sign(far_diff)))
    side = one_sided.astype(np.float32)
    before = cv2.remap(side, xx-ny, yy+nx, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    after = cv2.remap(side, xx+ny, yy-nx, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sustained = one_sided & (np.maximum(before,after) >= LIMITS["sustained_threshold"])
    paired = ((strength > 0) & (np.minimum(left,right) >= LIMITS["paired_near"])
              & (np.minimum(wide_left,wide_right) >= LIMITS["paired_far"]))
    return sustained, paired


def stop_predictions(rgb, low, connected, cv2, skeletonize):
    ink, _, (old_fill, _), _ = attachment_maps(rgb, low, connected, cv2, skeletonize)
    skeleton = skeletonize(np.ascontiguousarray(ink))
    stable = skeleton & (low==2) & (connected==2)
    rock_face, paired = section_tests(rgb, cv2)
    fills = [old_fill]
    kernel = np.ones((3,3),dtype=np.uint8)
    for allowed in (~rock_face, paired & ~rock_face):
        permitted = stable | (skeleton & allowed)
        recovered = trace_attached(permitted, stable, SETTINGS["maximum_length"])
        ribbon = cv2.dilate(recovered.astype(np.uint8), kernel).astype(bool)
        fills.append(ribbon & ink & allowed & (connected==1))
    return fills, rock_face, paired


def score_regions(truth, predictions, cv2):
    valid = truth != 255
    boundary = np.zeros_like(valid)
    for own,other in ((1,2),(2,1)):
        if np.any(truth==other):
            distance = cv2.distanceTransform((truth!=other).astype(np.uint8),cv2.DIST_L2,5)
            boundary |= (truth==own) & (distance<=8)
    return {
        mode: {name: np.bincount(truth[keep].astype(int)*4 + pred[keep],
                                minlength=16).reshape(4,4).tolist()
               for name,keep in (("all",valid),("boundary",boundary))}
        for mode,pred in zip(MODES,predictions)
    }


def diagnose(rows, net, torch, cv2, skeletonize, solver):
    from gongbi_training.source_scene_mapping import source_to_scene

    result = []
    branch = None
    board = Image.new("RGB",(1240,79+112*len(rows)),"#f4f1e8")
    draw = ImageDraw.Draw(board)
    label(draw,(12,12),"树石交界停止规则对照",25)
    label(draw,(12,46),"橙色为树误作石，紫色为石误作树。",16)
    for i,row in enumerate(rows):
        assert sha(row["image"])==row["image_sha256"] and sha(row["mask"])==row["mask_sha256"]
        with Image.open(row["image"]) as f:
            im = f.convert("RGB")
        rgb = np.asarray(im)
        low, connected, _, _ = scene_evidence(rgb,net,cv2,torch,solver)
        fills, rock_face, paired = stop_predictions(rgb,low,connected,cv2,skeletonize)
        predictions = [connected, *(np.where(fill,2,connected) for fill in fills)]
        with Image.open(row["mask"]) as source:
            truth = source_to_scene(np.asarray(source))
        if row["id"]=="zouzhe_xiangjiaoqiuwutu_43":
            branch = (im, predictions, rock_face, paired, fills)
        changes = {mode:dict(tree=int((fill & (truth==2)).sum()),
                             rock=int((fill & (truth==1)).sum()), total=int(fill.sum()))
                   for mode,fill in zip(MODES[1:],fills)}
        result.append(dict(id=row["id"],batch=row["batch"],cohort=row["cohort"],changes=changes,
                           metrics=score_regions(truth,predictions,cv2)))
        y=78+i*112
        label(draw,(12,y),row["id"],14)
        board.paste(ImageOps.contain(im,(218,75)),(12,y+20))
        for j,(pred,name) in enumerate(zip(predictions,NAMES),1):
            tile=np.asarray(ImageOps.grayscale(im).convert("RGB")).copy()
            tile[(truth==2)&(pred==1)]=(235,119,44)
            tile[(truth==1)&(pred==2)]=(158,70,184)
            x=12+244*j
            board.paste(ImageOps.contain(Image.fromarray(tile),(218,75)),(x,y+20))
            label(draw,(x,y+96),name,13)
        if (i+1)%6==0:
            print("OBJECTS",i+1,flush=True)
    board.save(OUT/"branch_stop_comparison.png")
    totals={}
    for cohort in ("development_regression","other_archive","additional_groups"):
        data=[r for r in result if r["cohort"]==cohort]
        totals[cohort]={mode:{loc:summarize(sum(np.array(row["metrics"][mode][loc]) for row in data))
                              for loc in ("all","boundary")} for mode in MODES}
    return result,totals,branch


def render_branch(detail, refs):
    im, preds, rock_face, paired, fills = detail
    bounds=(113,401,158,445)
    x0,y0,x1,y1=bounds
    zone=np.s_[y0:y1,x0:x1]
    zoom=6
    board=Image.new("RGB",(1240,375),"#f4f1e8")
    draw=ImageDraw.Draw(board)
    label(draw,(12,10),"枝梢和岩面停止位置核查",25)
    label(draw,(12,44),"青色为补回的树区，红色为持续单侧明暗变化，圆圈为同一枝条的墨芯。",16)
    for i,(name,fill) in enumerate(zip(("原图","上轮延伸","遇岩面侧差停止","要求双侧墨脊"),(None,*fills))):
        x,y=12+307*i,79
        tile=np.asarray(im.crop(bounds)).copy()
        if fill is not None:
            if i>=2:
                unstable=rock_face[zone]
                tile[unstable]=((tile[unstable]*.6)+np.array([224,51,43])*.4).astype("uint8")
            tile[fill[zone]]=(8,182,175)
        board.paste(Image.fromarray(tile).resize(((x1-x0)*zoom,(y1-y0)*zoom),Image.Resampling.NEAREST),(x,y))
        label(draw,(x,y+270),name,15)
        for ref in refs:
            px,py=ref["xy"]
            lx,ly=x+(px-x0+.5)*zoom,y+(py-y0+.5)*zoom
            draw.ellipse((lx-4,ly-4,lx+4,ly+4),outline="#e79448",width=2)
    board.save(OUT/"branch_stop_detail.png")
    return [dict(id=ref["id"],xy=ref["xy"],points={
        key:int(pred[ref["xy"][1],ref["xy"][0]]) for key,pred in zip(MODES,preds)},
        at_one_sided_edge=bool(rock_face[ref["xy"][1],ref["xy"][0]]),
        meets_paired_sides=bool(paired[ref["xy"][1],ref["xy"][0]])) for ref in refs]


def cunfa_conflicts(previous, current):
    """Route disagreement to review while preserving the original shape score."""
    latest={r["id"]:r for r in current}
    rows=[]
    for r in previous:
        new=latest[r["id"]]
        before=r["accepted"]["连通树区否决"]
        after=new["proposed"]
        status="待核归属" if before!=after else "皴法候选" if before else "未知"
        rows.append(dict(id=r["id"],status=status,reference=r["expected"] or r["reference_kind"],
                         shape_score=r["shape_score"],eligible_for_painting_or_training=False))
    return rows


def main():
    configure()
    import cv2
    import torch
    from skimage.graph import MCP_Geometric
    from skimage.morphology import skeletonize
    from gongbi_training.scene_objects import SceneObjects

    OUT.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    lock=verify_baselines()
    checkpoint=SCENE/"mapping_fixed.pt"
    assert sha(checkpoint)==read(SCENE/"checkpoint_freeze.json")["mapping_fixed"]
    previous=read(GAP/"results.json")
    attached=read(ATTACHED/"results.json")
    first=[dict(**r,cohort=r["batch"]) for r in read(GAP/"protocol.json")["object_refs"]]
    more=[dict(**r,cohort="additional_groups") for r in read(ATTACHED/"additional_groups/protocol.json")["rows"]]
    rows=first+more
    assert len({r["image_sha256"] for r in rows})==26
    save=dict(script_sha256=sha(__file__), settings=LIMITS, source_tracking_settings=SETTINGS,
              source_refs=rows, historical_regression=True, no_label_input_at_inference=True)
    write("protocol.json",save)
    model=SceneObjects()
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    model.eval()
    results,totals,branch=diagnose(rows,model,torch,cv2,skeletonize,MCP_Geometric)
    branch=render_branch(branch,previous["branch_review"][1:])
    conflicts=cunfa_conflicts(previous["role_checks"],attached["cunfa"])
    assert lock==verify_baselines()
    write("results.json",dict(records=results, totals=totals, branch=branch, cunfa_review=conflicts,
                              painting_baselines=lock, adopted=False, independent_validation=False,
                              robot_connected=False))
    print("DONE",len(results),"crops",flush=True)


if __name__=="__main__":
    main()
