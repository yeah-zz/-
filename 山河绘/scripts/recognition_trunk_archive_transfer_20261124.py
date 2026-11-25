# -*- coding: utf-8 -*-
"""Freeze original-image references and test the fixed entrance rule on other works."""

import json
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure
from recognition_full_trunk_20261123 import assess
from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines
from recognition_trunk_entrance_20261124 import alignment, classify
from recognition_trunk_envelope_20261123 import RULES, best_envelope
from recognition_trunk_identity_20261123 import check_candidate
from recognition_trunk_source_background_20261123 import source_background, source_rows


OUT = LAB / "trunk_archive_transfer_20261124"
ENTRANCE = LAB / "recognition_trunk_entrance_20261124"
VIEWS = (("原框",0,0),("上移",-4,-4),("下移",4,4),("加高",-4,4))
CASES = (
    (3, "柳干", "trunk", (384,287,424,385),
     ((330,403,410),(343,399,408),(361,398,409),(373,400,410))),
    (6, "林下浅干", "trunk", (206,247,228,300),
     ((264,212,219),(274,211,218),(286,211,218))),
    (8, "左侧墨干", "trunk", (330,98,373,181),
     ((106,337,348),(121,340,351),(139,344,358),(160,346,361),(173,344,361))),
    (8, "右侧细干", "trunk", (372,132,404,197),
     ((140,380,389),(154,381,390),(168,382,391),(181,380,392))),
    (19, "树身与树根交界", "mixed", (352,147,438,277), ()),
    (0, "深色岩纹", "rock", (214,94,263,174), ()),
    (10, "淡色崖面", "rock", (389,208,453,350), ()),
    (19, "树下坡边", "slope", (301,275,346,354), ()),
)


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def color_overlay(raw, box, path, sections):
    x0,y0,x1,y1=box
    tile=raw.crop(box)
    zoom=min(3.5,240/tile.height,255/tile.width)
    tile=tile.resize((round(tile.width*zoom),round(tile.height*zoom)),Image.Resampling.NEAREST)
    pen=ImageDraw.Draw(tile)
    for points,color in ((path,"#04c8ac"),(sections,"#f3a54e")):
        for yy,a,b,*_ in points:
            for xx in (a,b):
                if x0<=xx<x1 and y0<=yy<y1:
                    px,py=(xx-x0+.5)*zoom,(yy-y0+.5)*zoom
                    pen.ellipse((px-1,py-1,px+1,py+1),fill=color)
    return tile


def render(cases):
    board=Image.new("RGB",(1310,73+287*len(cases)),"#f4f1e8")
    pen=ImageDraw.Draw(board)
    label(pen,(14,11),"跨作品浅干和坡边复测",25)
    label(pen,(14,46),"橙点为原图人工参考，青色为通过检查的两侧外缘。",16)
    for i,case in enumerate(cases):
        y=73+i*287
        label(pen,(14,y),case["title"]+"　"+case["id"],15)
        with Image.open(case["image"]) as source:
            image=source.convert("RGB")
        for j,row in enumerate(case["views"]):
            x=14+325*j
            tile=color_overlay(image,row["crop"],row["candidate"]["retained"],case["sections"])
            board.paste(tile,(x,y+23))
            text=f"{row['view']} · 保留 {len(row['candidate']['retained'])} 行"
            if row["available_sections"]:
                text+=f" · 对齐 {row['aligned_after']}/{row['available_sections']}"
            label(pen,(x,y+267),text,14)
    board.save(OUT/"archive_transfer_comparison.png")


def main():
    configure()
    import cv2
    import torch
    from skimage.graph import MCP_Geometric
    from gongbi_training.scene_objects import SceneObjects
    from recognition_role_gap_20261123 import scene_evidence
    import recognition_trunk_entrance_20261124 as prior

    torch.set_num_threads(4)
    lock=verify_baselines()
    sources=read(OUT/"archive_candidates.json")
    older=read(ENTRANCE/"results.json")["cases"]
    used={sha(row["image"]) for row in older}
    train=read(SCENE/"dataset_manifest.json")["records"]
    trained={sha(row["image"]) for row in train if row["split"]=="train"}
    def work_key(value):
        return value.removeprefix("filename:").casefold()
    trained_groups={work_key(row["work_group"]) for row in train if row["split"]=="train"}
    references=[]
    for n,title,kind,box,sections in CASES:
        row=sources[n]
        source_hash=sha(row["image"])
        assert source_hash not in used
        references.append(dict(id=row["id"],title=title,role=kind,crop=box,sections=sections,
                               image=row["image"],image_sha256=source_hash,work_group=row["work_group"],
                               exact_training_image=source_hash in trained,
                               filename_group_in_training=work_key(row["work_group"]) in trained_groups))
    write("reference_freeze.json",dict(cases=references,source_only_manual_coordinates=True,
                                       independent_expert=False,local_only=True))
    weight=SCENE/"mapping_fixed.pt"
    assert sha(weight)==read(SCENE/"checkpoint_freeze.json")["mapping_fixed"]
    source_code=sha(prior.__file__)
    assert source_code==read(ENTRANCE/"protocol.json")["script_sha256"]
    write("protocol.json",dict(script_sha256=sha(__file__),reference_sha256=sha(OUT/"reference_freeze.json"),
                              checkpoint_sha256=sha(weight),entrance_script_sha256=source_code,
                              views=VIEWS,synthetic_perturbations_of_real_archived_images=True))
    model=SceneObjects()
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=False)["model"])
    model.eval()
    cache={}
    for case in references:
        image=case["image"]
        if image not in cache:
            with Image.open(image) as file:
                rgb=np.asarray(file.convert("RGB"))
            low,connected,_,_=scene_evidence(rgb,model,cv2,torch,MCP_Geometric)
            lab=cv2.GaussianBlur(cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB),(0,0),.6)
            _,profile=source_background(lab,cv2)
            cache[image]=lab,profile,(low==2)&(connected==2)
        lab,profile,tree=cache[image]

        @lru_cache(maxsize=None)
        def review(box):
            x0,y0,x1,y1=box
            run=best_envelope(list(source_rows(profile,lab,box)),y1-y0,appearance=True)
            estimate=assess(dict(crop=box,sections=[]),run,tree[y0:y1,x0:x1])
            center=[(left+right)/2 for _,left,right,_ in estimate["path"]]
            drift=float(np.ptp(center)/run["span"]) if center else 0.
            allowed=estimate["accepted"] and drift<=RULES["maximum_lateral_drift"]
            return check_candidate(estimate["path"],allowed,lab,profile,tree,review_body=True)

        x0,y0,x1,y1=case["crop"]
        scored=[]
        for view,top,bottom in VIEWS:
            box=(x0,y0+top,x1,y1+bottom)
            before=review(box)
            result=classify(box,before,review,lab.shape)
            scored.append(dict(view=view,crop=box,baseline=before,candidate=result,
                               available_sections=sum(box[1]<=y<box[3] for y,a,b in case["sections"]),
                               aligned_before=alignment(before["retained"],case["sections"]),
                               aligned_after=alignment(result["retained"],case["sections"])))
        case["views"]=scored
        print(case["title"],[(len(r["baseline"]["retained"]),len(r["candidate"]["retained"]),
                               r["aligned_after"]) for r in scored],flush=True)
    render(references)
    assert lock==verify_baselines()
    write("results.json",dict(cases=references,painting_baselines=lock,adopted=False,
                               project_independent_validation=False,robot_connected=False))


if __name__=="__main__":
    main()
