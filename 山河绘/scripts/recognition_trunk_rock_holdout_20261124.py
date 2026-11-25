# -*- coding: utf-8 -*-
"""Evaluate the frozen rock-conflict check on additional archived images."""

import json
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure
from recognition_full_trunk_20261123 import assess
from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines
from recognition_trunk_entrance_20261124 import classify
from recognition_trunk_envelope_20261123 import RULES, best_envelope
from recognition_trunk_identity_20261123 import check_candidate
from recognition_trunk_rock_context_20261124 import conflicts, measure, scene_probabilities
from recognition_trunk_source_background_20261123 import source_background, source_rows


OUT = LAB / "trunk_rock_context_holdout_20261124"
RULE_RUN = LAB / "trunk_rock_context_20261124"
VIEWS = (("原框",0,0),("上移",-4,-4),("下移",4,4),("加高",-4,4))
REFERENCES = (
    ("qiu","贴岩斜干","diagonal_trunk",(330,234,393,300),((250,360),(263,347),(274,336))),
    ("wang","山坡右侧细干","trunk",(378,235,411,312),((252,393),(270,392),(290,392))),
    ("museum_wang","前景浅树干","trunk",(454,334,490,408),((369,464),(381,466),(396,476))),
    ("zouzhe_xiangjiaoqiuwutu_43","岩边树干","trunk",(44,438,92,506),((469,54),(483,58),(498,65))),
    ("wang","褐色山坡墨纹","rock",(275,88,332,182),()),
    ("museum_zhai","中央岩壁","rock",(238,261,280,368),()),
    ("museum_wang","右下岩面","rock",(498,471,565,549),()),
    ("peaks","中山墨纹","rock",(102,134,148,214),()),
    ("wangximen_101","青绿岩纹","rock",(82,206,159,343),()),
    ("wangximen_65","青绿岩壁","rock",(266,192,326,314),()),
    ("qiu","树旁青绿岩纹","rock",(105,73,167,178),()),
)


def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),"utf-8")


def covers(path,points):
    by_y={y:(left,right) for y,left,right,_ in path}
    return sum(y in by_y and by_y[y][0]<=x<=by_y[y][1] for y,x in points)


def render(cases):
    sheet=Image.new("RGB",(1330,77+148*len(cases)),"#f4f1e8")
    draw=ImageDraw.Draw(sheet)
    label(draw,(12,11),"冻结规则在其他原图区域上的对照",24)
    label(draw,(12,47),"青色为检查前候选，橙色为岩面证据冲突。红点为人工核查的可见树身位置。",16)
    for i,case in enumerate(cases):
        y=75+i*148
        label(draw,(12,y),case["title"]+"  "+case["key"],14)
        with Image.open(case["image"]) as file:
            rgb=file.convert("RGB")
        for j,view in enumerate(case["views"]):
            x0,y0,x1,y1=view["crop"]
            tile=rgb.crop(view["crop"])
            zoom=min(2.7,93/tile.height,183/tile.width)
            tile=tile.resize((round(tile.width*zoom),round(tile.height*zoom)),Image.Resampling.NEAREST)
            pen=ImageDraw.Draw(tile)
            color="#f19a4f" if view["withheld"] else "#08c6ac"
            for yy,a,b,_ in view["path"]:
                for xx in (a,b):
                    tx=(xx-x0+.5)*zoom;ty=(yy-y0+.5)*zoom
                    pen.ellipse((tx-1,ty-1,tx+1,ty+1),fill=color)
            for yy,xx in case["points"]:
                tx=(xx-x0+.5)*zoom;ty=(yy-y0+.5)*zoom
                pen.ellipse((tx-1.4,ty-1.4,tx+1.4,ty+1.4),fill="#fa795c")
            x=12+j*330
            sheet.paste(tile,(x,y+21))
            status="转待核" if view["withheld"] else "保留候选" if view["path"] else "未产生候选"
            label(draw,(x,y+118),view["name"]+" · "+status,14)
    sheet.save(OUT/"holdout_comparison.png")


def main():
    configure()
    import cv2
    import torch
    from skimage.graph import MCP_Geometric
    from gongbi_training.scene_objects import SceneObjects
    from recognition_role_gap_20261123 import scene_evidence
    import recognition_trunk_rock_context_20261124 as frozen

    torch.set_num_threads(4)
    old=read(RULE_RUN/"protocol.json")
    assert old["script_sha256"]==sha(frozen.__file__) and old["mixed_scale_rule"]==frozen.MIXED
    weights=SCENE/"mapping_fixed.pt"
    assert old["checkpoint_sha256"]==sha(weights)
    previously_used={sha(row["image"]) for row in read(RULE_RUN/"results.json")["records"]}
    known_train={sha(row["image"]) for row in read(SCENE/"dataset_manifest.json")["records"]
                 if row["split"]=="train"}
    sources={row["key"]:row["image"] for row in read(OUT/"sources.json")}
    references=[]
    for key,title,role,box,points in REFERENCES:
        image=sources[key]
        digest=sha(image)
        assert digest not in previously_used
        references.append(dict(key=key,title=title,role=role,crop=box,points=points,image=image,
                               image_sha256=digest,exact_training_file=digest in known_train))
    write("reference_freeze.json",dict(cases=references,source_only_review=True,expert_review=False))
    write("protocol.json",dict(script_sha256=sha(__file__),checkpoint_sha256=sha(weights),
                              prior_rule_sha256=sha(frozen.__file__),mixed_scale_rule=frozen.MIXED,
                              reference_sha256=sha(OUT/"reference_freeze.json"),views=VIEWS,
                              disjoint_from_rock_rule_development_images=True))
    model=SceneObjects()
    model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=False)["model"])
    model.eval()
    lock=verify_baselines()
    cache={}
    for row in references:
        image=row["image"]
        if image not in cache:
            with Image.open(image) as file:
                rgb=np.asarray(file.convert("RGB"))
            low,connected,_,_=scene_evidence(rgb,model,cv2,torch,MCP_Geometric)
            lab=cv2.GaussianBlur(cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB),(0,0),.6)
            _,profile=source_background(lab,cv2)
            cache[image]=(lab,profile,(low==2)&(connected==2),
                          scene_probabilities(rgb,model,torch,cv2),cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
        lab,profile,tree,probs,gray=cache[image]

        @lru_cache(maxsize=None)
        def review(box):
            left,top,right,bottom=box
            band=best_envelope(list(source_rows(profile,lab,box)),bottom-top,appearance=True)
            estimate=assess(dict(crop=box,sections=[]),band,tree[top:bottom,left:right])
            middle=[(a+b)/2 for _,a,b,_ in estimate["path"]]
            drift=float(np.ptp(middle)/band["span"]) if middle else 0.
            ready=estimate["accepted"] and drift<=RULES["maximum_lateral_drift"]
            checked=check_candidate(estimate["path"],ready,lab,profile,tree,review_body=True)
            return checked,estimate["path"]

        x0,y0,x1,y1=row["crop"]
        views=[]
        for name,dy0,dy1 in VIEWS:
            box=(x0,y0+dy0,x1,y1+dy1)
            baseline,geometry=review(box)
            candidate=classify(box,baseline,lambda b:review(b)[0],lab.shape)
            path=candidate["retained"]
            evidence=measure(path,probs,gray)
            withheld=conflicts(evidence)["mixed_same_side"]
            # Record the raw geometry separately to diagnose upstream misses and unexercised conflicts.
            raw_evidence=measure(geometry,probs,gray)
            views.append(dict(name=name,crop=box,path=path,geometry_rows=len(geometry),
                              identity_status=baseline["status"],withheld=withheld,evidence=evidence,
                              raw_geometry_would_conflict=conflicts(raw_evidence)["mixed_same_side"],
                              raw_geometry_evidence=raw_evidence,anchor_hits=covers(path,row["points"])))
        row["views"]=views
        print(row["title"],[(len(r["path"]),r["withheld"],r["anchor_hits"],
                            r["raw_geometry_would_conflict"]) for r in views],flush=True)
    render(references)
    assert lock==verify_baselines()
    write("results.json",dict(cases=references,painting_baselines=lock,adopted=False,robot_connected=False))


if __name__=="__main__":
    main()
