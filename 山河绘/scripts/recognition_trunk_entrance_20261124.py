# -*- coding: utf-8 -*-
"""Retry the trunk identity check at several source rows and keep shared support."""

import json
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure
from recognition_full_trunk_20261123 import assess
from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines
from recognition_trunk_envelope_20261123 import RULES, best_envelope
from recognition_trunk_identity_20261123 import POINT_REFERENCES, check_candidate
from recognition_trunk_source_background_20261123 import source_background, source_rows


OUT = LAB / "recognition_trunk_entrance_20261124"
PREV = LAB / "recognition_trunk_identity_20261123"
GEOMETRY = LAB / "recognition_trunk_source_background_20261123"
SETTINGS = dict(entrance_offsets=(-12,-8,-4,4,8,12), minimum_votes=2,
                boundary_tolerance=2, minimum_continuous_rows=12)


def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),"utf-8")


def shared_sections(paths,box):
    """Keep measured sections from one real path when a second route supports them."""
    left,top,right,bottom=box
    maps=[{row[0]:row for row in path} for path in paths]
    best=[]
    best_strength=(-1,-1)
    for route in paths:
        run=[]
        votes_total=0
        for data in route:
            y,a,b,_=data
            tol=SETTINGS["boundary_tolerance"]
            votes=sum(y in alt and abs(a-alt[y][1])<=tol and abs(b-alt[y][2])<=tol for alt in maps)
            usable=top<=y<bottom and left<=a<b<right and votes>=SETTINGS["minimum_votes"]
            if not usable or (run and y!=run[-1][0]+1):
                run=[]
                votes_total=0
            if usable:
                run.append(data)
                votes_total+=votes
                strength=(len(run),votes_total)
                if strength>best_strength:
                    best,best_strength=list(run),strength
    if len(best)<SETTINGS["minimum_continuous_rows"]:
        return []
    return best


def classify(box,baseline,recheck,shape):
    retained=baseline["retained"]
    if retained:
        return dict(status="沿用已通过的树身",retained=retained,attempts=[])
    left,top,right,bottom=box
    attempts=[]
    entrances=sorted({max(0,top+offset) for offset in SETTINGS["entrance_offsets"]})
    for y in entrances:
        if y==top or bottom-y<SETTINGS["minimum_continuous_rows"] or bottom>shape[0]:
            continue
        decision=recheck((left,y,right,bottom))
        attempts.append(dict(offset=y-top,rows=len(decision["retained"]),status=decision["status"],
                             stop_row=decision["stop_row"],path=decision["retained"]))
    selected=shared_sections([v["path"] for v in attempts if v["rows"]],box)
    return dict(status="多个起点支持" if selected else "待核",retained=selected,attempts=attempts)


def alignment(path,refs):
    points={y:(left,right) for y,left,right,_ in path}
    tol=SETTINGS["boundary_tolerance"]
    return sum(y in points and abs(l-points[y][0])<=tol and abs(r-points[y][1])<=tol for y,l,r in refs)


def render(cases):
    chosen=[(c,v) for c in cases for v in c["views"]
            if c["id"] in ("T2/1","R2/5","R5/6") and v["group"]=="previous_vertical"]
    board=Image.new("RGB",(1040,75+182*len(chosen)),"#f4f1e8")
    draw=ImageDraw.Draw(board)
    label(draw,(12,10),"换一个起点检查同一段树身",24)
    label(draw,(12,42),"青色为保留的树干外缘，橙框为本次裁剪。",16)
    for index,(case,view) in enumerate(chosen):
        x0,y0,x1,y1=view["crop"]
        with Image.open(case["image"]) as source:
            crop=(max(0,x0-14),max(0,y0-14),min(source.width,x1+14),min(source.height,y1+5))
            tile=source.convert("RGB").crop(crop)
        zoom=min(3,128/tile.height,240/tile.width)
        tile=tile.resize((round(tile.width*zoom),round(tile.height*zoom)),Image.Resampling.NEAREST)
        y=75+index*182
        label(draw,(12,y),case["id"]+"　"+{"above":"上移","below":"下移","taller":"加高"}[view["view"]],15)
        for j,(name,path) in enumerate((("原图",[]),("原起点",view["before"]),("多个起点",view["after"]["retained"]))):
            frame=tile.copy()
            pen=ImageDraw.Draw(frame)
            pen.rectangle(tuple(v*zoom for v in (x0-crop[0],y0-crop[1],x1-crop[0]-1,y1-crop[1]-1)),
                          outline="#e58b45")
            for row,left,right,_ in path:
                for edge in (left,right):
                    px=(edge-crop[0]+.5)*zoom
                    py=(row-crop[1]+.5)*zoom
                    pen.ellipse((px-1,py-1,px+1,py+1),fill="#0bbda6")
            x=12+j*344
            board.paste(frame,(x,y+20))
            caption=name if j==0 else f"{name} · 保留 {len(path)} 行"
            label(draw,(x,y+154),caption,14)
    board.save(OUT/"trunk_entrance_comparison.png")


def main():
    configure()
    import cv2
    import torch
    from skimage.graph import MCP_Geometric
    from gongbi_training.scene_objects import SceneObjects
    from recognition_role_gap_20261123 import scene_evidence

    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True,parents=True)
    baselines=verify_baselines()
    previous=read(PREV/"results.json")
    old_geometry={c["id"]:c for c in read(GEOMETRY/"results.json")["cases"]}
    weight=SCENE/"mapping_fixed.pt"
    assert sha(weight)==read(SCENE/"checkpoint_freeze.json")["mapping_fixed"]
    write("protocol.json",dict(script_sha256=sha(__file__),checkpoint_sha256=sha(weight),
                                previous_sha256=sha(PREV/"results.json"),settings=SETTINGS,
                                added_probes=("up_6","down_6","grow_6"),references_used_for_scoring_only=True))
    model=SceneObjects()
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=False)["model"])
    model.eval()
    cache={}
    cases=[]
    for source in previous["cases"]:
        image=source["image"]
        refs=old_geometry[source["id"]]["sections"]
        assert sha(image)==old_geometry[source["id"]]["image_sha256"]
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
            geometry=assess(dict(crop=box,sections=[]),run,tree[y0:y1,x0:x1])
            center=[(a+b)/2 for _,a,b,_ in geometry["path"]]
            drift=float(np.ptp(center)/run["span"]) if center else 0.
            accepted=geometry["accepted"] and drift<=RULES["maximum_lateral_drift"]
            return check_candidate(geometry["path"],accepted,lab,profile,tree,review_body=True)

        x0,y0,x1,y1=source["crop"]
        bounds=[("up_6",[x0,y0-6,x1,y1-6]),("down_6",[x0,y0+6,x1,y1+6]),
                ("grow_6",[x0,y0-6,x1,y1+6])]
        groups=(("horizontal",source["views"]),("previous_vertical",source["vertical_probes"]),
                ("additional_vertical",[dict(view=name,crop=box,body_review=review(tuple(box)))
                                        for name,box in bounds]))
        views=[]
        for group,records in groups:
            for row in records:
                box=row["crop"]
                old=row["body_review"]
                proposal=classify(box,old,review,lab.shape)
                previous_y={r[0] for r in old["retained"]}
                proposed_y={r[0] for r in proposal["retained"]}
                checks=[dict(**pt,before=pt["y"] in previous_y,after=pt["y"] in proposed_y)
                        for pt in POINT_REFERENCES.get(source["id"],[]) if box[1]<=pt["y"]<box[3]]
                views.append(dict(view=row["view"],group=group,crop=box,before=old["retained"],after=proposal,
                                  previous_matches=alignment(old["retained"],refs),current_matches=alignment(proposal["retained"],refs),
                                  checks=checks))
        cases.append(dict(id=source["id"],role=source["role"],image=image,views=views))
        print(source["id"],[(r["view"],len(r["before"]),len(r["after"]["retained"]))
                            for r in views if r["group"]!="horizontal"],flush=True)
    render(cases)
    assert baselines==verify_baselines()
    write("results.json",dict(cases=cases,painting_baselines=baselines,adopted=False,
                               independent_validation=False,robot_connected=False))


if __name__=="__main__":
    main()
