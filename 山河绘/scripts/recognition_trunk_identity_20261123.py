# -*- coding: utf-8 -*-
"""Check a band for an attached upper trunk and stop a mixed lower tail."""

import json

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure
from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines
from recognition_trunk_source_background_20261123 import source_background, source_rows
from recognition_trunk_envelope_20261123 import RULES as ENVELOPE_RULES, assess, best_envelope


OUT = LAB / "recognition_trunk_identity_20261123"
PREV = LAB / "recognition_trunk_source_background_20261123"
POLICY = dict(search_above=28, sideways_margin=24, max_edge_step=3.0,
              max_parent_color_delta=13., minimum_parent_rows=9, minimum_parent_tree=.30,
              max_hops=3, tail_delta=11., tail_tree_fraction=.35, stop_run=3)
BODY_REVIEW = dict(head_rows=9, maximum_lightness_range=14., shorter_parent_rows=7,
                   shorter_parent_tree=.65)

# These are source review positions on the previously produced paths. They do
# not enter the continuation or stopping rules. The pixels at the interface
# can contain overlapping motifs, so disputed rows remain unscored.
POINT_REFERENCES = {
    "R2/5": [{"y":y,"role":"vegetation_or_slope"} for y in (305,308,312,315,320)],
    "R5/6": ([{"y":y,"role":"upper_trunk"} for y in (400,403,406,409)]
             + [{"y":y,"role":"root_boundary_uncertain"} for y in (412,416,420)]
             + [{"y":y,"role":"adjacent_rock"} for y in (425,431,436,440)]),
}


def write(name,value):
    (OUT / name).write_text(json.dumps(value,ensure_ascii=False,indent=2),"utf-8")


def median_body(lab,y,left,right):
    return np.median(lab[y,left+1:right],axis=0)


def delta_color(color,reference):
    return np.linalg.norm((color-reference)*np.array((.42,1.,1.)),axis=-1)


def tree_fraction(tree,y,left,right):
    return float(tree[y,left+1:right].mean())


def attach_parent(path,lab,profile,tree):
    """Start from the top of a band and trace a plausible parent in source pixels."""
    if not path or path[0][0]<2:
        return dict(path=[],support_rows=0,length=0,tree_fraction=0.)
    y0,l0,r0,_=path[0]
    ymin=max(0,y0-POLICY["search_above"])
    x0=max(0,l0-POLICY["sideways_margin"])
    x1=min(lab.shape[1],r0+POLICY["sideways_margin"]+1)
    head=path[:min(len(path),7)]
    root_color=np.median([median_body(lab,y,l,r) for y,l,r,_ in head],axis=0)
    root_width=np.median([right-left for _,left,right,_ in head])
    rows=list(source_rows(profile,lab,(x0,ymin,x1,y0)))[::-1]
    banks=[dict(y=y0,coords=np.array([[l0,r0]],dtype=int),score=np.zeros(1),parent=np.full((1,2),-1,dtype=int))]
    best=(0.,0,0)
    for row in rows:
        y=row["y"]+ymin
        coords=row["bounds"]+x0
        widths=coords[:,1]-coords[:,0]
        colors=delta_color(row["color"],root_color)
        available=(colors<=POLICY["max_parent_color_delta"]) & (widths>=max(3,root_width*.48))
        available &= widths<=max(6,root_width*1.8)
        local=.20*row["score"]-colors*.11+.4*row["supported"]
        coords,local=coords[available],local[available]
        if len(coords):
            local=local+np.array([tree_fraction(tree,y,l,r) for l,r in coords])
        scores=np.full(len(coords),-np.inf)
        parents=np.full((len(coords),2),-1,dtype=int)
        for i in range(max(0,len(banks)-POLICY["max_hops"]),len(banks)):
            prev=banks[i]
            if not (len(coords) and len(prev["coords"])):
                continue
            dy=prev["y"]-y
            move=np.abs(coords[:,None]-prev["coords"])
            comparison=prev["score"][None,:]-.6*move.sum(axis=2)-1.1*(dy-1)
            comparison[np.max(move,axis=2)>POLICY["max_edge_step"]*dy]=-np.inf
            columns=comparison.argmax(axis=1)
            links=comparison[np.arange(len(coords)),columns]+local
            improved=links>scores
            scores[improved]=links[improved]
            parents[improved,0]=i
            parents[improved,1]=columns[improved]
        banks.append(dict(y=y,coords=coords,score=scores,parent=parents))
        if len(scores) and scores.max()>best[0]:
            col=int(scores.argmax())
            best=(float(scores[col]),len(banks)-1,col)
    _,i,j=best
    parent=[]
    while i>0:
        row=banks[i]
        left,right=map(int,row["coords"][j])
        parent.append([int(row["y"]),left,right])
        i,j=row["parent"][j]
    support=float(np.mean([tree_fraction(tree,y,l,r) for y,l,r in parent])) if parent else 0.
    extent=int(y0-parent[0][0]) if parent else 0
    return dict(path=parent,support_rows=len(parent),length=extent,tree_fraction=support)


def split_tail(path,parent,lab,tree):
    """Mark the first sustained lower departure from a verified upper band."""
    body=[(y,l,r) for y,l,r,_ in path]
    top=body[:min(9,len(body))]
    reference_rows=top + [tuple(row) for row in parent["path"][-min(8,len(parent["path"])):]]
    reference=np.median([median_body(lab,y,l,r) for y,l,r in reference_rows],axis=0)
    samples=[];stop=None;dips=[]
    for y,left,right in body:
        delta=float(delta_color(median_body(lab,y,left,right),reference))
        fraction=tree_fraction(tree,y,left,right)
        outlier=delta>POLICY["tail_delta"] and fraction<POLICY["tail_tree_fraction"]
        if outlier:
            if dips and y!=dips[-1]+1:
                dips=[]
            dips.append(y)
        else:
            dips=[]
        if len(dips)>=POLICY["stop_run"] and stop is None:
            stop=int(dips[0])
        samples.append(dict(y=y,color_delta=delta,tree_fraction=fraction,body_disagreement=bool(outlier)))
    return stop,samples


def check_candidate(path,prior_accept,lab,profile,tree,parent=None,review_body=False):
    if not prior_accept or not path:
        return dict(status="待核", parent=dict(path=[],tree_fraction=0.,support_rows=0,length=0),
                    stop_row=None,retained=[],tail=[],row_signals=[],head_lightness_range=None)
    if parent is None:
        parent=attach_parent(path,lab,profile,tree)
    head=[median_body(lab,y,l,r)[0] for y,l,r,_ in path[:BODY_REVIEW["head_rows"]]]
    lightness_range=float(np.quantile(head,.90)-np.quantile(head,.10))
    established=parent["support_rows"]>=POLICY["minimum_parent_rows"]
    established &= parent["tree_fraction"]>=POLICY["minimum_parent_tree"]
    if review_body:
        if lightness_range>BODY_REVIEW["maximum_lightness_range"]:
            return dict(status="上段明暗变化过大",parent=parent,stop_row=None,retained=[],tail=path,
                        row_signals=[],head_lightness_range=lightness_range)
        short_parent=parent["support_rows"]>=BODY_REVIEW["shorter_parent_rows"]
        short_parent &= parent["tree_fraction"]>=BODY_REVIEW["shorter_parent_tree"]
        established |= short_parent
    if not established:
        return dict(status="缺少上接树干证据",parent=parent,stop_row=None,retained=[],tail=path,
                    row_signals=[],head_lightness_range=lightness_range)
    stop,samples=split_tail(path,parent,lab,tree)
    keep=path if stop is None else [r for r in path if r[0]<stop]
    exclude=[] if stop is None else [r for r in path if r[0]>=stop]
    return dict(status="上接树干，末端待核" if stop is not None else "上接树干候选",parent=parent,
                stop_row=stop,retained=keep,tail=exclude,row_signals=samples,
                head_lightness_range=lightness_range)


def draw(cases):
    board=Image.new("RGB",(1190,79+224*len(cases)),"#f4f1e8")
    d=ImageDraw.Draw(board)
    label(d,(12,11),"坡边、树根与上接树干的对照",25)
    label(d,(12,45),"青色为原候选或保留段，蓝色为向上连接证据，橙色为待核或停止的色带。",16)
    for i,case in enumerate(cases):
        y=77+i*224
        x0,y0,x1,y1=case["crop"]
        with Image.open(case["image"]) as f:
            im=f.convert("RGB")
        crop=(max(0,x0-24),max(0,y0-31),min(im.width,x1+24),min(im.height,y1+16))
        panel=im.crop(crop)
        zoom=min(3.3,159/panel.height,250/panel.width)
        tile=panel.resize((round(panel.width*zoom),round(panel.height*zoom)),Image.Resampling.NEAREST)
        label(d,(12,y),case["id"]+"　"+case["role"],15)
        result=case["views"][0]["body_review"]
        origin=case["views"][0]["original_path"]
        for j,(caption,sets) in enumerate((
                ("原图",[]),("上轮候选",[(origin,"#0cc6aa")]),
                (result["status"],[(result["retained"],"#0cc6aa"),(result["parent"]["path"],"#4c9fe0"),
                                  (result["tail"],"#ec903e")]))):
            frame=tile.copy();pen=ImageDraw.Draw(frame)
            for data,color in sets:
                for yy,a,b,*_ in data:
                    for xx in (a,b):
                        px=(xx-crop[0]+.5)*zoom;py=(yy-crop[1]+.5)*zoom
                        pen.ellipse((px-1,py-1,px+1,py+1),fill=color)
            x=12+j*389
            board.paste(frame,(x,y+28))
            status=caption
            if j==2 and result["parent"]["support_rows"]:
                status+=f" · 上接 {result['parent']['support_rows']} 行"
            label(d,(x,y+200),status,15)
    board.save(OUT/"trunk_identity_comparison.png")


def vertical_probes(case,lab,profile,tree):
    x0,y0,x1,y1=case["crop"]
    variants=(("above",[x0,y0-4,x1,y1-4]),("below",[x0,y0+4,x1,y1+4]),
              ("taller",[x0,y0-4,x1,y1+4]))
    results=[]
    for name,box in variants:
        left,top,right,bottom=box
        band=best_envelope(list(source_rows(profile,lab,box)),bottom-top,appearance=True)
        candidate=assess(dict(case,crop=box),band,tree[top:bottom,left:right])
        center=[(l+r)/2 for _,l,r,_ in candidate["path"]]
        drift=float(np.ptp(center)/candidate["span"]) if center else 0.
        accepted=bool(candidate["accepted"] and drift<=ENVELOPE_RULES["maximum_lateral_drift"])
        reviewed=check_candidate(candidate["path"],accepted,lab,profile,tree,review_body=True)
        positions={r[0] for r in reviewed["retained"]}
        matches=sum(ref["both_within_tolerance"] and ref["y"] in positions
                    for ref in candidate["reference_sections"])
        results.append(dict(view=name,crop=box,original_accepted=accepted,
                             filled_source_rows=len(positions),matched=matches,body_review=reviewed))
    return results


def main():
    configure()
    import cv2
    import torch
    from skimage.graph import MCP_Geometric
    from gongbi_training.scene_objects import SceneObjects
    from recognition_role_gap_20261123 import scene_evidence

    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True,parents=True)
    lock=verify_baselines()
    previous=read(PREV/"results.json")
    checkpoint=SCENE/"mapping_fixed.pt"
    assert sha(checkpoint)==read(SCENE/"checkpoint_freeze.json")["mapping_fixed"]
    write("protocol.json",dict(script_sha256=sha(__file__),previous_sha256=sha(PREV/"results.json"),
                               checkpoint_sha256=sha(checkpoint),settings=POLICY,
                               body_check=BODY_REVIEW,body_check_chosen_after_first_known_case_diagnostic=True,
                               manual_review_points=POINT_REFERENCES,point_references_only_for_scoring=True))
    model=SceneObjects()
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    model.eval()
    saved=[]
    cache={}
    for case in previous["cases"]:
        path=case["image"]
        assert sha(path)==case["image_sha256"]
        if path not in cache:
            with Image.open(path) as f:
                rgb=np.asarray(f.convert("RGB"))
            low,connected,_,_=scene_evidence(rgb,model,cv2,torch,MCP_Geometric)
            lab=cv2.GaussianBlur(cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB),(0,0),.6)
            _,profile=source_background(lab,cv2)
            cache[path]=lab,profile,(low==2)&(connected==2)
        lab,profile,tree=cache[path]
        views=[]
        for view in case["views"]:
            old=view["results"]["source_lab"]
            checked=check_candidate(old["path"],old["accepted"],lab,profile,tree)
            revised=check_candidate(old["path"],old["accepted"],lab,profile,tree,
                                    parent=checked["parent"],review_body=True)
            ys={point[0] for point in revised["retained"]}
            refs=set(p[0] for p in old["path"])
            manual=[dict(**r,previous=r["y"] in refs,current=r["y"] in ys)
                    for r in POINT_REFERENCES.get(case["id"],[])]
            mode_count=sum(check["both_within_tolerance"] and check["y"] in ys
                           for check in old["reference_sections"])
            views.append(dict(view=view["view"],crop=view["crop"],original_accepted=old["accepted"],
                              original_path=old["path"],original_matched=sum(r["both_within_tolerance"]
                                                                          for r in old["reference_sections"]),
                              identity=checked,body_review=revised,matched=mode_count,manual_points=manual))
        variation=vertical_probes(case,lab,profile,tree) if case["id"] in {
            "T2/1","T4/2","T4/3","R2/5","R5/6"} else []
        saved.append(dict(id=case["id"],image=path,crop=case["crop"],role=case["role"],views=views,
                          vertical_probes=variation))
        print(case["id"],[(r["body_review"]["status"],r["body_review"]["parent"]["support_rows"],
                           r["body_review"]["stop_row"],r["matched"]) for r in views],flush=True)
    draw(saved)
    assert lock==verify_baselines()
    write("results.json",dict(cases=saved,painting_baselines=lock,adopted=False,
                               independent_validation=False,robot_connected=False))


if __name__=="__main__":
    main()
