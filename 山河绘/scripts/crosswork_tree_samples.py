# -*- coding: utf-8 -*-
"""Trace reviewed visible tree geometry and fit cross-work structural profiles."""

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from gongbi_training.runtime import LAB, ROOT
from master_studio.brush import catmull_rom
from master_studio.digital_strokes import DigitalStroke
from recognition_groups_20261123 import read, sha
from digital_specialty_round import trunk_profile


REFERENCE = (
    ("41154",(19,243,315,518),(350,325),
     [(137,278),(138,243),(138,211),(141,178),(144,156),(155,132),(162,104),(159,74)],
     [("main",4,[(132,143),(124,137),(118,122)]),
      ("main",5,[(171,123),(189,112),(203,104)]),
      ("main",6,[(173,94),(188,78),(191,65)]),
      ("b1",2,[(196,94),(201,82)])]),
    ("45635",(63,268,348,587),(350,392),
     [(127,307),(128,271),(133,242),(139,215),(145,192),(155,173),(165,154),(177,147)],
     [("main",5,[(146,155),(141,142),(129,133)]),
      ("main",6,[(183,151),(196,149),(212,139)]),
      ("main",7,[(179,135),(186,127)]),
      ("b1",2,[(200,139),(205,130)])]),
    ("45636",(55,324,306,536),(350,296),
     [(203,238),(209,218),(212,191),(211,169),(210,144),(211,124),(214,104),(219,76)],
     [("main",4,[(225,138),(244,130),(258,125)]),
      ("main",5,[(195,119),(186,110),(178,99)]),
      ("main",6,[(225,94),(240,91),(255,88)]),
      ("b0",2,[(250,120),(255,112)])]),
)


def _arc(path):
    return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]


def trees_from_references():
    folder=ROOT/"master_studio"/"corpus"/"ni_zan"
    source={str(r["objectID"]):r for r in read(folder/"meta.json")}
    cohort=[]
    for ident,box,shape,trunk,branches in REFERENCE:
        file=folder/f"{ident}.jpg"
        scale=(np.asarray(box[2:])-box[:2])/shape
        paths=[{"id":"main","parent":None,"attach_index":None,
                "xy":(np.asarray(trunk)*scale+box[:2]).tolist()}]
        for k,(parent,anchor,points) in enumerate(branches):
            p=next(g for g in paths if g["id"]==parent)
            xy=np.vstack([p["xy"][anchor],np.asarray(points)*scale+box[:2]])
            paths.append(dict(id=f"b{k}",parent=parent,attach_index=anchor,xy=xy.tolist()))
        cohort.append(dict(work_id=f"met:{ident}",artist=source[ident]["artist"],
                           source=source[ident]["source"],image=str(file),sha256=sha(file),crop=box,paths=paths,
                           evidence="assistant source-overlay visible partial tree centerline, endpoints approximate",
                           root="lowest visible trunk point; hidden root unresolved",occlusion="missing continuations unknown",
                           complete_leafless_tree=False,project_blind=False))
    return cohort


def widths_and_curvature():
    files=[LAB/"trunk_archive_transfer_20261124"/"results.json",
           LAB/"recognition_trunk_source_background_20261123"/"results.json"]
    rows=[]
    for file in files:
        for record in read(file)["cases"]:
            if not record["sections"]:
                continue
            assert sha(record["image"])==record["image_sha256"]
            group=record.get("work_group",Path(record["image"]).stem.rsplit("_",1)[0])
            rows.append((record,group,trunk_profile(record)))
    grid=np.linspace(0,1,31)
    single=next(v for r,_,v in rows if r["id"]=="T2/1")
    width_scores=[]
    for r,group,(t,center,width) in rows:
        train=[(name,v) for _,name,v in rows if name!=group]
        works=sorted({name for name,_ in train})
        # Each work contributes a single profile, irrespective of the number
        # of measured subtrees cropped from that work.
        curves=[]
        for i in (1,2):
            means=[np.mean([np.interp(grid,v[0],v[i]) for name,v in train if name==work],axis=0) for work in works]
            curves.append(np.median(means,axis=0))
        profiles={"T1":(np.zeros_like(t),np.ones_like(t)),
                  "T1_taper":(np.zeros_like(t),1-.42*t),
                  "T2":(np.interp(t,single[0],single[1]),np.interp(t,single[0],single[2])),
                  "T3":(np.interp(t,grid,curves[0]),np.interp(t,grid,curves[1]))}
        metrics={}
        for name,(xy,w) in profiles.items():
            if name=="T2" and group==next(g for a,g,_ in rows if a["id"]=="T2/1"):
                continue
            metrics[name]={"center_mae":float(np.abs(center-xy).mean()),
                           "width_mae":float(np.abs(width-w).mean())}
        width_scores.append(dict(id=r["id"],work=group,source=r["image"],sha256=r["image_sha256"],
                                 sections=r["sections"],training_works=works,metrics=metrics))
    profiles=np.stack([np.interp(grid,v[0],v[2]) for _,_,v in rows])
    return grid,np.median(profiles,axis=0),width_scores


def descriptor(tree):
    trunk=np.asarray(tree["paths"][0]["xy"])
    height=max(float(trunk[0,1]-trunk[-1,1]),1.)
    result=[]
    for branch in tree["paths"][1:]:
        if branch["parent"]!="main":
            continue
        path=np.asarray(branch["xy"])
        delta=path[-1]-path[0]
        result.append([(trunk[0,1]-path[0,1])/height,float(np.arctan2(-delta[1],abs(delta[0])+1e-4)),
                       _arc(path)[-1]/height,float(np.sign(delta[0]))])
    return np.asarray(result)


def branch_validation(trees):
    fixed=np.array([[.46,.53,.37,-1],[.61,.57,.33,1],[.77,.62,.30,-1],[.86,.69,.27,1]])
    result=[]
    for tree in trees:
        truth=descriptor(tree)
        variants={"T0":fixed}
        others=[descriptor(r) for r in trees if r["work_id"]!=tree["work_id"]]
        variants["T3"]=min(others,key=lambda candidate:sum(_branch_distance(candidate,p) for p in others))
        result.append({"work":tree["work_id"],"truth":truth.tolist(),
                       "training_works":[r["work_id"] for r in trees if r["work_id"]!=tree["work_id"]],
                       "cost":{k:_branch_distance(v,truth) for k,v in variants.items()}})
    return result


def _branch_distance(predicted, truth):
    d=np.abs(predicted[:,None]-truth[None])
    distance=d[:,:,0]+d[:,:,1]/np.pi+d[:,:,2]+.23*(d[:,:,3]>1)
    r,c=linear_sum_assignment(distance)
    return float((distance[r,c].sum()+.30*abs(len(predicted)-len(truth)))/max(len(predicted),len(truth)))


def emit_tree(trees, width_grid, width_profile, layout, example, kind="learned"):
    """Transfer jointly observed branch geometry to a new trunk attachment."""
    center,height,lean=layout
    version=f"tree_{example}_{kind}"
    chosen=trees[example % len(trees)]
    if kind=="learned":
        original=np.asarray(chosen["paths"][0]["xy"])
        size=original[0,1]-original[-1,1]
        spatial=(original-original[0])/size
        guides=np.c_[center[0]+height*(spatial[:,0]+lean*spatial[:,1]**2),center[1]+height*spatial[:,1]]
    else:
        size=1.
        guides=np.asarray([center,[center[0]+5,center[1]-.35*height],
                           [center[0]-4,center[1]-.7*height],[center[0]+lean*height,center[1]-height]])
    points=np.asarray(catmull_rom(guides,8))
    t=_arc(points)/max(_arc(points)[-1],1e-6)
    widths=9*np.interp(t,width_grid,width_profile) if kind=="learned" else 9*(1-.48*t)
    path_id=f"{version}_trunk"
    records=[DigitalStroke(path_id,version,"trunk","trunk_profile",points.tolist(),widths.tolist(),
                           (55,49,42),.82,.25,.30,1,(),
                           {"geometry":"reference trace and handwritten layout" if kind=="learned" else "manual baseline",
                            "width":"multiwork median" if kind=="learned" else "manual taper",
                            "work":chosen["work_id"] if kind=="learned" else None,
                            "placement":"manual composition"})]
    if kind=="learned":
        graph={"main":(path_id,guides)}
        for i,piece in enumerate(chosen["paths"][1:]):
            parent,geometry=graph[piece["parent"]]
            start=geometry[piece["attach_index"]]
            offsets=(np.asarray(piece["xy"])-piece["xy"][0])*(height/size)
            offsets[:,0]*=1.05 if example%2==0 else .89
            controls=start+offsets
            idx=f"{version}_branch{i}"
            graph[piece["id"]]=(idx,controls)
            curve=catmull_rom(controls,8)
            order=1 if piece["parent"]=="main" else 2
            records.append(DigitalStroke(idx,version,f"order_{order}_branch","reference_branch_graph",
                    curve,np.linspace(3.4/order,1.05,len(curve)).tolist(),(55,49,42),.78,.29,.24,2,(parent,),
                    {"branch_geometry":"joint source graph", "work":chosen["work_id"],"width":"manual branch taper",
                     "scaled_layout_and_lean":"manual transform","connection":"source trace hypothesis"}))
    else:
        for i,(at,side) in enumerate(((.46,-1),(.61,1),(.77,-1),(.86,1))):
            start=points[round(at*(len(points)-1))]
            curve=catmull_rom([start,start+[side*19,-10],start+[side*(52-7*i),-28]],8)
            records.append(DigitalStroke(f"{version}_branch{i}",version,"order_1_branch","manual_regular",
                            curve,np.linspace(3.4,1.05,len(curve)).tolist(),(55,49,42),.78,.29,.24,2,(path_id,),
                            {"geometry":"manual branch positions and controls","width":"manual taper"}))
    return records
