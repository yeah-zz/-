# -*- coding: utf-8 -*-
"""Verify frozen inputs, expected evaluation scope and replay determinism."""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from gongbi_training.runtime import LAB, configure
from master_studio.digital_strokes import replay
from recognition_groups_20261123 import SCENE, read, sha, verify_baselines
from recognition_scene_topology_20261124 import write


def observe_replays(plans):
    for path in plans:
        doc=read(path)
        output,frames=replay(path,frames=True)
        with Image.open(path.with_suffix(".png")) as actual:
            assert actual.convert("RGB").tobytes()==output.tobytes(),path
        assert len(frames)==len(doc["strokes"])+1 and len(set(frames[0].getdata()))==1
        for row in doc["strokes"]:
            assert len(row["xy"])==len(row["widths_px"])
            assert all(len(xy)==2 and all(math.isfinite(v) for v in xy) for xy in row["xy"])
        gif=path.with_suffix(".gif")
        if gif.exists():
            with Image.open(gif) as movie:
                assert movie.n_frames==len(frames)
    return len(plans)


def main():
    configure()
    import cv2
    from gongbi_training.cunfa_continuity import join_marks
    from gongbi_training import structure_candidates
    a=LAB/"crosswork_structure_20261124"
    b=LAB/"crosswork_cunfa_20261124"
    d=LAB/"crosswork_drawing_20261124"
    ma=read(a/"structure.json")
    mb=read(b/"cunfa.json")
    md=read(d/"drawing_results.json")
    old=LAB/"visual_structure_round_20261124"
    old_draw=LAB/"digital_specialty_round_20261124"
    assert sha(structure_candidates.__file__)==ma["implementation"]
    assert ma["checkpoint"]==sha(SCENE/"mapping_fixed.pt")==read(old/"protocol.json")["checkpoint"]
    baselines=verify_baselines()
    assert baselines==ma["baselines"]==mb["checkpoints"]==md["baseline_hashes"]==read(old_draw/"results.json")["painting_baselines"]
    cases=ma["results"]
    assert len(cases)==51 and sum(len(r["points_yx"]) for r in cases)==12
    neg=[r for r in cases if r["kind"] in ("rough_rock","root_pending")]
    assert len(neg)==40
    assert [sum(r["modes"][m]["rough_counts"]["tree"] for r in neg) for m in ("A0","A1","A4")]==[2246,3860,2516]
    assert [sum(r["modes"][m]["tree_hits"] for r in cases) for m in ("A0","A1","A4")]==[9,11,11]
    assert len(mb["rows"])==45 and sum(mb["variants"]["group"]["validation"]["confusion"][k][k] for k in (0,1))==18
    reference=read(d/"tree_references.json")
    assert len(reference["samples"])==3
    for r in reference["samples"]:
        assert sha(r["image"])==r["sha256"]
        assert all(x["parent"] in {y["id"] for y in r["paths"][:i]} for i,x in enumerate(r["paths"]) if i)
    assert all(r["work"] not in r["training_works"] for r in md["widths"]+md["branch"])

    gray=np.full((64,64),.89,np.float32)
    left=np.c_[np.arange(13,29),np.full(16,30)]
    right=np.c_[np.arange(37,53),np.full(16,30)]
    gray[29:32,13:29]=.35
    gray[29:32,37:53]=.35
    marks=[(p,np.full(len(p),3.),float(len(p)-1),float(len(p)-1),np.full(len(p),.5)) for p in (left,right)]
    _,blank=join_marks(marks,gray,cv2)
    gray[29:32,29:37]=.42
    paths,evidence=join_marks(marks,gray,cv2)
    assert blank and not any(r["selected"] for r in blank)
    assert len(paths)==1 and sum(r["selected"] for r in evidence)==1
    assert md["painting_strokes"]==len(read(d/"tree_rock.json")["strokes"])==len(read(old_draw/"composition.json")["strokes"])==24
    doc=read(d/"tree_rock.json")
    old_r=read(d/"same_strokes_legacy.json")
    assert doc["strokes"]==old_r["strokes"] and doc["renderer"]!=old_r["renderer"]
    legacy_names=read(old_draw/"results.json")["sequence_sizes"]
    replay_old=observe_replays([old_draw/f"{k}.json" for k in legacy_names])
    replay_new=observe_replays([p for p in d.glob("*.json") if p.with_suffix(".png").exists() and "strokes" in read(p)])
    assert (a/"annotation_browser_check.json").is_file()
    result=dict(legacy_replays=replay_old,new_replays=replay_new,
                baseline_hashes=baselines,blank_gap_rejected=True,dark_gap_joined=True,
                processor_manifest={str(p):sha(p) for p in map(Path,[
                    "scripts/structure_candidate_round.py","scripts/cunfa_group_open_round.py",
                    "scripts/crosswork_drawing_round.py","scripts/crosswork_tree_samples.py",
                    "gongbi_training/cunfa_continuity.py","master_studio/digital_strokes.py",
                    "tools/structure_reference_review.html"])},independent_artwork_validation=False)
    write(d/"verification.json",result)
    print(f"Verified {replay_old} frozen and {replay_new} new JSON replays, baselines, matched evaluation numbers, work isolation and gap evidence.")


if __name__=="__main__":
    main()
