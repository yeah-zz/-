"""Match source structures and evaluate tree/rock evidence on continuous paths."""

import numpy as np
from scipy.spatial import cKDTree

from gongbi_training.scene_topology import continuations, ink_paths


def rolling(values, radius=4):
    kernel = np.ones(2*radius+1) / (2*radius+1)
    return np.convolve(np.pad(values, radius, mode="edge"), kernel, mode="valid")


def candidate_fusion(rgb, cues, mean, detail, cv2):
    """Return six ablations and per-path source/evidence records in image coordinates."""
    margin = mean[2]-mean[1]
    relevant = np.maximum(mean[2],mean[1]) >= np.maximum(mean[0],mean[3])
    shape = margin.shape
    both = cues["dark"] | cues["light"] | cues["paired"]

    def classify(support, signed=margin):
        return np.where(support & relevant, np.where(signed>=.18,2,np.where(signed<=-.18,1,255)),255).astype(np.uint8)

    modes = {"A0":(cues["dark"],classify(cues["dark"])),"A1":(both,classify(both))}
    names = ("dark","light","paired")
    lab = cv2.cvtColor(rgb.astype(np.float32)/255, cv2.COLOR_RGB2LAB)/100
    feature_maps = [np.full(shape,np.nan,np.float32) for _ in range(3)]
    geometry = []
    seen = np.zeros(shape,bool)
    for name in names:
        skeleton=cues[name]
        for path in ink_paths(skeleton):
            xy=path[:,::-1]
            at=tuple(path.T)
            local=margin[at]
            length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
            tangents=np.gradient(xy.astype(float),axis=0)
            norm=np.maximum(np.linalg.norm(tangents,axis=1),1e-6)
            unit=tangents/norm[:,None]
            rock=float(mean[1][at].mean())
            tree=float(mean[2][at].mean())
            width=cues["half_width"][at].astype(float)*2
            signal=cues["pair_score"][at]
            paired_path=(signal>=.034)&(width>0)
            hue=lab[at]
            color=float(np.mean(np.linalg.norm(np.diff(hue,axis=0),axis=1)))
            turn=float(np.mean(1-np.abs(np.sum(unit[:-1]*unit[1:],axis=1))))
            east=np.pi*cues["angle"][at]/8
            tangent=np.c_[-np.sin(east),np.cos(east)]
            along_pair=paired_path & (np.abs(np.sum(unit*tangent,axis=1))>=.78)
            valid_width=np.where(along_pair,width,0)
            widths=valid_width[valid_width>0]
            width_cv=float(widths.std()/widths.mean()) if widths.size else None
            boundary_turn=None
            if np.count_nonzero(along_pair)>=2:
                geometry_offsets=np.c_[np.cos(east),np.sin(east)]*valid_width[:,None]*.5
                centers=xy[along_pair]
                offsets=geometry_offsets[along_pair].copy()
                for i in range(1,len(offsets)):
                    if offsets[i]@offsets[i-1]<0: offsets[i]*=-1
                delta=np.diff(centers,axis=0)
                a=np.diff(centers-offsets,axis=0)
                b=np.diff(centers+offsets,axis=0)
                dl=np.maximum(np.linalg.norm(delta,axis=1),1e-5)
                ca=np.sum(a*delta,axis=1)/(np.maximum(np.linalg.norm(a,axis=1),1e-5)*dl)
                cb=np.sum(b*delta,axis=1)/(np.maximum(np.linalg.norm(b,axis=1),1e-5)*dl)
                boundary_turn=float(np.minimum(ca,cb).mean())
            geometry.append(dict(id=len(geometry),source=name,path=path,xy=xy,unit=unit,length=length,
                                 rock=rock,tree=tree,local=local,width_cv=width_cv,color=color,turn=turn,
                                 boundary_direction=boundary_turn,mean_width=float(widths.mean()) if widths.size else None,
                                 paired_fraction=float(along_pair.mean()),
                                 side_evidence=float(signal.mean()),chroma_evidence=float(cues["chroma_score"][at].mean()),
                                 ridge_evidence=float(cues["ridge_score"][at].mean()),
                                 dark_fraction=float(cues["dark"][at].mean()),matched=[]))
            seen[at]=True
    # Compare each source pair at original resolution. Nearest-point candidates
    # must follow the same direction and overlap for a substantial part of a path.
    for earlier,later in (("dark","light"),("dark","paired"),("light","paired")):
        rows=[g for g in geometry if g["source"]==earlier and g["length"]>=5]
        if not rows:
            continue
        points=np.concatenate([g["xy"] for g in rows])
        dirs=np.concatenate([g["unit"] for g in rows])
        owner=np.repeat([g["id"] for g in rows],[len(g["xy"]) for g in rows])
        lookup=cKDTree(points)
        for item in (g for g in geometry if g["source"]==later and g["length"]>=5):
            distance,index=lookup.query(item["xy"])
            correct=(distance<=2.25) & (np.abs(np.sum(item["unit"]*dirs[index],axis=1))>=.80)
            targets,count=np.unique(owner[index[correct]],return_counts=True)
            if not len(targets):
                continue
            best=int(targets[count.argmax()])
            mate=geometry[best]
            support=correct&(owner[index]==best)
            expected=item["mean_width"] or mate["mean_width"]
            widths_agree=not expected or not item["mean_width"] or not mate["mean_width"] or (
                min(item["mean_width"],mate["mean_width"])>=.55*max(item["mean_width"],mate["mean_width"]))
            overlap=float(support.mean())
            if overlap>=.48 and widths_agree:
                item["matched"].append((best,overlap))
                mate["matched"].append((item["id"],overlap))
                if later!="paired" or mate["side_evidence"]>=.025:
                    item["duplicate"] = item.get("duplicate",np.zeros(len(support),bool)) | support
    kept=both&~seen
    for g in geometry:
        unique = ~g.get("duplicate",np.zeros(len(g["path"]),bool))
        g["unique"]=unique
        kept[tuple(g["path"][unique].T)]=True
    modes["A2"]=(kept,classify(kept))
    links=continuations([g["path"] for g in geometry])
    for g in geometry:
        points=g["path"]
        at=tuple(points.T)
        local = np.where(np.abs(g["local"])>=.4, g["local"], .65*g["local"]+.35*rolling(g["local"]))
        persistent=(g["length"]>=13 and g["paired_fraction"]>=.42 and g["color"]<=.15
                    and (g["width_cv"] is None or g["width_cv"]<=.65)
                    and (g["boundary_direction"] is None or g["boundary_direction"]>=.45))
        multiple=bool(g["matched"])
        # Weak new responses remain candidates with explicit uncertain ownership.
        eligible = g["source"]=="dark" or persistent or multiple or (g["length"]>=9 and g["tree"]>=.72)
        signed=local.copy()
        if not eligible:
            signed=np.minimum(signed,.12)
        detail_margin=detail[2][at]-detail[1][at]
        rival=(detail_margin<-.10)&(mean[1][at]>=.20)
        veto=(rolling(rival.astype(float))>=.55) | (
            (g["source"]!="dark" and not persistent and g["rock"]>.28) & (local<.48))
        refined=np.where(veto & (np.abs(signed)<.63),np.minimum(signed,.10),signed)
        parents=[links.get((g["id"],side)) for side in (0,1)]
        resolved=refined.copy()
        for side,other in enumerate(parents):
            if other is None:
                continue
            neighbor=geometry[other[0]]
            # A geometric connection can pass evidence only when the two
            # segments are compatible; an opposing rock path is not a parent.
            if neighbor["source"]!=g["source"] or neighbor["tree"]-neighbor["rock"]<.36:
                continue
            idx=np.arange(min(8,len(points))) if side==0 else np.arange(max(0,len(points)-8),len(points))
            allowed=(local[idx]>=.02)&(detail_margin[idx]>=-.08)&~veto[idx]
            resolved[idx[allowed]]+= 0.10 if eligible else 0.05
        for field,values in zip(feature_maps,(signed,refined,resolved)):
            pixel=tuple(points[g["unique"]].T)
            previous=field[pixel]
            next_values=values[g["unique"]]
            # Overlapping sections with opposing roles abstain at the junction.
            field[pixel]=np.where(np.isnan(previous),next_values,
                np.where((previous>=.18)&(next_values<=-.18)|(previous<=-.18)&(next_values>=.18),
                         0., np.where(np.abs(previous)>=np.abs(next_values),previous,next_values)))
        g["persistent_pair"]=persistent
        g["geometric_links"]=parents
    for mode,field in zip(("A3","A4","A5"),feature_maps):
        modes[mode]=(kept,classify(kept,np.where(np.isnan(field),margin,field)))
    return modes, geometry
