"""Paired fixed-config v4/v5 run on the preselected new image set."""
import stroke_completeness_v5 as new
import split_detail_v4_frozen as old
from rock_v15_common import read,jsave,digest
from pathlib import Path
import json,time,datetime,sys
import numpy as np,cv2
from scipy.ndimage import distance_transform_edt
from PIL import Image
ROOT=new.ROOT

def freeze():
    files=[Path(new.__file__),Path(old.__file__),Path(new.__file__).with_name('source_ink_baseline_20261028.py')]
    cfg=ROOT/'fixed_protocol.json'
    if cfg.exists():
        doc=read(cfg)
        for r in doc['code']:
            if digest(r['path'])!=r['sha256']:raise RuntimeError('Frozen extraction code changed: '+r['path'])
        return doc
    doc=dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),config=new.CFG,code=[dict(path=str(p),sha256=digest(p)) for p in files],selection_manifest_sha256=digest(ROOT/'validation_manifest.json'),new_images_used_for_parameter_selection=False,algorithm_updates_are_classical_not_neural=True,parameters_fixed_for_all_validation_images=True,manual_stroke_gold=False,reference_labels_read=False,showcase_ids_prespecified=['cma_132840','met_63207','met_36055'])
    jsave(cfg,doc)
    for p in files:
        dest=ROOT/'frozen_code'/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    return doc

def metric(folder):
    data=read(folder/'paths.json');rgb=np.asarray(Image.open(folder/'input.png').convert('RGB'))
    # Analyze exact pixel geometry; thresholding antialiased display PNGs can
    # break diagonal strokes and falsely turn every small cycle into an opening.
    selected=new.raster(rgb.shape[:2],[np.asarray(p['points'],np.float32) for p in data['paths']])
    raw=np.asarray(Image.open(folder/'raw_skeleton.png').convert('L'))<128
    colored=np.asarray(Image.open(folder/'dense_regions.png').convert('RGB'));dense=np.any(colored!=rgb,axis=2)
    distances=distance_transform_edt(~selected);skeleton_coverage=float((distances[raw]<=2).mean()) if raw.any() else 1.
    n,labels,stats,_=cv2.connectedComponentsWithStats((~selected).astype(np.uint8),connectivity=4)
    small=0
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area>64 or x==0 or y==0 or x+w==rgb.shape[1] or y+h==rgb.shape[0]:continue
        ys,xs=np.where(labels[y:y+h,x:x+w]==i);ys+=y;xs+=x
        if not dense[ys,xs].any():small+=1
    structures=[np.asarray(p['points']) for p in data['paths'] if p['role']=='long_structure']
    lengths=[new.plen(p) for p in structures]
    return dict(source_skeleton_retention_within_2px=skeleton_coverage,structure_path_count=len(structures),median_structure_length=float(np.median(lengths)) if lengths else 0.,small_enclosed_regions_outside_detail=small,selected_center_pixels=int(selected.sum()),detail_center_pixels=int((np.asarray(Image.open(folder/'detail_lines.png').convert('L'))<128).sum()),accepted_joins=len(data['joins']),statistics=read(folder/'statistics.json'))

if __name__=='__main__':
    freeze();rows=read(ROOT/'validation_manifest.json')['records'];results=[]
    for i,r in enumerate(rows):
        freeze();source=Path(r['selected_image'])
        if digest(source)!=r['working_sha256']:raise RuntimeError('Validation input changed')
        item=dict(id=r['id'],source=r['source_page'],title=r['title'],source_sha256=r['working_sha256']);start=time.perf_counter()
        for name,module in [('v4',old),('v5',new)]:
            folder=ROOT/'validation_outputs'/r['id']/name
            if not (folder/'statistics.json').exists():module.run(source,folder)
            item[name]=metric(folder)
        item['elapsed_seconds']=time.perf_counter()-start
        results.append(item);jsave(ROOT/'validation_results.json',dict(records=results,complete=len(results)==len(rows),expected_count=len(rows),parameters_not_tuned_on_validation=True,metrics_are_no_gold_diagnostics_not_accuracy=True))
        print('VALIDATION_PAIR',i+1,len(rows),r['id'],'seconds',round(item['elapsed_seconds'],1),'cycles',item['v4']['small_enclosed_regions_outside_detail'],item['v5']['small_enclosed_regions_outside_detail'],flush=True)
    print('VALIDATION_COMPLETE',len(results),flush=True)
