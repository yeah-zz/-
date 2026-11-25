# -*- coding: utf-8 -*-
"""Fit a weak quality screen only to three previously user-accepted outputs.

This is positive image-level feedback, never exact stroke gold. It does not
certify correctness and never fits thresholds to the new holdout predictions.
"""
from evaluate_simple_landscape_specialist_r2 import ROOT,LAB,qa,baseline,read,jsave,digest
from PIL import Image
import numpy as np

def freeze():
    file=ROOT/'accepted_reference_quality.json'
    if file.exists():return read(file)
    rows=[];old=LAB/'generated_simple_landscapes_20261028';accepted=LAB/'split_detail_structure_20261028'/'v4'
    for sid in ['01_rocks','02_pine','03_pavilion']:
        src=old/'originals'/(sid+'.png');path=accepted/sid/'paths.json';im=Image.open(src).convert('RGB');original=np.asarray(im.size);im=im.resize((496,496),Image.Resampling.LANCZOS);rgb=np.asarray(im);doc=read(path)
        scale=np.array([496/original[0],496/original[1]],np.float32);paths=[np.asarray(r['points'],np.float32)*scale for r in doc['paths']];mask=baseline.raster(rgb.shape[:2],paths);q=qa(mask,rgb);q['loop_density_per_1000_pixels']=q['small_closed_regions']/max(1,q['pixel_count'])*1000
        rows.append(dict(id=sid,source=str(src),source_sha256=digest(src),accepted_paths=str(path),accepted_paths_sha256=digest(path),qa=q))
    rules=dict(min_image_support=max(.5,min(r['qa']['image_support'] for r in rows)-.03),min_source_coverage=max(.2,min(r['qa']['source_pixel_coverage'] for r in rows)-.08),max_loop_density_per_1000=max(r['qa']['loop_density_per_1000_pixels'] for r in rows)*1.2+.5)
    doc=dict(reference_records=rows,normalized_long_edge=496,rules=rules,source='three user-accepted v4 extractions; image-level acceptance only',new_holdout_images_or_outputs_used_for_fitting=False,artist_stroke_accuracy_unknown=True,positive_references_are_not_exact_gold=True,method_sha256=digest(__file__))
    jsave(file,doc);print('ACCEPTED_REFERENCE_GATE_FROZEN',rules,flush=True);return doc

def passes_reference(q,config):
    rules=config['rules'];density=q['small_closed_regions']/max(1,q['pixel_count'])*1000
    return q['pixel_count']>5 and q['image_support']>=rules['min_image_support'] and q['source_pixel_coverage']>=rules['min_source_coverage'] and density<=rules['max_loop_density_per_1000']

if __name__=='__main__':freeze()
