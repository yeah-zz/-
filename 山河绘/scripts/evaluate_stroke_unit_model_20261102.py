# -*- coding: utf-8 -*-
"""Batch offline transfer tests. Pixel candidates stay frozen; relation model is new."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import argparse,hashlib,json,time
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from stroke_unit_core_20261102 import process,union_raster,length
from train_stroke_unit_model_20261102 import ROOT,RelationNet,probabilities


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def catalog():
    rows=[]
    for p in sorted((LAB/'mountain_complete_lines_20261101'/'outputs').glob('G*/complete_paths.json')):
        rows.append(dict(id=p.parent.name,image=str(p.parent/'input.png'),paths=str(p),
            split='development' if p.parent.name in ['G03_01','G01_01','G05_03'] else 'old_scene_transfer',
            upstream='frozen complete extraction 496px',is_new_generated_image=False))
    for p in sorted((LAB/'native_fine_faint_20261101'/'outputs_r3').glob('T*/paths.json')):
        rows.append(dict(id=p.parent.name,image=str(LAB/'stability_brush_preview_20261101'/'originals'/(p.parent.name+'.png')),
            paths=str(p),split='native_transfer',upstream='frozen native fine/faint r3',is_new_generated_image=False))
    for p in sorted((LAB/'faint_center_direction_20261101'/'fresh_outputs').glob('N*/paths.json')):
        rows.append(dict(id='F_'+p.parent.name,image=str(LAB/'faint_center_direction_20261101'/'fresh_inputs'/(p.parent.name+'.png')),
            paths=str(p),split='locked_final_transfer',upstream='frozen faint center/direction fusion',is_new_generated_image=False))
    for r in rows:
        r['image_sha256']=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()
        r['paths_sha256']=hashlib.sha256(Path(r['paths']).read_bytes()).hexdigest()
    return rows


def render(shape,paths):
    return Image.fromarray(np.where(union_raster(shape,paths),0,255).astype(np.uint8)).convert('RGB')


def evaluate(rows):
    cv2.setNumThreads(2);torch.set_num_threads(4)
    model=RelationNet().cuda();saved=torch.load(ROOT/'model.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model']);model.eval()
    thresholds=json.loads((ROOT/'thresholds.json').read_text('utf-8'))
    infer=lambda x:probabilities(model,x)
    all_stats=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for record in rows:
        out=ROOT/'outputs'/record['id'];out.mkdir(parents=True,exist_ok=True)
        rgb=np.array(Image.open(record['image']).convert('RGB'));shape=rgb.shape[:2]
        d=json.loads(Path(record['paths']).read_text('utf-8'))
        old=[np.array(p['points'] if isinstance(p,dict) else p,float) for p in d['paths']]
        start=time.monotonic();new,decisions,stats=process(rgb,old,infer,thresholds)
        raw_scores=decisions.pop('probability_arrays');np.savez_compressed(out/'relation_probabilities.npz',**raw_scores)
        save(out/'decisions.json',decisions)
        newpaths=[r['points'] for r in new]
        old_mask=union_raster(shape,old);new_mask=union_raster(shape,newpaths)
        change=np.repeat(np.where(old_mask,0,255).astype(np.uint8)[:,:,None],3,axis=2)
        change[new_mask&~old_mask]=[30,120,200];change[old_mask&~new_mask]=[210,65,65]
        Image.fromarray(rgb).save(out/'source.png');render(shape,old).save(out/'old.png')
        render(shape,newpaths).save(out/'new.png');Image.fromarray(change).save(out/'changes.png')
        # Preserve every curve in the complete output. Short units are flagged,
        # not silently exported as ready-to-execute brush strokes.
        factor=496/max(shape)
        save(out/'curves.json',dict(paths=[dict(points=r['points'].tolist(),source_ids=r['source_ids'],
            short_unit_pending=length(r['points'])*factor*186.8/495<2.,artist_stroke_identity_verified=False) for r in new],
            coordinate_system='source_image_pixels',width=shape[1],height=shape[0],
            ready_for_robot=False,actually_drawn=False,physical_z=None,physical_speed=None))
        stats.update(id=record['id'],split=record['split'],inference_seconds=time.monotonic()-start,
            upstream=record['upstream'],source_sha256=record['image_sha256'],upstream_paths_sha256=record['paths_sha256'],
            partial_labels_read_at_inference=False,robot_connected=False)
        save(out/'statistics.json',stats);all_stats.append(stats)
        board=Image.new('RGB',(1536,550),'#f5f3eb');draw=ImageDraw.Draw(board)
        for i,(im,title) in enumerate([(Image.fromarray(rgb),'原图'),(render(shape,old),f'旧像素段 {len(old)}'),
                (render(shape,newpaths),f'新关系模型＋墨带检查 {len(newpaths)}')]):
            im.thumbnail((496,496),Image.Resampling.LANCZOS)
            draw.text((i*512+6,6),record['id']+' '+title,font=font,fill='black');board.paste(im,(i*512,45))
        board.save(out/'comparison.jpg',quality=96)
        print('RESULT',json.dumps(stats,ensure_ascii=False),flush=True)
    save(ROOT/('results_'+('_'.join(r['id'] for r in rows) if len(rows)<10 else 'batch')+'.json'),dict(records=all_stats))
    return all_stats


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--ids',nargs='*');a=p.parse_args()
    rows=catalog()
    if a.development:rows=[r for r in rows if r['split']=='development']
    if a.ids:rows=[r for r in rows if r['id'] in a.ids]
    save(ROOT/'image_catalog.json',dict(records=catalog(),new_image_generation_calls=0))
    evaluate(rows)
