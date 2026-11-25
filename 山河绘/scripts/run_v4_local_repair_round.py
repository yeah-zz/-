"""Grouped fixed comparison of source-supported local repairs against frozen v4."""
import sys,json,hashlib,argparse,html,shutil
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
import v4_local_repair_candidates as repair
from audit_v4_local_repairs import OUT,R2,dump,sha
import split_detail_v4_frozen as v4

def save_pending(target,doc,raw,new):
    left=raw&~new
    Image.fromarray(np.where(left,0,255).astype(np.uint8)).save(target/'pending.png')
    all_evidence=np.repeat(np.where(new,0,255).astype(np.uint8)[:,:,None],3,2)
    all_evidence[left]=[185,185,185];Image.fromarray(all_evidence).save(target/'all_evidence.png')
    # Original ridge-only pending paths are historical, some may now be selected.
    doc['baseline_ridge_pending_paths_for_history']=doc.pop('pending_paths',[])
    doc.update(unresolved_original_skeleton_mask='pending.png',pending_paths_cover_ridge_candidates_only=False,
               pending_geometry_storage='complete unresolved source-skeleton pixel mask; no stroke identity assigned',
               all_evidence_preview='all_evidence.png')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--groups',default='1,2,3');args=parser.parse_args()
    groups=list(map(int,args.groups.split(',')));manifest=json.loads((OUT/'baseline_manifest.json').read_text(encoding='utf-8'))
    records=[r for r in manifest['records'] if r['group'] in groups];results=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for r in records:
        source=Path(r['baseline']);target=OUT/'candidate'/r['id'];target.mkdir(parents=True,exist_ok=True)
        rgb=np.asarray(Image.open(source/'input.png').convert('RGB'));raw=np.asarray(Image.open(source/'raw_skeleton.png').convert('L'))<128
        doc=json.loads((source/'paths.json').read_text(encoding='utf-8'));original=doc['paths']
        additions,fan,decisions=repair.recover(rgb,original,raw)
        base=v4.raster(raw.shape,[np.asarray(p['points'],np.float32) for p in original]);added=v4.raster(raw.shape,[np.asarray(p['points'],np.float32) for p in additions]);new=base|added
        assert not (base&~new).any()
        bw=lambda mask:np.where(mask,0,255).astype(np.uint8)
        Image.fromarray(bw(new)).save(target/'lines.png')
        diff=np.repeat(bw(base)[:,:,None],3,2);diff[added&~base]=[0,135,190];Image.fromarray(diff).save(target/'changes.png')
        overlay=rgb.copy();overlay[added&~base]=[0,110,235];Image.fromarray(overlay).save(target/'additions_on_source.png')
        fan_overlay=rgb.copy();fan_overlay[fan]=(fan_overlay[fan]*.6+np.array([60,180,70])*.4).astype(np.uint8);Image.fromarray(fan_overlay).save(target/'bundle_detection.png')
        doc.update(paths=original+additions,config=dict(baseline=doc.get('config'),local_repair=repair.CFG),ready_for_robot=False,actually_drawn=False,reference_labels_used=False,new_training_steps=0)
        save_pending(target,doc,raw,new)
        dump(target/'paths.json',doc);dump(target/'decisions.json',decisions)
        row=dict(id=r['id'],split=r['split'],baseline_paths=len(original),added_paths=len(additions),baseline_pixels=int(base.sum()),added_pixels=int((added&~base).sum()),removed_pixels=int((base&~new).sum()),original_paths_exactly_preserved=doc['paths'][:len(original)]==original)
        results.append(row);dump(target/'statistics.json',row)
        canvas=Image.new('RGB',(1600,450),'#f3f1eb');draw=ImageDraw.Draw(canvas)
        for j,(name,label) in enumerate([(source/'input.png','原图'),(source/'lines.png','冻结 v4'),(target/'lines.png','局部补回候选'),(target/'changes.png','青色：新增 / 红色：删除')]):
            draw.text((400*j+6,6),r['id']+' '+label,font=font,fill='black');im=Image.open(name).convert('RGB');im.thumbnail((394,394));canvas.paste(im,(400*j,38))
        canvas.save(target/'comparison.jpg',quality=94)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    dump(OUT/('results_groups_'+args.groups.replace(',','_')+'.json'),dict(config=repair.CFG,records=results,code_sha256=sha(Path(repair.__file__)),scope='Automatic image-evidence decisions, not precision/recall or expert acceptance. Existing v4 paths immutable; no new neural weight training.'))
if __name__=='__main__':main()
