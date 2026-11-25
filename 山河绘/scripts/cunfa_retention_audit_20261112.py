# -*- coding: utf-8 -*-
"""Audit scope refinement losses before changing any experiment or labels."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_retention_v4_20261112';PREV=LAB/'cunfa_band_checked_20261112'
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'audit').mkdir(exist_ok=True);manifest=json.loads((PREV/'dataset_manifest.json').read_text('utf8'));rows=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
 for r in manifest['records']:
  z=np.load(r['path']);t=z['core_targets'];im=z['image'];old=np.load(PREV/'predictions'/(r['id']+'_v2_frozen.npz'))['prediction'];new=np.load(PREV/'predictions'/(r['id']+'_frozen_role_refiner.npz'))['prediction'];known=t!=255
  dropped=known&(t>0)&(old>0)&(new==0);newly_lost_correct=known&(t>0)&(old==t)&(new!=t);semantic_flip=known&(old>0)&(new>0)&(old!=new);recovered=known&(t>0)&(old==0)&(new>0);gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype('float32');contrast=cv2.GaussianBlur(gray,(0,0),4)-gray
  points=np.column_stack(np.nonzero(dropped)[::-1]);n,labels,stats,_=cv2.connectedComponentsWithStats(dropped.astype('uint8'),8);components=[]
  for k in range(1,n):
   x,y,w,h,area=stats[k];components.append(dict(box=[int(x),int(y),int(x+w),int(y+h)],pixels=int(area)))
  components.sort(key=lambda c:c['pixels'],reverse=True);info=dict(id=r['id'],split=r['split'],source_family=r['source_family'],newly_suppressed_ink=int(dropped.sum()),newly_lost_previously_correct_by_role={str(k):int((newly_lost_correct&(t==k)).sum()) for k in [1,2]},recovered_ink=int(recovered.sum()),retained_ink_semantic_flip=int(semantic_flip.sum()),median_lost_local_contrast=float(np.median(contrast[dropped])) if dropped.any() else None,components=components,all_losses_from_foreground_suppression=not semantic_flip.any(),reference_labels_changed=False);rows.append(info)
  overlay=im.copy();overlay[dropped]=[215,48,48];overlay[recovered]=[35,150,80];board=Image.new('RGB',(1200,430),'#f4f1e8');draw=ImageDraw.Draw(board)
  for j,(image,title) in enumerate([(Image.fromarray(im),'原始画面'),(Image.fromarray(overlay),'红=新增遗漏，绿=找回（参考内）'),(Image.fromarray((dropped*255).astype('uint8')).convert('RGB'),'遗漏位置 / 只用于复核')]):
   image.thumbnail((380,340));board.paste(image,(j*400+6,38));draw.text((j*400+6,5),title,font=font,fill='#243e32')
  draw.text((8,393),r['id']+'：参考参与遗漏统计；不向推理提供参考。该画面不是新盲测。',font=font,fill='#815238');board.save(ROOT/'audit'/(r['id']+'.jpg'),quality=95)
 report=dict(records=rows,previous_results_sha256=hashlib.sha256((PREV/'results.json').read_bytes()).hexdigest(),diagnostic_repeated=True,old_references_unchanged=True,robot_connected=False);(ROOT/'retention_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(rows,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
