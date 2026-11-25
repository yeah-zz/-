"""Quarantine an ambiguous assistant reference, preserving pre-inference files."""
import sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from guohua_boundary_reference_20261116 import ROOT,read,write,sha
refs=copy.deepcopy(read(ROOT/'references.json'));results=copy.deepcopy(read(ROOT/'results.json'))
review=dict(original_reference_sha256=sha(ROOT/'references.json'),original_results_sha256=sha(ROOT/'results.json'),after_prediction_review=True,reason='Source image visual recheck: foreground rock face may mix pine trunk and rock; do not treat it as confirmed rock.',changes=[dict(id='dai-jin_returning-late-from-a-spring-outing_6',region='foreground_rock_face',old_object='rock',new_object='unknown',action='quarantine from all metrics and future training, coordinates unchanged')],model_changed=False,expert_verified=False)
for c in refs['cases']:
 if c['id']==review['changes'][0]['id']:
  r=next(r for r in c['regions'] if r['name']=='foreground_rock_face');r.update(object='unknown',kind='uncertain_polygon',note='二次目视核查仍难区分松树干与岩面，降为待定，不作山石真值')
for c in results['records']:
 if c['id']==review['changes'][0]['id']:
  r=next(r for r in c['regions'] if r['name']=='foreground_rock_face');r.update(object='unknown',kind='uncertain_polygon',scores=None,note='二次核查降为待定；不参与计分')
refs.update(frozen_before_new_predictions=False,review_revision='v2_quarantine_only',original_pre_prediction_reference_sha256=review['original_reference_sha256'],review_after_predictions_disclosed=True)
colors={'rock':(183,134,75),'tree':(45,161,91),'unknown':(196,64,162)}
for c in refs['cases']:
 im=Image.open(c['image']).convert('RGBA');core=Image.new('L',im.size,255)
 for r in sorted(c['regions'],key=lambda r:r['object']!='unknown'):
  mask=Image.new('L',im.size,0);d=ImageDraw.Draw(mask)
  if r['kind']=='contour_corridor':d.line([tuple(p) for p in r['points']],fill=255,width=r['corridor_width_px'],joint='curve')
  else:d.polygon([tuple(p) for p in r['points']],fill=255)
  layer=Image.new('RGBA',im.size,(*colors[r['object']],0));layer.putalpha(mask.point(lambda a:a*85//255));im=Image.alpha_composite(im,layer);d=ImageDraw.Draw(im);d.line([tuple(p) for p in r['points']]+([] if r['kind']=='contour_corridor' else [tuple(r['points'][0])]),fill=(*colors[r['object']],230),width=2)
  if r['kind']=='core_polygon':core.paste(1 if r['object']=='rock' else 2,mask=mask)
 im.convert('RGB').save(ROOT/'views'/(c['id']+'_reviewed_reference.jpg'),quality=95);core.save(ROOT/'masks'/(c['id']+'_reviewed_sparse_core.png'));next(r for r in results['records'] if r['id']==c['id'])['reference']='views/'+c['id']+'_reviewed_reference.jpg'
write('reference_review.json',review);write('reviewed_references.json',refs);results['reference_sha256']=sha(ROOT/'reviewed_references.json');results['original_results_sha256']=review['original_results_sha256'];write('reviewed_results.json',results)
board=Image.new('RGB',(1280,4*360+75),'#f4f1e8');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
for i,r in enumerate(results['records']):
 for j,(path,title) in enumerate([(r['input'],'原图'),(r['reference'],'助手参考：棕山石／绿树木／粉待定'),(r['predictions']['expanded'],'冻结模型：上轮扩充数据'),(r['predictions']['core_loss'],'冻结模型：失败的新约束')]):
  im=Image.open(ROOT/path);im.thumbnail((308,280));board.paste(im,(j*320+5,i*360+42));d.text((j*320+5,i*360+8),title,font=font,fill='#294735')
 d.text((5,i*360+327),r['id'][:95],font=font,fill='#79523a')
d.text((5,4*360+6),'25处候选：20处对象归属参考、5处待定；非专家真值、精确墨带或模型提取结果。',font=font,fill='#79523a');board.save(ROOT/'reference_comparison.jpg',quality=94)
print(json.dumps(review,ensure_ascii=False))
