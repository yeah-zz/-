# -*- coding: utf-8 -*-
import sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'local_rock_patch_20261117';APPROVED=LAB/'drawing_budgets_20261105/G03_01_500';WORKING=LAB/'structural_roles_20261107/G03_01'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'baseline_lock.json').exists();a=json.loads((APPROVED/'plan.json').read_text('utf8'));b=json.loads((WORKING/'plan.json').read_text('utf8'));assert len(a['curves'])==500 and len(b['curves'])==489
 source=Path(b['source']);assert sha(source)==a['source_sha256']==b['source_sha256'];ids=[i for i,r in enumerate(b['curves']) if r['module']==4];assert len(ids)==21
 lock=dict(approved500=str(APPROVED/'plan.json'),approved500_sha256=sha(APPROVED/'plan.json'),working489=str(WORKING/'plan.json'),working489_sha256=sha(WORKING/'plan.json'),source=str(source),source_sha256=sha(source),selected_module=4,selected_name=b['modules'][4]['name'],selected_curve_ids=ids,
  source_box=[108,368,215,417],coordinate_system='496x496 source image pixels',all_source_plans_immutable=True,
  no_curve_insertion_deletion_join=True,no_training_labels_from_baseline=True,experiment='image-constrained local curve fitting and stroke-band reconstruction; no new recognition training',adopted=False,robot_connected=False)
 (ROOT/'baseline_lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2),encoding='utf8');(ROOT/'baseline489.json').write_text(json.dumps(b,ensure_ascii=False,indent=2),encoding='utf8')
 im=Image.open(source).convert('RGB');im.save(ROOT/'source.png');im.crop(lock['source_box']).resize((856,392),Image.Resampling.NEAREST).save(ROOT/'rock_source.png')
 board=Image.new('RGB',(1400,1020),'#f3f0e7');dr=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
 for n,i in enumerate(ids):
  r=b['curves'][i];crop=im.crop(lock['source_box']).resize((214,98),Image.Resampling.NEAREST);pd=ImageDraw.Draw(crop);p=np.array(r['points']);q=(p-np.array(lock['source_box'][:2]))*2;pd.line([tuple(v) for v in q],fill='#d44231',width=2)
  x=n%5*280+5;y=n//5*200+5;board.paste(crop,(x,y));dr.text((x,y+110),str(i)+' '+r.get('stroke_role','unknown'),font=font,fill='#243f35');dr.text((x,y+143),'phase '+str(r['phase']),font=font,fill='#805333')
 board.save(ROOT/'selected_paths.jpg');print('LOCKED',len(ids),flush=True)
if __name__=='__main__':main()
