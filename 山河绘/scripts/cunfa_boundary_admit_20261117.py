# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image
ROOT=LAB/'cunfa_boundary_relations_20261117'
def main():
 assert not (ROOT/'reference_manifest.json').exists();d=json.loads((ROOT/'candidate_references.json').read_text('utf8'));rows=d['records'];changes=[]
 for r in rows:
  if r['id']=='nizan_4':
   r.update(context_role_target=None,role_reference_checked=False);r['visual_evidence']+='；近看有大量细墨迹，淡墨标签改为待定';changes.append(dict(id=r['id'],reason='Mixed fine marks; no diffuse-tone target'))
  if r['id']=='nizan_7':
   # First crop included a red seal at its edge; replace before any fitting.
   im=Image.open(r['reference_image']);r['box']=[round(v*im.size[j%2]/[772,1400][j%2]) for j,v in enumerate([512,686,548,711])];im.crop(r['box']).save(r['image']);r['image_sha256']=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest();r['visual_evidence']='水面未着墨局部；原候选边缘含印章，已在预测前重选';changes.append(dict(id=r['id'],reason='Repositioned paper crop away from seal before predictions'))
 for c in d['curves']:
  c['checked']=True
  if c['id']=='qingpu_26_new_0':c['open']=[];changes.append(dict(id=c['id'],reason='Left side has pale mountain plane; do not label it open space'))
 d.update(changes_before_predictions=changes,status='frozen assistant sparse references, not expert ink edges',reference_input_prohibited=True,production_replaced=False,robot_connected=False)
 (ROOT/'reference_manifest.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8');print('ADMITTED',len(d['curves']))
if __name__=='__main__':main()
