import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,numpy as np,torch
from PIL import Image
from gongbi_training.narrow_select import load_clip
ROOT=LAB/'object_hierarchy_20261107';SRC=LAB/'module_stroke_planner_20261104/outputs/G03_01'
def main():
 ROOT.mkdir(exist_ok=True);torch.set_num_threads(4)
 clip,net,transform=load_clip()
 names=['tree','pavilion','rock','water','background']
 prompts=['a traditional Chinese painting of a pine tree, branches and needles','a traditional Chinese painting of a pavilion building with a roof and columns','a traditional Chinese painting of a blue green rock or a mountain cliff','a traditional Chinese painting of water ripples on a river','empty white paper background']
 rgb=np.array(Image.open(SRC/'source.png').convert('RGB'));meta=json.loads((SRC/'sam_manifest_seeded.json').read_text('utf8'));masks=np.load(SRC/'sam_masks_seeded.npz');rows=[];views=[]
 for r in meta['records']:
  m=masks[r['key']];yy,xx=np.where(m);box=(max(0,xx.min()-10),max(0,yy.min()-10),min(496,xx.max()+11),min(496,yy.max()+11))
  isolated=rgb.copy();isolated[~m]=255
  views.append(transform(Image.fromarray(isolated).crop(box)));rows.append(dict(**r,box=box))
 with torch.no_grad():
  text=net.encode_text(clip.tokenize(prompts).cuda());text/=text.norm(dim=-1,keepdim=True);scores=[]
  for start in range(0,len(views),16):
   f=net.encode_image(torch.stack(views[start:start+16]).cuda());f/=f.norm(dim=-1,keepdim=True);scores.extend((f@text.T).float().cpu().numpy())
 for r,s in zip(rows,scores):
  r['scores']=dict(zip(names,map(float,s)));r['class']=names[int(np.argmax(s))];print(r['key'],r['area'],r['box'],r['class'],r['scores'],flush=True)
 (ROOT/'clip_mask_roles.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else list(v)),encoding='utf8')
if __name__=='__main__':main()
