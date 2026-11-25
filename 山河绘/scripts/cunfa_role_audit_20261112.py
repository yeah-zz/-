# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont,ImageEnhance
ROOT=LAB/'cunfa_role_reference_20261112';PREV=LAB/'cunfa_fresh_validation_20261111'
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'audit').mkdir(exist_ok=True)
 d=json.loads((PREV/'test_manifest.json').read_text('utf8'));rows=[r for r in d['records'] if r['work_id']=='230757'];src=Image.open(rows[0]['source_image']).convert('RGB');font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 for r in rows:
  x0,y0,x1,y1=r['source_box'];a=src.crop((x0,y0,x1,y1)).resize((380,420));b=ImageEnhance.Contrast(a).enhance(2.0)
  context_box=[max(0,x0-110),max(0,y0-110),min(src.width,x1+110),min(src.height,y1+110)];ctx=src.crop(context_box);dr=ImageDraw.Draw(ctx);dr.rectangle([x0-context_box[0],y0-context_box[1],x1-context_box[0],y1-context_box[1]],outline='#197a93',width=2);ctx.thumbnail((500,440))
  out=Image.new('RGB',(1320,525),'#f4f1e8');dd=ImageDraw.Draw(out);dd.text((12,7),r['id']+' 原始裁剪 / 固定增强（仅阅读）/ 原作位置',font=font(23),fill='#243e32');out.paste(a,(10,55));out.paste(b,(410,55));out.paste(ctx,(810,55));dd.text((12,482),'增强图不进入模型；馆藏作品标签不能替代局部笔迹归属。',font=font(21),fill='#8c432f');out.save(ROOT/'audit'/(r['id']+'.jpg'),quality=94)
 print('AUDIT_PANELS',len(rows),flush=True)
if __name__=='__main__':main()
