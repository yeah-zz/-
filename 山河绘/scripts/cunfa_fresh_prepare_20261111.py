# -*- coding: utf-8 -*-
import sys,json,hashlib,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_fresh_validation_20261111'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
SPECS=[
 ('231531','披麻皴',[[110,235,185,315],[195,250,270,330],[95,555,175,635],[30,640,105,720]],'Museum work description explicitly names hemp fibre for mountains; cropped faces reviewed by assistant, not expert.'),
 ('234350','斧劈皴',[[210,300,320,400],[58,515,135,605],[80,680,170,790],[70,805,165,905]],'Museum work description explicitly names axe-cut on rocks; plum branches and birds are not intended class targets.'),
 ('230757','斧劈皴',[[725,305,820,410],[835,290,935,400],[190,835,285,920],[55,705,145,805]],'Museum describes small axe-cut and short blunt-brush rock marks; exclude snow/sky/buildings/trees where dominant.')]
def main():
 assert not (ROOT/'predictions.json').exists(),'Do not revise samples after prediction.'
 sources={r['id']:r for r in json.loads((ROOT/'source_manifest.json').read_text('utf8'))};npm=json.loads((ROOT/'npm_source_manifest.json').read_text('utf8'));freeze=json.loads((ROOT/'freeze_manifest.json').read_text('utf8'));(ROOT/'crops').mkdir(exist_ok=True);rows=[]
 def add(key,label,artist,work,path,box,url,scope,extra=None):
  source=Path(path);im=Image.open(source).convert('RGB');x0,y0,x1,y1=box;assert 0<=x0<x1<=im.width and 0<=y0<y1<=im.height
  ident='fresh_'+key+'_'+str(sum(r['work_id']==work for r in rows));dest=ROOT/'crops'/(ident+'.png');im.crop(box).save(dest)
  assert artist not in freeze['train_authors']
  rows.append(dict(id=ident,work_id=work,label=label,artist_group=artist,publisher='NPM' if key.startswith('npm') else 'DPM',image=str(dest),source_image=str(source),source_box=box,
   source_url=url,crop_sha256=sha(dest),source_sha256=sha(source),label_scope=scope,label_status='source-backed provisional; assistant local visual screening, no expert certification',
   expert_local_review=False,training_eligible=False,is_augmentation=False,**(extra or {})))
 for key,label,boxes,scope in SPECS:
  s=sources[key];assert label in s['article'].replace(' ','')
  for b in boxes:add(key,label,s['artist'],key,s['images'][0]['path'],b,s['url'],scope)
 # Content-only source views exclude colour chart, mount and photographic frame.
 # The raw preview and exact content crop are retained; no CAPTCHA download used.
 for idx,boxes in [(7,[[210,250,280,330],[290,285,365,365]]),(10,[[375,90,450,165],[453,113,530,190]])]:
  raw=Path(npm['images'][idx]['path']);clean=ROOT/'sources'/f'npm_5621_{idx}_content.png';content_box=[0,46,595,374];Image.open(raw).crop(content_box).save(clean)
  for b in boxes:
   b=[b[0],b[1]-46,b[2],b[3]-46]
   add('npm5621','披麻皴','Wen_Zhengming','npm5621',clean,b,npm['url'],npm['evidence'],dict(raw_source_image=str(raw),raw_sha256=sha(raw),source_content_box=content_box,attribution=npm['text_attribution']))
 exclusions=[dict(id=k,reason='Technique term is absent from the work description; artist biography, glossary or imitation lineage is insufficient for pure-class labels.') for k in ['233555','231419','231773','229559']]
 exclusions.extend([dict(id='npm11259',url='https://digitalarchive.npm.gov.tw/Collection/Detail/11259?dep=P',reason='Museum techniques table lists both hemp-fibre and axe-cut; pure local class cannot be assigned without further review.'),dict(id='npm5621_preview0_to6',reason='Whole-scroll preview too small; first photographic views are text/mounts, not mountain texture. Excluded before prediction.')])
 local_exclusions={'fresh_234350_0':'Plum branches dominate; cannot certify local texture from whole-work label.','fresh_234350_1':'Crop contains a red collector seal; exclude before prediction to prevent label shortcuts.','fresh_234350_2':'Too little clear internal texture; local style remains uncertain.'}
 for r in rows:
  r['admission']='excluded_pre_prediction' if r['id'] in local_exclusions else 'source_label_test'
  r['exclusion_reason']=local_exclusions.get(r['id'])
 d=dict(records=rows,reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze_manifest_sha256=sha(ROOT/'freeze_manifest.json'),expert_local_test_samples=0,
  all_labels_provisional=True,admission_frozen_before_prediction=True,sample_selection_did_not_use_model=True,exclusions=exclusions,robot_connected=False)
 (ROOT/'test_manifest.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n);canvas=Image.new('RGB',(1400,4*280+60),'#f4f1e8');dr=ImageDraw.Draw(canvas);dr.text((15,10),'新作者16例：预测前冻结的区域与来源候选标签',font=font(25),fill='#233e32')
 for i,r in enumerate(rows):
  x=i%4*350;y=i//4*280+60;im=Image.open(r['image']);im.thumbnail((325,190));canvas.paste(im,(x+10,y));dr.text((x+10,y+197),r['id'],font=font(18),fill='black');dr.text((x+10,y+224),r['label']+' '+r['artist_group'],font=font(19),fill='black')
 canvas.save(ROOT/'pre_prediction_review.jpg',quality=94);print('PREPARED',len(rows),'ARTISTS',sorted({r['artist_group'] for r in rows}),flush=True)
if __name__=='__main__':main()
