import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_structure_20261111';OLD=LAB/'cunfa_gabor_baseline_20261110'
SPEC=[
 ('230692',6,'Dong_Qichang','Dong_album_leaf7','披麻皴',[[140,75,214,200],[218,75,294,200],[190,295,285,365]],'Album leaf 7 only; the museum specifically associates this leaf with hemp-fibre brushwork.'),
 ('230678',0,'Dong_Qichang','Dong_scroll','披麻皴',[[242,18,355,113],[365,18,475,113],[485,22,595,117]],'Only distant mountains; museum explicitly assigns hemp-fibre texture to the distant mountains, not the foreground.'),
 ('232500',0,'Liu_Jue','Xia_Yun_Yu_Yu','披麻皴',[[175,240,240,325],[245,220,310,310],[235,350,300,455],[352,320,420,440],[180,480,255,570],[250,565,325,650]],'Museum describes almost all mountains as long hemp fibre; assistant-selected rock ROIs remain provisional.'),
 ('228140',0,'Wang_Lu','Hua_Shan_album','斧劈皴',[[100,185,210,295],[240,180,350,290],[365,185,475,295],[500,360,610,470],[620,400,730,510]],'Museum describes album rocks as small axe-cut; no expert per-crop annotation.'),
 ('228140',1,'Wang_Lu','Hua_Shan_album','斧劈皴',[[110,185,220,295],[250,220,360,330],[345,320,415,445],[660,470,770,550]],'Same album and author remain one group; leaf image is not an independent author.'),
 ('234462',0,'Liu_Songnian','Si_Jing_Shan_Shui','斧劈皴',[[1030,235,1100,345],[1040,355,1110,465]],'Museum describes rocks as small axe-cut; painting around buildings and trees is excluded.'),
]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 (ROOT/'crops').mkdir(exist_ok=True);sources={r['id']:r for r in json.loads((ROOT/'source_manifest.json').read_text('utf8'))};rows=[]
 for key,idx,artist,work,label,boxes,scope in SPEC:
  src=sources[key];p=next(Path(i['path']) for i in src['images'] if i['index']==idx);im=Image.open(p).convert('RGB')
  article=src['main_text'].split('关键词')[0]
  assert label in article.replace(' ','')
  if key=='230692':article=article[article.index('第七开'):]
  for n,box in enumerate(boxes):
   ident=f'new_{key}_{idx}_{n}';dest=ROOT/'crops'/(ident+'.png');im.crop(box).save(dest)
   rows.append(dict(id=ident,image=str(dest),source_image=str(p),source_box=box,label=label,artist_group=artist,work_group=work,publisher_group='DPM',source_url=src['url'],source_sha256=sha(p),crop_sha256=sha(dest),label_scope=scope,label_evidence_grade='museum named local/whole-work technique + assistant ROI; not expert local gold',evidence_paraphrase=scope,expert_local_annotation=False,split='external_holdout',is_augmentation=False))
 # Mixed descriptions cannot establish local pure-class labels; retain audit only.
 exclusions=[dict(id='228153',reason='Museum explicitly describes mixed dry-brush and axe-cut; cannot assign pure local axe labels without expert review.'),dict(id='230692_other_leaves',reason='Only leaf7 has relevant explicit description; no whole-album label propagation.'),dict(id='initial_ROI_review',reason='Before any prediction: four Wang Lu boxes moved off vegetation to visible cliff faces; Liu Songnian third box removed because roof/railings contaminate rock. No prediction-dependent selection.')]
 old=json.loads((OLD/'dataset_manifest.json').read_text('utf8'));train=[r for r in old['records'] if r['admission']=='pilot_only'];unknown=[r for r in old['records'] if r['admission']=='unknown_probe']
 assert {r['artist_group'] for r in rows}.isdisjoint({r['artist_group'] for r in train})
 assert len(rows)==23
 data=dict(development_records=train,external_holdout=rows,previous_unknown_probes=unknown,exclusions=exclusions,expert_test_samples=0,
  source_labels_are_provisional=True,crops_frozen_before_prediction=True,old_manifest_sha256=sha(OLD/'dataset_manifest.json'),new_source_manifest_sha256=sha(ROOT/'source_manifest.json'),robot_connected=False)
 (ROOT/'dataset_manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
 out=Image.new('RGB',(1400,6*245+60),'#f4f1e8');d=ImageDraw.Draw(out);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 d.text((15,8),'新增独立作者留出：先核对裁剪，再运行预测（尚非专家真值）',font=font(23),fill='black')
 for i,r in enumerate(rows):
  x=i%4*350;y=i//4*245+55;im=Image.open(r['image']);im.thumbnail((320,160));out.paste(im,(x+12,y));d.text((x+12,y+165),r['id'],font=font(18),fill='black');d.text((x+12,y+190),r['label']+' '+r['artist_group'],font=font(19),fill='black')
 out.save(ROOT/'new_regions_review.jpg',quality=94);print('NEW',len(rows),'ARTISTS',len({r['artist_group'] for r in rows}),flush=True)
if __name__=='__main__':main()
