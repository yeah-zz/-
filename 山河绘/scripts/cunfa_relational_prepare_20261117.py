import sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(LAB/'guohua_system_review_20261110/deps'))
import pymupdf
ROOT=LAB/'cunfa_relational_roles_20261117';SRC=LAB/'cunfa_cross_source_20261117';VIDEO=LAB/'cunfa_temporal_roles_20261112'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
VIDEO_SPECS={
 'T_structure':[
  ([5,45,110,110],'outline_context','初始勾结构阶段，左侧外缘和山面转折。'),
  ([200,35,310,95],'outline_context','初始勾结构阶段，上部山石折向轮廓。'),
  ([100,90,175,165],'outline_context','初始勾结构阶段，中下折向边与留白交界。'),
  ([260,85,365,145],'outline_context','初始勾结构阶段，右下外缘与内部折向。'),
  ([5,3,70,18],'unmarked_surface','旧参考已核查的未着墨核心附近，视频纸面保留真实纹理。')],
 'T_cun_group1':[
  ([65,100,115,155],'interior_mark_context','同镜头加皴阶段，左侧新增短线组附近；不提供差分或目标掩膜给模型。'),
  ([115,110,165,155],'interior_mark_context','同镜头加皴阶段，中部新增墨迹组附近，仍无逐笔真值。'),
  ([260,130,295,165],'outline_context','原结构已存在的右侧外缘，按旧部分参考与原图复核。'),
  ([270,50,290,78],'unmarked_surface','视频纸面小留白，按原图复核而非模型预测标注。')]
}
def main():
 (ROOT/'sources').mkdir(parents=True,exist_ok=True);(ROOT/'views').mkdir(exist_ok=True)
 assert not (ROOT/'protocol.json').exists(),'Frozen training admission'
 data=read(SRC/'reference_manifest.json');rows=copy.deepcopy(data['records'])
 for r in rows:r['role_fold_group']=r['reference_unit'];r['added_this_round']=False
 vm=read(VIDEO/'dataset_manifest.json');lookup={r['id']:r for r in vm['records']}
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 for ident,specs in VIDEO_SPECS.items():
  old=lookup[ident];assert sha(old['path'])==old['sha256'];z=np.load(old['path']);image=Image.fromarray(z['image']);ref=ROOT/'sources'/(ident+'.png');image.save(ref)
  for j,(box,role,note) in enumerate(specs):
   p=ROOT/'views'/(ident+'_new_'+str(j)+'.png');image.crop(box).save(p)
   target={'outline_context':1,'interior_mark_context':2,'unmarked_surface':0}[role]
   target_counts={str(k):int((z['targets'][box[1]:box[3],box[0]:box[2]]==k).sum()) for k in [0,1,2,255]}
   rows.append(dict(id=ident+'_new_'+str(j),image=str(p),image_sha256=sha(p),reference_image=str(ref),reference_image_sha256=sha(ref),
    reference_unit=ident,source_group='mixiaolu_short_pima_work',role_fold_group='mixiaolu_short_pima_work',source_url=vm['source_url'],
    source_evidence=old['reference_kind'],source_reference_sha256=old['sha256'],source_reference_counts=target_counts,
    before_seconds=old['before_seconds'],after_seconds=old['after_seconds'],box=box,split='train',object_type='paper' if role=='unmarked_surface' else 'rock',
    object_reference_checked=True,context_role_target=role,role_reference_checked=True,role_reference_scope='region_context_not_pixel_or_stroke_truth',
    visual_evidence=note,reviewer='assistant_phase_and_static_review_20261117',expert_verified=False,
    automatic_prediction_as_label=False,evaluation_scope='reused_video_development',added_this_round=True,
    local_style_target=None,complete_curve=None,stroke_order=None,pressure=None,model_input='static after image only'))
 # A separate cliff-slope teaching figure adds long edges and broad tone faces.
 doc=pymupdf.open(SRC/'sources/qingpu.pdf');page=doc[25];images=[doc.extract_image(x[0]) for x in page.get_images(full=True)]
 imdata=max(images,key=lambda r:r['width']*r['height']);path=ROOT/'sources'/('qingpu_26.'+imdata['ext']);path.write_bytes(imdata['image'])
 image=Image.open(path).convert('RGB');print('CLIFF_IMAGE',image.size,flush=True)
 # Initial source inspection is exported before any new-model predictions.
 image.save(ROOT/'cliff_source.png')
 write('preliminary_manifest.json',dict(records=rows,cliff_image=str(path),cliff_size=list(image.size),
  cliff_source_pdf_sha256=sha(SRC/'sources/qingpu.pdf'),cliff_page=26,video_work_not_new_independent_author=True,
  old_source_models_untouched=True,expert_labels=0,robot_connected=False))
 board=Image.new('RGB',(1440,3*270+90),'#f3f0e7');d=ImageDraw.Draw(board)
 d.text((15,12),'复用视频阶段参考：只看静态原图，时序用于参考核查',font=font(28),fill='#243f35')
 for i,r in enumerate([r for r in rows if r['added_this_round']]):
  x=i%3*480+10;y=i//3*270+70;im=Image.open(r['image']);scale=min(440/im.width,170/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)))
  board.paste(im,(x,y));d.text((x,y+180),r['id'],font=font(19),fill='#243f35');d.text((x,y+213),r['context_role_target'],font=font(18),fill='#805333')
 board.save(ROOT/'video_references.jpg',quality=94)
 print('VIDEO_REGIONS',9,flush=True)
if __name__=='__main__':main()
