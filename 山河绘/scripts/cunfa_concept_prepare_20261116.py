# -*- coding: utf-8 -*-
"""Freeze assistant-reviewed region morphology BEFORE fitting concept heads."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure, LAB
configure()
from PIL import Image, ImageDraw, ImageFont
from gongbi_training.cunfa_concept_supervision import concept_targets, validate_concept_split

ROOT = LAB/'cunfa_concept_supervision_20261116'
OLD = LAB/'cunfa_role_reference_20261112'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n, v): (ROOT/n).write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf8')

# Coordinates and visual judgements are assistant annotations, NOT model output.
# Two attributes can co-occur. A thin outline in an axe illustration is not
# automatically labelled an axe stroke. Crop edges are not stroke endpoints.
SPECS = {
 'aam_short_hemp': [
  ([90,70,270,310], [1,0,1], '左侧多条弯曲长墨带沿同一弧向排列；不声称逐笔归属。'),
  ([425,245,730,480], [1,0,1], '右下可见相邻弯曲细长墨带及断续边缘。'),
  ([235,150,430,390], [None,None,None], '深墨交叠，单笔与块面边界无法可靠区分，全部待定。'),
  ([340,0,710,170], [1,None,None], '上缘细长斜向墨带可见；深墨与底层曲带交叠，宽块和独立弧组均待定。')],
 'aam_long_hemp': [
  ([60,50,310,390], [1,0,1], '左侧长墨带同向弯曲，保留带间亮缝。'),
  ([200,40,570,260], [1,0,1], '中上多条连续弯曲墨带，不是尖角宽块。'),
  ([225,260,610,500], [1,0,1], '中下成组长线沿弧面排列，断续不等于多次落笔。'),
  ([510,100,730,430], [1,0,1], '右侧邻近墨带向下弯曲；笔顺仍未知。')],
 'aam_small_axe': [
  ([38,35,150,140], [1,1,0], '同一区域有宽三角墨块及细长拖尾，两类属性同时存在。'),
  ([75,85,340,290], [0,1,0], '主要为有尖角或平顶的宽墨块；未见独立成组弯曲细线。'),
  ([305,215,540,435], [0,1,0], '中下多块宽角状墨迹叠列；不是完整分笔标签。'),
  ([495,100,730,235], [1,1,0], '右侧小宽角墨块和横向细长墨迹并存。'),
  ([200,4,530,60], [1,0,0], '上方细长横向边界；不能因示例名为斧劈就标成宽块。')],
 'aam_large_axe': [
  ([120,130,480,420], [0,1,0], '大面积角状宽墨迹及灰墨面；交叠笔序未知。'),
  ([470,5,710,150], [0,1,0], '右上明显三角宽墨面，非细线笔组。'),
  ([385,240,710,490], [1,None,0], '右下细长近直斜墨带可见；弱墨面角形不确定。'),
  ([530,70,730,185], [1,0,0], '右上长黑边线，属于细长形态；作用可能为轮廓。'),
  ([0,50,135,250], [1,1,0], '左侧单条弯边及灰墨角状面并存，不满足两条以上弯曲笔组条件。')],
 'hkma_axe': [
  ([20,12,117,122], [1,1,0], '左上练习区域有短宽角状笔头，也有细长拖尾，两个属性同时为真。'),
  ([22,75,98,155], [1,1,0], '练习区有宽笔头与细长拖尾，属性可以同时为真。'),
  ([125,125,218,200], [1,0,1], '同一斧劈教学照片的石头内部含相邻弧形细线，不能仅按名称判形。'),
  ([33,7,79,73], [1,1,0], '局部短宽墨块和细长笔迹并存，无成组弯曲结构。')],
 'hkma_hemp': [
  ([10,12,88,104], [1,0,1], '山石左侧多条相邻弯曲墨带。'),
  ([68,38,146,92], [1,0,1], '山石中部细长墨带成组弯曲，保留空隙。'),
  ([92,2,172,66], [1,None,0], '顶部细长折向墨迹可见；交叠宽部不确定。')]
}

def main():
 ROOT.mkdir(exist_ok=True); (ROOT/'views').mkdir(exist_ok=True)
 assert not (ROOT/'protocol.json').exists(), 'Frozen experiment; do not relabel after evaluation'
 refs=json.loads((OLD/'source_references.json').read_text('utf8'))['records']; records=[]
 font=lambda n: ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 for ref in refs:
  im=Image.open(ref['image']).convert('RGB'); assert sha(ref['image'])==ref['crop_sha256']
  if ref['id'] not in SPECS: continue
  for j,(box,values,evidence) in enumerate(SPECS[ref['id']]):
   assert 0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height
   ident=ref['id']+'_'+str(j); crop=im.crop(box); dest=ROOT/'views'/(ident+'.png'); crop.save(dest)
   r=dict(id=ident, image=str(dest), image_sha256=sha(dest), reference_image=ref['image'],
    reference_unit=ref['id'], source_sha256=ref['source_sha256'], source_group=ref['source_group'],
    source_url=ref['source_url'], source_evidence=ref['source_evidence'], box=box,
    split='test' if ref['publisher']=='HKMA' else 'train', object_type='rock',
    object_reference_checked=True, object_detector_input=False,
    label_scope='observable_region_morphology', concept_origin='visual_region_review',
    concepts=dict(zip(('elongated_mark','angular_broad_mark','curved_parallel_group'),values)),
    visual_evidence=evidence, reviewer='assistant_visual_review_20261116',
    concept_reference_checked=True, expert_verified=False, automatic_prediction_as_label=False,
    role='unknown', local_style_target=None, pixel_mask=None, complete_curve=None,
    stroke_instance=None, stroke_order=None, brush_pressure=None, reference_is_not_pixel_truth=True)
   r['targets']=concept_targets(r).tolist();records.append(r)
 validate_concept_split(records)
 # Two old NPM references are not clear enough for these attribute labels.
 review=[dict(id=r['id'],image=r['image'],source_url=r['source_url'],concepts={k:None for k in records[0]['concepts']},
              reason='低清成品局部，墨带与染墨难分，未给予概念目标。',split='review_only') for r in refs if r['publisher']=='NPM']
 write('reference_manifest.json',dict(records=records,review_only=review,original_named_reference_units=8,
  morphology_labelled_reference_units=6,train_source_groups=1,test_source_groups=1,
  unique_reference_units_not_crop_count=True,concepts_are_assistant_references=True,
  fully_resolved_roles=0,complete_stroke_labels=0,expert_annotations=0,
  new_independent_works=0,automatically_generated_labels=0,robot_connected=False))
 write('definitions.json',dict(
  elongated_mark='可见延伸的线带形墨迹；可为轮廓或皴笔，不由皴法名推定。',
  angular_broad_mark='可见有尖角/平顶/楔状面的宽墨形；不把所有粗线都当作此类。',
  curved_parallel_group='至少两条可区分的墨带沿相近弧向排列；不包含笔顺或逐笔归属。',
  unknown='遮挡、混墨、低清或角色不明的位置不作负例；形态不是物理动作。'))
 rows=(len(records)+3)//4;board=Image.new('RGB',(1440,rows*250+110),'#f3f0e7');d=ImageDraw.Draw(board)
 d.text((15,10),'概念参考：形态可并存，名称不能替代观察',font=font(27),fill='#243f35')
 d.text((15,55),'助手逐区参考；非模型输出、逐笔真值或实画。AAM训练 / HKMA留出。',font=font(19),fill='#805333')
 for i,r in enumerate(records):
  x=i%4*360;y=i//4*250+105;im=Image.open(r['image']);im.thumbnail((330,150));board.paste(im,(x+12,y))
  d.text((x+12,y+156),r['id']+' / '+r['split'],font=font(16),fill='#243f35')
  values=['?' if v is None else str(v) for v in r['concepts'].values()]
  d.text((x+12,y+186),'长线 / 角块 / 弧组：'+' / '.join(values),font=font(17),fill='#805333')
 board.save(ROOT/'reference_board.jpg',quality=94)
 print(json.dumps(dict(region_views=len(records),train=sum(r['split']=='train' for r in records),test=sum(r['split']=='test' for r in records),reference_units=6,complete_strokes=0),ensure_ascii=False),flush=True)

if __name__=='__main__': main()
