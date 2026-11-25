# -*- coding: utf-8 -*-
"""Actual new local reference annotations, with reused-source lineage explicit."""
import sys,json,hashlib,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.source_identity import source_identity,check_previously_seen
ROOT=LAB/'clear_rock_sources_20261117'
PROJECT=Path(__file__).resolve().parents[1]

# All coordinates are native 1280x720. Regions describe context, not every ink pixel.
REGIONS=[
 ('F360_pine_needles',360,[600,515,677,557],'tree','foliage','松枝周围可见放射状松针；禁止提供山石皴笔目标。'),
 ('F360_mountain_face',360,[310,421,421,467],'rock','unknown','山面内部的浅色面与结构，不能把留白全当纸面，也不确定具体皴法。'),
 ('F1500_pine_needles',1500,[565,368,646,404],'tree','foliage','前景松枝上的成组细叶；该细碎纹理属于树木。'),
 ('F1500_pine_trunk',1500,[661,407,694,457],'tree','trunk','松枝汇合后向下延伸的树干区域；边缘及邻近岩面不作像素标签。'),
 ('F1500_foreground_rock',1500,[579,457,696,537],'rock','unknown','前景主石内部及部分结构；单个区域不等于完整石块边界。'),
 ('F1500_right_needles',1500,[1020,385,1115,429],'tree','foliage','右前景松树枝叶；不作披麻、斧劈或负纸面标签。'),
 ('F1875_wash_core',1875,[654,116,707,174],'rock','wash','背后山体中连续灰墨铺开的内部；只确认区域为染墨面，不确认内部每个像素归属。'),
 ('F1875_tree',1875,[349,526,394,578],'tree','unknown','前方小树群，细节不足以逐笔区分枝干与叶片。'),
 ('F1500_tree_rock_overlap',1500,[642,424,714,472],'unknown','unknown','树根与石顶交接，树干、结构线和重叠墨迹难以唯一归属，保持待定。'),
 ('F1875_wash_tree_overlap',1875,[647,198,691,259],'unknown','unknown','背景染墨与前方树木交叠，不能整块归为皴笔或树叶。'),
]
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(name,v):(ROOT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    (ROOT/'views').mkdir(exist_ok=True)
    meta=read(ROOT/'fuchun_metadata.json');manifest=read(ROOT/'fuchun_manifest.json')
    prior=[]
    for p in (LAB/'pima_independent_20261109').glob('*/metadata.json'):
        m=read(p);prior.append(dict(url=m['url'],metadata=str(p),title=m['name']))
    duplicate=check_previously_seen(meta['url'],prior)
    assert duplicate['previously_seen']
    write('source_preflight.json',dict(**duplicate,source_new_to_project=False,
        first_identity_lookup_performed_after_download=True,
        correction='本轮初始检索误当新来源；旧清单核查后纠正。720p补充不计独立新作品。',
        old_media_native_size=[854,480],new_media_native_size=[1280,720],
        evidence_scope='higher_native_resolution_on_existing_work_only'))
    frame_by_t={r['requested_seconds']:r for r in manifest['frames']};rows=[]
    for ident,t,box,obj,role,reason in REGIONS:
        r=frame_by_t[t];im=Image.open(r['path']).convert('RGB');x0,y0,x1,y1=box
        assert im.size==(1280,720) and 0<=x0<x1<=1280 and 0<=y0<y1<=720
        crop=im.crop(box);crop.save(ROOT/f'views/{ident}.png')
        marked=im.copy();d=ImageDraw.Draw(marked);d.rectangle(box,outline='#a44791' if obj=='unknown' else '#c27628',width=3)
        marked.thumbnail((640,360));marked.save(ROOT/f'views/{ident}_context.jpg',quality=93)
        rows.append(dict(id=ident,source_identity=source_identity(meta['url']),source_url=meta['url'],
            work_group='fuchun_example10_same_as_pima_independent_20261109_new_fuchun',
            source_group='fuchun_ganjing_channel_1gbpdvdtp0i2vkrtF5lvmscUY1qs0c',
            frame=r['path'],frame_sha256=sha(r['path']),frame_index=r['frame_index'],nominal_seconds=r['nominal_seconds'],
            native_size=[1280,720],box=box,object_type=obj,role=role,visual_evidence=reason,
            label_scope='assistant_regional_context_not_dense_ink_or_stroke',
            reviewer='assistant_visual_review_20261117',expert_verified=False,
            new_local_annotation=True,new_independent_work=False,source_seen_before_round=True,
            object_checked=obj!='unknown',role_checked=role in {'foliage','trunk','wash'},
            region_is_ink_mask=False,unmarked_is_background=False,
            local_cunfa='unknown',style_training_target=None,complete_curve=None,
            stroke_order=None,pressure_mm=None,training_admitted=False))
    write('local_references.json',dict(records=rows,review_status='assistant_native_crops_rechecked_not_expert_gold',
        total=10,confirmed_object_regions=8,uncertain_regions=2,source_works=1,
        new_independent_works=0,new_complete_artist_strokes=0,new_model_trained=False,robot_connected=False))
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    board=Image.new('RGB',(1500,950),'#f5f2e9');d=ImageDraw.Draw(board)
    picked=['F360_pine_needles','F1500_pine_trunk','F1500_foreground_rock','F1875_wash_core','F1500_tree_rock_overlap','F1875_wash_tree_overlap']
    labels=['松针：不能标成皴笔','树干区域：不是逐像素墨迹','山石：皴法仍未知','连续染墨面：不是细线皴笔','树石交界：待定','染墨与树木交叠：待定']
    for i,(ident,title) in enumerate(zip(picked,labels)):
        r=next(x for x in rows if x['id']==ident);x=i%3*500;y=i//3*475
        d.text((x+12,y+8),title,font=font(21),fill='#234035')
        im=Image.open(ROOT/f'views/{ident}_context.jpg');im.thumbnail((480,270));board.paste(im,(x+10,y+46))
        crop=Image.open(ROOT/f'views/{ident}.png');scale=min(440/crop.width,112/crop.height)
        crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.NEAREST)
        board.paste(crop,(x+10,y+325));d.text((x+12,y+450),f"{r['nominal_seconds']:.0f}秒 · 助手区域参考；非模型输出",font=font(15),fill='#78523c')
    board.save(ROOT/'local_reference_review.jpg',quality=94)
    write('source_access_audit.json',dict(
        checked='2026-11-17',ordinary_public_requests_only=True,
        attempts=[dict(id='jiangmei',url='https://www.5minutes.com.cn/Web/Course/CourseDetail.aspx?id=159d16ff-4e5d-452a-9ee1-2e72b3513ace',http_status=403,media_acquired=False),
                  dict(id='jingxin',url='https://www.bilibili.com/video/BV1Rf4y1Y7Pv/',http_status=412,media_acquired=False),
                  dict(id='textbook',url='https://r3-ndr.ykt.cbern.com.cn/edu_product/esp/assets_document/a4a312f4-1689-4e56-ab95-660905db81ec.pkg/pdf.pdf',http_status=403,media_acquired=False),
                  dict(id='textbook_alternate_official_link',url='https://v3.ykt.cbern.com.cn/65/document/f7a761c407d34224a489ce4b0df0e105/pdf.pdf',http_status=403,media_acquired=False),
                  dict(id='tong_museum',url='https://hk.art.museum/en/web/ma/exhibitions-and-events/virtually-at-hkmoa.html',http_status=200,media_acquired=False,reason='官方页面列出视频，实际链接为YouTube；本轮未取得视频，不算已核查逐笔内容。'),
                  dict(id='fuchun',url=meta['url'],http_status=200,media_acquired=True,new_independent_work=False)],
        complete_rock_demonstration_goal_met=False))
    write('review_status.json',dict(primary_goals=[1,2],main_goal_complete_stroke_reference_met=False,
        new_local_reference_regions=10,new_independent_works=0,new_complete_strokes=0,
        source_quality='720p native; paper/hand/zoom changes remain; title does not label all strokes',
        new_model_trained=False,production_replaced=False,robot_connected=False,
        next_required='清晰、作者可核查、单块山石从起形到加皴的连续过程，逐笔归属可靠后再开放完整笔触训练。'))
    cards=''.join(f'<article><h3>{html.escape(r["id"])}</h3><img src="views/{r["id"]}_context.jpg"><img src="views/{r["id"]}.png"><p>{html.escape(r["visual_evidence"])}</p><p>对象：{r["object_type"]}；角色：{r["role"]}；皴法：未知。<br>原生框：{r["box"]}；第{r["frame_index"]}帧。</p></article>' for r in rows)
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>高清示范来源与新增局部参考</title><style>body{font:17px/1.7 system-ui;max-width:1500px;margin:auto;padding:24px;background:#f5f2e9;color:#243f34}img{max-width:100%;height:auto}.notice{padding:16px;background:#e9dfc9}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(330px,100%),1fr));gap:18px}article{padding:16px;background:white}article img{max-height:350px;object-fit:contain}h1{font-size:30px}a{color:#265e79}</style><h1>高清示范核查：树木、山石、染墨与待定</h1><p class="notice">这是来源核查和助手区域参考，不是模型效果。已有《富春山居图》讲解视频补取720p，不能算新增独立作品；完整真实分笔示范仍未补齐。</p><p>新增局部参考与旧作品分开计数。8处对象参考、2处待定；所有局部均不提供皴法类别、完整曲线、笔顺或机械臂压力。未训练新模型。</p><img src="local_reference_review.jpg" alt="对象和角色的局部参考"><h2>完整上下文与原生局部</h2><div class="grid">CARDS</div><h2>视频取样核查</h2><img src="fuchun_overview.jpg" alt="顺序解码抽查"><p>视频约58分49秒，本轮全片顺序解码，抽查主要覆盖前31分15秒；这不是看完每一帧的人工审查。标题卡、缩放、手部遮挡和混合阶段均存在；不据此认定完整单笔真值。</p><p><a href="SOURCE">打开发布页</a> · <a href="local_references.json">逐区域参考</a> · <a href="source_preflight.json">重复来源核查</a> · <a href="source_access_audit.json">访问与取得结果</a> · <a href="verification.json">完整性检查</a></p></html>'''
    (ROOT/'index.html').write_text(doc.replace('CARDS',cards).replace('SOURCE',html.escape(meta['url'])),encoding='utf8')
    print(json.dumps(dict(regions=len(rows),previously_seen=duplicate['previously_seen'],output=str(ROOT)),ensure_ascii=False))
if __name__=='__main__':main()
