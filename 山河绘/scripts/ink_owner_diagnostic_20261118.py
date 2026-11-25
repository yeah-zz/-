# -*- coding: utf-8 -*-
"""Post-hoc ink-core ownership references, never whole artist strokes or new test truth."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'ink_owner_diagnostic_20261118';SOURCE=LAB/'object_prospective_20261118'
ARMS=['global256','detail512','fixed_mean','learned_gate'];NAMES=['冻结256','冻结512','固定平均','学习融合']
SPECS=[('43','R01',194,108,'rock'),('43','R02',219,169,'rock'),('43','R03',145,267,'rock'),('43','R04',178,307,'rock'),('43','R05',292,333,'rock'),('43','R06',370,438,'rock'),
('43','T01',96,470,'tree'),('43','T02',87,443,'tree'),('43','T03',115,431,'tree'),('43','T04',104,400,'tree'),
('11','T05',83,90,'tree'),('11','T06',137,112,'tree'),('11','T07',182,179,'tree'),('11','T08',324,168,'tree'),('11','T09',376,207,'tree'),('11','T10',403,243,'tree'),
('43','U01',126,409,'unknown'),('11','U02',385,273,'unknown'),('43','U03',125,380,'unknown'),('11','U04',272,268,'unknown')]
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,v):(ROOT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def prepare():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'draft_references.json').exists(),'Draft exists'
    rows=[];board=Image.new('RGB',(1280,1100),'#f5f2e9');d=ImageDraw.Draw(board);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    for i,(suffix,ident,x,y,obj) in enumerate(SPECS):
        image_id='zouzhe_xiangjiaoqiuwutu_'+suffix;path=SOURCE/'raw'/(image_id+'.jpg');im=Image.open(path).convert('RGB');gray=np.array(im.convert('L'))
        # Small displacement only locates a visible dark core; ownership still needs visual review.
        if obj!='unknown':
            patch=gray[y-2:y+3,x-2:x+3];yy,xx=np.unravel_index(patch.argmin(),patch.shape);px,py=x-2+int(xx),y-2+int(yy)
        else:px,py=x,y
        rows.append(dict(id=ident,image_id=image_id,image=str(path),image_sha256=sha(path),native_size=list(im.size),seed_xy=[x,y],xy=[px,py],
            seed_adjustment='darkest pixel within2px, assistant crop review required',object_candidate=obj,
            annotation_scope='visible_ink_core_point_not_complete_stroke_or_boundary',
            source_seen_before_annotation=True,prior_predictions_already_inspected=True,expert_verified=False,
            training_allowed=False,cunfa='unknown',role='unknown',stroke_identity=None,stroke_order=None,pressure_mm=None))
        cx=i%4*320;cy=i//4*220
        tile=im.crop((px-20,py-20,px+21,py+21)).resize((164,164),Image.Resampling.NEAREST)
        draw=ImageDraw.Draw(tile);draw.ellipse((77,77,87,87),outline='#f2473b',width=2)
        board.paste(tile,(cx+10,cy+36));d.text((cx+8,cy+5),f'{ident} · {obj} · {px},{py}',font=font(16),fill='#243f35')
        context=im.crop((max(0,px-50),max(0,py-50),min(512,px+51),min(512,py+51)));context.thumbnail((130,160));board.paste(context,(cx+180,cy+38))
    board.save(ROOT/'draft_contact.jpg',quality=95);write('draft_references.json',dict(records=rows,reference_status='awaiting_native_crop_review',posthoc_diagnostic=True,new_independent_test=False))
    print('DRAFT',len(rows))
def evaluate():
    assert not (ROOT/'results.json').exists(),'Do not relabel after point scoring'
    data=read(ROOT/'reviewed_references.json');assert data['review_status']=='assistant_crops_checked_not_expert'
    rows=[]
    for r in data['records']:
        assert sha(r['image'])==r['image_sha256'];x,y=r['xy'];obj=r['object_type'];out=dict(r,predictions={})
        for arm in ARMS:
            pred=np.load(SOURCE/f'predictions/{r["image_id"]}_{arm}.npz')['prediction'];cls=1 if obj=='rock' else 2 if obj=='tree' else None
            out['predictions'][arm]=dict(object_id=int(pred[y,x]),correct=None if cls is None else bool(pred[y,x]==cls),
                neighborhood_fraction=None if cls is None else float((pred[y-1:y+2,x-1:x+2]==cls).mean()))
        rows.append(out)
    stats={}
    for arm in ARMS:
        stats[arm]={}
        for obj in ['rock','tree']:
            chosen=[r for r in rows if r['object_type']==obj]
            stats[arm][obj]=dict(core_points=len(chosen),correct=sum(r['predictions'][arm]['correct'] for r in chosen),
                stable_wrong=sum(r['predictions'][arm]['neighborhood_fraction']==0 for r in chosen),
                location_sensitive=sum(0<r['predictions'][arm]['neighborhood_fraction']<1 for r in chosen))
    write('results.json',dict(records=rows,summary=stats,scoring='point center primary,3x3 is location sensitivity only;9pixels are not9references',
        evaluated_points=sum(r['object_type']!='unknown' for r in rows),unknown_points=sum(r['object_type']=='unknown' for r in rows),
        posthoc_diagnostic=True,new_independent_test=False,new_model_trained=False,adopted=False,robot_connected=False))
    print(json.dumps(stats,ensure_ascii=False))
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:evaluate()
