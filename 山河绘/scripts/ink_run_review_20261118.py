"""Source-only review, then frozen cached-output diagnosis. No training or robot API."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure, LAB
configure()
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gongbi_training.ink_path_diagnostics import score_visible_runs
ROOT = LAB / 'ink_run_review_20261118'
SOURCE = LAB / 'object_prospective_20261118'
PROJECT = Path(__file__).resolve().parents[1]
ARMS = ['global256', 'detail512', 'fixed_mean', 'learned_gate']
NAMES = ['冻结256', '冻结512', '固定平均', '学习融合']
COLORS = {0: '#9ea7aa', 1: '#bf802c', 2: '#208b62', 3: '#935e9e'}
FONT = 'C:/Windows/Fonts/msyh.ttc'


def read(path): return json.loads(Path(path).read_text('utf8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name, data): (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')


SPECS = [
    ('R_A', '43', 'rock', [[387,304],[390,312],[390,324],[390,336],[391,341],[395,349],[398,356],[399,365],[400,371]], [[403,311],[404,325],[405,339],[409,354]], '岩面外侧可见细墨线片段；不推断真实起收笔'),
    ('R_B', '43', 'rock', [[304,321],[314,314],[325,309],[336,307],[346,309],[355,312],[362,320],[367,330]], [[315,304],[329,298],[345,299],[357,303]], '岩面弧形墨带片段；不能据此认定皴法'),
    ('R_C', '43', 'rock', [[199,245],[205,254],[210,264],[215,273],[219,283]], [[202,241],[211,250],[217,260],[222,271]], '岩面内侧斜向墨痕片段；角色待定'),
    ('T_A', '11', 'tree', [[82,62],[84,68],[87,75],[91,83],[96,88],[103,92],[112,96],[121,98],[128,100]], [[90,63],[93,70],[98,78],[104,83]], '树枝可见片段；到相接处结束，不决定穿越关系'),
    ('T_B', '43', 'tree', [[97,493],[94,483],[95,475],[97,467],[97,459],[99,450],[99,443],[101,438]], [[104,484],[105,476],[106,466]], '树干可见墨带内参考；不作为真实中心线或笔宽答案'),
    ('T_C', '43', 'tree', [[102,450],[110,443],[118,438],[126,434],[135,430]], [[112,450],[121,444],[132,439]], '向右分枝可见片段；交叉归属另列待定')]


def source(r): return Image.open(r['image']).convert('RGB')


def review_view(r):
    pts=np.array([p for run in r['runs'] for p in run] + r['adjacent_low_ink_path']);x0,y0=np.maximum(pts.min(0)-12,0).astype(int);x1,y1=np.minimum(pts.max(0)+13,512).astype(int)
    box=[int(x0),int(y0),int(x1),int(y1)];crop=source(r).crop(box);scale=5
    a=crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.NEAREST);b=a.copy();d=ImageDraw.Draw(b)
    for run in r['runs']:
        p=[((x-x0)*scale,(y-y0)*scale) for x,y in run];d.line(p,fill='#00cce5',width=2)
        for x,y in p:d.ellipse((x-2,y-2,x+2,y+2),fill='#00cce5')
    p=[((x-x0)*scale,(y-y0)*scale) for x,y in r['adjacent_low_ink_path']];d.line(p,fill='#db4fad',width=2)
    board=Image.new('RGB',(max(720,a.width*2+12),a.height+50),'#f6f3eb');board.paste(a,(0,50));board.paste(b,(a.width+12,50))
    ImageDraw.Draw(board).text((8,8),r['id']+' 原图 / 青=助手候选段 紫=相邻对照（非纸面真值、非负标签）',font=ImageFont.truetype(FONT,16),fill='#253f36')
    board.save(ROOT/'views'/(r['id']+'_source_review.png'))
    return box


def prepare():
    ROOT.mkdir(exist_ok=True);(ROOT/'views').mkdir(exist_ok=True)
    assert not (ROOT/'draft_references.json').exists(), 'Keep draft immutable'
    rows=[]
    for ident,suffix,obj,path,adj,note in SPECS:
        iid='zouzhe_xiangjiaoqiuwutu_'+suffix;image=SOURCE/'raw'/(iid+'.jpg')
        r=dict(id=ident,image_id=iid,image=str(image),source_sha256=sha(image),object_type=obj,runs=[path],adjacent_low_ink_path=adj,
               native_size=[512,512],review_note=note,scope='assistant_visible_fragment_not_artist_complete_stroke',
               training_allowed=False,expert_verified=False,source_and_predictions_previously_seen=True,
               cunfa='unknown',role='unknown',artist_stroke_identity=None,order=None,pressure=None,
               adjacent_path_is_negative_label=False,review_status='pending_source_only_review')
        r['view_box']=review_view(r);rows.append(r)
    unknown=[dict(id='U_leaf',image_id='zouzhe_xiangjiaoqiuwutu_11',box=[350,185,400,230],reason='叶团交叠，不能按单笔分开'),
             dict(id='U_junction',image_id='zouzhe_xiangjiaoqiuwutu_43',box=[94,425,111,447],reason='分枝交汇；不得跨越连接或补笔序')]
    write('draft_references.json',dict(records=rows,unknown_regions=unknown,posthoc_diagnostic=True,independent_validation=False))
    print('Prepared six source-only run candidates')


def freeze():
    path=ROOT/'reviewed_references.json';data=read(path)
    assert all(r['review_status']=='assistant_source_reviewed' for r in data['records'])
    assert not (ROOT/'results.json').exists(), 'Cannot revise labels after scoring'
    write('reference_freeze.json',dict(reference_sha256=sha(path),source_and_predictions_previously_seen=True,
          frozen_before_this_run_scoring=True,independent_validation=False,training_allowed=False))
    print('Frozen',len(data['records']))


def evaluate():
    refs=read(ROOT/'reviewed_references.json');f=read(ROOT/'reference_freeze.json');assert sha(ROOT/'reviewed_references.json')==f['reference_sha256']
    assert not (ROOT/'results.json').exists(), 'Preserve scored diagnosis'
    rows=[]
    for r in refs['records']:
        assert sha(r['image'])==r['source_sha256'];expected={'rock':1,'tree':2,'unknown':None}[r['object_type']]
        out=dict(r,outputs={})
        for arm in ARMS:
            pred=np.load(SOURCE/'predictions'/(r['image_id']+'_'+arm+'.npz'))['prediction'];assert pred.shape==(512,512)
            out['outputs'][arm]=score_visible_runs(pred,r['runs'],expected)
            if expected is not None:
                shifted=[]
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    runs=[[[x+dx,y+dy] for x,y in run] for run in r['runs']]
                    shifted.append(score_visible_runs(pred,runs,expected)['coverage'])
                out['outputs'][arm]['one_pixel_shift_coverage_range']=[min(shifted),max(shifted)]
        rows.append(out)
    write('results.json',dict(records=rows,unknown_regions=refs['unknown_regions'],posthoc_diagnostic=True,
          new_model_trained=False,independent_validation=False,adopted=False,robot_connected=False,
          scope='ownership along assistant supplied visible runs, not automatic path extraction',
          sampling_unit='native image arc length; samples not independent labels; do not pool as new test accuracy'))
    print(json.dumps({r['id']:{a:None if v['coverage'] is None else round(v['coverage'],3) for a,v in r['outputs'].items()} for r in rows}))


if __name__=='__main__':
    {'prepare':prepare,'freeze':freeze,'evaluate':evaluate}[sys.argv[1]]()
