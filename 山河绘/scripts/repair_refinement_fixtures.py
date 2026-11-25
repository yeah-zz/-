import sys
sys.dont_write_bytecode=True
from paper_center_refinement_data import ROOT
from corrected_stroke_fixture import fixture
from joint_ink_structure_data import write_record,hard_fixture
from rock_v15_common import read,jsave,digest
import numpy as np,cv2,shutil
from PIL import Image,ImageDraw,ImageFont
STAGE=ROOT/'corrected_renderer_stage3'
def prepare():
    if (STAGE/'manifest.json').exists():raise FileExistsError('Preserve corrected dataset')
    STAGE.mkdir(exist_ok=True);manifest=read(ROOT/'manifest.json');rows=[]
    for r in manifest['records']:
        if r['kind']!='hard_declared':rows.append(r);continue
        old=read(r['reference']);rgb,t,m,role,rel,rm,meta=fixture(old['seed'],old['case']);meta.update(split=r['split'],kind='hard_declared',source_group=r['source_group'],replaces_image_sha256=r['sha256'])
        rows.append(write_record(STAGE/'corrected_hard'/r['split'],r['id'],rgb,t,m,role,rel,rm,meta))
    manifest.update(records=rows,historical_fixture_bug='two-vertex needle paths were rendered as endpoints; old models and evaluations may be affected',historical_pine_test_valid=False,source_sha256=digest(__import__('corrected_stroke_fixture').__file__))
    jsave(STAGE/'manifest.json',manifest);checks=[]
    for seed in range(920000,920008):
        rgb,t,m,role,rel,rm,doc=fixture(seed,'needles')
        # Test RGB at each declared segment midpoint, not only target masks.
        for i,row in enumerate(doc['strokes']):
            q=np.rint(np.mean(row['points'],axis=0)).astype(int);painted=bool(t[1,q[1],q[0]]>.5);checks.append(dict(seed=seed,stroke=i,midpoint_painted=painted));assert painted
    a=hard_fixture(920000,'needles')[0];b=fixture(920000,'needles')[0];f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);board=Image.new('RGB',(768,300),'#f3f1eb');d=ImageDraw.Draw(board)
    for j,(im,label) in enumerate([(a,'旧程序输入：仅端点'),(b,'修正输入：完整线段'),(np.repeat(np.where(fixture(920000,'needles')[1][0]>.3,0,255).astype(np.uint8)[:,:,None],3,2),'声明线条参考')]):d.text((j*256+3,4),label,font=f,fill='black');board.paste(Image.fromarray(im),(j*256,36))
    board.save(STAGE/'RENDERER_AUDIT.jpg',quality=95);jsave(STAGE/'renderer_checks.json',dict(checks=checks,all_passed=True,midpoint_checks=len(checks),rendered_vs_label_consistency=True))
    archive=ROOT/'iterations/invalid_sparse_fixture_metrics';archive.mkdir(parents=True,exist_ok=True)
    for name in ['summary.json','index.html']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,archive/name)
    print('CORRECTED_FIXTURES',len(rows),len(checks),flush=True)
if __name__=='__main__':prepare()
