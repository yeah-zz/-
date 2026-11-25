"""Remove reviewed creek foliage frames; keep open twigs and unchanged color paths.

Edits the preview candidate only, never the Web catalog or a running task.
"""
from pathlib import Path
import sys,copy,json,hashlib,socket
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image
from ink_control import refresh_stats
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.preview_color_coverage_20261009 import narrow_render,make_board,aligned_reference

FOLDER=ROOT/'artifacts/color_coverage_moderate_20261009/02_松岸曲水'
TARGET=FOLDER/'任务资料/painting.json'
BASELINE=FOLDER/'历史版本/去叶团黑框前/painting.json'
LEAF_IDS=('00363','00336','00362','00105','00333','00231','00264','00265','00054')
TREES={'左侧主松树','左下小树','近岸右侧小树','右下岛上树'}
# Sparse, open supports in source coordinates. No closed leaf contour is added.
TWIGS=[
 ('左侧主松树',[(221,488),(199,480),(176,468),(147,463)]),
 ('左侧主松树',[(190,475),(182,455)]),
 ('左侧主松树',[(278,463),(285,444),(309,425),(337,416)]),
 ('左侧主松树',[(288,441),(263,429),(246,419)]),
 ('左侧主松树',[(423,524),(441,510),(471,503),(510,496)]),
 ('左侧主松树',[(470,503),(459,488)]),
 ('左侧主松树',[(547,581),(572,585),(601,578),(634,570)]),
 ('左侧主松树',[(600,579),(590,563)]),
 ('左下小树',[(97,720),(91,696),(88,670)]),
 ('左下小树',[(92,700),(70,686)]),
 ('近岸右侧小树',[(874,845),(878,823),(894,801)]),
 ('近岸右侧小树',[(878,823),(850,807)]),
 ('右下岛上树',[(1046,573),(1047,549),(1053,526)]),
 ('右下岛上树',[(1047,550),(1023,539)]),
 ('右下岛上树',[(1158,567),(1164,549),(1174,534)]),
 ('右下岛上树',[(1164,550),(1184,550)]),
]

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    def offline(*a,**k):raise AssertionError('This revision cannot connect to hardware')
    socket.socket.connect=offline;socket.socket.connect_ex=offline
    if not BASELINE.exists():
        BASELINE.parent.mkdir(parents=True,exist_ok=True);BASELINE.write_bytes(TARGET.read_bytes())
    old=read(BASELINE);new=copy.deepcopy(old)
    canonical=ROOT/'待画作品/02_松岸曲水_竖题山河绘_绘字修复版/任务资料/painting.json'
    canonical_hash=sha(canonical)
    meta=old['ink_job']['meta'];scale=meta['ink_execution']['mm_per_px']
    tr=old['workflow']['source_revision']['source_transform'];r=tr['scale'];off=np.array(tr['offset'])
    ink=meta['ink_execution']['base_strokes'];w,h=meta['image_size']
    mask=np.zeros((h,w),np.uint8)
    removal={s['id'] for s in ink if s['id'].split('-')[-1] in LEAF_IDS}
    assert len(removal)==9
    for s in ink:
        if s['id'] not in removal:continue
        q=np.rint(s['points']).astype(np.int32)
        cv2.fillConvexPoly(mask,cv2.convexHull(q),1)
    mask=cv2.dilate(mask,np.ones((5,5),np.uint8))
    output=[];clipped=[];dropped=[]
    for s in ink:
        if s['id'] in removal:dropped.append(s['id']);continue
        if s.get('drawing_region') not in TREES:
            output.append(copy.deepcopy(s));continue
        q=np.rint(s['points']).astype(int)
        inside=mask[q[:,1],q[:,0]]>0
        if not inside.any():output.append(copy.deepcopy(s));continue
        clipped.append(s['id'])
        good=~inside;edges=np.diff(np.r_[False,good,False].astype(int))
        for j,(a,b) in enumerate(zip(np.where(edges==1)[0],np.where(edges==-1)[0])):
            if b-a<2:continue
            part=copy.deepcopy(s)
            for key,value in s.items():
                if isinstance(value,list) and len(value)==len(s['points']):part[key]=value[a:b]
            part.update(id=s['id']+'-open-%d'%j,parent_stroke_id=s['id'])
            refresh_stats(part,scale)
            if part['length_mm']>=.8:output.append(part)
    for i,(region,guide) in enumerate(TWIGS):
        p=np.asarray(guide,float)*r+off
        # Resample each open segment at <=.25 mm without changing the guide shape.
        q=[]
        for a,b in zip(p[:-1],p[1:]):
            q.extend(np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)*scale/.25))+1))[:-1])
        q=np.vstack([*q,p[-1]]);n=len(q)
        example=next(s for s in ink if s.get('drawing_region')==region)
        st=dict(id=f'creek-open-twig-{i:02d}',role='branch',drawing_region=region,layer=0,tail_mm=0,
                subject_order=example.get('subject_order',0),points=q.tolist(),
                press=[.15]*n,speed=[40.]*n,recipe='open-branch-no-foliage-frame')
        output.append(refresh_stats(st,scale))
    # Insert twigs after the existing strokes of their tree; preserve the exact
    # relative sequence of every unrelated stroke, including the final title.
    existing=[s for s in output if not s['id'].startswith('creek-open-twig-')]
    added=[s for s in output if s['id'].startswith('creek-open-twig-')]
    last={s.get('drawing_region'):i for i,s in enumerate(existing)}
    output=[];inserted=set()
    for i,s in enumerate(existing):
        output.append(s);region=s.get('drawing_region')
        if i==last[region]:
            output.extend(t for t in added if t['drawing_region']==region);inserted.add(region)
    # A small shrub may have consisted entirely of its removed leaf contour.
    # Anchor its new twigs at its original place rather than after the title.
    for region in TREES-inserted:
        before_ids=[s['id'] for s in ink[:next(i for i,s in enumerate(ink) if s.get('drawing_region')==region)]]
        at=max((i+1 for i,s in enumerate(output) if s['id'] in before_ids or s.get('parent_stroke_id') in before_ids),default=0)
        output[at:at]=[t for t in added if t['drawing_region']==region]
    from web.execution import prepare
    from artwork_order import ink_payload
    prepare(TARGET)
    new['ink_job']=ink_payload(old['ink_job'],output)
    plan=new['workflow'];plan['modules']['outline'].update(strokes=copy.deepcopy(output),stroke_count=len(output))
    plan['source_revision']['stroke_data_sha256']=digest(stroke_data(new))
    plan['open_foliage_revision']=dict(date='2026-10-09',removed_foliage_frame_ids=sorted(removal),
        clipped_tree_stroke_ids=clipped,added_open_twigs=len(TWIGS),twig_pressure_mm=.15,
        color_strokes_exactly_preserved=True,status='preview-only-not-installed')
    save(TARGET,new)
    # Validate the exact plan Web will load. No changes to color/reloading/order.
    from run_json_painting import load_artwork
    from web.motion_profile import apply_to_artwork,DEFAULTS
    from run_color_experiment import module_lines
    from color_profile import ColorProfile
    prepared=apply_to_artwork(load_artwork(TARGET),DEFAULTS)
    color=prepared[1]['modules']['color'];profile=ColorProfile(**color['motion_candidate']).to_dict()
    for c in ('blue','green'):
        _,skipped=module_lines(color,c,profile);assert not skipped
    assert old['workflow']['modules']['color']==new['workflow']['modules']['color']
    old_untouched={s['id']:s for s in ink if s.get('drawing_region') not in TREES}
    assert old_untouched=={s['id']:s for s in output if s.get('drawing_region') not in TREES}
    assert [s['id'] for s in ink if s.get('drawing_region') not in TREES]==[s['id'] for s in output if s.get('drawing_region') not in TREES]
    assert not any(s['id'] in removal for s in output)
    assert len({s['id'] for s in output})==len(output)
    assert sha(canonical)==canonical_hash
    report=dict(removed_leaf_frames=len(removal),removed_or_clipped_internal_paths=len(clipped),
                added_open_twigs=len(TWIGS),old_ink_strokes=len(ink),new_ink_strokes=len(output),
                non_tree_ink_and_title_exactly_preserved=True,color_module_exactly_preserved=True,
                canonical_unchanged=True,offline_web_load_passed=True,hardware_connections=0,
                baseline_sha256=sha(BASELINE),candidate_sha256=sha(TARGET))
    save(FOLDER/'去叶团黑框检查.json',report)
    before=narrow_render(old);after=narrow_render(new)
    board=make_board([before,after],['修改前：叶团黑框与内部碎线','修改后：细枝支撑，绿色阶段画叶团'],
                     '第二幅松岸曲水｜松树与灌木去黑框',1800,730)
    board.save(FOLDER/'枝叶去黑框前后对比.jpg',quality=96)
    after.save(FOLDER/'候选轨迹预览.png')
    black=copy.deepcopy(new);black['workflow']['modules']['color']['strokes']=[]
    narrow_render(black).save(FOLDER/'新黑线预览.png')
    source=cv2.imdecode(np.fromfile(FOLDER/'参考原图.png',np.uint8),cv2.IMREAD_COLOR)
    ref=aligned_reference(source,old)
    make_board([ref,before,after],['原图（对齐画布）','原候选：叶团带黑框','新候选：细枝＋绿色叶团'],
               '02 松岸曲水｜枝叶去框修订').save(FOLDER/'原图与轨迹对比.jpg',quality=95)
    root=FOLDER.parent
    folders=sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2].isdigit())
    contact=Image.new('RGB',(1500,493*5),'white')
    for i,folder in enumerate(folders):
        pic=Image.open(folder/'原图与轨迹对比.jpg');contact.paste(pic.resize((1500,493),Image.Resampling.LANCZOS),(0,i*493))
    contact.save(root/'五幅上色对比总览.jpg',quality=95)
    audit=read(root/'检查汇总.json')
    audit['artworks'][1].update(black_and_title_exactly_preserved=False,
        non_tree_ink_and_title_exactly_preserved=True,open_foliage_revision=report)
    save(root/'检查汇总.json',audit)
    per=read(FOLDER/'检查报告.json');per.update(black_and_title_exactly_preserved=False,
        non_tree_ink_and_title_exactly_preserved=True,open_foliage_revision=report);save(FOLDER/'检查报告.json',per)
    webaudit=read(root/'网页加载与补蘸校验.json')
    webaudit['artworks'][1].update(black_unchanged=False,non_tree_ink_and_title_unchanged=True,open_foliage_revision=True)
    save(root/'网页加载与补蘸校验.json',webaudit)
    notes=root/'说明.md'
    notes.write_text(notes.read_text(encoding='utf-8').split('\n## 第二幅后续修订')[0]+'\n## 第二幅后续修订\n按用户圈选移除9条叶团封边，清理叶团内部碎轮廓，新增16笔0.15mm开放细枝。山石、水纹、题字以及全部上色、补蘸设置保持原候选。最新黑线数据以去叶团黑框检查.json为准；仍未安装到Web正式画稿。\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
