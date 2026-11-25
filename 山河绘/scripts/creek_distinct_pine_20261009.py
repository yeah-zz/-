"""A distinct creek pine: original creek boughs, fine forks, seven uneven crowns."""
from pathlib import Path
import sys,copy,json,socket
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image
from ink_control import refresh_stats
from shuimo import press_to_width_mm
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.creek_open_branches_20261009 import FOLDER,TARGET,read,save,sha
from scripts.creek_clean_foliage_20261009 import curve
from scripts.preview_color_coverage_20261009 import narrow_render,make_board,aligned_reference,connect_region

# Retain the second artwork's composition and its own original branching.
BASELINE=FOLDER/'历史版本/松树对齐第一幅前/painting.json'
ORIGINAL_BRANCHES=FOLDER/'历史版本/枝叶整理前/painting.json'
RESTORE={'song-an-qu-shui-ink-00295-open-0','song-an-qu-shui-ink-00335',
         'song-an-qu-shui-ink-00283','song-an-qu-shui-ink-00103-open-0',
         'song-an-qu-shui-ink-00106-open-0','song-an-qu-shui-ink-00137-open-0'}
# The positions and outlines are specific to artwork 2, not transformed copies
# of artwork 1. Uneven lower edges avoid the previous flat cloud-symbol shape.
CROWNS=[
 ('左展松冠',[(106,476),(115,463),(136,461),(148,450),(171,452),(184,458),(207,459),(225,470),(214,483),(195,484),(180,480),(165,486),(146,483),(130,487),(117,483)]),
 ('顶松冠',[(217,441),(230,425),(249,427),(264,410),(285,403),(300,410),(318,409),(333,418),(352,421),(370,437),(357,448),(338,446),(321,452),(302,447),(284,454),(262,447),(243,451)]),
 ('上侧松冠',[(333,470),(341,457),(359,454),(368,441),(386,444),(397,452),(412,454),(422,467),(409,476),(390,471),(375,480),(358,473),(345,476)]),
 ('中展松冠',[(405,514),(415,499),(431,500),(446,487),(466,486),(481,497),(502,493),(518,504),(531,515),(513,525),(494,520),(477,530),(458,523),(441,528),(423,522)]),
 ('左下小松冠',[(102,507),(112,497),(130,495),(140,498),(156,499),(163,508),(148,516),(132,512),(118,517)]),
 ('右侧小松冠',[(488,548),(500,537),(515,535),(526,526),(542,530),(553,537),(566,540),(571,551),(556,558),(539,553),(524,561),(507,555)]),
 ('末端松冠',[(566,589),(576,575),(593,575),(606,558),(625,557),(638,568),(655,566),(668,582),(660,595),(641,593),(625,601),(609,595),(592,602),(577,596)]),
]
TWIGS=[[(283,456),(285,444),(305,429)],[(286,445),(263,432)],
       [(222,480),(196,478),(170,465)],[(208,481),(184,492),(159,502),(136,503)],
       [(348,533),(365,508),(367,478),(378,463)],
       [(448,524),(465,514),(486,503)],
       [(510,567),(516,554),(534,545)],
       [(568,594),(591,591),(620,578)]]

def main():
    def offline(*a,**k):raise AssertionError('No hardware connection in preview generation')
    socket.socket.connect=offline;socket.socket.connect_ex=offline
    old=read(BASELINE);new=copy.deepcopy(old);original=read(ORIGINAL_BRANCHES)
    reference_path=FOLDER.parent/'01_松瀑亭舟/任务资料/painting.json';reference=read(reference_path)
    protected=[reference_path,ROOT/'待画作品/02_松岸曲水_竖题山河绘_绘字修复版/任务资料/painting.json']
    hashes={str(p):sha(p) for p in protected}
    meta=old['ink_job']['meta'];scale=meta['ink_execution']['mm_per_px']
    tr=old['workflow']['source_revision']['source_transform'];r=tr['scale'];off=np.array(tr['offset']);pxmm=r*scale
    ink=meta['ink_execution']['base_strokes'];kept=[copy.deepcopy(s) for s in ink if not (s['id'].startswith('creek-clean-branch-') and s['drawing_region']=='左侧主松树')]
    additions=[]
    for s in original['ink_job']['meta']['ink_execution']['base_strokes']:
        if s['id'] not in RESTORE:continue
        st=copy.deepcopy(s);st.update(id='creek-own-bough-'+s['id'],role='branch')
        st['press']=[.15]*len(st['points']);additions.append(refresh_stats(st,scale))
    assert len(additions)==len(RESTORE)
    for i,guide in enumerate(TWIGS):
        p=curve(guide)*r+off;n=len(p)
        s=dict(id=f'creek-distinct-twig-{i:02d}',role='branch',drawing_region='左侧主松树',
            layer=0,tail_mm=0,points=p.tolist(),press=[.12]*n,speed=[40.]*n)
        additions.append(refresh_stats(s,scale))
    at=max(i+1 for i,s in enumerate(kept) if s.get('drawing_region')=='左侧主松树')
    newink=kept[:at]+additions+kept[at:]
    crowns=[];stats=[]
    for k,(name,anchors) in enumerate(CROWNS):
        mask=np.zeros((1024,1536),np.uint8);polygon=curve(anchors,True)
        cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],1)
        safe=(cv2.distanceTransform(mask,cv2.DIST_L2,5)>(press_to_width_mm(.22)/2+.04)/pxmm).astype(np.uint8)
        yy,xx=np.nonzero(safe);paths=[]
        for j,y in enumerate(np.arange(yy.min(),yy.max()+1,.60/pxmm)):
            xs=np.flatnonzero(safe[int(round(y))])
            if len(xs)<3:continue
            a=xs.min()+.30/pxmm;b=xs.max()-.30/pxmm
            if (b-a)*pxmm<1:continue
            x=np.linspace(a,b,max(3,int(np.ceil((b-a)*pxmm/.25))+1))
            p=np.column_stack((x,np.full(len(x),y)))
            if j%2:p=p[::-1]
            p=p*r+off;n=len(p)
            s=dict(id=f'creek-distinct-crown-{k:02d}-{j:02d}',color='green',role='canopy_fill',
                drawing_region='左侧主松树',region_id=f'creek-distinct-canopy-{k}',canopy_index=k,
                vegetation_cluster=name,points=p.tolist(),press=[.22]*n,speed=[40.]*n,layer=0,
                pressure_mm=None,pigment_recipe_calibrated=False)
            paths.append(refresh_stats(s,scale))
        linked,joins=connect_region(paths,safe,r,off,scale);crowns.extend(linked)
        stats.append(dict(crown=name,contacts=len(linked),joins=joins))
    oldcolor=old['workflow']['modules']['color']['strokes'];newcolor=[];inserted=False
    for s in oldcolor:
        if s['color']=='green' and s.get('drawing_region')=='左侧主松树':
            if not inserted:newcolor.extend(crowns);inserted=True
        else:newcolor.append(copy.deepcopy(s))
    from web.execution import prepare
    from artwork_order import ink_payload
    prepare(TARGET);new['ink_job']=ink_payload(old['ink_job'],newink)
    plan=new['workflow'];plan['modules']['outline'].update(strokes=copy.deepcopy(newink),stroke_count=len(newink))
    plan['modules']['color'].update(strokes=newcolor,stroke_count=len(newcolor))
    plan['source_revision']['stroke_data_sha256']=digest(stroke_data(new))
    plan['distinct_pine_revision']=dict(date='2026-10-09',status='preview-only-not-installed',
        own_branch_source=str(ORIGINAL_BRANCHES),restored_bough_ids=sorted(RESTORE),
        new_crowns=7,canopy_stats=stats,fine_twig_count=8,
        note='保留第二幅自己的树干与主枝走势，借鉴第一幅曲折分叉的画法，重新设计七组不等大的松冠，不使用第一幅的几何轨迹。')
    save(TARGET,new)
    from run_json_painting import load_artwork,manifest_for
    from web.motion_profile import apply_to_artwork,DEFAULTS
    from run_color_experiment import module_lines,reload_stroke_limit
    from color_profile import ColorProfile
    from itertools import groupby
    prepared=apply_to_artwork(load_artwork(TARGET),DEFAULTS);manifest_for(prepared[2],prepared[1],prepared[3])
    module=prepared[1]['modules']['color'];profile=ColorProfile(**module['motion_candidate']).to_dict()
    for c in ('blue','green'):
        _,skipped=module_lines(module,c,profile);assert not skipped
        names=[name for name,_ in groupby(s['drawing_region'] for s in newcolor if s['color']==c)]
        assert len(names)==len(set(names))
    assert reload_stroke_limit(module)==8
    assert all(.2<=v<=.3 for s in newcolor for v in s['press'])
    assert [s for s in ink if not (s['id'].startswith('creek-clean-branch-') and s['drawing_region']=='左侧主松树')]==kept
    assert [s for s in oldcolor if s.get('drawing_region')!='左侧主松树']==[s for s in newcolor if s.get('drawing_region')!='左侧主松树']
    assert [s for s in ink if s['role']=='title']==[s for s in newink if s['role']=='title']
    assert not any(s['id'].startswith('creek-first-pine') for s in newink+newcolor)
    assert hashes=={str(p):sha(Path(p)) for p in hashes}
    report=dict(offline_web_load_passed=True,canonical_and_first_artwork_unchanged=True,hardware_connections=0,
        uses_own_creek_bough_geometry=True,uses_first_artwork_geometry=False,new_canopies=7,
        new_pine_contacts=len(crowns),title_shrubs_and_remaining_scene_preserved=True,
        candidate_sha256=sha(TARGET),baseline_sha256=sha(BASELINE))
    save(FOLDER/'松树独立造型校验.json',report)
    before=narrow_render(old);after=narrow_render(new);ref=narrow_render(reference)
    make_board([ref.crop((205,355,785,710)),after.crop((205,410,730,712))],
        ['第一幅：参考枝干画法','第二幅：独立的枝向与七组松冠'],
        '借鉴枝条画法，保留不同造型',1800,710).save(FOLDER/'松树不同造型对照.jpg',quality=96)
    make_board([before,after],['上一版：过于简化','新方案：第二幅自己的枝向与松冠'],
        '第二幅松岸曲水｜独立松树造型',1800,730).save(FOLDER/'松树独立造型前后对比.jpg',quality=96)
    after.save(FOLDER/'候选轨迹预览.png')
    black=copy.deepcopy(new);black['workflow']['modules']['color']['strokes']=[]
    narrow_render(black).save(FOLDER/'新黑线预览.png')
    source=cv2.imdecode(np.fromfile(FOLDER/'参考原图.png',np.uint8),1)
    make_board([aligned_reference(source,new),before,after],['原参考图','简化版候选','独立松树造型候选'],
        '02 松岸曲水｜不同造型，相近笔法').save(FOLDER/'原图与轨迹对比.jpg',quality=95)
    root=FOLDER.parent;folders=sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2].isdigit())
    overview=Image.new('RGB',(1500,2465),'white')
    for i,p in enumerate(folders):overview.paste(Image.open(p/'原图与轨迹对比.jpg').resize((1500,493),Image.Resampling.LANCZOS),(0,i*493))
    overview.save(root/'五幅上色对比总览.jpg',quality=95)
    for path in (root/'检查汇总.json',FOLDER/'检查报告.json'):
        audit=read(path);row=audit['artworks'][1] if 'artworks' in audit else audit
        row.pop('pine_matched_to_first',None)
        row.update(distinct_pine_revision=report,current_candidate_sha256=sha(TARGET),
            counts={c:sum(s['color']==c for s in newcolor) for c in ('blue','green')})
        save(path,audit)
    audit=read(root/'网页加载与补蘸校验.json');row=audit['artworks'][1]
    row.pop('pine_matched_to_first',None);row.update(distinct_pine_revision=True,current_candidate_sha256=sha(TARGET),web_load=True)
    save(root/'网页加载与补蘸校验.json',audit)
    notes=root/'说明.md';notes.write_text(notes.read_text(encoding='utf-8').split('\n## 第二幅松树对齐第一幅')[0]+
        '\n## 第二幅独立松树造型\n根据用户纠正，不采用第一幅的树形或轨迹。保留第二幅原有主枝走势，重新设计七组错落松冠，借鉴第一幅曲折分叉的画法。沿用当前横题、删顶排远山、三处灌木和补蘸设置。最新校验见松树独立造型校验.json；复制第一幅的中间方案不采用。尚未安装到Web正式任务。\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
