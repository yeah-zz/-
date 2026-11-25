"""Review-only redesign of creek vegetation, generated from executable strokes."""
from pathlib import Path
import sys,copy,json,socket,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from scipy.interpolate import CubicSpline,PchipInterpolator
from PIL import Image
from ink_control import refresh_stats
from shuimo import press_to_width_mm
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.preview_color_coverage_20261009 import narrow_render,make_board,aligned_reference,connect_region
from scripts.creek_open_branches_20261009 import FOLDER,TARGET,TREES,read,save,sha

BASELINE=FOLDER/'历史版本/枝叶整理前/painting.json'
# Small, coherent crowns: source-coordinate anchors, not black outline paths.
CROWNS=[
 ('左侧主松树','松冠左',[(110,475),(117,461),(141,458),(156,449),(178,449),(191,455),(210,456),(223,470),(204,480),(158,484),(126,482)]),
 ('左侧主松树','松冠顶',[(220,440),(234,424),(254,424),(268,409),(291,405),(311,413),(335,411),(353,421),(375,427),(385,442),(357,449),(290,451),(244,447)]),
 ('左侧主松树','松冠中',[(385,512),(394,496),(417,494),(434,483),(457,481),(474,489),(497,489),(511,498),(532,503),(539,515),(516,520),(447,522),(405,519)]),
 ('左侧主松树','松冠右',[(536,582),(544,568),(564,566),(579,555),(602,553),(618,561),(638,563),(652,578),(644,588),(602,591),(558,589)]),
 ('左下小树','岸边灌木',[(60,708),(64,693),(77,691),(84,677),(101,676),(110,687),(124,690),(130,703),(116,712),(86,714)]),
 ('近岸右侧小树','近岸灌木',[(837,831),(844,817),(859,814),(870,802),(889,799),(904,811),(921,811),(934,825),(928,836),(887,840),(854,837)]),
 ('右下岛上树','岛上左簇',[(1008,553),(1018,540),(1035,537),(1046,525),(1062,525),(1074,539),(1085,544),(1083,557),(1050,560),(1023,558)]),
 ('右下岛上树','岛上右簇',[(1090,557),(1097,544),(1110,541),(1120,532),(1135,534),(1142,544),(1156,549),(1158,559),(1133,565),(1103,563)]),
]
BRANCHES=[
 ('左侧主松树',[(299,548),(324,534),(319,514),(287,486),(283,461),(303,433)]),
 ('左侧主松树',[(287,486),(250,485),(219,478),(180,467)]),
 ('左侧主松树',[(452,519),(454,506),(468,497)]),
 ('左侧主松树',[(452,519),(480,548),(523,576),(559,583),(590,575)]),
 ('左下小树',[(96,716),(95,700),(94,690)]),
 ('左下小树',[(95,700),(79,694)]),
 ('近岸右侧小树',[(881,844),(881,828),(884,815)]),
 ('近岸右侧小树',[(881,828),(861,820)]),
 ('右下岛上树',[(1049,571),(1051,552),(1049,538)]),
 ('右下岛上树',[(1129,571),(1123,555),(1113,545)]),
]
# Visually checked plant-interior fragments mislabelled as rock ridges in the
# source tracing. They lie inside the two shrub groups, not the shore perimeter.
PLANT_FRAGMENTS={'00230','00233','00232','00055','00085','00093','00262','00263'}

def curve(points,closed=False):
    p=np.asarray(points,float)
    if closed:p=np.vstack((p,p[0]))
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    t=np.linspace(0,arc[-1],int(np.ceil(arc[-1]))+1)
    return CubicSpline(arc,p,bc_type='periodic')(t) if closed else PchipInterpolator(arc,p)(t)

def main():
    def offline(*a,**k):raise AssertionError('Offline candidate only')
    socket.socket.connect=offline;socket.socket.connect_ex=offline
    if not BASELINE.exists():BASELINE.parent.mkdir(parents=True,exist_ok=True);BASELINE.write_bytes(TARGET.read_bytes())
    old=read(BASELINE);new=copy.deepcopy(old);meta=old['ink_job']['meta']
    r=old['workflow']['source_revision']['source_transform']['scale']
    off=np.asarray(old['workflow']['source_revision']['source_transform']['offset'])
    scale=meta['ink_execution']['mm_per_px'];pxmm=r*scale
    canonical=ROOT/'待画作品/02_松岸曲水_竖题山河绘_绘字修复版/任务资料/painting.json';canon_hash=sha(canonical)
    oldink=meta['ink_execution']['base_strokes'];newink=[];removed=[]
    for s in oldink:
        p=(np.asarray(s['points'])-off)/r
        # The large lower contours share a tree label with adjacent rock: keep
        # them. Only replace the upper branching and the previous added twigs.
        drop=s['id'].split('-')[-1] in PLANT_FRAGMENTS or s['id'].startswith('creek-open-twig-') or (
            s.get('drawing_region')=='左侧主松树' and p[:,1].max()<615.5
            and not s['id'].startswith('song-an-qu-shui-ink-00278'))
        if drop:removed.append(s['id'])
        else:newink.append(copy.deepcopy(s))
    added=[]
    for i,(region,guide) in enumerate(BRANCHES):
        p=curve(guide)*r+off;n=len(p)
        s=dict(id=f'creek-clean-branch-{i:02d}',role='branch',drawing_region=region,
            layer=0,tail_mm=0,points=p.tolist(),press=[.12]*n,speed=[40.]*n)
        added.append(refresh_stats(s,scale))
    # Insert replacement branches at each original tree's last retained stroke.
    # For a shrub without retained strokes use its original predecessor.
    for region in [x for x in old['workflow']['execution_order']['subjects'] if x in TREES]:
        selected=[s for s in added if s['drawing_region']==region]
        indices=[i+1 for i,s in enumerate(newink) if s.get('drawing_region')==region]
        if indices:at=max(indices)
        else:
            first=next(i for i,s in enumerate(oldink) if s.get('drawing_region')==region)
            before={s['id'] for s in oldink[:first]}
            at=max((i+1 for i,s in enumerate(newink) if s['id'] in before),default=0)
        newink[at:at]=selected
    oldcolor=old['workflow']['modules']['color']['strokes'];replacements=[];cluster_stats=[]
    h,w=1024,1536
    for k,(region,label,anchors) in enumerate(CROWNS):
        polygon=curve(anchors,True);mask=np.zeros((h,w),np.uint8)
        cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],1)
        safe=(cv2.distanceTransform(mask,cv2.DIST_L2,5)>(press_to_width_mm(.22)/2+.05)/pxmm).astype(np.uint8)
        ys,xs=np.nonzero(safe);rows=[]
        for j,y in enumerate(np.arange(ys.min(),ys.max()+1,.60/pxmm)):
            x=np.flatnonzero(safe[int(round(y))])
            if len(x)<4:continue
            # Reserve the curved return inside the crown instead of turning
            # outside its edge. Stroke footprint rounds the ends of the pad.
            a=x.min()+.40/pxmm;b=x.max()-.40/pxmm
            if (b-a)*pxmm<1:continue
            xx=np.linspace(a,b,max(3,int(np.ceil((b-a)*pxmm/.25))+1))
            p=np.column_stack((xx,np.full(len(xx),y)))
            if j%2:p=p[::-1]
            p=p*r+off;n=len(p)
            s=dict(id=f'creek-leaf-{k:02d}-{j:03d}',color='green',role='foliage_wash',
                drawing_region=region,vegetation_cluster=label,region_id=f'clean-canopy-{k}',
                points=p.tolist(),press=[.22]*n,speed=[40.]*n,layer=0,
                pressure_mm=None,recipe='simple-canopy-continuous-fill',pigment_recipe_calibrated=False)
            rows.append(refresh_stats(s,scale))
        linked,joins=connect_region(rows,safe,r,off,scale)
        replacements.extend(linked)
        cluster_stats.append(dict(cluster=label,contacts=len(linked),joins=joins,
            total_length_mm=sum(s['length_mm'] for s in linked)))
    revised=[];inserted=set()
    for s in oldcolor:
        region=s.get('drawing_region')
        if s.get('color')=='green' and region in TREES:
            if region not in inserted:
                revised.extend(t for t in replacements if t['drawing_region']==region);inserted.add(region)
        else:revised.append(copy.deepcopy(s))
    assert inserted==TREES
    from web.execution import prepare
    from artwork_order import ink_payload
    prepare(TARGET);new['ink_job']=ink_payload(old['ink_job'],newink)
    plan=new['workflow'];plan['modules']['outline'].update(strokes=copy.deepcopy(newink),stroke_count=len(newink))
    plan['modules']['color'].update(strokes=revised,stroke_count=len(revised))
    plan['source_revision']['stroke_data_sha256']=digest(stroke_data(new))
    plan['clean_vegetation_revision']=dict(status='preview-only-not-installed',date='2026-10-09',
        black_branch_pressure_mm=.12,leaf_pressure_mm=.22,leaf_clusters=8,branches=10,
        removed_upper_branch_ids=removed,clusters=cluster_stats,
        note='四组松冠，三处灌木；仅调整可执行画稿的树枝和绿色叶团，参考原图保留。')
    save(TARGET,new)
    from run_json_painting import load_artwork
    from web.motion_profile import apply_to_artwork,DEFAULTS
    from run_color_experiment import module_lines,reload_stroke_limit
    from color_profile import ColorProfile
    from itertools import groupby
    prepared=apply_to_artwork(load_artwork(TARGET),DEFAULTS);m=prepared[1]['modules']['color']
    profile=ColorProfile(**m['motion_candidate']).to_dict()
    for c in ('blue','green'):
        _,skipped=module_lines(m,c,profile);assert not skipped
        ss=[s for s in revised if s['color']==c]
        regions=[a for a,_ in groupby(s['drawing_region'] for s in ss)]
        assert len(regions)==len(set(regions))
    assert reload_stroke_limit(m)==8
    assert all(.2<=v<=.3 for s in revised for v in s['press'])
    assert all(s['length_mm']<=80.000001 for s in replacements)
    assert [s for s in oldcolor if not (s['color']=='green' and s.get('drawing_region') in TREES)]==[s for s in revised if not (s['color']=='green' and s.get('drawing_region') in TREES)]
    assert [s for s in oldink if s['id'] not in removed]==[s for s in newink if not s['id'].startswith('creek-clean-branch-')]
    assert canon_hash==sha(canonical)
    report=dict(offline_web_load_passed=True,canonical_unchanged=True,hardware_connections=0,
        preserved_other_ink_and_colors=True,previous_vegetation_contacts=sum(s['color']=='green' and s.get('drawing_region') in TREES for s in oldcolor),
        new_vegetation_contacts=len(replacements),cluster_stats=cluster_stats,
        branches=10,leaf_pressure_mm=.22,branch_pressure_mm=.12,
        candidate_sha256=sha(TARGET),baseline_sha256=sha(BASELINE))
    save(FOLDER/'枝叶简化校验.json',report)
    before=narrow_render(old);after=narrow_render(new)
    make_board([before,after],['上一版：细枝、叶片仍较碎','整理后：四组松冠，三处灌木'],
        '第二幅松岸曲水｜枝叶与灌木简化',1800,730).save(FOLDER/'枝叶简化前后对比.jpg',quality=96)
    # A closer view makes crown spacing and the remaining branch structure visible.
    box=tuple(np.rint(np.array([50,395,690,735])*np.array([r,r,r,r])+np.tile(off,2)).astype(int))
    make_board([before.crop(box),after.crop(box)],['上一版','整理后'],
        '松树与左岸灌木局部',1600,680).save(FOLDER/'松树简化局部对比.jpg',quality=96)
    after.save(FOLDER/'候选轨迹预览.png')
    black=copy.deepcopy(new);black['workflow']['modules']['color']['strokes']=[]
    narrow_render(black).save(FOLDER/'新黑线预览.png')
    source=cv2.imdecode(np.fromfile(FOLDER/'参考原图.png',np.uint8),1)
    make_board([aligned_reference(source,new),before,after],['原参考图','上一版候选','简化枝叶候选'],
        '02 松岸曲水｜清理枝叶与灌木').save(FOLDER/'原图与轨迹对比.jpg',quality=95)
    root=FOLDER.parent;folders=sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2].isdigit())
    board=Image.new('RGB',(1500,2465),'white')
    for i,p in enumerate(folders):board.paste(Image.open(p/'原图与轨迹对比.jpg').resize((1500,493),Image.Resampling.LANCZOS),(0,i*493))
    board.save(root/'五幅上色对比总览.jpg',quality=95)
    for path in (root/'检查汇总.json',FOLDER/'检查报告.json'):
        audit=read(path);row=audit['artworks'][1] if 'artworks' in audit else audit
        row.update(clean_vegetation_revision=report,counts={c:sum(s['color']==c for s in revised) for c in ('blue','green')},
                   current_candidate_sha256=sha(TARGET),prior_coverage_metrics_are_historical=True,
                   non_tree_ink_and_title_exactly_preserved=False,title_exactly_preserved=True,
                   note='当前修订也清理了灌木内部误标为山石的碎线，具体变化以枝叶简化校验及脚本清单为准。')
        save(path,audit)
    webaudit=read(root/'网页加载与补蘸校验.json')
    webaudit['artworks'][1].update(non_tree_ink_and_title_unchanged=False,
        title_unchanged=True,clean_vegetation_revision=True,web_load=True,
        no_region_revisits=True,reload_limit=8,current_candidate_sha256=sha(TARGET))
    save(root/'网页加载与补蘸校验.json',webaudit)
    notes=root/'说明.md'
    notes.write_text(notes.read_text(encoding='utf-8').split('\n## 第二幅枝叶简化')[0]+'\n## 第二幅枝叶简化\n最新候选重画上部松枝与绿色叶团，保留四组松冠和三处灌木。绿色0.22mm、细枝0.12mm；逐色逐区域、每次最多8笔及原补蘸长度/时间条件保留。最新校验见第二幅的枝叶简化校验.json。早期色彩完全不变和覆盖率报告为历史记录；尚未替换正式Web任务。\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
