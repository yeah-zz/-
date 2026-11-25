"""Restore the creek reference's four crown groups in executable candidate paths."""
from pathlib import Path
import sys,copy,json,socket
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image
from ink_control import refresh_stats
from shuimo import press_to_width_mm
from scripts.creek_open_branches_20261009 import FOLDER,TARGET,read,save,sha
from scripts.creek_clean_foliage_20261009 import curve
from scripts.preview_color_coverage_20261009 import narrow_render,make_board,aligned_reference,connect_region
from scripts.consolidate_selected_artworks import digest,stroke_data

BASELINE=FOLDER/'历史版本/原图四组松冠调整前/painting.json'
# Actual source-image crown extents, not invented cloud silhouettes.
CROWNS=[('左松冠',(94,440,244,495)),('顶松冠',(208,391,404,465)),
        ('中松冠',(375,466,553,537)),('右松冠',(513,541,675,614))]
# Open, connected forks follow the reference. No foliage outline is traced.
TWIGS=[
 [(223,475),(208,483),(186,480),(169,474),(157,463)],
 [(169,474),(177,466),(181,454)],
 [(186,480),(202,472),(213,460)],
 [(269,478),(281,458),(276,443),(263,432)],
 [(276,443),(277,426),(282,418)],
 [(281,458),(302,450),(317,436),(321,421)],
 [(302,450),(329,453),(350,448),(366,439)],
 [(329,453),(339,438),(340,427)],
 [(405,523),(420,512),(425,495),(434,486)],
 [(420,512),(405,505),(398,496)],
 [(429,521),(452,523),(469,510),(479,492)],
 [(469,510),(491,516),(510,507)],
 [(555,594),(562,581),(560,566),(552,558)],
 [(562,581),(576,570),(579,556)],
 [(578,595),(603,591),(616,579),(622,565)],
 [(603,591),(628,596),(650,586)],
]

def main():
    def offline(*a,**k):raise AssertionError('Offline candidate generation only')
    socket.socket.connect=offline;socket.socket.connect_ex=offline
    if not BASELINE.exists():save(BASELINE,read(TARGET))
    old=read(BASELINE);new=copy.deepcopy(old)
    protected=[ROOT/'待画作品/02_松岸曲水_竖题山河绘_绘字修复版/任务资料/painting.json',FOLDER.parent/'01_松瀑亭舟/任务资料/painting.json',FOLDER/'参考原图.png']
    hashes={str(p):sha(p) for p in protected}
    src=cv2.imdecode(np.fromfile(FOLDER/'参考原图.png',np.uint8),1)
    meta=old['ink_job']['meta'];scale=meta['ink_execution']['mm_per_px']
    tr=old['workflow']['source_revision']['source_transform'];r=tr['scale'];off=np.asarray(tr['offset']);pxmm=r*scale
    ink=meta['ink_execution']['base_strokes']
    kept=[copy.deepcopy(s) for s in ink if not s['id'].startswith('creek-distinct-twig-')]
    twigs=[]
    for i,guide in enumerate(TWIGS):
        p=curve(guide)*r+off;n=len(p)
        # Fine tips taper; short forks never become a black leaf perimeter.
        press=np.linspace(.12,.10,n).tolist()
        twigs.append(refresh_stats(dict(id=f'creek-source-twig-{i:02d}',role='branch',drawing_region='左侧主松树',
            layer=0,tail_mm=0,points=p.tolist(),press=press,speed=[40.]*n),scale))
    at=max(i+1 for i,s in enumerate(kept) if s.get('drawing_region')=='左侧主松树')
    newink=kept[:at]+twigs+kept[at:]
    # Reserve room for black forks so the green path preview does not cover them.
    branches=np.zeros(src.shape[:2],np.uint8)
    for s in newink:
        if s.get('drawing_region')!='左侧主松树':continue
        p=(np.asarray(s['points'])-off)/r
        width=(max(press_to_width_mm(v) for v in s['press'])+.8)/pxmm
        cv2.polylines(branches,[np.rint(p).astype(np.int32)],False,1,max(1,int(np.ceil(width))))
    b,g,red=src.astype(float).transpose(2,0,1)
    pigment=((np.maximum(b,g)-red>15)&(g>55)&(red<205)).astype(np.uint8)
    crowns=[];counts=[];masks=[]
    for k,(label,(x0,y0,x1,y1)) in enumerate(CROWNS):
        mask=np.zeros(src.shape[:2],np.uint8);mask[y0:y1,x0:x1]=pigment[y0:y1,x0:x1]
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        mask[:]=0
        for contour in contours:
            if cv2.contourArea(contour)>35:cv2.drawContours(mask,[contour],-1,1,-1)
        masks.append(mask)
        safe=((cv2.distanceTransform(mask,cv2.DIST_L2,5)>(max(.8,press_to_width_mm(.22))/2+.05)/pxmm)&(branches==0)).astype(np.uint8)
        yy,xx=np.nonzero(safe);assert len(xx)>0
        rows=[]
        # Split each row at genuine mask/branch gaps; never cross empty space.
        for j,y in enumerate(np.arange(yy.min(),yy.max()+.01,.68/pxmm)):
            yi=int(round(y));edges=np.diff(np.r_[0,safe[yi],0].astype(int))
            spans=list(zip(np.where(edges==1)[0],np.where(edges==-1)[0]))
            if j%2:spans.reverse()
            for z,(a,b) in enumerate(spans):
                a+=.16/pxmm;b-=.16/pxmm
                if (b-a)*pxmm<1.0:continue
                x=np.linspace(a,b,max(3,int(np.ceil((b-a)*pxmm/.25))+1))
                p=np.column_stack((x,np.full(len(x),yi)))
                if j%2:p=p[::-1]
                p=p*r+off;n=len(p)
                s=dict(id=f'creek-source-crown-{k:02d}-{j:02d}-{z:02d}',color='green',role='canopy_fill',
                    drawing_region='左侧主松树',region_id=f'creek-source-canopy-{k}',canopy_index=k,
                    vegetation_cluster=label,points=p.tolist(),press=[.22]*n,speed=[40.]*n,layer=0,
                    pressure_mm=None,pigment_recipe_calibrated=False)
                rows.append(refresh_stats(s,scale))
        linked,joins=connect_region(rows,safe,r,off,scale);crowns.extend(linked)
        counts.append(dict(crown=label,contacts=len(linked),continuous_joins=joins,source_bounds=[x0,y0,x1,y1]))
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
    revision=dict(status='preview-only-not-installed',date='2026-10-09',reference_sha256=sha(FOLDER/'参考原图.png'),
        crowns=counts,open_forks=len(twigs),color_pressure_mm=.22,twig_pressure_range_mm=[.10,.12],
        note='以第二幅原图的四组松冠提取范围取代七组设计叶团，沿原图添细分枝，不描叶簇黑框。')
    plan['source_aligned_pine_revision']=revision;save(TARGET,new)
    from run_json_painting import load_artwork,manifest_for
    from web.motion_profile import apply_to_artwork,DEFAULTS
    from run_color_experiment import module_lines,reload_stroke_limit
    from color_profile import ColorProfile
    from itertools import groupby
    prepared=apply_to_artwork(load_artwork(TARGET),DEFAULTS);manifest_for(prepared[2],prepared[1],prepared[3])
    module=prepared[1]['modules']['color'];profile=ColorProfile(**module['motion_candidate']).to_dict()
    for c in ('blue','green'):
        _,skipped=module_lines(module,c,profile);assert not skipped
        regions=[v for v,_ in groupby(s['drawing_region'] for s in newcolor if s['color']==c)]
        assert len(regions)==len(set(regions))
    assert reload_stroke_limit(module)==8
    assert all(.2<=p<=.3 for s in newcolor for p in s['press'])
    assert all(s['length_mm']<=80.0001 for s in crowns)
    assert [s for s in ink if s.get('drawing_region')!='左侧主松树']==[s for s in newink if s.get('drawing_region')!='左侧主松树']
    assert [s for s in oldcolor if s.get('drawing_region')!='左侧主松树']==[s for s in newcolor if s.get('drawing_region')!='左侧主松树']
    assert hashes=={str(p):sha(Path(p)) for p in hashes}
    report=dict(offline_web_load_passed=True,hardware_connections=0,canonical_and_reference_unchanged=True,
        other_geometry_preserved=True,crowns=counts,green_pine_contacts=len(crowns),candidate_sha256=sha(TARGET))
    save(FOLDER/'原图松树对齐校验.json',report)
    before=narrow_render(old);after=narrow_render(new);ref=aligned_reference(src,new)
    crop=(205,426,715,713)
    make_board([ref.crop(crop),before.crop(crop),after.crop(crop)],
        ['第二幅原图','上一版：七组叶团','新版：按原图四组松冠和枝向'],
        '第二幅松树｜原图与实际轨迹对照',2100,730).save(FOLDER/'原图松树修正对照.jpg',quality=96)
    after.save(FOLDER/'候选轨迹预览.png')
    black=copy.deepcopy(new);black['workflow']['modules']['color']['strokes']=[]
    narrow_render(black).save(FOLDER/'新黑线预览.png')
    make_board([ref,before,after],['原参考图','上一版候选','按原图松树调整后的候选'],
        '02 松岸曲水｜保留横题和删远山，松树按原图修正').save(FOLDER/'原图与轨迹对比.jpg',quality=96)
    root=FOLDER.parent;folders=sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2].isdigit())
    overview=Image.new('RGB',(1500,2465),'white')
    for i,p in enumerate(folders):overview.paste(Image.open(p/'原图与轨迹对比.jpg').resize((1500,493),Image.Resampling.LANCZOS),(0,i*493))
    overview.save(root/'五幅上色对比总览.jpg',quality=95)
    for path in (root/'检查汇总.json',FOLDER/'检查报告.json',root/'网页加载与补蘸校验.json'):
        audit=read(path);row=audit['artworks'][1] if 'artworks' in audit else audit
        row.update(source_aligned_pine_revision=report,current_candidate_sha256=sha(TARGET),
            counts={c:sum(s['color']==c for s in newcolor) for c in ('blue','green')})
        save(path,audit)
    notes=root/'说明.md';notes.write_text(notes.read_text(encoding='utf-8').split('\n## 第二幅松树按原图修正')[0]+
        '\n## 第二幅松树按原图修正\n最新候选以第二幅原图四组松冠和枝向为依据，替代七组平顶叶团。原参考图未修改。保留横题、删除顶排远山和其余主体的既有修改。预览由候选JSON渲染；未安装到正式Web任务。见原图松树修正对照.jpg与原图松树对齐校验.json。\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
