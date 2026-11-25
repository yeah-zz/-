"""Offline curved pigment ribbons for the three reviewed landscape packages."""
from pathlib import Path
import sys, json, copy, hashlib
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from web.catalog import ARTWORKS
from painting_session import save
from scripts.source_aligned_washes import pigment_masks, ribbon_strokes, join_ribbon_sweeps
from scripts.consolidate_selected_artworks import digest, stroke_data
from scripts.source_wash_comparison import render
from scripts.order_selected_artworks import ORDERS
from shuimo import press_to_width_mm

NEAR_RIBBON_WIDTH_FACTOR=2.4
FAR_RIBBON_WIDTH_FACTOR=1.6

# Source-image coordinates. Each guide follows a pictured ridge or rock face.
# region, pigment, curved guide, variable half width in source pixels.
GUIDES={
1:[
 ('左侧中景山','green',[(350,282),(300,320),(243,353),(190,370)],[12,24,25,3]),
 ('左侧中景山','blue',[(384,283),(410,330),(450,373),(494,409)],[12,25,25,3]),
 ('左侧中景山','green',[(356,336),(386,371),(422,405)],[8,18,3]),
 ('中间中景山','green',[(756,279),(705,329),(646,371),(596,400)],[12,28,29,4]),
 ('中间中景山','blue',[(795,278),(828,330),(864,379),(910,420)],[12,26,29,4]),
 ('中间中景山','green',[(780,343),(739,390),(704,421)],[9,20,4]),
 ('右侧中景山','green',[(1239,313),(1200,353),(1156,399),(1117,429)],[10,22,23,3]),
 ('右侧中景山','blue',[(1284,312),(1326,352),(1370,388),(1410,408)],[10,22,24,3]),
 ('右侧中景山','green',[(1436,451),(1390,472),(1347,491),(1310,507)],[8,17,16,3]),
 ('右侧小岛','green',[(1350,783),(1316,806),(1295,831)],[7,15,3]),
 ('右侧小岛','blue',[(1391,789),(1380,814),(1370,836)],[8,16,3]),
 ('左前景山石','green',[(310,665),(244,700),(179,744),(105,788)],[10,30,29,4]),
 ('左前景山石','blue',[(364,681),(343,753),(316,827),(278,893)],[13,32,32,4]),
 ('左前景山石','green',[(408,685),(403,744),(397,805),(386,851)],[7,16,18,3]),
 ('中前景山石','green',[(531,814),(499,845),(467,885),(434,916)],[9,21,20,3]),
 ('中前景山石','blue',[(557,823),(565,855),(575,888)],[7,17,3]),
 ('中前景山石','green',[(700,722),(650,751),(599,786),(575,806)],[10,25,25,3]),
 ('中前景山石','blue',[(760,747),(743,797),(717,850),(692,879)],[10,28,27,3]),
 ('中前景山石','green',[(790,761),(788,806),(795,866)],[7,15,3]),
 ('右前景山石','blue',[(638,893),(644,917),(649,938)],[7,16,3]),
 ('右前景山石','green',[(695,920),(728,935),(756,946)],[8,16,3]),
 ('右前景山石','green',[(920,842),(882,860),(850,873)],[7,16,3]),
 ('右前景山石','blue',[(990,850),(1011,873),(1032,895)],[8,17,3]),
 ('右侧中景山','green',[(1100,566),(1069,584),(1034,596)],[7,15,3]),
 ('左侧远山','green',[(213,148),(178,168),(133,184)],[5,12,3]),
 ('左侧远山','blue',[(239,151),(277,174),(322,187)],[5,12,3]),
 ('中间远山','green',[(620,160),(581,182),(535,197)],[5,13,3]),
 ('中间远山','blue',[(661,161),(697,183),(739,203)],[5,12,3]),
 ('中间远山','blue',[(949,89),(987,121),(1021,145)],[6,15,3]),
 ('中间远山','green',[(913,102),(872,131),(829,154)],[6,16,3]),
 ('右侧远山','blue',[(953,189),(991,220),(1031,240)],[5,13,3]),
 ('右侧远山','blue',[(1214,190),(1252,216),(1287,232)],[5,12,3]),
],
2:[
 ('左侧高山','green',[(146,139),(108,170),(64,207),(12,246)],[8,23,23,3]),
 ('左侧高山','blue',[(227,264),(242,309),(259,351)],[6,16,3]),
 ('左侧高山','green',[(282,297),(308,342),(331,388)],[7,18,3]),
 ('左侧高山','blue',[(68,406),(33,428),(4,449)],[8,25,3]),
 ('左侧高山','blue',[(120,447),(136,489),(150,537)],[9,22,3]),
 ('亭下完整山石','blue',[(396,536),(423,571),(445,621),(454,660)],[8,24,25,3]),
 ('亭下完整山石','green',[(497,537),(516,562),(530,591)],[7,16,3]),
 ('亭下完整山石','green',[(704,469),(696,496),(688,519)],[8,17,3]),
 ('亭下完整山石','green',[(770,481),(777,535),(777,591),(760,641)],[8,23,24,3]),
 ('亭下完整山石','blue',[(800,604),(819,642),(834,683),(841,714)],[6,18,20,3]),
 ('左下山石','blue',[(114,627),(151,658),(179,703),(182,749)],[10,29,29,3]),
 ('左下山石','blue',[(239,677),(258,702),(278,728)],[9,24,3]),
 ('左下山石','green',[(320,756),(363,783),(389,810)],[10,28,3]),
 ('左下山石','blue',[(426,780),(452,801),(477,824)],[7,20,3]),
 ('左下山石','green',[(472,854),(515,849),(557,863)],[6,15,3]),
 ('左下山石','blue',[(636,812),(659,818),(680,823)],[5,12,3]),
 ('右侧山石','green',[(1533,341),(1573,361),(1615,386),(1658,402)],[7,20,20,3]),
 ('右侧山石','blue',[(1451,405),(1433,425),(1418,452)],[6,16,3]),
 ('右侧山石','blue',[(1510,571),(1498,611),(1485,649)],[7,18,3]),
 ('右侧山石','green',[(1677,489),(1665,535),(1651,581),(1636,630)],[8,22,24,3]),
 ('右侧山石','blue',[(1579,551),(1571,593),(1560,634)],[6,16,3]),
 ('左侧远山','blue',[(372,226),(394,253),(418,276)],[5,12,3]),
 ('中间远山','blue',[(908,529),(936,551),(965,575)],[5,12,3]),
 ('中间远山','blue',[(1279,487),(1307,510),(1345,532)],[5,13,3]),
 ('右侧远山','blue',[(1393,381),(1418,397),(1437,414)],[5,11,3]),
],
3:[
 ('左上岸石','blue',[(117,563),(140,590),(162,621)],[8,23,3]),
 ('左上岸石','blue',[(224,604),(244,624),(261,643)],[8,22,3]),
 ('左下岸石','blue',[(20,723),(52,766),(69,806)],[10,27,3]),
 ('左下岸石','blue',[(148,760),(167,791),(180,822)],[8,24,3]),
 ('左下岸石','green',[(0,587),(73,610),(148,633),(223,658)],[12,29,30,5]),
 ('中间岸石','green',[(244,665),(318,685),(391,704),(471,726)],[10,28,29,4]),
 ('中间岸石','blue',[(355,669),(389,686),(423,701)],[6,15,3]),
 ('中间岸石','blue',[(310,752),(345,765),(375,779)],[5,14,3]),
 ('右端岸石','green',[(477,731),(541,748),(605,766),(656,777)],[9,22,23,3]),
 ('右端岸石','blue',[(577,711),(590,731),(603,747)],[7,17,3]),
 ('右端岸石','blue',[(727,747),(744,757),(759,767)],[6,16,3]),
 ('左侧远山','blue',[(553,224),(518,250),(478,278),(438,300)],[7,20,23,3]),
 ('左侧远山','green',[(585,241),(620,268),(658,294),(702,315)],[7,19,21,3]),
 ('中间远山','blue',[(891,312),(865,326),(837,341)],[5,12,3]),
 ('中间远山','blue',[(1088,263),(1120,281),(1158,297)],[5,12,3]),
 ('右侧远山','blue',[(1334,269),(1293,291),(1250,314)],[7,19,3]),
 ('右侧远山','green',[(1392,281),(1368,303),(1330,329)],[6,16,3]),
 ('右侧远山','blue',[(1503,346),(1466,365),(1418,382)],[6,15,3]),
 ('左侧小岛','green',[(601,360),(570,372),(537,382)],[5,13,3]),
 ('中间小岛','green',[(804,388),(842,402),(876,409)],[5,12,3]),
 ('右侧岛岸','blue',[(1105,342),(1133,355),(1160,363)],[5,13,3]),
],
}


def box(mask,coords):
    x0,y0,x1,y1=coords
    cv2.rectangle(mask,(x0,y0),(x1,y1),1,-1)


def previews(old,new,destination,title,black_cleanup=False,canopy_fill=False):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    ink=new['ink_job']['meta']['ink_execution']['base_strokes']
    colors=new['workflow']['modules']['color']['strokes']
    pts=np.vstack([s['points'] for b in (old,new) for s in stroke_data(b)['black']+stroke_data(b)['color']])
    lo=np.floor(pts.min(0)-18).astype(int);hi=np.ceil(pts.max(0)+18).astype(int)
    board=Image.new('RGB',(1800,850),'#f7f7f0');d=ImageDraw.Draw(board)
    d.text((25,15),title+(' · 叶团填色前后对照' if canopy_fill else ' · 提线修复前后对照' if black_cleanup else ' · 实际上色轨迹前后对照'),font=font,fill='#304c3b')
    for i,(b,label) in enumerate(((old,'修改前'),(new,'修改后 · 叶团重叠铺满' if canopy_fill else '修改后 · 清理碎线与轮廓凸起' if black_cleanup else '修改后 · 顺山势弯曲色带'))):
        pic=render(b).crop((*lo,*hi));pic.thumbnail((855,665),Image.Resampling.LANCZOS)
        x=25+890*i;d.text((x,65),label,font=font,fill='#304c3b');board.paste(pic,(x+(855-pic.width)//2,110+(665-pic.height)//2))
    d.text((25,803),'叶团下压0.19mm，中心笔距约0.64mm；线宽、颜色与纸上渗色为未实测模型。' if canopy_fill else '山石下压0.30mm，中心笔距1.0mm；线宽、颜色与纸上渗色为未实测模型。',font=small,fill='#687562')
    board.save(destination/('叶团补满前后对照.png' if canopy_fill else '提线修复前后对照.png' if black_cleanup else '原图色带前后对照.png'))
    render(new).save(destination/'任务预览_线宽色彩为估计.png')
    board=Image.new('RGB',(1440,1080),'#f7f7f0');d=ImageDraw.Draw(board)
    d.text((25,15),title+' · 顺皴刷色（实际执行轨迹）',font=font,fill='#304c3b')
    stages=[('山水勾勒',[s for s in ink if s.get('role')!='title']),('蓝色顺势刷色',[s for s in colors if s['color']=='blue']),('绿色顺势刷色',[s for s in colors if s['color']=='green']),('最后题字',[s for s in ink if s.get('role')=='title'])]
    m=new['ink_job']['meta'];w,h=m['image_size'];scale=m['ink_execution']['mm_per_px'];canvas=np.full((h,w,3),255,np.uint8)
    for i,(label,ss) in enumerate(stages):
        color=[(40,40,40),(175,125,72),(97,152,97),(40,40,40)][i]
        for s in ss:
            cv2.polylines(canvas,[np.rint(s['points']).astype(np.int32)],False,color,max(1,round(press_to_width_mm(s['press'][0])/scale)),cv2.LINE_AA)
        pic=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)).crop((*lo,*hi));pic.thumbnail((680,415),Image.Resampling.LANCZOS)
        x=25+(i%2)*710;y=65+(i//2)*500;d.text((x,y),'%d · %s（%d笔）'%(i+1,label,len(ss)),font=font,fill='#304c3b');board.paste(pic,(x+(680-pic.width)//2,y+45+(415-pic.height)//2))
    d.text((25,1045),'按真实 JSON 绘制；云雾、树干及建筑留白，实际墨晕与渗色需纸上确认。',font=small,fill='#687562')
    board.save(destination/'顺皴刷色预览.png')


def build(number):
    path=ROOT/ARTWORKS[number][3];package=path.parent.parent
    from web.execution import prepare
    prepare(path)
    old=json.loads(path.read_text(encoding='utf-8'));new=copy.deepcopy(old)
    audit=json.loads((path.parent/'离线检查.json').read_text(encoding='utf-8'))
    transform=audit['source_transform'];ratio=transform['scale'];offset=np.array(transform['offset'])
    scale=new['ink_job']['meta']['ink_execution']['mm_per_px'];pixel_mm=scale*ratio
    image=cv2.imdecode(np.fromfile(str(package/'参考原图.png'),np.uint8),1);shape=image.shape[:2]
    ink=new['ink_job']['meta']['ink_execution']['base_strokes']
    raw=pigment_masks(image);inset=1.5+press_to_width_mm(.30)/2;radius=int(np.ceil(inset/pixel_mm))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    # Complete dark source footprints also exclude broad tree trunks and title.
    dark=(image.max(axis=2)<85).astype(np.uint8)
    branch=np.zeros(shape,np.uint8)
    lines=np.zeros(shape,np.uint8)
    for s in ink:
        if s.get('role') in ('mountain_outline','near_outline','far_outline','branch','pavilion','stairs','boat','title'):
            q=np.rint((np.asarray(s['points'])-offset)/ratio).astype(np.int32)
            cv2.polylines(lines,[q],False,1,1)
            if s.get('role')=='branch':cv2.polylines(branch,[q],False,1,1)
    objects=np.zeros(shape,np.uint8)
    if number==2:
        box(objects,(505,280,716,443));box(objects,(386,429,554,509))
    if number==3:box(objects,(1030,459,1337,546))
    old_colors=old['workflow']['modules']['color']['strokes']
    kept=[copy.deepcopy(s) for s in old_colors if s.get('role') not in ('mountain_color','distant_color','terrain_wash')]
    for s in kept:
        if s.get('role')=='leaf_color':
            q=np.rint((np.asarray(s['points'])-offset)/ratio).astype(np.int32)
            cv2.polylines(objects,[q],False,1,max(1,int(np.ceil(s.get('brush_width_mm_estimate',1.1)/pixel_mm))))
    contour_inset=.5+press_to_width_mm(.30)/2
    contour_radius=int(np.ceil(contour_inset/pixel_mm))
    contour_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*contour_radius+1,2*contour_radius+1))
    protected=cv2.dilate((dark&cv2.dilate(branch,kernel))|objects,kernel)|cv2.dilate(lines,contour_kernel)
    # The broad pigment envelope controls the faded mountain feet. Individual
    # blue/green masks determine hue; source white channels stay unpainted.
    envelope=cv2.morphologyEx(raw['blue']|raw['green'],cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(61,61)))
    envelope=cv2.dilate(envelope,kernel)
    safe=cv2.erode(envelope,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    safe[protected>0]=0
    new_colors=[];masks={'safe':safe,'protected':protected,'pigment_envelope':envelope}
    blue_contacts=np.zeros(shape,np.uint8);constraints=[]
    gap=press_to_width_mm(.30)+.75;rr=int(np.ceil(gap/pixel_mm));gap_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rr+1,2*rr+1))
    for j,(region,color,guide,widths) in sorted(enumerate(GUIDES[number]),key=lambda item:('blue','green').index(item[1][1])):
        factor=FAR_RIBBON_WIDTH_FACTOR if '远山' in region else NEAR_RIBBON_WIDTH_FACTOR
        widths=[v*factor for v in widths]
        allowed=safe.copy()
        if color=='green':allowed[cv2.dilate(blue_contacts,gap_kernel)>0]=0
        paths,mask=ribbon_strokes(guide,widths,allowed,raw[color],color,region,j,j,scale,ratio,offset)
        paths=join_ribbon_sweeps(paths,mask,scale,ratio,offset)
        for k,s in enumerate(paths):
            s['id']='threewash-%d-%02d-%s-%03d'%(number,j,color,k)
            s['subject_order']=ORDERS[number].index(region)
            if color=='blue':
                q=np.rint((np.asarray(s['points'])-offset)/ratio).astype(np.int32);cv2.polylines(blue_contacts,[q],False,1,1)
        new_colors.extend(paths);masks['ribbon_%d'%j]=mask
        constraints.append(dict(region=region,color=color,guide=guide,half_width_px=widths,strokes=len(paths)))
    assert new_colors and all(any(s['color']==c for s in new_colors) for c in ('blue','green'))
    colors=kept+new_colors
    colors.sort(key=lambda s:(('blue','green').index(s['color']),ORDERS[number].index(s['drawing_region']),s.get('source_ribbon_index',-1),s['points'][0][1]))
    plan=new['workflow'];plan['modules']['color'].update(strokes=colors,stroke_count=len(colors))
    count=Counter(s['color'] for s in colors)
    plan['fill_report']={c:dict(strokes=n,terrain_wash_strokes=sum(s.get('role')=='terrain_wash' and s['color']==c for s in colors),coverage_is_model_estimate=True) for c,n in count.items()}
    plan['color_geometry'].update(pitch_mm=1.0,pitch_px=1.0/pixel_mm,max_stroke_mm=50,terrain_pressure_mm=.30,pressure_range_mm=[min(v for s in colors for v in s['press']),max(v for s in colors for v in s['press'])],source='source hue masks intersected with curved mountain and rock-face ribbons')
    note='按用户反馈扩大蓝绿覆盖：近中景色带宽度为首版的2.4倍，远山1.6倍，仍按原图颜料分区与山势裁切；山石下压0.30mm、中心笔距1.0mm，同色带最多三趟安全来回刷、接触路径不超过50mm；山脚收窄，保留云雾、树干、建筑与水面留白；保留原勾线、题字、树叶和舟体上色；蓝色完成后换绿色，题字最后。'
    plan['source_revision']=dict(terrain_fill='three_source_aligned_curved_ribbons',artwork_number=number,source_transform=transform,
        terrain_centerline_inset_mm=inset,terrain_bleed_allowance_mm=1.5,terrain_blue_green_centerline_gap_mm=gap,
        terrain_contour_centerline_inset_mm=contour_inset,terrain_contour_bleed_allowance_mm=.5,
        terrain_strip_pitch_mm=1.0,terrain_pressure_mm=.30,terrain_ribbon_guides=constraints,terrain_direction='curved-source-folds',
        near_ribbon_width_factor=NEAR_RIBBON_WIDTH_FACTOR,far_ribbon_width_factor=FAR_RIBBON_WIDTH_FACTOR,
        source_sha256=hashlib.sha256((package/'参考原图.png').read_bytes()).hexdigest(),note=note,
        stroke_data_sha256=digest(stroke_data(new)))
    plan['note']=note
    from run_color_from_job import validate_plan
    validate_plan(plan,new['ink_job'])
    from run_json_painting import manifest_for
    manifest_for(ink,plan,new['ink_job']['meta'])
    pending=path.parent/'待应用_顺皴刷色';pending.mkdir(exist_ok=True)
    save(pending/'painting.json',new);np.savez_compressed(pending/'原图色带禁涂掩膜.npz',**masks)
    from scripts.validate_three_source_washes import audit_candidate
    result=audit_candidate(pending/'painting.json',path);save(pending/'真实轨迹离线校验.json',result)
    previews(old,new,pending,ARTWORKS[number][1])
    (pending/'原图色带修订说明.md').write_text('# '+ARTWORKS[number][1]+' · 顺皴刷色修订\n\n'+note+'\n\n黑线、题字、叶色及舟体保留逐笔核对；真实接触段与转弯按0.05mm独立采样检查。笔宽和渗色仍需实测。\n',encoding='utf-8')
    save(pending/'应用记录.json',dict(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False,policy='Apply before next new run only when painting and wash locks are absent'))
    print(ARTWORKS[number][1],json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    for number in (1,2,3):build(number)
