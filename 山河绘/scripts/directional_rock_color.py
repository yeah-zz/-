"""Offline source-pigment-aligned curved washes with inset boundaries and bare bark."""
from pathlib import Path
import sys,json,copy,hashlib
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
TERRAIN_PRESS_MM=.30
TERRAIN_PITCH_MM=1.0
NEAR_RIBBON_WIDTH_FACTOR=2.4
FAR_RIBBON_WIDTH_FACTOR=1.6
from web.catalog import ARTWORKS
from painting_session import save
from artwork_order import ink_payload
from ink_control import refresh_stats
from shuimo import press_to_width_mm
from scripts.revise_pine_waterfall import order
from scripts.consolidate_selected_artworks import digest,stroke_data,index_for
from scripts.source_aligned_washes import (RIBBONS,pigment_masks,ribbon_strokes,
    join_ribbon_sweeps,dense_path,points_inside)

REGIONS=[
 ('左岸山石',0,[(0,360),(15,370),(40,455),(70,515),(95,570),(145,657),(115,706),(0,738)]),
 ('左前景山石',1,[(172,661),(198,644),(246,620),(296,603),(338,631),(361,693),(394,732),(420,807),(430,931),(381,907),(333,855),(290,776),(260,706),(230,680)]),
 ('左前景山石',3,[(0,728),(65,719),(151,733),(214,781),(248,833),(263,867),(279,915),(293,962),(312,1059),(0,1000)]),
 ('中前景山石',2,[(431,819),(466,803),(488,812),(522,845),(554,873),(587,932),(570,953),(495,947),(455,901)]),
 ('主峰',11,[(558,384),(637,308),(679,288),(704,299),(733,271),(756,195),(779,148),(813,133),(863,110),(886,127),(925,171),(951,194),(985,201),(1008,196),(1036,247),(1084,277),(1105,306),(1158,332),(1178,340),(1187,390),(1060,428),(950,485),(790,493),(610,466)]),
 ('主峰右侧山峰',12,[(790,450),(842,366),(884,337),(911,280),(951,257),(981,272),(1010,329),(1063,363),(1116,380),(1110,432),(1010,502),(880,530),(790,500)]),
 ('主峰',10,[(425,413),(478,380),(540,389),(558,384),(609,333),(660,291),(679,288),(701,299),(704,350),(658,405),(565,456),(470,462)]),
 ('瀑布与右侧崖壁',6,[(991,531),(1008,461),(1034,417),(1085,401),(1134,389),(1170,382),(1150,430),(1115,568),(1060,578),(990,557)]),
 ('瀑布与右侧崖壁',7,[(1233,404),(1255,377),(1305,332),(1340,263),(1375,220),(1422,210),(1447,216),(1447,590),(1380,610),(1270,558),(1220,534)]),
 ('亭下山岛',8,[(1100,761),(1141,718),(1182,688),(1221,684),(1247,677),(1310,680),(1384,683),(1447,682),(1447,781),(1330,781),(1200,777)]),
 ('主峰右侧山峰',9,[(710,689),(785,653),(837,608),(875,575),(914,551),(949,574),(1002,616),(1062,661),(1104,689),(1050,698),(850,697),(710,699)]),
]
TRUNK=[(19,730),(83,696),(121,647),(151,589),(185,554),(223,535),(252,512),(263,486),(235,451),(241,431),(253,416),(242,402),
       (263,402),(286,449),(310,487),(291,500),(281,528),(260,553),(222,573),(193,602),(178,638),(154,680),(151,707),(177,724)]
BARK=[[(276,493),(273,515),(258,535),(237,552),(215,575),(196,601),(179,627),(164,656),(150,683),(124,706)],
      [(266,521),(251,540),(227,559),(207,582),(186,611),(171,639),(158,667)],
      [(163,617),(148,642),(136,666),(121,687),(100,705)]]


def smooth_curve(points,step_px):
    """Cubic arc parameterization, retaining the branch/trunk direction."""
    from scipy.interpolate import CubicSpline
    p=np.asarray(points,float);t=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    samples=np.linspace(0,t[-1],max(3,int(np.ceil(t[-1]/step_px))+1))
    return CubicSpline(t,p,bc_type='natural')(samples)


def mask_poly(shape,points):
    m=np.zeros(shape,np.uint8);cv2.fillPoly(m,[np.rint(points).astype(np.int32)],1);return m


def pavilion_mask(shape,ink,ratio,offset,radius):
    """Exclude the complete pavilion interior, not just its drawn edges."""
    q=np.concatenate([(np.asarray(s['points'])-offset)/ratio for s in ink if s.get('drawing_region')=='亭子'])
    lo=np.floor(q.min(0)-radius).astype(int);hi=np.ceil(q.max(0)+radius).astype(int)
    return mask_poly(shape,[(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])])


def main():
    path=ROOT/ARTWORKS[0][3];package=path.parent.parent
    baseline=path.read_bytes();bundle=json.loads(baseline);prior=copy.deepcopy(bundle)
    from web.execution import prepare
    prepare(path)
    revision=bundle['workflow']['source_revision'];ratio=revision['source_transform']['scale'];offset=np.array(revision['source_transform']['offset'])
    scale=bundle['ink_job']['meta']['ink_execution']['mm_per_px'];pixel_mm=scale*ratio
    image=cv2.imdecode(np.fromfile(str(package/'松叶简化版原图.png'),np.uint8),1);h,w=image.shape[:2]
    raw=pigment_masks(image)
    boundary=np.zeros((h,w),np.uint8)
    ink=prior['ink_job']['meta']['ink_execution']['base_strokes']
    for st in ink:
        if (st.get('source_edge_index') is not None or st.get('distant_ridge_index') is not None) and st.get('role') in ('coarse','far','trunk','branch'):
            q=np.rint((np.array(st['points'])-offset)/ratio).astype(np.int32)
            cv2.polylines(boundary,[q],False,1,1)
    # Centerline allowance includes predicted half brush width plus 1.5mm
    # bleed/registration allowance. Real brush width still needs paper measurement.
    inset_mm=1.5+press_to_width_mm(TERRAIN_PRESS_MM)/2
    radius=int(np.ceil(inset_mm/pixel_mm));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    trunk=cv2.dilate(mask_poly((h,w),TRUNK),kernel)
    line_clear=cv2.dilate(boundary,kernel)
    pavilion=pavilion_mask((h,w),ink,ratio,offset,radius+8)
    existing=prior['workflow']['modules']['color']['strokes']
    colors=[];removed=[]
    # Retain the reviewed canopy and lake lines. Replace the old terrain
    # strokes instead of layering a second set over their geometric patches.
    for st in existing:
        q=(np.array(st['points'])-offset)/ratio;x,y=np.mean(q,axis=0)
        keep=st.get('role')=='canopy_fill' or (st.get('drawing_region')=='湖面水纹' and y>740 and x>650)
        if keep:colors.append(copy.deepcopy(st))
        else:removed.append(st['id'])
    constraints=[]
    green_middle=mask_poly((h,w),REGIONS[5][2])
    # Restrained blue inside the waterfall and green moss beside the pine root.
    accents=[('瀑布与右侧崖壁',-1,[(1159,445),(1180,414),(1226,403),(1216,454),(1188,548),(1165,601),(1145,598)],'blue'),
             ('左岸山石',-2,[(15,748),(80,733),(138,750),(185,795),(167,830),(112,830),(45,805),(12,776)],'green')]
    safety={};envelopes={};paint_masks={}
    far_regions=[
        ('远山',-3,[(354,353),(431,320),(490,292),(543,277),(597,314),(550,366),(385,394)],None),
        ('远山',-4,[(577,282),(622,250),(648,240),(687,218),(728,239),(715,282),(665,325),(577,346)],None),
        ('远山',-5,[(1045,254),(1094,223),(1138,248),(1184,267),(1198,320),(1130,309),(1078,286)],None),
        ('远山',-6,[(649,650),(700,595),(792,568),(833,548),(922,526),(951,557),(912,620),(747,682),(649,682)],None)]
    regions=[(*r,None) for r in REGIONS]+accents+far_regions
    for i,item in enumerate(regions):
        region,edge,polygon,accent_color=item
        area=mask_poly((h,w),polygon)
        if edge==11:area[green_middle>0]=0
        safe=cv2.erode(area,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        safe[(trunk>0)|(line_clear>0)|(pavilion>0)]=0
        safety[i]=safe;envelopes[i]=area
    blue_contacts=np.zeros((h,w),np.uint8)
    pigment_gap_mm=press_to_width_mm(TERRAIN_PRESS_MM)+.75
    gap_radius=int(np.ceil(pigment_gap_mm/pixel_mm))
    gap_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*gap_radius+1,2*gap_radius+1))
    # Build blue first so green ribbons can preserve a measured model gap
    # instead of painting both pigments over an ambiguous teal source pixel.
    for ribbon_index,(i,color,guide,widths) in sorted(enumerate(RIBBONS),key=lambda item:('blue','green').index(item[1][1])):
        region,edge,polygon,accent_color=regions[i]
        factor=FAR_RIBBON_WIDTH_FACTOR if region=='远山' else NEAR_RIBBON_WIDTH_FACTOR
        widths=[v*factor for v in widths]
        pigment=np.ones((h,w),np.uint8) if accent_color else raw[color]
        safe=safety[i].copy()
        if color=='green':safe[cv2.dilate(blue_contacts,gap_kernel)>0]=0
        paths,m=ribbon_strokes(guide,widths,safe,pigment,color,region,i,ribbon_index,scale,ratio,offset)
        paths=join_ribbon_sweeps(paths,m,scale,ratio,offset)
        for j,st in enumerate(paths):
            st['id']='sourcewash-%02d-%02d-%s-%03d'%(i,ribbon_index,color,j)
            q=(dense_path(st['points'],.10/scale)-offset)/ratio
            assert np.all(points_inside(q,safety[i]))
            assert not np.any(points_inside(q,trunk))
            assert not np.any(points_inside(q,pavilion))
            assert np.all(points_inside(q,m))
            if color=='blue':cv2.polylines(blue_contacts,[np.rint(q).astype(np.int32)],False,1,1)
        colors.extend(paths);paint_masks[ribbon_index]=m
        constraints.append(dict(region=region,color=color,count=len(paths),source_ribbon_index=ribbon_index,
            source_guide=guide,half_width_px=widths,source_pigment_intersection=not bool(accent_color)))
    template=copy.deepcopy(next(s for s in ink if s.get('role')=='texture'))
    ink=[copy.deepcopy(s) for s in ink if s.get('role')!='bark_texture']
    for i,line in enumerate(BARK):
        p=smooth_curve(line,.35/pixel_mm)*ratio+offset
        st=copy.deepcopy(template)
        for key in ('parent_stroke_id','segment','segments','source_edge_index','source_path_ids'):st.pop(key,None)
        st.update(id='bare-bark-%d'%i,role='bark_texture',points=p.tolist(),press=[.15]*len(p),speed=[40.]*len(p),drawing_region='整棵松树')
        ink.append(refresh_stats(st,scale))
    ink=order(ink,ratio,offset,True)
    # Explicit complete-subject order and top-to-bottom strip order.
    from scripts.order_selected_artworks import ORDERS
    for st in ink:
        if st.get('role')=='bark_texture':st['drawing_region']='整棵松树';st['subject_order']=0
    ink.sort(key=lambda s:ORDERS[0].index(s['drawing_region']))
    for st in colors:
        st['subject_order']=ORDERS[0].index(st['drawing_region'])
    colors.sort(key=lambda s:(('blue','green').index(s['color']),s['subject_order'],s.get('canopy_index',-1),s.get('terrain_region_index',-1),s['points'][0][0] if s.get('role')=='terrain_wash' else 0))
    bundle['ink_job']=ink_payload(prior['ink_job'],ink)
    plan=bundle['workflow'];plan['modules']['outline'].update(strokes=copy.deepcopy(ink),stroke_count=len(ink));plan['modules']['color'].update(strokes=colors,stroke_count=len(colors))
    from collections import Counter
    count_by_color=Counter(st['color'] for st in colors)
    plan['fill_report']={color:dict(strokes=count,terrain_wash_strokes=sum(st.get('role')=='terrain_wash' and st['color']==color for st in colors),
        coverage_is_model_estimate=True) for color,count in count_by_color.items()}
    pressures=[v for st in colors for v in st['press']]
    plan['color_geometry'].update(pitch_mm=TERRAIN_PITCH_MM,pitch_px=TERRAIN_PITCH_MM/pixel_mm,max_stroke_mm=50,
        pressure_range_mm=[min(pressures),max(pressures)],source='source pigment masks intersected with tapered source-fold ribbons',coverage_is_model_estimate=True)
    revision.update(terrain_fill='source_aligned_curved_ribbons',tree_trunk_fill=False,bark_texture_strokes=3,
        near_ribbon_width_factor=NEAR_RIBBON_WIDTH_FACTOR,far_ribbon_width_factor=FAR_RIBBON_WIDTH_FACTOR,
        terrain_centerline_inset_mm=float(inset_mm),terrain_bleed_allowance_mm=1.5,
        bark_style='longitudinal-curved',terrain_strip_pitch_mm=TERRAIN_PITCH_MM,terrain_pressure_mm=TERRAIN_PRESS_MM,
        terrain_envelope_smoothing_px=7,terrain_pigment_mask_smoothing_px=4,
        terrain_white_channels=True,continuous_color_sweeps=True,pavilion_interior_excluded=True,
        terrain_removed_old_color_ids=removed,after_counts=dict(black=len(ink),color=len(colors)),
        terrain_source_pigment_masks=True,terrain_ribbon_count=len(RIBBONS),terrain_ribbon_guides=constraints,
        terrain_source_sha256=hashlib.sha256((package/'松叶简化版原图.png').read_bytes()).hexdigest(),
        terrain_direction='curved-source-folds',terrain_dense_validation_step_mm=.10,
        terrain_blue_green_centerline_gap_mm=pigment_gap_mm,
        note='按用户反馈扩大山石蓝绿覆盖：近中景色带宽度为首版的2.4倍，远山1.6倍，仍由原图蓝绿颜料与岩面边界裁切；山石下压0.30mm、中心笔距1.0mm，山脚渐窄，保留原图云雾与主要留白；主峰左坡向左下、右坡向右下，前景沿斜向岩面刷色；同一色带内最多三笔安全连接来回刷，单笔不超过50mm；亭子内部及周围禁涂，保留原黑线、叶团、水纹、题字及三条0.15mm树干纹理。颜料浓淡和晕染未实测，预览按均匀颜料显示。')
    revision['stroke_data_sha256']=digest(stroke_data(bundle));plan['note']=revision['note']
    from run_color_from_job import validate_plan
    validate_plan(plan,bundle['ink_job'])
    from run_json_painting import manifest_for
    manifest_for(ink,plan,bundle['ink_job']['meta'])
    pending=path.parent/'待应用_顺皴刷色';pending.mkdir(exist_ok=True)
    save(pending/'painting.json',bundle)
    save(pending/'绘画顺序索引.json',dict(stroke_data_sha256=digest(stroke_data(bundle)),stages=index_for(bundle)))
    # Persist independent binary masks used to validate the actual contact
    # segments and rounded return connectors, not just the sampled vertices.
    np.savez_compressed(pending/'原图色带禁涂掩膜.npz',trunk=trunk,pavilion=pavilion,line_clear=line_clear,
        **{'safe_%d'%i:m for i,m in safety.items()},**{'envelope_%d'%i:m for i,m in envelopes.items()},
        **{'ribbon_%d'%i:m for i,m in paint_masks.items()})
    save(pending/'上色边界校验.json',dict(robot_connected=False,terrain_direction='curved-source-folds-and-safe-return-sweeps',terrain_paths_inside_inset=True,
        trunk_excluded=True,centerline_inset_mm=float(inset_mm),regions=constraints,
        pavilion_interior_excluded=True,white_channels_preserved=True,max_connected_length_mm=50,dense_contact_step_mm=.10,
        strip_pitch_mm=TERRAIN_PITCH_MM,terrain_pressure_mm=TERRAIN_PRESS_MM,bark_style='longitudinal-curved',floating_point_brush_paths=True,
        actual_brush_width_and_bleed_need_measurement=True))
    # Four execution stages, generated from candidate geometry (not an image edit).
    from PIL import Image,ImageDraw,ImageFont
    board=Image.new('RGB',(1440,1080),'#f7f7f0');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    cw,ch=bundle['ink_job']['meta']['image_size'];canvas=np.full((ch,cw,3),255,np.uint8)
    parts=[('山水勾勒',[s for s in ink if s.get('role')!='title'],(40,40,40)),('蓝色顺势刷色',[s for s in colors if s['color']=='blue'],(175,125,72)),('绿色顺势刷色',[s for s in colors if s['color']=='green'],(97,152,97)),('最后题字',[s for s in ink if s.get('role')=='title'],(40,40,40))]
    draw.text((25,15),'松瀑亭舟 · 原图色带重建（真实执行轨迹）',font=font,fill='#304c3b')
    for i,(label,ss,color) in enumerate(parts):
        for st in ss:cv2.polylines(canvas,[np.rint(st['points']).astype(np.int32)],False,color,max(1,round(press_to_width_mm(st['press'][0])/scale)),cv2.LINE_AA)
        pic=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));pic.thumbnail((690,425));x=25+(i%2)*710;y=65+(i//2)*500
        draw.text((x,y),'%d · %s（%d笔）'%(i+1,label,len(ss)),font=font,fill='#304c3b');board.paste(pic,(x+(690-pic.width)//2,y+40))
    draw.text((25,1045),'轨迹与估计线宽示意；模型内收不能保证消除实物渗色，需确认实际笔宽。',font=font,fill='#687562')
    board.save(pending/'顺皴刷色预览.png')
    cv2.imencode('.png',canvas)[1].tofile(str(pending/'任务预览_线宽色彩为估计.png'))
    from scripts.source_wash_comparison import comparison
    comparison(prior,bundle,pending/'原图色带前后对照.png')
    from scripts.validate_source_washes import audit_candidate
    audit=audit_candidate(pending/'painting.json',path)
    save(pending/'真实轨迹离线校验.json',audit)
    counts=audit['counts']
    (pending/'原图色带修订说明.md').write_text('# 松瀑亭舟 · 扩大原图色带覆盖\n\n'+revision['note']+'\n\n当前笔数：山水勾勒 %(black)d、蓝色 %(blue)d、绿色 %(green)d、题字 %(title)d。\n'%counts+'\n黑线、叶团、水纹及题字逐笔保留；接触线段与连接段按0.05mm采样校验。实际笔宽、颜料浓淡和纸张渗色仍需实测。\n',encoding='utf-8')
    spec=dict(baseline_sha256=hashlib.sha256(baseline).hexdigest(),candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False,
        policy='Apply automatically before the next new web run, only when painting and wash locks are absent')
    save(pending/'应用记录.json',spec)
    (pending/'说明.md').write_text('新版通过离线检查，空闲且无会话锁时用现有发布流程应用。\n'+revision['note']+'\n同一原图色带内最多三笔来回连接，每笔最长50mm；对真实接触段和转弯连接段每0.10mm检查，不穿过留白、树干或亭子。\n色边内收包含估计半笔宽与1.5mm渗色余量；需实测笔宽，不保证纸上绝不越界。\n',encoding='utf-8')
    print({label:len(ss) for label,ss,_ in parts})


if __name__=='__main__':main()
