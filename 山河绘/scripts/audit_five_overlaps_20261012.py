"""Read-only geometry/brush-model overlap audit of the first five artworks."""
from collections import defaultdict
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw,ImageFont
from scripts.first_color_edge_gap_20261012 import h,read,encode
from scripts.source_aligned_washes import dense_path
from shuimo import press_to_width_mm
from web.catalog import ARTWORKS

OUT=ROOT/'artifacts/五幅重复与重叠检查_20261012'


def exact_paths(strokes):
    groups=defaultdict(list)
    for s in strokes:
        p=tuple(map(tuple,np.round(s['points'],6)))
        groups[min(p,p[::-1])].append(s['id'])
    return [ids for ids in groups.values() if len(ids)>1]


def straight_retraces(strokes,scale):
    segments=[]
    for s in strokes:
        p=np.asarray(s['points'],float)
        if len(p)!=2:continue
        d=p[1]-p[0]
        if abs(d[0])<1e-8:axis=1
        elif abs(d[1])<1e-8:axis=0
        else:continue
        segments.append((s,axis,p[0,1-axis],min(p[:,axis]),max(p[:,axis])))
    result=[]
    for a,b in itertools.combinations(segments,2):
        sa,axis,line,lo,hi=a;sb,other,line2,lo2,hi2=b
        if axis!=other or sa.get('color')!=sb.get('color'):continue
        transverse=abs(line-line2)*scale
        if transverse>.15:continue
        shared=(min(hi,hi2)-max(lo,lo2))*scale
        if shared<1.:continue
        fraction=shared/(min(hi-lo,hi2-lo2)*scale)
        if fraction<.70 and shared<5.:continue
        result.append(dict(ids=[sa['id'],sb['id']],regions=[sa.get('drawing_region'),sb.get('drawing_region')],
                           color=sa.get('color'),centerline_separation_mm=transverse,shared_length_mm=shared,
                           shorter_stroke_fraction=fraction,exact_collinear=transverse<1e-6))
    return result


def black_retraces(strokes,scale):
    sample=[]
    for s in strokes:
        q=dense_path(s['points'],.12/scale);t=np.gradient(q,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-12)
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]*scale
        interior=(arc>=.8)&(arc<=arc[-1]-.8)
        sample.append((q,t,interior,cKDTree(q),q.min(0),q.max(0)))
    result=[]
    for i,j in itertools.combinations(range(len(strokes)),2):
        a,b=sample[i],sample[j]
        if np.any(a[4]>b[5]+.15/scale) or np.any(b[4]>a[5]+.15/scale):continue
        # Compare the shorter path to avoid treating one overlapping section as
        # a duplicate of the entire longer contour.
        if strokes[i]['length_mm']<=strokes[j]['length_mm']:short,long,ii,jj=a,b,i,j
        else:short,long,ii,jj=b,a,j,i
        d,idx=long[3].query(short[0]);aligned=np.abs(np.sum(short[1]*long[1][idx],axis=1))>.985
        match=(d*scale<=.15)&aligned&short[2]
        length=float(match.sum()*.12)
        fraction=float(match.sum()/max(1,short[2].sum()))
        if length<3 or (fraction<.50 and length<8):continue
        result.append(dict(ids=[strokes[ii]['id'],strokes[jj]['id']],
                           regions=[strokes[ii].get('drawing_region'),strokes[jj].get('drawing_region')],
                           roles=[strokes[ii].get('role'),strokes[jj].get('role')],
                           matched_interior_length_mm=length,shorter_path_fraction=fraction,
                           involves_converted_outline=bool(strokes[ii].get('source_color_stroke_id') or strokes[jj].get('source_color_stroke_id'))))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True);reports=[]
    board=Image.new('RGB',(1550,5*650),'#faf8f0');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    for number,item in enumerate(ARTWORKS[:5],1):
        path=ROOT/item[3];before=hashlib.sha256(path.read_bytes()).hexdigest();bundle=read(path)
        ink=bundle['ink_job']['meta']['ink_execution']['base_strokes'];colors=bundle['workflow']['modules']['color']['strokes']
        w,height=bundle['ink_job']['meta']['image_size'];shape=(height,w);scale=bundle['ink_job']['meta']['ink_execution']['mm_per_px']
        masks={};regions=list(dict.fromkeys((s['color'],s.get('drawing_region')) for s in colors))
        for key in regions:masks[key]=h.footprint([s for s in colors if (s['color'],s.get('drawing_region'))==key],shape,scale)
        pigment={c:np.zeros(shape,np.uint8) for c in ('blue','green')}
        multiplicity={c:np.zeros(shape,np.uint16) for c in ('blue','green')}
        for key,mask in masks.items():pigment[key[0]]|=mask;multiplicity[key[0]]+=mask
        region_pairs=[]
        for a,b in itertools.combinations(regions,2):
            if a[0]!=b[0]:continue
            overlap=int(np.count_nonzero(masks[a]&masks[b]))*scale**2
            if overlap>=2.:
                region_pairs.append(dict(color=a[0],regions=[a[1],b[1]],overlap_model_mm2=overlap))
        region_pairs.sort(key=lambda p:p['overlap_model_mm2'],reverse=True)
        mixed=pigment['blue']&pigment['green']
        cross_region=((multiplicity['blue']>1)|(multiplicity['green']>1)).astype(np.uint8)
        exact_ink=exact_paths(ink);exact_color=exact_paths(colors)
        near_color=straight_retraces(colors,scale);near_black=black_retraces(ink,scale)
        firstcolor=next(s for s in colors if s.get('nominal_pitch_mm'))
        pitch=firstcolor['nominal_pitch_mm'];width=press_to_width_mm(max(firstcolor['press']))
        record=dict(number=number,title=item[1],source=str(path),source_sha256=before,
                    exact_black_duplicates=exact_ink,exact_color_duplicates=exact_color,
                    same_or_near_color_centerline_retraces=near_color,near_black_centerline_retraces=near_black,
                    same_color_cross_region_overlaps=region_pairs,
                    cross_region_same_color_model_area_mm2=float(cross_region.sum()*scale**2),
                    blue_green_model_overlap_mm2=float(mixed.sum()*scale**2),
                    main_pitch_mm=pitch,main_model_width_mm=width,
                    adjacent_row_model_overlap_width_mm=max(0.,width-pitch),
                    tolerance_mm=.15,physical_overlap_measured=False,robot_connected=False,
                    note='中心线近重描使用0.15mm距离与平行方向门限，排除短交叉和共同端点。跨部位和蓝绿面积按估计笔宽统计，未代表实物面积。')
        reports.append(record);folder=OUT/str(number);folder.mkdir(exist_ok=True);(folder/'report.json').write_bytes(encode(record))
        picture=np.asarray(Image.open(io.BytesIO(h.render(bundle)))).copy()
        picture[cross_region>0]=np.rint(picture[cross_region>0]*.25+np.array([235,85,60])*.75).astype(np.uint8)
        picture[mixed>0]=np.rint(picture[mixed>0]*.25+np.array([177,70,190])*.75).astype(np.uint8)
        for x,data,label in [(20,Image.open(io.BytesIO(h.render(bundle))),'当前画稿'),(800,Image.fromarray(picture),'红：同色跨部位；紫：蓝绿笔宽交叠')]:
            data.thumbnail((730,550));board.paste(data,(x,(number-1)*650+65));draw.text((x,(number-1)*650+15),f'{number:02d} {item[1]} · {label}',font=font,fill='#34463a')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
        print(encode(dict(number=number,exact_black=len(exact_ink),exact_color=len(exact_color),near_color_pairs=len(near_color),near_black_pairs=len(near_black),converted_black_pairs=sum(p['involves_converted_outline'] for p in near_black),same_color_cross_region_mm2=round(record['cross_region_same_color_model_area_mm2'],1),blue_green_mm2=round(record['blue_green_model_overlap_mm2'],1),largest_cross_region_pairs=region_pairs[:3])).decode('utf-8'),flush=True)
    board.save(OUT/'五幅重叠位置.png');(OUT/'report.json').write_bytes(encode(reports))


if __name__=='__main__':main()
