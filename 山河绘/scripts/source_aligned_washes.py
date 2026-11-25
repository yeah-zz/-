"""Source pigment envelopes and curved rock-face ribbons, entirely offline.

Guide coordinates are in the untouched user's source image. The broad region
polygons are exclusion envelopes, never the shapes used to paint a color block.
Actual color shapes intersect the source pigment and tapered curved ribbons.
"""
import copy
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from ink_control import refresh_stats
from shuimo import press_to_width_mm

# region, pigment, source fold, varying half-width (source pixels).
RIBBONS = [
 (0,'green',[(8,414),(22,470),(43,525),(71,584),(94,633)],[9,18,19,17,4]),
 (0,'blue',[(47,482),(60,551),(89,612),(114,657)],[8,16,20,5]),
 (1,'blue',[(268,648),(290,714),(318,780),(347,850),(375,922)],[14,34,33,27,5]),
 (1,'blue',[(330,667),(353,742),(387,826),(411,909)],[7,15,19,4]),
 (2,'green',[(30,827),(45,852),(60,876)],[9,17,3]),
 (2,'blue',[(75,858),(104,907),(135,958),(153,990)],[10,25,23,3]),
 (2,'green',[(123,857),(150,916),(181,974),(213,1018)],[12,23,24,3]),
 (3,'blue',[(466,839),(492,875),(515,927)],[11,20,4]),
 (3,'green',[(508,860),(534,894),(558,938)],[9,22,5]),
 (4,'blue',[(839,154),(807,215),(773,284),(727,352),(674,401)],[10,22,24,20,4]),
 (4,'blue',[(873,161),(864,227),(834,296),(798,357),(753,412)],[8,21,24,20,4]),
 (4,'blue',[(909,183),(948,244),(1001,302),(1067,341),(1141,367)],[9,24,29,25,4]),
 (5,'green',[(938,292),(907,350),(866,402),(810,454)],[11,25,30,4]),
 (5,'green',[(972,292),(990,347),(1037,398),(1100,435)],[9,25,29,4]),
 (6,'green',[(675,316),(643,351),(600,389),(558,420)],[10,21,23,4]),
 (7,'green',[(1080,439),(1052,490),(1024,546),(1007,566)],[11,23,20,3]),
 (7,'blue',[(1132,419),(1106,478),(1091,535),(1079,568)],[8,18,20,4]),
 (8,'blue',[(1299,354),(1270,410),(1248,469),(1235,516)],[10,21,20,4]),
 (8,'green',[(1417,245),(1383,302),(1351,363),(1318,439),(1291,493)],[12,31,32,25,4]),
 (8,'green',[(1450,294),(1428,356),(1395,418),(1364,478),(1329,531)],[12,27,32,24,4]),
 (9,'blue',[(1175,738),(1146,761),(1118,768)],[8,13,4]),
 (9,'green',[(1258,709),(1229,740),(1208,762)],[12,21,4]),
 (9,'blue',[(1324,713),(1300,744),(1273,764)],[10,18,4]),
 (9,'green',[(1396,712),(1365,748),(1342,770)],[13,23,4]),
 (9,'green',[(1453,718),(1427,751),(1401,771)],[10,20,4]),
 (10,'green',[(918,584),(880,621),(829,658),(773,681)],[9,25,24,4]),
 (10,'green',[(946,586),(974,627),(1021,665),(1074,684)],[8,23,21,4]),
 (10,'blue',[(933,601),(958,641),(995,676)],[5,10,3]),
 (11,'blue',[(1190,440),(1179,486),(1163,543),(1151,581)],[5,8,7,3]),
 (12,'green',[(30,748),(65,768),(107,790),(147,802)],[10,17,17,3]),
 (13,'green',[(527,306),(489,328),(449,346)],[5,10,3]),
 (14,'blue',[(679,262),(640,289),(600,310)],[5,10,3]),
 (15,'blue',[(1099,254),(1133,275),(1170,289)],[4,9,3]),
 (16,'blue',[(833,581),(791,610),(750,641)],[5,11,3]),
]


def pigment_masks(image):
    """Classify blurred pigment, avoiding holes from watercolor paper texture."""
    sm=cv2.GaussianBlur(image.astype(np.float32),(0,0),7)
    b,g,r=sm.transpose(2,0,1)
    raw={'blue':((b-r>17)&(b-g>4)&(r<234)).astype(np.uint8),
         'green':((g-r>14)&(b-g<15)&(r<239)).astype(np.uint8)}
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    return {c:(cv2.GaussianBlur(cv2.morphologyEx(m,cv2.MORPH_CLOSE,kernel).astype(np.float32),(0,0),4)>.48).astype(np.uint8)
            for c,m in raw.items()}


def ribbon_geometry(guide,widths,pixel_mm):
    p=np.asarray(guide,float)
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    spline=CubicSpline(arc,p,bc_type='natural')
    t=np.linspace(0,arc[-1],max(4,int(np.ceil(arc[-1]*pixel_mm/.10))+1))
    center=spline(t);v=spline(t,1);v/=np.linalg.norm(v,axis=1)[:,None]
    normal=np.column_stack((v[:,1],-v[:,0]))
    width=np.maximum(1,CubicSpline(arc,widths,bc_type='natural')(t))
    return center,normal,width


def points_inside(points,mask):
    q=np.rint(points).astype(int);h,w=mask.shape
    valid=(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    out=np.zeros(len(q),bool);out[valid]=mask[q[valid,1],q[valid,0]]>0
    return out


def dense_path(points,step_px):
    """Sample all connecting line segments as executed by the robot."""
    p=np.asarray(points,float)
    lengths=np.linalg.norm(np.diff(p,axis=0),axis=1)
    pieces=[np.linspace(a,b,max(2,int(np.ceil(d/step_px))+1))[:-1]
            for a,b,d in zip(p[:-1],p[1:],lengths)]
    return np.vstack([*pieces,p[-1:]])


def ribbon_strokes(guide,widths,safe,pigment,color,region,index,ribbon_index,scale,ratio,offset):
    pixel_mm=scale*ratio;center,normal,width=ribbon_geometry(guide,widths,pixel_mm)
    shape=safe.shape
    envelope=np.zeros(shape,np.uint8)
    polygon=np.vstack([center+normal*width[:,None],(center-normal*width[:,None])[::-1]])
    cv2.fillPoly(envelope,[np.rint(polygon).astype(np.int32)],1)
    # The ribbon shapes, broad source clouds and source hue boundaries are
    # shared by all passes; white intervals remain unpainted by the connector.
    mask=envelope&safe&pigment
    mask=(cv2.GaussianBlur(mask.astype(np.float32),(0,0),2)>.55).astype(np.uint8)&mask
    # One source pixel of interpolation reserve prevents a line segment
    # between two accepted raster samples from grazing a diagonal mask corner.
    contact_mask=cv2.erode(mask,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    output=[];pitch=1.0/pixel_mm
    count=max(1,int(np.floor(2*width.max()/pitch)))
    for k in range(count):
        shift=(k-(count-1)/2)*pitch
        path=center+normal*shift
        good=points_inside(path,contact_mask)&(abs(shift)<width-1)
        edges=np.diff(np.r_[False,good,False].astype(int));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0]
        for a,b in zip(starts,ends):
            q=path[a:b]
            if len(q)<3:continue
            length=np.linalg.norm(np.diff(q,axis=0),axis=1).sum()*pixel_mm
            if length<2.5:continue
            # Split long wet contacts before the runtime's pigment-load limit.
            arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]*pixel_mm
            breaks=np.linspace(0,arc[-1],int(np.ceil(arc[-1]/45))+1)
            for start,end in zip(breaks[:-1],breaks[1:]):
                t=np.unique(np.r_[start,np.arange(start,end,.20),end])
                local=np.column_stack([np.interp(t,arc,q[:,axis]) for axis in (0,1)])
                if not np.all(points_inside(dense_path(local,.10/pixel_mm),mask)):continue
                p=local*ratio+offset;n=len(p)
                st=dict(id='sourcewash-%02d-%02d-%s-%03d'%(index,ribbon_index,color,len(output)),color=color,role='terrain_wash',
                    region_id='rock-'+region,points=p.tolist(),press=[.30]*n,speed=[40.]*n,
                    recipe='source-aligned-cun-ribbon',path_style='source-fold-sweep',drawing_region=region,
                    brush_width_mm_estimate=press_to_width_mm(.30),target_width_mm=[press_to_width_mm(.30)]*n,
                    target_opacity=[.40]*n,pressure_mm=None,pigment_recipe_calibrated=False,layer=0,
                    terrain_region_index=index,source_ribbon_index=ribbon_index,source_guide=guide,
                    source_ribbon_half_width_px=widths)
                output.append(refresh_stats(st,scale))
    return output,mask


def join_ribbon_sweeps(paths,mask,scale,ratio,offset):
    """Join neighboring returns through a rounded turn inside the same ribbon."""
    result=[];current=None;legs=0
    turn_mask=cv2.erode(mask,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    def finish():
        nonlocal current
        if current is not None:result.append(refresh_stats(current,scale));current=None
    for st in paths:
        if current is None:current=copy.deepcopy(st);legs=1;continue
        p=np.asarray(current['points']);q=np.asarray(st['points'])
        if np.linalg.norm(p[-1]-q[-1])<np.linalg.norm(p[-1]-q[0]):q=q[::-1]
        distance=np.linalg.norm(p[-1]-q[0])*scale
        length=(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()+np.linalg.norm(np.diff(q,axis=0),axis=1).sum())*scale+distance
        # Hermite tangents make the wet turnaround smooth. Shrink tangent
        # lengths so all turning points can be checked against the same mask.
        turn=None
        if legs<3 and distance<2.6 and length<48:
            a=p[-1];b=q[0];ta=p[-1]-p[-2];tb=q[1]-q[0]
            ta/=np.linalg.norm(ta);tb/=np.linalg.norm(tb)
            t=np.linspace(0,1,41)[:,None];span=np.linalg.norm(b-a)
            for fraction in (.4,.2,0):
                candidate=(2*t**3-3*t**2+1)*a+(t**3-2*t**2+t)*ta*span*fraction+(-2*t**3+3*t**2)*b+(t**3-t**2)*tb*span*fraction
                local=(dense_path(candidate,.10/scale)-offset)/ratio
                if np.all(points_inside(local,turn_mask)):turn=candidate;break
        if turn is not None:
            merged=np.vstack([p,turn[1:],q[1:]]);n=len(merged)
            current.update(points=merged.tolist(),press=[.30]*n,speed=[40.]*n,
                target_width_mm=[press_to_width_mm(.30)]*n,target_opacity=[.40]*n,
                path_style='continuous-source-fold-return',connected_passes=legs+1)
            legs+=1
        else:finish();current=copy.deepcopy(st);legs=1
    finish();return result
