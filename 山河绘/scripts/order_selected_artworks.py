"""Reorder exact reviewed strokes by complete pictured subjects; no hardware."""
from collections import Counter, defaultdict
from pathlib import Path
import copy
import csv
import hashlib
import io
import json
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import cv2
from web.catalog import ARTWORKS
from painting_session import save, session_lock
from artwork_order import ink_payload
from web.execution import prepare
from scripts.consolidate_selected_artworks import index_for, stroke_data, digest

ORDERS = [
 ['整棵松树','左岸山石','左前景山石','中前景山石','亭子','亭下山岛','小舟','瀑布与右侧崖壁','主峰','主峰右侧山峰','远山','湖面水纹','题字'],
 ['左侧主松树','左下小树','近岸中间小树','近岸右侧小树','中间岛上树','右上岸边树','右下岛上树','左前景山石','中前景山石','右前景山石','右侧小岛','左侧中景山','中间中景山','右侧中景山','左侧远山','中间远山','右侧远山','曲水水纹','题字'],
 ['亭左松树','亭右松树','近景松树','亭子','台阶','亭下完整山石','左下山石','左侧高山','右侧山石','左侧远山','中间远山','右侧远山','湖面水纹','题字'],
 ['整棵柳树','左上岸石','左下岸石','中间岸石','右端岸石','完整渔舟','左侧小岛','中间小岛','右侧岛岸','左侧远山','中间远山','右侧远山','湖面水纹','题字'],
 ['左上完整山体','整片梯田与田埂','左下临水山石','中下近景山石','右下近景山石','左侧远山','中间远山','右侧远山','湖面水纹','题字'],
]


def nearest(xy, anchors):
    return min(range(len(anchors)), key=lambda i: np.linalg.norm(xy-np.array(anchors[i])))


def inside(xy, polygon):
    return cv2.pointPolygonTest(np.array(polygon,dtype=np.float32),tuple(map(float,xy)),False)>=0


def classify(number, points, stroke):
    x,y=np.median(points,axis=0); role=stroke.get('role',''); rid=str(stroke.get('region_id',''))
    if role=='title': return '题字'
    if number==0:
        pine=[(216,706),(294,629),(325,546),(370,485),(260,470),(240,424),(285,375),(430,372),(500,438),(604,480),(747,549),(760,603),(662,624),(544,570),(460,540),(416,594),(366,667),(292,706),(257,728)]
        if role=='pine_needle' or 'pine-cluster' in rid or inside((x,y),pine): return '整棵松树'
        if 1160<x<1355 and 535<y<662:return '亭子'
        if 807<x<978 and 714<y<791:return '小舟'
        region=int(stroke.get('semantic_region',12))
        if region==4:return '左岸山石' if x<300 else '左前景山石'
        return {1:'远山',2:'主峰',3:'主峰右侧山峰',5:'瀑布与右侧崖壁',6:'亭下山岛',7:'左岸山石',8:'左前景山石',9:'中前景山石',10:'远山',11:'湖面水纹',12:'湖面水纹',13:'湖面水纹'}.get(region,'湖面水纹')
    if number==1:
        trees=['左侧主松树','左下小树','近岸中间小树','近岸右侧小树','中间岛上树','右上岸边树','右下岛上树']
        anchors=[(380,550),(100,698),(491,772),(875,828),(646,449),(1330,441),(1100,555)]
        pine=[(60,455),(160,408),(295,403),(405,440),(545,465),(545,512),(680,527),(697,604),(510,624),(392,629),(425,680),(412,741),(240,741),(250,680),(70,757),(50,704),(133,627),(159,552),(165,510),(60,514)]
        if x<145 and y>640 and role in ('branch','foliage_outline','leaf_color'):return '左下小树'
        if inside((x,y),pine):return '左侧主松树'
        if role in ('branch','foliage_outline','leaf_color') or inside((x,y),pine):return trees[nearest(np.array([x,y]),anchors)]
        if role=='water' or stroke.get('role')=='water_color':return '曲水水纹'
        if y>620:
            if x>1150:return '右侧小岛'
            return ['左前景山石','中前景山石','右前景山石'][nearest(np.array([x,y]),[(270,755),(690,825),(960,909)])]
        if y<250 or role=='distant_color':return ['左侧远山','中间远山','右侧远山'][min(2,int(x/512))]
        return ['左侧中景山','中间中景山','右侧中景山'][0 if x<510 else 1 if x<1050 else 2]
    if number==2:
        if role.startswith('pavilion'):return '亭子'
        if role=='stairs':return '台阶'
        trees=['亭左松树','亭右松树','近景松树']; anchors=[(421,400),(760,416),(385,702)]
        boxes=[(320,330,510,451),(690,345,846,455),(260,620,530,755)]
        if role in ('branch','foliage_outline','leaf_color') or any(a<x<c and b<y<d for a,b,c,d in boxes):return trees[nearest(np.array([x,y]),anchors)]
        if role.startswith('pavilion') or 512<x<714 and 283<y<440:return '亭子'
        if role=='stairs' or 389<x<540 and 427<y<513:return '台阶'
        if role=='water' or role=='water_color':return '湖面水纹'
        if role=='far_outline' or role=='distant_color':return ['左侧远山','中间远山','右侧远山'][0 if x<650 else 1 if x<1250 else 2]
        if x>1200:return '右侧山石'
        if x<300 and y<490:return '左侧高山'
        if y>650 and x<820:return '左下山石'
        return '亭下完整山石'
    if number==3:
        willow=[(0,90),(140,100),(225,161),(320,299),(367,306),(447,442),(437,633),(389,582),(333,472),(245,454),(168,443),(120,562),(80,570),(0,560)]
        if role in ('branch','willow_fine','foliage_outline','leaf_color') or inside((x,y),willow) and role!='far_outline':return '整棵柳树'
        if role=='boat' or 1035<x<1330 and 459<y<542:return '完整渔舟'
        if role=='water' or role=='water_color':return '湖面水纹'
        if x<880 and y>539:
            return ['左上岸石','左下岸石','中间岸石','右端岸石'][0 if y<720 else 1 if x<260 else 2 if x<560 else 3]
        if 335<y<459:
            return ['左侧小岛','中间小岛','右侧岛岸'][0 if x<720 else 1 if x<1080 else 2]
        if y<440:return ['左侧远山','中间远山','右侧远山'][0 if x<720 else 1 if x<1110 else 2]
        return '湖面水纹'
    old=stroke.get('drawing_region','')
    if role=='water' or role=='water_color' or old=='水纹':return '湖面水纹'
    if old=='左上山体':return '左上完整山体'
    if old=='梯田':return '整片梯田与田埂'
    if old=='远山' or role=='far_outline' or role=='distant_color':return ['左侧远山','中间远山','右侧远山'][nearest(np.array([x,y]),[(450,350),(870,490),(1260,580)])]
    return ['左下临水山石','中下近景山石','右下近景山石'][0 if x<310 else 1 if x<880 else 2]


def signature(stroke):
    return json.dumps({k:v for k,v in stroke.items() if k not in ('drawing_region','title_character','subject_order')},sort_keys=True,ensure_ascii=False)


def reorder(strokes, number, scale, offset, ink):
    # Keep split pieces of one parent together, in their original segment order.
    parents=defaultdict(list)
    for st in copy.deepcopy(strokes):parents[st.get('parent_stroke_id',st['id']) if ink else st['id']].append(st)
    groups=defaultdict(list)
    for children in parents.values():
        children.sort(key=lambda s:s.get('segment',0))
        points=(np.concatenate([np.array(s['points'],float) for s in children])-offset)/scale
        center=np.median(points,axis=0); subject=classify(number,points,children[0])
        character=None
        if subject=='题字':
            limits=[(416,540),(143,240),(1000,1260),(1000,1150),(970,1140)][number]
            position=center[1] if number==1 else center[0]
            character=0 if position<limits[0] else 1 if position<limits[1] else 2
        role=children[0].get('role','')
        phase=character if character is not None else 2 if role=='pine_needle' else 0 if ('outline' in role or role in ('coarse','branch','boat','pavilion','stairs')) else 1
        for st in children:
            st['drawing_region']=subject;st['subject_order']=ORDERS[number].index(subject)
            if character is not None:st['title_character']='山河绘'[character]
        groups[(subject,phase)].append(children)
    output=[]
    for subject in ORDERS[number]:
        phases=sorted(p for s,p in groups if s==subject)
        current=None
        for phase in phases:
            pending=groups[(subject,phase)]
            while pending:
                if current is None:
                    take=min(range(len(pending)),key=lambda i:(np.mean(pending[i][0]['points'],axis=0)[1],np.mean(pending[i][0]['points'],axis=0)[0],pending[i][0]['id']))
                else:take=min(range(len(pending)),key=lambda i:np.linalg.norm(np.array(pending[i][0]['points'][0])-current))
                children=pending.pop(take);output.extend(children);current=np.array(children[-1]['points'][-1])
    assert Counter(map(signature,strokes))==Counter(map(signature,output))
    return output


def main(only=None):
    with session_lock(ROOT/'robot_runs'):
        if (ROOT/'robot_runs'/'blue_wash_motion.lock').exists():raise ValueError('Drawing is active')
        for number,(folder,title,_,source) in enumerate(ARTWORKS):
            if only is not None and number != only:continue
            path=ROOT/source; package=path.parent.parent; task=path.parent
            if json.loads(path.read_text(encoding='utf-8-sig'))['workflow'].get('source_revision'):
                raise ValueError('画稿已按用户原图修订，请使用对应修订工具，不能用旧笔序备份覆盖。')
            history=package/'历史版本'/'20261006_笔序调整前'; history.mkdir(parents=True,exist_ok=True)
            backup=history/'painting.json'
            if not backup.exists():shutil.copy2(path,backup)
            original=json.loads(backup.read_text(encoding='utf-8-sig'));bundle=copy.deepcopy(original)
            audit=json.loads((task/'离线检查.json').read_text(encoding='utf-8-sig'))
            transform=audit.get('source_transform',{'scale':1,'offset':[0,0]});scale=transform['scale'];offset=np.array(transform['offset'])
            old_ink=original['ink_job']['meta']['ink_execution']['base_strokes']; old_colors=original['workflow']['modules']['color']['strokes']
            strokes=reorder(old_ink,number,scale,offset,True)
            colors=[]
            for color in dict.fromkeys(s['color'] for s in old_colors):
                colors.extend(reorder([s for s in old_colors if s['color']==color],number,scale,offset,False))
            prepare(path)
            bundle['ink_job']=ink_payload(original['ink_job'],strokes)
            plan=bundle['workflow'];plan['modules']['outline']['strokes']=copy.deepcopy(strokes);plan['modules']['color']['strokes']=colors
            plan['execution_order']={'format':'shanhe-subject-order-v1','title_last':True,'color_mode':'one_color_all_subjects','subjects':ORDERS[number],'note':'整部位完成后再画下一部位；长笔分段连续；全部上色后最后按山、河、绘题字。'}
            from run_color_from_job import validate_plan
            from run_json_painting import manifest_for
            validate_plan(plan,bundle['ink_job']); manifest_for(strokes,plan,bundle['ink_job']['meta'])
            save(path,bundle);save(task/'black_only.json',bundle['ink_job'])
            for p in task.glob('*_黑线任务.json'):save(p,bundle['ink_job'])
            for p in task.glob('*_计划.json'):save(p,plan)
            index=index_for(bundle);save(task/'绘画顺序索引.json',{'format':'shanhe-stroke-order-index-v1','stroke_data_sha256':digest(stroke_data(bundle)),'counts':{k:len(v) for k,v in index.items()},'stages':index})
            save(task/'部位顺序.json',plan['execution_order'])
            save(task/'笔序修改校验.json',{'date':'2026-10-06','robot_connected':False,'all_strokes_ids_geometry_press_speeds_preserved':True,'parents_and_segments_contiguous':True,'title_after_all_colors':True,'ink_count':len(strokes),'color_count':len(colors),'subjects':ORDERS[number]})
            print(title+': subject order updated, title after all colors.',flush=True)


if __name__=='__main__':main()
