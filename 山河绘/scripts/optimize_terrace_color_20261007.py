"""Connect existing color sweeps inside source pigment; keep every pressure sample."""
from pathlib import Path
import sys,json,copy,hashlib
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from painting_session import save
from web.catalog import ARTWORKS
from scripts.source_aligned_washes import dense_path,points_inside
from scripts.source_wash_comparison import render
from scripts.consolidate_selected_artworks import digest,stroke_data
from ink_control import refresh_stats
from shuimo import press_to_width_mm

ARRAYS=('points','press','speed','target_width_mm','target_opacity')

def audit(candidate,baseline):
    candidate=Path(candidate);old=json.loads(Path(baseline).read_text(encoding='utf-8'));new=json.loads(candidate.read_text(encoding='utf-8'))
    assert new['ink_job']==old['ink_job'] and new['workflow']['modules']['outline']==old['workflow']['modules']['outline']
    before=old['workflow']['modules']['color'];after=new['workflow']['modules']['color'];assert before['motion_candidate']==after['motion_candidate'] and before['reload_strokes_per_load']==after['reload_strokes_per_load']
    sources={s['id']:s for s in before['strokes']};seen=[];scale=before['mm_per_px'];masks=np.load(candidate.parent/'连续接笔安全掩膜.npz');connectors=0
    for s in after['strokes']:
        if 'source_spans' not in s:assert s==sources[s['id']];seen.append(s['id']);continue
        assert s['length_mm']<=45.001
        for span in s['source_spans']:
            prior=sources[span['id']];seen.append(prior['id']);a,b=span['range']
            for k in ARRAYS:
                if k in prior:assert s[k][a:b]==(prior[k][::-1] if span['reversed'] else prior[k])
        for a,b in s['connector_spans']:
            q=dense_path(s['points'][a:b],.05/scale);width=max(1,round(press_to_width_mm(max(s['press'][a:b]))/scale));assert np.all(points_inside(q,masks[s['color']+'-%d'%width]))
            assert np.linalg.norm(np.diff(s['points'][a:b],axis=0),axis=1).sum()*scale<=5.001
            assert max(s['press'][a:b])<=.280001;connectors+=1
    assert Counter(seen)==Counter(sources.keys())
    assert new['workflow']['source_revision']['stroke_data_sha256']==digest(stroke_data(new))
    from web.execution import prepare
    from run_json_painting import load_artwork,manifest_for
    prepare(candidate);ink,plan,strokes,meta=load_artwork(candidate);manifest_for(strokes,plan,meta)
    old_length=sum(s['length_mm'] for s in before['strokes']);new_length=sum(s['length_mm'] for s in after['strokes'])
    # Explicit per-lift scenarios, not an uncalibrated claim of machine duration.
    saved_lifts=len(before['strokes'])-len(after['strokes']);extra=new_length-old_length
    return dict(robot_connected=False,ink_and_outline_exactly_preserved=True,all_original_color_points_and_pressure_samples_preserved=True,source_connectors_inside_half_width_eroded_color_mask=True,
        color_before=len(before['strokes']),color_after=len(after['strokes']),saved_lift_cycles=saved_lifts,wet_length_before_mm=round(old_length,2),wet_length_after_mm=round(new_length,2),new_contact_limit_mm=45,
        connector_count=connectors,counts=dict(black=sum(s.get('role')!='title' for s in strokes),title=sum(s.get('role')=='title' for s in strokes),**dict(Counter(s['color'] for s in after['strokes']))),
        estimated_saved_seconds_at_draw_3_5_mm_s={str(t):round(saved_lifts*t-extra/3.5,1) for t in (1,2,3)},timing_is_unmeasured_scenario=True,runtime_validation_passed=True)

def main():
    path=ROOT/ARTWORKS[4][3];old=json.loads(path.read_text(encoding='utf-8'));new=copy.deepcopy(old);module=new['workflow']['modules']['color'];scale=module['mm_per_px'];groups=defaultdict(list)
    for s in module['strokes']:groups[(s['color'],s['drawing_region'],s['region_id'],s['role'])].append(s)
    masks={}
    for color,name in [('blue','蓝色色区.png'),('green','青绿色区.png')]:
        raw=cv2.imdecode(np.fromfile(str(ROOT/'输出/横题山河绘_青绿梯田云山_20261005'/name),np.uint8),0)<128
        # Existing strokes already reach the source mask edge. Preserve their
        # modeled footprint when assessing a safe turnaround at those ends.
        allowed=raw.astype(np.uint8)
        for s in module['strokes']:
            if s['color']!=color:continue
            q=np.rint(s['points']).astype(np.int32)
            for k,(a,b) in enumerate(zip(q[:-1],q[1:])):
                width=max(1,round(max(press_to_width_mm(s['press'][k]),press_to_width_mm(s['press'][k+1]))/scale))
                cv2.line(allowed,tuple(a),tuple(b),1,width)
        for width in range(4,8):
            radius=max(1,width//2)
            kernel=np.zeros((radius*2+1,radius*2+1),np.uint8);cv2.circle(kernel,(radius,radius),radius,1,-1)
            masks[color+'-%d'%width]=cv2.erode(allowed,kernel)
    result=[]
    for (color,region,zone,role),ss in groups.items():
        pending=list(ss)
        while pending:
            current=copy.deepcopy(pending.pop(0));length=current['length_mm'];spans=[dict(id=current['id'],range=[0,len(current['points'])],reversed=False)];turns=[]
            while pending:
                end=np.array(current['points'][-1]);options=[]
                for j,s in enumerate(pending):
                    if length+s['length_mm']>45:continue
                    for reverse in (False,True):
                        start=np.array(s['points'][-1] if reverse else s['points'][0]);distance=np.linalg.norm(start-end)*scale
                        if distance<=5.0 and length+s['length_mm']+distance<=45:options.append((distance,j,reverse))
                found=None
                for distance,j,reverse in sorted(options):
                    s=pending[j];a=end;b=np.array(s['points'][-1] if reverse else s['points'][0]);n=max(2,int(np.ceil(distance/.15))+1);turn=np.linspace(a,b,n)
                    width=max(1,round(press_to_width_mm(max(current['press'][-1],s['press'][-1] if reverse else s['press'][0]))/scale))
                    if np.all(points_inside(dense_path(turn,.05/scale),masks[color+'-%d'%width])):found=(distance,j,reverse,turn);break
                if found is None:break
                distance,j,reverse,turn=found;s=pending.pop(j);start=len(current['points']);n=len(turn)
                for k in ARRAYS:
                    if k not in current:continue
                    values=s[k][::-1] if reverse else s[k]
                    bridge=turn.tolist() if k=='points' else np.linspace(current[k][-1],values[0],n).tolist()
                    current[k].extend(bridge[1:]);current[k].extend(values)
                turns.append([start-1,start+n-1]);spans.append(dict(id=s['id'],range=[start+n-1,start+n-1+len(s['points'])],reversed=reverse));length+=s['length_mm']+distance
            if len(spans)>1:
                current.update(id='terrace-continuous-%04d'%len(result),source_spans=spans,connector_spans=turns,path_style='continuous-same-color-source-zone',source_path_ids=[v['id'] for v in spans]);refresh_stats(current,scale)
                current['brush_width_mm_estimate']=max(current['target_width_mm'])
            result.append(current)
    module.update(strokes=result,stroke_count=len(result));plan=new['workflow'];note='第五幅仅合并同颜色、同部位、同源色区的相邻上色路径；所有原色笔点列、逐点下压量和速度逐项保留，可反向连续回扫。新增短接笔必须在按半笔宽内收的原色区中，最长5mm，合并后整笔最多45mm；白云、田埂和水纹间隔不跨越。勾线、行笔速度和每4笔补色设置保持。'
    transform=json.loads((path.parent/'离线检查.json').read_text(encoding='utf-8'))['source_transform']
    plan['source_revision']=dict(reference_source_file=str(path.parent.parent/'参考原图.png'),source_sha256=hashlib.sha256((path.parent.parent/'参考原图.png').read_bytes()).hexdigest(),source_transform=transform,stroke_data_sha256=digest(stroke_data(new)),note=note,color_connection_optimization=True,original_pressures_preserved=True);plan['note']+=' '+note
    pending=path.parent/'待应用_第五幅连续上色';pending.mkdir(exist_ok=True);save(pending/'painting.json',new);np.savez_compressed(pending/'连续接笔安全掩膜.npz',**masks)
    report=audit(pending/'painting.json',path);save(pending/'连续上色校验.json',report)
    (pending/'连续上色说明.md').write_text('# 第五幅连续上色\n\n'+note+'\n\n'+json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ('任务预览_线宽色彩为估计.png','黑线加蓝绿_实际路径.png'):render(new).save(pending/name)
    board=Image.new('RGB',(2000,960),'#f7f7f0');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28)
    for i,(bundle,label) in enumerate(((old,'修改前 · 上色706笔'),(new,'连续上色 · %d笔'%len(result)))):
        pic=render(bundle,True);pic.thumbnail((930,740));x=35+i*1000;draw.text((x,25),label,font=font,fill='#304c3b');board.paste(pic,(x,100))
    draw.text((35,850),'保留原勾线、全部上色色带和逐点下压量；仅增加同色区内的短接笔。',font=font,fill='#304c3b')
    draw.text((35,895),'预览笔宽与色阶为模型估计；节省用时取决于实际抬笔、往返和补色耗时。',font=font,fill='#687562');board.save(pending/'第五幅连续上色前后对照.png')
    save(pending/'应用记录.json',dict(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False))
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
