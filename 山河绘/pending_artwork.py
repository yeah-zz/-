"""Install a reviewed pending material revision only before an idle new run."""
import json,hashlib,shutil
from pathlib import Path
from painting_session import save,session_lock
ROOT=Path(__file__).resolve().parent


def apply_pending(path,pending_name='待应用_顺皴刷色'):
    if pending_name not in ('待应用_顺皴刷色','待应用_提线清理','待应用_标注下压','待应用_叶团补满','待应用_松枝细节','待应用_原图松枝','待应用_第三幅调整','待应用_远山云雾','待应用_远山细勾','待应用_第五幅连续上色','待应用_第五幅顺坡上色','待应用_右峰黑线'):raise ValueError('Unknown pending material directory')
    path=Path(path).resolve();pending=path.parent/pending_name;request=pending/'应用记录.json'
    if not request.exists():return False
    root=ROOT
    if not path.is_relative_to(root/'待画作品'):return False
    logs=root/'robot_runs'
    if any((logs/n).exists() for n in ('painting_session.lock','blue_wash_motion.lock')):return False
    with session_lock(logs):
        if (logs/'blue_wash_motion.lock').exists():return False
        spec=json.loads(request.read_text(encoding='utf-8'))
        if spec.get('applied'):return False
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['baseline_sha256']:
            raise ValueError('待应用画稿的旧版已改变，请重新核对，未自动覆盖。')
        candidate=pending/'painting.json'
        if hashlib.sha256(candidate.read_bytes()).hexdigest()!=spec['candidate_sha256']:
            raise ValueError('待应用画稿校验不一致，请重新核对。')
        # Final validation uses the exact runtime profile and current hardware
        # registration, but performs no network or robot connection.
        from web.execution import prepare
        from run_json_painting import load_artwork
        prepare(candidate);load_artwork(candidate)
        data=json.loads(candidate.read_text(encoding='utf-8'))
        if pending_name=='待应用_右峰黑线':
            from scripts.repair_pavilion_inner_ridge import audit
            save(pending/'右峰黑线校验.json',audit(candidate,path))
        elif pending_name=='待应用_第五幅顺坡上色':
            from scripts.refine_terrace_summits_20261007 import audit
            save(pending/'顺山坡上色校验.json',audit(candidate,path))
        elif pending_name=='待应用_第五幅连续上色':
            from scripts.optimize_terrace_color_20261007 import audit
            save(pending/'连续上色校验.json',audit(candidate,path))
        elif pending_name=='待应用_远山细勾':
            from scripts.refine_distance_ridges_20261007 import audit
            save(pending/'远山细勾校验.json',audit(candidate,path))
        elif pending_name=='待应用_远山云雾':
            from scripts.refine_pavilion_mist_20261007 import audit
            save(pending/'远山云雾校验.json',audit(candidate,path))
        elif pending_name=='待应用_第三幅调整':
            from scripts.revise_pavilion_landscape import audit
            save(pending/'第三幅调整校验.json',audit(candidate,path))
        elif pending_name=='待应用_原图松枝':
            from scripts.source_faithful_pine_branches import audit
            save(pending/'原图松枝校验.json',audit(candidate,path))
        elif pending_name=='待应用_松枝细节':
            from scripts.refine_waterfall_pine_branches import audit
            save(pending/'松枝细节校验.json',audit(candidate,path))
        elif pending_name=='待应用_叶团补满':
            from scripts.repair_creek_canopy_fill import audit
            save(pending/'叶团补满校验.json',audit(candidate,path))
        elif pending_name=='待应用_标注下压':
            from scripts.annotated_creek_pressures import audit
            save(pending/'标注下压校验.json',audit(candidate,path))
        elif pending_name=='待应用_提线清理':
            from scripts.clean_landscape_ink import audit_cleanup
            save(pending/'提线修复校验.json',audit_cleanup(candidate,path))
        elif data['workflow'].get('source_revision',{}).get('terrain_fill')=='source_aligned_curved_ribbons':
            from scripts.validate_source_washes import audit_candidate
            save(pending/'真实轨迹离线校验.json',audit_candidate(candidate,path))
        elif data['workflow'].get('source_revision',{}).get('terrain_fill')=='three_source_aligned_curved_ribbons':
            from scripts.validate_three_source_washes import audit_candidate
            save(pending/'真实轨迹离线校验.json',audit_candidate(candidate,path))
        history_name={'待应用_提线清理':'20261006_提线修复前','待应用_标注下压':'20261006_用户标注下压前','待应用_叶团补满':'20261006_叶团补满前','待应用_松枝细节':'20261006_松枝细节调整前','待应用_原图松枝':'20261006_原图松枝重提前','待应用_第三幅调整':'20261006_第三幅原图风格调整前'}.get(pending_name,'20261006_顺皴刷色前')
        if pending_name=='待应用_远山云雾':history_name='20261007_远山云雾调整前'
        if pending_name=='待应用_远山细勾':history_name='20261007_远山细勾调整前'
        if pending_name=='待应用_第五幅连续上色':history_name='20261007_第五幅连续上色前'
        if pending_name=='待应用_第五幅顺坡上色':history_name='20261007_第五幅顺坡上色前'
        if pending_name=='待应用_右峰黑线':history_name='20261007_右峰黑线补齐前'
        history=path.parent.parent/'历史版本'/history_name;history.mkdir(parents=True,exist_ok=True)
        if not (history/'painting.json').exists():shutil.copy2(path,history/'painting.json')
        elif (history/'painting.json').read_bytes()!=path.read_bytes():
            version=history/spec['baseline_sha256'][:12];version.mkdir(exist_ok=True)
            if not (version/'painting.json').exists():shutil.copy2(path,version/'painting.json')
        if pending_name=='待应用_标注下压':
            source=Path(data['workflow']['source_revision']['reference_source_file']).resolve()
            if not source.is_relative_to(root/'参考图'):raise ValueError('Source image outside reference directory')
            if hashlib.sha256(source.read_bytes()).hexdigest()!=data['workflow']['source_revision']['source_sha256']:raise ValueError('Reference source changed')
            reference=path.parent.parent/'参考原图.png'
            if not (history/'参考原图.png').exists():shutil.copy2(reference,history/'参考原图.png')
            shutil.copy2(source,reference)
        save(path,data);save(path.parent/'black_only.json',data['ink_job'])
        for p in path.parent.glob('*_黑线任务.json'):save(p,data['ink_job'])
        for p in path.parent.glob('*_计划.json'):save(p,data['workflow'])
        ink=data['ink_job']['meta']['ink_execution']['base_strokes'];colors=data['workflow']['modules']['color']['strokes']
        counts=dict(black=sum(s.get('role')!='title' for s in ink),blue=sum(s['color']=='blue' for s in colors),
                    green=sum(s['color']=='green' for s in colors),title=sum(s.get('role')=='title' for s in ink))
        save(path.parent/'原图修订与下压说明.json',data['workflow']['source_revision'])
        save(path.parent/'笔序修改校验.json',dict(robot_connected=False,user_requested_source_revision=True,
            title_after_all_colors=True,terrain_direction=data['workflow']['source_revision'].get('terrain_direction','downward-and-safe-return-sweeps'),tree_trunk_fill=False,counts=counts))
        for name in ('上色边界校验.json','真实轨迹离线校验.json','原图色带禁涂掩膜.npz','原图色带修订说明.md','提线修复校验.json','提线修复说明.md','标注下压校验.json','标注下压说明.md','叶团填色掩膜.npz','叶团补满校验.json','叶团补满说明.md','松枝细节校验.json','松枝细节说明.md','原图松枝校验.json','原图松枝说明.md','原图松枝提线掩膜.npz','第三幅调整校验.json','第三幅调整说明.md','第三幅原图约束掩膜.npz'):
            if (pending/name).exists():shutil.copy2(pending/name,path.parent/name)
        if pending_name=='待应用_远山云雾':
            for name in ('远山云雾校验.json','远山云雾说明.md','远山云雾覆盖掩膜.npz'):
                shutil.copy2(pending/name,path.parent/name)
        if pending_name=='待应用_远山细勾':
            for name in ('远山细勾校验.json','各部位下压量.md','山顶少色掩膜.npz'):
                shutil.copy2(pending/name,path.parent/name)
        if pending_name=='待应用_第五幅连续上色':
            for name in ('连续上色校验.json','连续上色说明.md','连续接笔安全掩膜.npz'):
                shutil.copy2(pending/name,path.parent/name)
        if pending_name=='待应用_第五幅顺坡上色':
            for name in ('顺山坡上色校验.json','顺山坡上色说明.md','顺山坡上色掩膜.npz'):
                shutil.copy2(pending/name,path.parent/name)
        from collections import Counter
        audit=path.parent/'离线检查.json'
        if audit.exists():
            values=json.loads(audit.read_text(encoding='utf-8'))
            values.update(ink_strokes=len(ink),source_revision=data['workflow']['source_revision'],
                          colors={c:dict(strokes=counts[c]) for c in ('blue','green')},role_counts=dict(Counter(s['role'] for s in ink)))
            save(audit,values)
        preview=path.parent.parent/'预览与下压标注';preview.mkdir(exist_ok=True)
        if (pending/'任务预览_线宽色彩为估计.png').exists():
            for name in ('任务预览_线宽色彩为估计.png','执行路径预览_非实测线宽.png'):
                shutil.copy2(pending/'任务预览_线宽色彩为估计.png',preview/name)
        for name in ('顺皴刷色预览.png','原图色带前后对照.png','提线修复前后对照.png','纯勾线.png','新版原图与标注下压对照.png','黑线加蓝绿_实际路径.png','画布参考.png','叶团补满前后对照.png','松枝细节前后对照.png','原图松枝与黑线对照.png','原图与第三幅调整对照.png'):
            if (pending/name).exists():shutil.copy2(pending/name,preview/name)
        if pending_name=='待应用_远山云雾':
            for name in ('远山云雾与原图对照.png','原图与第三幅调整对照.png','新版原图与标注下压对照.png'):
                shutil.copy2(pending/'远山云雾与原图对照.png',preview/name)
        if pending_name=='待应用_远山细勾':
            for name in ('远山细勾与下压量对照.png','原图与第三幅调整对照.png','新版原图与标注下压对照.png'):
                shutil.copy2(pending/'远山细勾与下压量对照.png',preview/name)
        if pending_name=='待应用_第五幅连续上色':
            shutil.copy2(pending/'第五幅连续上色前后对照.png',preview/'第五幅连续上色前后对照.png')
        if pending_name=='待应用_第五幅顺坡上色':
            shutil.copy2(pending/'第五幅顺山坡上色前后对照.png',preview/'第五幅顺山坡上色前后对照.png')
            if (pending/'第五幅远山扩色与原图对照.png').exists():shutil.copy2(pending/'第五幅远山扩色与原图对照.png',preview/'第五幅远山扩色与原图对照.png')
        import re
        readme=path.parent.parent/'阅读说明.md'
        if readme.exists():
            text=readme.read_text(encoding='utf-8').replace('当前当前笔数：','当前笔数：')
            text=re.sub(r'当前笔数：.*?。','当前笔数：山水勾勒 %(black)d 笔，蓝色 %(blue)d 笔，绿色 %(green)d 笔，最后题字 %(title)d 笔。'%counts,text)
            revision=data['workflow']['source_revision']
            margin_note='山石中心线按半笔宽与1.5mm余量内收，仍需实测纸上笔宽和渗色。'
            if 'terrain_contour_bleed_allowance_mm' in revision:
                margin_note='山石轮廓按估计半笔宽与%.2fmm余量内收；树干与建筑周围保留1.5mm余量。笔宽和渗色仍需实测。'%revision['terrain_contour_bleed_allowance_mm']
            note='\n\n## 顺皴刷色\n\n'+revision['note']+'\n'+margin_note+'\n'
            if '## 顺皴刷色' not in text:text+=note
            else:text=re.sub(r'\n+## 顺皴刷色\n.*?(?=\n## |\Z)',note,text,flags=re.S)
            readme.write_text(text,encoding='utf-8')
        directory=root/'待画作品'/'作品目录.md'
        if directory.exists():
            text=directory.read_text(encoding='utf-8')
            from web.catalog import ARTWORKS
            for number,(_,title,_,canonical) in enumerate(ARTWORKS,1):
                if (root/canonical).resolve()==path:
                    prefix='| %02d | %s |'%(number,title)
                    text=re.sub(re.escape(prefix)+r'.*?\| \[painting.json\]',
                        prefix+' %(black)d | %(blue)d | %(green)d | %(title)d | [painting.json]'%counts,text)
                    break
            directory.write_text(text,encoding='utf-8')
        spec['applied']=True;save(request,spec)
        from scripts.report_subject_order import main
        main()
        return True
