"""Isolated, versioned artwork drafts. All planning happens offline in a child."""
import argparse
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from painting_session import save

LABELS = {'blue':'蓝色', 'green':'绿色', 'red':'红色', 'yellow':'黄色', 'blue2':'淡蓝'}
BGR = {'blue':(160,105,48), 'green':(72,112,65), 'red':(30,48,245),
       'yellow':(38,186,248), 'blue2':(205,170,125)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def studio_error(error):
    text=str(error)
    if '\\' in text or '\n' in text or ':/' in text or '.py' in text or not any('\u4e00' <= c <= '\u9fff' for c in text):
        return '画稿制作未完成，请调整参数或换一张浅底清晰的参考图后再试。详细诊断已保留。'
    return text


def numeric(value, low, high, label):
    if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('%s须在 %s 到 %s 之间。' % (label, low, high))
    return float(value)


def options(body):
    order = body.get('order', 'original')
    if order not in ('original', 'top', 'left'):
        raise ValueError('请选择原有顺序、从上到下或从左到右。')
    excluded = body.get('excluded', [])
    ordered = body.get('ordered', [])
    if not isinstance(excluded, list) or not isinstance(ordered, list) or any(not isinstance(v,str) or len(v)>100 for v in excluded+ordered):
        raise ValueError('笔触选择无效。')
    colors = body.get('colors', ['blue','green'])
    if not isinstance(colors,list) or len(set(colors)) != len(colors) or any(c not in LABELS for c in colors):
        raise ValueError('请选择可用颜料颜色。')
    return dict(min_length=numeric(body.get('min_length',1.2),.2,10,'短线过滤长度'),
                edit_only=body.get('edit_only') is True,
                pressure_scale=numeric(body.get('pressure_scale',1),.5,1.1,'勾勒笔压倍率'),
                pitch=numeric(body.get('pitch',1.5),.6,4,'铺色间距'),
                angle=numeric(body.get('angle',0),-90,90,'铺色方向'),
                color_press=numeric(body.get('color_press',.2),.05,.3,'上色下压量'),
                opacity=numeric(body.get('opacity',.5),.15,.8,'预览浓淡'),
                order=order, excluded=excluded, ordered=ordered, colors=colors)


def reorder(strokes, opts):
    ids = {s['id'] for s in strokes}
    if not set(opts['excluded']).issubset(ids) or not set(opts['ordered']).issubset(ids) or len(set(opts['ordered'])) != len(opts['ordered']):
        raise ValueError('笔触列表已改变，请刷新预览后调整。')
    result = [copy.deepcopy(s) for s in strokes if s['id'] not in opts['excluded']]
    if opts['order'] != 'original':
        axis = 1 if opts['order'] == 'top' else 0
        result.sort(key=lambda s:sum(p[axis] for p in s['points'])/len(s['points']))
    if opts['ordered']:
        rank = {v:i for i,v in enumerate(opts['ordered'])}
        result.sort(key=lambda s:rank.get(s['id'],len(rank)))
    return result


class Studio:
    def __init__(self, app, root, store):
        self.app, self.root, self.folder = app, Path(root), Path(store)/'studio'
        self.lock = threading.RLock()
        self.jobs = {}
        # Published materials are sufficient for the library, even when draft
        # intermediates are later removed. Keep this index with the materials.
        for path in (self.root/'待画作品'/'新作品').glob('*/制作参数.json'):
            try:
                info=read(path)
                if not re.fullmatch('studio-[a-f0-9]{12}',info['artworkId']):
                    continue
                self.app.add(path.parent/'painting.json',info['title'],path.parent/'参考原图.png',
                             ident=info['artworkId'],featured=False,description='新制作画稿 · 已检查')
            except (ValueError,OSError,KeyError,TypeError):
                continue
        for path in self.folder.glob('*/record.json'):
            try:
                record = read(path)
                if record.get('busy'):
                    record.update(busy=False,error='上次制作被中断，已保留最后完成的版本。')
                self.jobs[record['id']] = record
                if record.get('published'):
                    self.load_published(record)
            except (ValueError, OSError, KeyError):
                continue

    def load_published(self, record):
        info = record['published']
        path = self.root/info['path']
        if not path.resolve().is_relative_to((self.root/'待画作品'/'新作品').resolve()):
            raise ValueError('画稿路径无效。')
        if path.exists():
            self.app.add(path, record['title'], path.parent/'参考原图.png',
                         ident=info['artworkId'], featured=False, description='新制作画稿 · 已检查')

    def record(self, ident):
        if not isinstance(ident,str) or not re.fullmatch('[a-f0-9]{12}',ident) or ident not in self.jobs:
            raise ValueError('画稿草稿不存在，请重新选择。')
        return self.jobs[ident]

    def public(self, record):
        return {k:copy.deepcopy(record.get(k)) for k in
                ('id','title','revision','stage','busy','error','counts','settings','checked','availableColors','published')}

    def persist(self, record):
        save(self.folder/record['id']/'record.json',{k:v for k,v in record.items() if not k.startswith('_')})

    def rename(self, body):
        with self.app.lock, self.lock:
            record = self.record(body.get('id'))
            if record.get('abandoned') or record['busy'] or body.get('revision') != record['revision']:
                raise ValueError('画稿正在处理或已更新，请刷新后修改名称。')
            title = body.get('title')
            if not isinstance(title, str) or not title.strip() or len(title.strip()) > 50 or any(ord(c) < 32 for c in title):
                raise ValueError('请输入 1 至 50 个字符的画作名称。')
            record.update(title=title.strip(), revision=record['revision']+1)
            if record.get('published'):
                info = record['published']
                path = self.root/info['path']
                index = path.parent/'制作参数.json'
                if index.exists():
                    metadata = read(index)
                    metadata['title'] = record['title']
                    save(index, metadata)
                self.load_published(record)
            self.persist(record)
            return self.public(record)

    def abandon(self, body):
        with self.lock:
            record = self.record(body.get('id'))
            if body.get('revision') != record['revision']:
                raise ValueError('画稿已更新，请刷新后再放弃。')
            record.update(abandoned=True, busy=False)
            process = record.get('_process')
            if process and process.poll() is None:
                process.terminate()
            self.persist(record)
            return {'abandoned': record['id']}

    def upload(self, body):
        suffix = Path(body.get('name','')).suffix.lower()
        if suffix not in ('.png','.jpg','.jpeg','.webp'):
            raise ValueError('请上传 PNG、JPG 或 WEBP 图片。')
        try:
            raw = base64.b64decode(body['data'],validate=True)
        except (ValueError,KeyError):
            raise ValueError('图片数据无效。')
        if not 0 < len(raw) <= 20*1024*1024:
            raise ValueError('图片须小于 20 MB。')
        import cv2
        import numpy as np
        image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if image is None or max(image.shape[:2])>8000 or min(image.shape[:2])<32:
            raise ValueError('图片无法读取，边长须在 32 至 8000 像素内。')
        ident = uuid.uuid4().hex[:12]
        folder = self.folder/ident
        folder.mkdir(parents=True)
        import imgio
        imgio.imwrite(str(folder/'参考原图.png'),image)
        record = dict(id=ident,title=Path(body['name']).stem[:50] or '新画作',revision=0,
                      stage='uploaded',busy=False,error=None,settings=options({}),counts={},checked=False)
        with self.lock:
            self.jobs[ident]=record
            self.persist(record)
            return self.public(record)

    def start(self, body):
        with self.lock:
            record = self.record(body.get('id'))
            if record.get('abandoned'):
                raise ValueError('这份草稿已放弃，请选择新图片。')
            if record['busy']:
                raise ValueError('本幅画稿正在处理，请等待完成。')
            if sum(bool(r.get('busy')) for r in self.jobs.values()) >= 2:
                raise ValueError('已有两幅画稿正在处理，请等待其中一幅完成。')
            if body.get('revision') != record['revision']:
                raise ValueError('画稿已更新，请刷新后再操作。')
            action = body.get('action')
            if action not in ('extract','outline','ink_only','color','color_edit','validate'):
                raise ValueError('制作步骤无效。')
            if action != 'extract' and not record.get('version'):
                raise ValueError('请先提取线稿。')
            if action in ('color_edit','validate') and record['stage'] not in ('colored','ready'):
                raise ValueError('请先生成上色路径；黑白画也请确认无上色步骤。')
            settings = options(body.get('settings',record['settings']))
            record.update(busy=True,error=None)
            self.persist(record)
            thread = threading.Thread(target=self.process,args=(record,action,settings),daemon=True)
            thread.start()
            return self.public(record)

    def process(self, record, action, settings):
        folder = self.folder/record['id']
        version = uuid.uuid4().hex
        out = folder/'versions'/version
        out.mkdir(parents=True)
        request = dict(action=action,settings=settings,source=str(folder/'参考原图.png'),
                       previous=str(folder/'versions'/record['version']) if record.get('version') else None)
        save(out/'request.json',request)
        args = [sys.executable,str(Path(__file__)),'--build',str(out)]
        saved = self.folder.parent/'site_settings.json'
        if saved.exists():
            args += ['--settings',str(saved)]
        try:
            with self.lock:
                if record.get('abandoned'): return
                process = subprocess.Popen(args,cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                           encoding='utf-8',errors='replace',env=os.environ|{'PYTHONIOENCODING':'utf-8'})
                record['_process'] = process
            try:
                stdout, stderr = process.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise ValueError('画稿处理超时，请简化图片后重试。')
            finally:
                with self.lock: record.pop('_process', None)
            save(out/'diagnostics.json',dict(output=stdout,error=stderr))
            data = read(out/'result.json')
            if process.returncode or data.get('error'):
                raise ValueError(data.get('error','画稿处理未完成，请调整后重新生成。'))
            with self.lock:
                if record.get('abandoned'): return
                record.update(version=version,revision=record['revision']+1,stage=data['stage'],
                              counts=data['counts'],availableColors=data['availableColors'],
                              checked=action=='validate',settings=data.get('settings',settings),busy=False,error=None)
                record.pop('published',None)
                self.persist(record)
        except Exception as exc:
            with self.lock:
                if record.get('abandoned'): return
                record.update(busy=False,error=studio_error(exc))
                self.persist(record)

    def scene(self, ident):
        with self.lock:
            record=self.record(ident)
            if not record.get('version'):
                return None
            return read(self.folder/ident/'versions'/record['version']/'scene.json')

    def image(self, ident, prepared=False):
        record=self.record(ident)
        if prepared and record.get('version'):
            return (self.folder/ident/'versions'/record['version']/'prepared.png').read_bytes()
        return (self.folder/record['id']/'参考原图.png').read_bytes()

    def publish(self, body):
        with self.app.lock, self.lock:
            record=self.record(body.get('id'))
            if record.get('abandoned') or record['busy'] or not record.get('checked') or body.get('revision')!=record['revision']:
                raise ValueError('请先完成最终画稿检查，再加入画库。')
            deleted = getattr(self.app, 'deleted_artworks', {})
            if record.get('published') and not (isinstance(deleted, dict)
                    and record['published']['artworkId'] in deleted):
                return self.public(record)
            # Always create a new material folder. Never overwrite an existing artwork.
            artwork_id='studio-'+uuid.uuid4().hex[:12]
            title=re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',record['title']).strip(' .')[:60] or '新画作'
            target=self.root/'待画作品'/'新作品'/(title+'_'+artwork_id[-6:])
            target.mkdir(parents=True)
            source=self.folder/record['id']/'versions'/record['version']
            data=read(source/'painting.json')
            reference=str(target/'参考原图.png')
            data['workflow']['reference_image']=reference
            data['ink_job']['meta']['image']=reference
            save(target/'painting.json',data)
            shutil.copy2(self.folder/record['id']/'参考原图.png',target/'参考原图.png')
            save(target/'制作参数.json',dict(artworkId=artwork_id,title=record['title'],settings=record['settings'],counts=record['counts']))
            record['published']=dict(artworkId=artwork_id,path=str((target/'painting.json').relative_to(self.root)))
            self.load_published(record)
            self.persist(record)
            return self.public(record)


def studio_for(app):
    from web import server
    with app.lock:
        if not hasattr(app,'studio'):
            app.studio=Studio(app,server.ROOT,server.STORE)
        return app.studio


def route(handler, method, body=None):
    try:
        return _route(handler,method,body)
    except Exception as exc:
        handler.send({'error':studio_error(exc)},400)
        return True


def _route(handler, method, body=None):
    """Additive hook for the existing authenticated localhost HTTP server."""
    from urllib.parse import urlparse,parse_qs
    parsed=urlparse(handler.path)
    static={'/studio':'studio.html','/studio.js':'studio.js','/studio.css':'studio.css'}
    if method=='GET' and parsed.path in static:
        import mimetypes
        path=Path(__file__).with_name(static[parsed.path])
        handler.send(path.read_bytes(),content_type=mimetypes.guess_type(path)[0] or 'text/plain')
        return True
    if not parsed.path.startswith('/api/studio/'):
        return False
    studio=studio_for(handler.server.app)
    endpoint=parsed.path.removeprefix('/api/studio/')
    if method=='GET':
        q=parse_qs(parsed.query)
        ident=q.get('id',[None])[0]
        if endpoint=='jobs':
            with studio.lock: jobs=[studio.public(r) for r in studio.jobs.values() if not r.get('abandoned')]
            handler.send({'jobs':jobs})
        elif endpoint=='job': handler.send(studio.public(studio.record(ident)))
        elif endpoint=='scene': handler.send(studio.scene(ident))
        elif endpoint in ('image','prepared'): handler.send(studio.image(ident,endpoint=='prepared'),content_type='image/png')
        else: raise ValueError('制作页面接口不存在。')
    elif endpoint=='upload': handler.send(studio.upload(body))
    elif endpoint=='step': handler.send(studio.start(body))
    elif endpoint=='publish': handler.send(studio.publish(body))
    elif endpoint=='rename': handler.send(studio.rename(body))
    elif endpoint=='abandon': handler.send(studio.abandon(body))
    else: raise ValueError('制作页面接口不存在。')
    return True


def build(folder, settings_file=None):
    """Offline only. Network is disabled before loading the planning modules."""
    import socket
    def no_network(*args,**kwargs): raise RuntimeError('画稿生成阶段禁止连接机械臂。')
    socket.socket.connect=no_network
    socket.create_connection=no_network
    import cv2
    import numpy as np
    import imgio,vision,job,shuimo
    from config import ArmConfig,ShuimoConfig
    from ink_control import InkPolicy,build_plan,split_long_strokes,refresh_stats
    from geometry import CanvasMap
    from run_mountain_v4_robot import safe_transfers
    from run_json_painting import make_bundle,load_artwork,manifest_for
    from run_color_experiment import module_lines
    from unified_painting import color_station
    from web.site_settings import apply,load_record
    if settings_file: apply(load_record(settings_file))
    available=[]
    # Match the cups currently editable in the workbench. Other historical
    # measurements may belong to an older brush and are not offered here.
    for color in ('blue','green'):
        try: color_station(color,ask=lambda _: '');available.append(color)
        except (ValueError,KeyError): pass
    folder=Path(folder);request=read(folder/'request.json');opts=request['settings'];action=request['action']
    if action=='extract':
        from unified_painting import build_plan as initial_plan
        plan,work,_,points,meta=initial_plan(request['source'])
        for i,s in enumerate(meta['ink_execution']['base_strokes']):s['id']='outline-studio-%05d'%i
        data=make_bundle(plan,dict(format='arm-painting job',version=1,meta=meta,waypoints=points))
        save(folder/'raw.json',data)
        imgio.imwrite(str(folder/'prepared.png'),work)
    else:
        previous=Path(request['previous'])
        data=read(previous/'painting.json')
        shutil.copy2(previous/'raw.json',folder/'raw.json')
        shutil.copy2(previous/'prepared.png',folder/'prepared.png')
    raw=read(folder/'raw.json')
    ink=data['ink_job'];meta=ink['meta'];plan=data['workflow'];module=plan['modules']['color']
    scale=meta['ink_execution']['mm_per_px'];mapper=CanvasMap(meta['image_size'])
    if action in ('extract','outline'):
        base=meta['ink_execution']['base_strokes'] if opts['edit_only'] else raw['ink_job']['meta']['ink_execution']['base_strokes']
        strokes=reorder(base,opts)
        if not opts['edit_only']:strokes=[s for s in strokes if s['length_mm']>=opts['min_length']]
        if not strokes: raise ValueError('过滤后没有可画的黑线，请降低短线过滤长度。')
        if not opts['edit_only']:
            for s in strokes:
                s['press']=[min(ShuimoConfig.PRESS_MAX_MM,p*opts['pressure_scale']) for p in s['press']]
                refresh_stats(s,scale)
        policy=InkPolicy(**meta['ink_execution']['policy'])
        if not opts['edit_only']:
            strokes=split_long_strokes(strokes,scale,policy)
            for s in strokes:
                if s['segments']>1:s['id']=s['id']+'-part'+str(s['segment'])
        meta['ink_execution']['base_strokes']=strokes;meta['strokes']=len(strokes)
        steps,_=build_plan(strokes,shuimo.ink_palette_index(),scale,policy)
        points,_=job.steps_to_waypoints(steps,mapper);ink['waypoints']=safe_transfers(points)
        module['strokes']=[]
        # Keep IDs in the editor tied to the currently visible geometry.
        opts['excluded']=[];opts['ordered']=[]
        stage='outlined'
    elif action=='ink_only':
        module['strokes']=[]
        opts['colors']=[]
        opts['excluded']=[];opts['ordered']=[]
        plan['ready_for_execution']=False
        stage='colored'
    elif action=='color':
        chosen=opts['colors']
        if any(c not in available for c in chosen):
            raise ValueError('所选颜色杯尚未通过现场参数检查，请先在工作台设置杯位。')
        if opts['color_press']>ShuimoConfig.PRESS_MAX_MM:
            raise ValueError('上色下压量超过当前毛笔允许范围，请减小。')
        work,_=vision.prepare_image(vision.load_image(request['source']),work_size=1800)
        gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
        lab=cv2.cvtColor(work,cv2.COLOR_BGR2LAB).astype(float)
        chroma=np.linalg.norm(lab[:,:,1:]-128,axis=2)
        colored=(chroma>10)&(gray<247)&(gray>70)
        exclude=np.zeros(gray.shape,np.uint8)
        for s in meta['ink_execution']['base_strokes']:
            cv2.polylines(exclude,[np.rint(s['points']).astype(np.int32)],False,1,max(1,int(np.ceil(s['width_mm']/scale))))
        exclude=cv2.dilate(exclude,np.ones((5,5),np.uint8))>0
        colored &= ~exclude
        margin=int(np.ceil(21/scale));colored[:margin]=False;colored[-margin:]=False;colored[:,:margin]=False;colored[:,-margin:]=False
        module['motion_candidate']['press_mm']=opts['color_press']
        from color_profile import ColorProfile
        profile=ColorProfile(**module['motion_candidate']).to_dict()
        from scripts import build_filled_landscape as fill
        fill.MAX_STROKE_MM=40.;fill.BRIDGE_MAX_MM=.8;fill.BRIDGE_MIN_INSIDE=1.
        strokes=[]
        if chosen:
            palette=np.array([BGR[c] for c in chosen],np.uint8).reshape(1,-1,3)
            palette=cv2.cvtColor(palette,cv2.COLOR_BGR2LAB).astype(float).reshape(-1,3)
            nearest=np.argmin(np.sum((lab[:,:,None,1:]-palette[None,None,:,1:])**2,axis=3),axis=2)
            for index,color in enumerate(chosen):
                mask=(colored&(nearest==index)).astype(np.uint8)
                mask=cv2.erode(mask,np.ones((3,3),np.uint8))
                paths,_=fill.fill_paths_rot(mask,scale,color,max(1,round(opts['pitch']/scale)),
                                            float(shuimo.press_to_width_mm(opts['color_press'])),opts['opacity'],opts['angle'])
                for s in paths:
                    n=len(s['points']);s.update(press=[opts['color_press']]*n,speed=[profile['draw_velocity_percent']]*n)
                strokes.extend(paths)
        module.update(strokes=strokes,stroke_count=len(strokes))
        permitted=set()
        for color in chosen:
            paths,_=module_lines(module,color,profile);permitted.update(p['id'] for p in paths)
        module['strokes']=[s for s in strokes if s['id'] in permitted]
        save(folder/'color_raw.json',copy.deepcopy(module['strokes']))
        stage='colored'
    else:
        previous=Path(request['previous'])
        if (previous/'color_raw.json').exists():shutil.copy2(previous/'color_raw.json',folder/'color_raw.json')
        if action=='color_edit':
            module['strokes']=reorder(module['strokes'],opts)
            module['strokes']=[s for s in module['strokes'] if s['color'] in opts['colors']]
            rank={c:i for i,c in enumerate(opts['colors'])}
            module['strokes'].sort(key=lambda s:rank[s['color']])
            opts['excluded']=[];opts['ordered']=[]
            stage='colored'
        else:
            stage='ready'
            plan['ready_for_execution']=True
    if action!='color' and request.get('previous') and not (folder/'color_raw.json').exists():
        old=Path(request['previous'])/'color_raw.json'
        if old.exists():shutil.copy2(old,folder/'color_raw.json')
    # Black edits also invalidate previously generated color and final checks.
    module['stroke_count']=len(module['strokes'])
    path=folder/'painting.json';save(path,data)
    artwork=load_artwork(path);manifest_for(artwork[2],artwork[1],artwork[3])
    if action=='validate':
        for c in set(s['color'] for s in module['strokes']):
            if c not in available:raise ValueError('画稿使用了尚未接入现场参数页面的颜料，请重新规划上色。')
            color_station(c,ask=lambda _: '')
    from web.server import artwork_data
    geometry=artwork_data(path)
    scene=dict(size=geometry['size'],scale=scale,outline=[dict(id=s['id'],length=s['length_mm'],**view) for s,view in zip(meta['ink_execution']['base_strokes'],geometry['stages']['black'])],
               color=[dict(id=s['id'],color=s['color'],points=s['points'],length=s['length_mm'],width=s['brush_width_mm_estimate']/scale,opacity=s.get('target_opacity',.5)) for s in module['strokes']])
    save(folder/'scene.json',scene)
    save(folder/'result.json',dict(stage=stage,counts=geometry['counts'],availableColors=available,settings=opts))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build',required=True);parser.add_argument('--settings');args=parser.parse_args()
    try:
        build(args.build,args.settings)
    except Exception as exc:
        save(Path(args.build)/'result.json',dict(error=studio_error(exc)))
        import traceback
        traceback.print_exc()
        raise SystemExit(2)
