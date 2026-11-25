# -*- coding: utf-8 -*-
"""Public pretrained vectorizer comparison; offline robot, all outputs on E."""
from rock_v15_common import LAB, read, jsave, digest, FONT, SMALL
from pathlib import Path
import os,sys,json,time,urllib.request,zipfile,shutil,importlib.util
ROOT=LAB/'public_vectorizers_v30_20261027'
DEPS=ROOT/'dependencies'
RESULTS=ROOT/'results'
PAD_MODE='constant'
MODEL_CACHE={}
sys.path.insert(1,str(DEPS))

def fetch(url,dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():return dest
    req=urllib.request.Request(url,headers={'User-Agent':'Gongbi-public-research-benchmark/1.0'})
    started=time.perf_counter()
    with urllib.request.urlopen(req,timeout=90) as r,open(str(dest)+'.part','wb') as f:
        shutil.copyfileobj(r,f,1024*1024)
    Path(str(dest)+'.part').replace(dest)
    print('DOWNLOADED',dest.name,dest.stat().st_size,round(time.perf_counter()-started,1),flush=True)
    return dest

def bootstrap():
    ROOT.mkdir(parents=True,exist_ok=True)
    snapshots=[]
    for repo,name in [('Nauhcnay/Deep-Sketch-Vectorization','DeepSketch'),('CISLab-HKUST/SketchKit','SketchKit')]:
        meta=ROOT/'sources'/(name+'_commit.json')
        if not meta.exists():fetch('https://api.github.com/repos/'+repo+'/commits/main',meta)
        commit=read(meta)['sha'];archive=fetch('https://codeload.github.com/'+repo+'/zip/'+commit,ROOT/'downloads'/(name+'_'+commit[:12]+'.zip'))
        target=ROOT/'sources'/name
        if not target.exists():
            with zipfile.ZipFile(archive) as z:
                prefix=z.namelist()[0].split('/')[0]+'/'
                for entry in z.infolist():
                    rel=entry.filename[len(prefix):]
                    if not rel or entry.is_dir():continue
                    path=(target/rel).resolve()
                    if target.resolve() not in path.parents:raise ValueError('Unsafe archive path')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(entry) as src,open(path,'wb') as dst:shutil.copyfileobj(src,dst)
        snapshots.append(dict(repository=repo,commit=commit,archive=str(archive),sha256=digest(archive)))
    import torch
    a=torch.nn.Conv2d(2,3,3).cuda();x=torch.randn(1,2,16,16,device='cuda');a(x).sum().backward()
    modules={m:bool(importlib.util.find_spec(m)) for m in ['cv2','numpy','scipy','skimage','sklearn','svgpathtools','rdp','aabbtree','edge_distance_aabb','ndjson','einops','torchvision','gdown','networkx']}
    jsave(ROOT/'bootstrap.json',dict(sources=snapshots,python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),cuda_backward_passed=True,available=modules,dependencies_target=str(DEPS),robot_connected=False))
    print('ENVIRONMENT',torch.__version__,torch.cuda.get_device_name(0),modules,flush=True)

def assets():
    from concurrent.futures import ThreadPoolExecutor
    urls=[('https://cislab.hkust-gz.edu.cn/projects/sketchkit/weights/LineDrawer/500_rough.pth',ROOT/'weights'/'LineDrawer'/'500_rough.pth'),('https://cislab.hkust-gz.edu.cn/projects/sketchkit/weights/LineDrawer/500_ref.pth',ROOT/'weights'/'LineDrawer'/'500_ref.pth')]
    meta=ROOT/'sources'/'DeepSketch_weights.json'
    if not meta.exists():fetch('https://huggingface.co/api/models/waterheater/deepsketch',meta)
    info=read(meta)
    print('DEEPSKETCH_FILES',[r['rfilename'] for r in info['siblings']],flush=True)
    for name in ['udf_full.pth','ndc_full.pth']:
        urls.append(('https://huggingface.co/waterheater/deepsketch/resolve/'+info['sha']+'/'+name,ROOT/'weights'/'DeepSketch'/name))
    records=[]
    def task(pair):
        url,dest=pair
        try:fetch(url,dest);return dict(url=url,path=str(dest),bytes=dest.stat().st_size,sha256=digest(dest),status='downloaded')
        except Exception as e:return dict(url=url,path=str(dest),status='error',error=repr(e))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(task,urls):records.append(row);print('ASSET',row,flush=True)
    jsave(ROOT/'assets.json',dict(records=records,huggingface_revision=info['sha']))

def install_deps():
    import subprocess
    DEPS.mkdir(exist_ok=True)
    packages=['svgpathtools==1.6.1','svgwrite==1.4.3','rdp==0.8','aabbtree==2.8.1','ndjson==0.3.1','einops==0.8.0','edge-distance-aabb']
    cmd=[sys.executable,'-B','-m','pip','install','--disable-pip-version-check','--no-deps','--target',str(DEPS)]+packages
    cp=subprocess.run(cmd,text=True,encoding='utf-8',errors='replace',capture_output=True)
    (ROOT/'dependency_install.log').write_text(cp.stdout+'\n'+cp.stderr,encoding='utf-8')
    print(cp.stdout[-5000:],cp.stderr[-3000:],flush=True)
    if cp.returncode:raise RuntimeError('Dependency installation failed; see E log')

def prepare():
    import v27_separation_faint as old
    import numpy as np,cv2,hashlib
    from PIL import Image
    all_cases=old.cases('development')
    print('AVAILABLE_CASES',[(r['id'],r['split'],rgb.shape) for r,rgb,refs,valid,bg in all_cases],flush=True)
    chosen=['R03','R04','H01','H02','N10','N11','X01','X02','X03','X04']
    clear=[r['id'] for r,rgb,refs,valid,bg in all_cases if r['split']=='development_holdout'][:2]
    chosen+=clear
    if (ROOT/'manifest.json').exists():raise FileExistsError('Preserve predeclared comparison set')
    rows=[]
    for r,rgb,refs,valid,bg in all_cases:
        if r['id'] not in chosen:continue
        key=r['id'];dest=ROOT/'inputs'/key;dest.mkdir(parents=True,exist_ok=True)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        smooth=cv2.bilateralFilter(gray,5,.035,3)
        paper=cv2.GaussianBlur(cv2.dilate(smooth,np.ones((41,41),np.uint8)),(0,0),9)
        alpha=np.clip((paper-smooth)/np.maximum(paper,.1),0,1)
        pre=np.uint8(np.rint(255*(1-np.clip(alpha*5,0,1))))
        Image.fromarray(rgb).save(dest/'raw.png');Image.fromarray(pre).save(dest/'normalized.png')
        np.savez_compressed(dest/'reference.npz',rgb=rgb,valid=valid,bg=bg)
        jsave(dest/'reference.json',dict(paths=[np.asarray(p).tolist() for p in refs],partial=True,not_complete_ink_gold=True))
        rows.append(dict(id=key,group=r['split'],shape=list(rgb.shape[:2]),raw=str(dest/'raw.png'),normalized=str(dest/'normalized.png'),raw_sha256=digest(dest/'raw.png'),normalized_sha256=digest(dest/'normalized.png'),reference_file=str(dest/'reference.json'),mask_file=str(dest/'reference.npz'),historic_development_not_new_blind=True))
    if len(rows)!=12:raise RuntimeError('Expected 12 fixed images: '+str(chosen))
    jsave(ROOT/'manifest.json',dict(records=rows,preprocessing=dict(bilateral_diameter=5,sigma_color=.035,sigma_space=3,paper_dilation=41,paper_gaussian=9,relative_contrast_gain=5,chosen_before_results=True),pretrained_models_only=True,no_finetuning=True,inference_never_reads_reference=True,maximum_input_long_edge=512,pad_white_to_multiple_8=True,robot_connected=False))
    print('PREPARED',[r['id'] for r in rows],flush=True)

def import_line():
    import types,importlib,torch
    if 'linedrawer' in MODEL_CACHE:return MODEL_CACHE['linedrawer']
    path=ROOT/'sources'/'SketchKit'/'sketchkit'/'vectorization'/'LineDrawer'/'CNNVE'
    package=types.ModuleType('v30_cnnve');package.__path__=[str(path)];sys.modules['v30_cnnve']=package
    module=importlib.import_module('v30_cnnve.CNNVE')
    model=importlib.import_module('v30_cnnve.network.model')
    def local_weights(url,map_location=None,**unused):
        file=ROOT/'weights'/'LineDrawer'/url.rsplit('/',1)[-1]
        if not file.is_file():raise FileNotFoundError(file)
        return torch.load(file,map_location=map_location,weights_only=True)
    model.load_state_dict_from_url=local_weights
    MODEL_CACHE['linedrawer']=module.CNNVE(device='cuda')
    return MODEL_CACHE['linedrawer']

def import_deep():
    import importlib
    path=ROOT/'sources'/'DeepSketch';sys.path.insert(1,str(path));sys.path.insert(2,str(path/'utils'))
    return importlib.import_module('predict_s1')

def smoke():
    import torch,traceback
    torch.set_num_threads(4)
    result={}
    for name,loader in [('LineDrawer',import_line),('DeepSketch',import_deep)]:
        try:
            obj=loader()
            if name=='DeepSketch':
                from types import SimpleNamespace
                models=obj.load_model(SimpleNamespace(model_udf=str(ROOT/'weights'/'DeepSketch'/'udf_full.pth'),model_ndc=str(ROOT/'weights'/'DeepSketch'/'ndc_full.pth')),'cuda')
                result[name]=dict(status='loaded',device=str(next(models[0].parameters()).device),udf_args=vars(models[1]),ndc_args=vars(models[3]))
            else:result[name]=dict(status='loaded',device=str(next(obj.model.parameters()).device))
            print('SMOKE',name,result[name],flush=True)
            del obj
        except Exception:result[name]=dict(status='error',traceback=traceback.format_exc());print(result[name]['traceback'],flush=True)
        torch.cuda.empty_cache()
    jsave(ROOT/'model_load_check.json',result)

def model_input(row,view):
    import cv2,numpy as np
    from PIL import Image
    a=np.asarray(Image.open(row[view]).convert('L'));h,w=a.shape
    scale=min(1.,512/max(h,w));hh=max(8,int(round(h*scale)));ww=max(8,int(round(w*scale)))
    b=cv2.resize(a,(ww,hh),interpolation=cv2.INTER_AREA) if (hh,ww)!=(h,w) else a.copy()
    b=np.pad(b,((0,(-hh)%8),(0,(-ww)%8)),mode='edge') if PAD_MODE=='edge' else np.pad(b,((0,(-hh)%8),(0,(-ww)%8)),constant_values=255)
    return b,dict(original_shape=[h,w],resized_shape=[hh,ww],padded_shape=list(b.shape),scale_xy=[ww/w,hh/h],padding_mode=PAD_MODE)

def extra_probes():
    import numpy as np,cv2
    from PIL import Image
    if (ROOT/'extra_probes.json').exists():raise FileExistsError('Preserve extra probes')
    rows=[]
    for key,sid,box in [('Q06_full','user_qianli_06',None),('Q07_mountain','user_qianli_07',(.28,.08,.63,.53))]:
        source=LAB/'data'/'processed'/sid/'image.png';rgb=np.asarray(Image.open(source).convert('RGB'))
        if box:
            h,w=rgb.shape[:2];a,b,c,d=box;rgb=rgb[round(b*h):round(d*h),round(a*w):round(c*w)]
        h,w=rgb.shape[:2];ratio=min(1.,512/max(h,w));rgb=cv2.resize(rgb,(round(w*ratio),round(h*ratio)),interpolation=cv2.INTER_AREA)
        dest=ROOT/'inputs'/key;dest.mkdir(parents=True,exist_ok=True);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;smooth=cv2.bilateralFilter(gray,5,.035,3);paper=cv2.GaussianBlur(cv2.dilate(smooth,np.ones((41,41),np.uint8)),(0,0),9);alpha=np.clip((paper-smooth)/np.maximum(paper,.1),0,1);pre=np.uint8(np.rint(255*(1-np.clip(alpha*5,0,1))))
        Image.fromarray(rgb).save(dest/'raw.png');Image.fromarray(pre).save(dest/'normalized.png')
        rows.append(dict(id=key,group='qualitative_colored_ancient_painting_probe',shape=list(rgb.shape[:2]),raw=str(dest/'raw.png'),normalized=str(dest/'normalized.png'),source=str(source),source_sha256=digest(source),source_box_fraction=box,raw_sha256=digest(dest/'raw.png'),normalized_sha256=digest(dest/'normalized.png'),no_ground_truth=True,not_independent_test=True))
    jsave(ROOT/'extra_probes.json',dict(records=rows,not_part_of_fixed_12_quantitative_set=True,no_parameters_changed=True))

def all_rows():
    rows=read(ROOT/'manifest.json')['records']
    if (ROOT/'extra_probes.json').exists():rows+=read(ROOT/'extra_probes.json')['records']
    return rows

def native_line_paths(dic):
    """Keep native MergeList membership; join only degree-two endpoints.

    Unlike SketchKit's convenience SVG adapter this uses final merged groups.
    Multiway branches remain ambiguous, not an invented single traversal.
    """
    import numpy as np
    from collections import defaultdict
    from scipy.ndimage import distance_transform_edt
    edges=[];lookup=dic['EdgeDic'];groups=dic['MergeList'];masks=dic['SplitMaskDic']
    membership={int(edge):i for i,g in enumerate(groups) for edge in g}
    for record in dic['simpleGraphs']:
        for u,v,data in record['graph'].edges(data=True):
            obj=data.get('object')
            if obj is None or not obj.pixels:continue
            name='%d,%d_%d,%d'%(u.point[1],u.point[0],v.point[1],v.point[0]);idx=lookup.get(name,-1)
            if idx<0 or idx not in membership:continue
            a=np.asarray(u.point);b=np.asarray(v.point);mid=np.asarray(obj.pixels)
            forward=np.linalg.norm(mid[0]-a)+np.linalg.norm(mid[-1]-b);backward=np.linalg.norm(mid[-1]-a)+np.linalg.norm(mid[0]-b)
            if backward<forward:mid=mid[::-1]
            q=np.concatenate([a[None],mid,b[None]]).astype(np.float32)[:,::-1]
            q=q[np.r_[True,np.any(np.diff(q,axis=0)!=0,axis=1)]]
            edges.append(dict(group=membership[idx],q=q,start=tuple(q[0]),end=tuple(q[-1]),mask_id=int(idx)))
    out=[];outgroups=[];widths=[];mask_ids=[]
    shape=next(iter(masks.values()))['update_mask'].shape[:2] if masks else dic['skeletonize'].shape
    union=np.zeros(shape,np.uint8)
    for gid,ids in enumerate(groups):
        mask=np.zeros(shape,np.uint8)
        for idx in ids:mask=np.maximum(mask,masks[idx]['update_mask'][:,:,0])
        union=np.maximum(union,mask);dist=distance_transform_edt(mask>0)
        ee=[e for e in edges if e['group']==gid];node=defaultdict(list)
        for j,e in enumerate(ee):node[e['start']].append((j,0));node[e['end']].append((j,1))
        used=set()
        starts=[(j,end) for j,e in enumerate(ee) for end,key in enumerate([e['start'],e['end']]) if len(node[key])!=2]
        starts += [(j,0) for j in range(len(ee))]
        for j,end in starts:
            if j in used:continue
            q=[];source_ids=[]
            while j not in used:
                used.add(j);e=ee[j];p=e['q'] if end==0 else e['q'][::-1];q.extend(p.tolist() if not q else p[1:].tolist());source_ids.append(e['mask_id']);key=e['end'] if end==0 else e['start'];choices=[v for v in node[key] if v[0] not in used]
                if len(node[key])!=2 or len(choices)!=1:break
                j,end=choices[0]
            p=np.asarray(q,np.float32)
            if len(p)<2:continue
            xy=np.rint(p).astype(int);xy[:,0]=np.clip(xy[:,0],0,shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,shape[0]-1)
            out.append(p);outgroups.append(gid);widths.append(np.maximum(1,2*dist[xy[:,1],xy[:,0]]));mask_ids.append(source_ids)
    return out,widths,union,dict(native_merged_groups=len(groups),native_edges=len(edges),path_group_ids=outgroups,source_mask_ids=mask_ids,centerline_readout='native_skeleton_edges_with_native_MergeList_and_degree_two_traversal',width_source='distance_to_boundary_of_native_binary_stroke_mask_not_pressure_prediction')

def save_result(folder,paths,widths,ink,meta,geometry,seconds):
    import numpy as np,cv2
    from PIL import Image
    h,w=geometry['original_shape'];sx,sy=geometry['scale_xy'];hh,ww=geometry['resized_shape'];out=[];ow=[]
    for p,width in zip(paths,widths):
        p=np.asarray(p,np.float32)/[sx,sy];width=np.asarray(width)/((sx+sy)*.5)
        keep=(p[:,0]>=0)&(p[:,0]<=w-1)&(p[:,1]>=0)&(p[:,1]<=h-1)
        starts=np.flatnonzero(keep&~np.r_[False,keep[:-1]]);ends=np.flatnonzero(keep&~np.r_[keep[1:],False])+1
        for aa,bb in zip(starts,ends):
            if bb-aa<2:continue
            out.append(p[aa:bb].tolist());ow.append(width[aa:bb].tolist())
    ink=cv2.resize(ink[:hh,:ww],(w,h),interpolation=cv2.INTER_LINEAR)
    Image.fromarray(np.uint8(255-ink)).save(folder/'native_ink.png')
    jsave(folder/'trajectory.json',dict(paths=out,apparent_width_px=ow,coordinates='original_image_pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,robot_connected=False,geometry=geometry,metadata=meta,runtime_seconds=seconds,reference_labels_used=False,padding_crop_policy='drop_outside_contiguous_runs_no_boundary_clamping'))

def deep_paths(folder):
    import numpy as np
    from svgpathtools import svg2paths
    svg,_=svg2paths(str(folder/'output_final.svg'));paths=[]
    for path in svg:
        q=[]
        for seg in path:
            length=seg.length();n=max(2,int(np.ceil(length/2))+1);z=[seg.point(t) for t in np.linspace(0,1,n)];q.extend([[p.real/2,p.imag/2] for p in z][int(bool(q)):])
        if len(q)>1:paths.append(np.asarray(q,np.float32))
    return paths

def worker(method,key,view):
    import torch,numpy as np,cv2,traceback
    from PIL import Image
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(303010);np.random.seed(303010)
    row=next(r for r in all_rows() if r['id']==key);folder=RESULTS/method/view/key;folder.mkdir(parents=True,exist_ok=True)
    blocked={str(Path(row[k]).resolve()).lower() for k in ['reference_file','mask_file'] if k in row}
    def no_reference_access(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            if str(Path(os.fsdecode(args[0])).resolve()).lower() in blocked:raise RuntimeError('Reference access forbidden during inference')
    sys.addaudithook(no_reference_access)
    if (folder/'trajectory.json').exists():print('EXISTS',method,key,view);return
    image,geometry=model_input(row,view);Image.fromarray(image).save(folder/'model_input.png')
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    if method=='linedrawer':
        obj=import_line();loaded=time.perf_counter()
        # Native CNNVE graph/mask pipeline. Save merged masks, not convenience
        # wrapper SVG which currently discards MergeList and variable width.
        obj.process_single_image(str(folder/'model_input.png'),str(folder))
        import gzip,pickle
        with gzip.open(folder/'model_input.graph_merge','rb') as f:dic=pickle.load(f)
        paths,widths,ink,meta=native_line_paths(dic)
        meta.update(model='LineDrawer_CNNVE_public_pretrained',actual_device=str(next(obj.model.parameters()).device),weights_used=['500_rough.pth','500_ref.pth'],official_algorithm_unchanged=True)
        np.savez_compressed(folder/'native_masks.npz',**{'mask_'+str(i):m['update_mask'][:,:,0] for i,m in dic['SplitMaskDic'].items()})
    elif method=='deepsketch':
        obj=import_deep()
        from types import SimpleNamespace
        if 'deepmodels' not in MODEL_CACHE:MODEL_CACHE['deepmodels']=obj.load_model(SimpleNamespace(model_udf=str(ROOT/'weights'/'DeepSketch'/'udf_full.pth'),model_ndc=str(ROOT/'weights'/'DeepSketch'/'ndc_full.pth')),'cuda')
        a,aa,b,bb=MODEL_CACHE['deepmodels'];loaded=time.perf_counter()
        img,img_np,canvas,name=obj.load_img(str(folder/'model_input.png'),'cuda',aa.up_scale,resize=False,ui_mode=False)
        with torch.inference_mode():
            udf,usm,keypts,keyvis=obj.predict_UDF(img,img_np,a,str(folder),'output')
            lx,ly,pt,emap=obj.predict_SVG(udf,b,aa,bb,canvas,str(folder),True,'output',False)
        lines,_,usma,usmu=obj.refine_topology(obj.D(emap),pt,usm,lx,ly,keypts,down_rate=8,downsample=True,full_auto_mode=True)
        obj.lines_to_svg(lines*2,canvas[1]*2,canvas[0]*2,str(folder/'output_refine.svg'))
        obj.simplify_SVG(str(folder/'output_refine.svg'),keypts,bezier=False,rdp_simplify=True,epsilon=.4,skip_len=4)
        paths=deep_paths(folder)
        widths=[np.ones(len(p),np.float32) for p in paths];ink=np.zeros(image.shape,np.uint8)
        for p in paths:cv2.polylines(ink,[np.rint(p).astype(np.int32)],False,255,1,cv2.LINE_AA)
        meta=dict(model='DeepSketch_SIGGRAPH2024_full_pretrained',actual_device=str(next(a.parameters()).device),official_pipeline='UDF_NDC_refine_topology_simplify_SVG_RDP',refine=True,rdp_epsilon=.4,skip_len=4,down_rate=8,actual_variable_width_not_provided=True,width_source='unit_width_display_only_not_estimated_ink_width')
    else:raise ValueError(method)
    torch.cuda.synchronize();ended=time.perf_counter();meta.update(model_load_seconds=loaded-started,inference_and_postprocess_seconds=ended-loaded,peak_gpu_bytes=torch.cuda.max_memory_allocated(),torch_version=torch.__version__,seed=303010)
    save_result(folder,paths,widths,ink,meta,geometry,ended-started)
    print('COMPLETED',method,key,view,'paths',len(paths),'seconds',round(ended-started,2),flush=True)

def locked_worker(method,key,view):
    # Both launchers may prepare jobs concurrently, but the full models must
    # not compete for the laptop's GPU/Windows commit budget.
    import msvcrt
    lock=ROOT/'gpu_inference.lock'
    with lock.open('a+b') as handle:
        handle.seek(0);handle.write(b'1');handle.flush()
        while True:
            try:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:time.sleep(1)
        try:worker(method,key,view)
        finally:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)

def refresh_readout():
    # Re-export already completed native results without neural inference.
    # This fixes only padded-image cropping; it does not alter predictions.
    import types,importlib,numpy as np,cv2,gzip,pickle
    package=types.ModuleType('v30_cnnve');package.__path__=[str(ROOT/'sources'/'SketchKit'/'sketchkit'/'vectorization'/'LineDrawer'/'CNNVE')];sys.modules['v30_cnnve']=package
    importlib.import_module('v30_cnnve.utils.graph_utils')
    for row in read(ROOT/'manifest.json')['records']:
        for method in ['linedrawer','deepsketch']:
            for view in ['raw','normalized']:
                folder=ROOT/'results'/method/view/row['id'];file=folder/'trajectory.json'
                if not file.exists():continue
                doc=read(file)
                if doc.get('padding_crop_policy'):continue
                image,geom=model_input(row,view)
                if method=='linedrawer':
                    with gzip.open(folder/'model_input.graph_merge','rb') as f:dic=pickle.load(f)
                    paths,widths,ink,newmeta=native_line_paths(dic);doc['metadata'].update(newmeta)
                else:
                    paths=deep_paths(folder);widths=[np.ones(len(p)) for p in paths];ink=np.zeros(image.shape,np.uint8)
                    for p in paths:cv2.polylines(ink,[np.rint(p).astype(np.int32)],False,255,1,cv2.LINE_AA)
                shutil.copy2(file,folder/'trajectory_before_padding_crop_fix.json')
                save_result(folder,paths,widths,ink,doc['metadata'],geom,doc['runtime_seconds'])
                print('READOUT_REFRESHED',method,row['id'],view,flush=True)

def batch(method,extra=False):
    import subprocess
    if extra and (ROOT/'protocol_v2_pending').exists():print('LEGACY_EXTRA_SKIPPED_USE_EDGE_PADDING_PROTOCOL',flush=True);return
    records=[]
    chosen=read(ROOT/'extra_probes.json' if extra else ROOT/'manifest.json')['records']
    for row in chosen:
        for view in ['raw','normalized']:
            folder=ROOT/'results'/method/view/row['id'];folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
            if (folder/'trajectory.json').exists():
                records.append(dict(id=row['id'],view=view,status='ok_cached',wall_seconds=0,log=str(folder/'execution.log')))
                continue
            if (folder/'execution.log').exists():shutil.copy2(folder/'execution.log',folder/('execution_before_retry_'+str(time.time_ns())+'.log'))
            with (folder/'execution.log').open('w',encoding='utf-8') as log:
                try:
                    cp=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker',method,row['id'],view],stdout=log,stderr=subprocess.STDOUT,timeout=300)
                    status='ok' if cp.returncode==0 and (folder/'trajectory.json').exists() else 'failed'
                except subprocess.TimeoutExpired:status='timeout_300_seconds'
            records.append(dict(id=row['id'],view=view,status=status,wall_seconds=time.perf_counter()-start,log=str(folder/'execution.log')))
            jsave(ROOT/(method+('_extra' if extra else '')+'_batch.json'),dict(records=records,per_case_timeout_seconds=300,robot_connected=False))
            print('BATCH',method,row['id'],view,status,round(records[-1]['wall_seconds'],1),flush=True)

def baseline():
    import numpy as np,cv2
    from PIL import Image
    # v28 and v29 share exactly the same ink geometry. This is the version
    # in the user's latest comparison, not a claim of deployed v16 promotion.
    source=LAB/'targeted_relations_v29_20261027'/'trajectories'/'v28';rows=[]
    for row in read(ROOT/'manifest.json')['records']:
        folder=RESULTS/'v28'/'raw'/row['id'];folder.mkdir(parents=True,exist_ok=True);file=source/(row['id']+'.json');doc=read(file)
        jsave(folder/'trajectory.json',dict(doc,comparison_role='previous_research_baseline_not_deployed_v16',source=str(file),source_sha256=digest(file),reference_labels_used=False,physical_z=None,ready_for_robot=False,actually_drawn=False,robot_connected=False))
        a=np.zeros(row['shape'],np.uint8)
        for p in doc['paths']:cv2.polylines(a,[np.rint(p).astype(np.int32)],False,255,1,cv2.LINE_AA)
        Image.fromarray(255-a).save(folder/'native_ink.png');rows.append(dict(id=row['id'],sha256=digest(file)))
    jsave(ROOT/'baseline.json',dict(records=rows,v29_same_ink_geometry=True,deployed_v16_unchanged=True))

def line_image(shape,paths,overlay=None,color=False):
    import numpy as np,cv2
    a=np.full((*shape,3),255,np.uint8) if overlay is None else overlay.copy()
    for i,p in enumerate(paths):
        q=np.rint(np.asarray(p)*3).astype(np.int32)
        c=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[i*37%180,210,185]]]),cv2.COLOR_HSV2RGB)[0,0]) if color else ((225,20,35) if overlay is not None else (15,15,15))
        if overlay is not None:q=np.rint(np.asarray(p)).astype(np.int32);cv2.polylines(a,[q],False,c,1,cv2.LINE_AA)
        else:
            if i==0:a=cv2.resize(a,(shape[1]*3,shape[0]*3))
            cv2.polylines(a,[q],False,c,3,cv2.LINE_AA)
    if overlay is None and len(paths):a=cv2.resize(a,(shape[1],shape[0]),interpolation=cv2.INTER_AREA)
    return a

def evaluate():
    import numpy as np,cv2
    from PIL import Image,ImageDraw
    import v26_tone_coverage as metrics
    from scipy.ndimage import distance_transform_edt
    manifest=read(ROOT/'manifest.json');results=[]
    for row in manifest['records']:
        z=np.load(row['mask_file']);refs=read(row['reference_file'])['paths'];rgb=z['rgb'];h,w=rgb.shape[:2];panels=[('原图',rgb),('统一预处理',np.asarray(Image.open(row['normalized']).convert('RGB')))];rec=dict(id=row['id'],group=row['group'],metrics={})
        for method,view in [('v28','raw'),('linedrawer','raw'),('linedrawer','normalized'),('deepsketch','raw'),('deepsketch','normalized')]:
            folder=RESULTS/method/view/row['id'];file=folder/'trajectory.json';name=method+'_'+view
            if not file.exists():rec['metrics'][name]=dict(status='not_completed');continue
            doc=read(file);paths=[np.asarray(p,np.float32) for p in doc['paths']];m=metrics.evaluate_paths(refs,paths,z['valid'],z['bg'],rgb)
            m.update(status='ok',runtime_seconds=doc.get('runtime_seconds'),native_model_variable_width=not doc.get('metadata',{}).get('actual_variable_width_not_provided',False) if method!='v28' else False)
            # The annotations are partial. Never call unmatched model ink false
            # positives or report full-image precision from these references.
            m.update(full_image_precision_unavailable=True,reference_is_partial=True)
            pruned=[p for p in paths if np.linalg.norm(np.diff(p,axis=0),axis=1).sum()>=6]
            pm=metrics.evaluate_paths(refs,pruned,z['valid'],z['bg'],rgb)
            m['shared_min6_readout']={k:pm[k] for k in ['point_coverage_3px','mean_best_single_path_coverage','path_count','mixed_reference_paths_flagged']}
            m['shared_min6_readout']['not_model_selection']=True
            rec['metrics'][name]=m
            a=line_image((h,w),paths);ov=line_image((h,w),paths,rgb)
            Image.fromarray(a).save(folder/'centerlines.png');Image.fromarray(ov).save(folder/'overlay.png')
            panels.append((name+' 线条数 '+str(len(paths)),a))
        board=Image.new('RGB',(1680,600),'#f2efe7');d=ImageDraw.Draw(board);d.text((7,5),row['id']+' 公开预训练模型对照（未微调）；黑线统一笔宽；全部离线',font=FONT,fill='black')
        for i,(title,a) in enumerate(panels):
            col=i%4;rr=i//4;d.text((col*420+6,35+rr*280),title,font=SMALL,fill='black');metrics.v21.paste(board,a,col*420+6,60+rr*280,408,245)
        (ROOT/'comparisons').mkdir(exist_ok=True);board.save(ROOT/'comparisons'/(row['id']+'.jpg'),quality=96);results.append(rec)
    jsave(ROOT/'evaluation.json',dict(records=results,no_threshold_or_model_selection_on_test=True,no_finetuning=True,partial_references_not_full_extraction_accuracy=True,robot_connected=False))
    summary={}
    for name in ['v28_raw','linedrawer_raw','linedrawer_normalized','deepsketch_raw','deepsketch_normalized']:
        rr=[r['metrics'][name] for r in results if r['metrics'].get(name,{}).get('status')=='ok'];summary[name]=dict(completed=len(rr),point_coverage_3px=float(np.mean([r['point_coverage_3px'] for r in rr])) if rr else None,single_path_coverage=float(np.mean([r['mean_best_single_path_coverage'] for r in rr])) if rr else None,paths=float(np.mean([r['path_count'] for r in rr])) if rr else None,mixed_proxy=float(np.mean([r['mixed_reference_paths_flagged'] for r in rr])) if rr else None,seconds=float(np.mean([r['runtime_seconds'] for r in rr if r['runtime_seconds'] is not None])) if rr and any(r['runtime_seconds'] is not None for r in rr) else None)
    for name in summary:
        rr=[r['metrics'][name] for r in results if r['metrics'].get(name,{}).get('status')=='ok']
        summary[name]['shared_min6_readout']={k:float(np.mean([r['shared_min6_readout'][k] for r in rr])) if rr else None for k in ['point_coverage_3px','mean_best_single_path_coverage','path_count','mixed_reference_paths_flagged']}
    jsave(ROOT/'summary.json',summary);print('RESULTS',json.dumps(summary),flush=True)
    selected=['R04','H02','X03','N11'];board=Image.new('RGB',(1600,1200),'#f3f0e8');d=ImageDraw.Draw(board)
    d.text((6,5),'公开预训练方法：原图 / v28旧轨迹 / LineDrawer统一预处理 / DeepSketch统一预处理',font=FONT,fill='black')
    for i,key in enumerate(selected):
        row=next(r for r in manifest['records'] if r['id']==key);a=np.asarray(Image.open(row['raw']).convert('RGB'));items=[('原图',a)]
        for method,view,title in [('v28','raw','v28 旧轨迹'),('linedrawer','normalized','LineDrawer'),('deepsketch','normalized','DeepSketch')]:
            file=RESULTS/method/view/key/'centerlines.png';items.append((title,np.asarray(Image.open(file).convert('RGB')) if file.exists() else np.full_like(a,240)))
        for j,(title,im) in enumerate(items):d.text((j*400+6,35+i*290),key+' '+title,font=SMALL,fill='black');metrics.v21.paste(board,im,j*400+5,65+i*290,390,250)
    board.save(ROOT/'RESULT_COMPARISON.jpg',quality=97)

def persistent(method):
    global RESULTS,PAD_MODE
    import torch,traceback,gc,msvcrt,contextlib
    RESULTS=ROOT/'results_edge_padding';PAD_MODE='edge'
    jsave(ROOT/'padding_protocol_v2.json',dict(reason='constant_white_padding_introduced_artificial_paper_boundary_edges',correction='replicate_edge_pixels_without_new_boundary_contrast',all_12_and_both_views_rerun=True,weights_thresholds_and_preprocessing_unchanged=True,initial_results_preserved=str(ROOT/'results'),final_results=str(RESULTS),maximum_long_edge=512,reference_labels_used=False))
    rows=[]
    with (ROOT/'gpu_inference.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'1');lock.flush()
        while True:
            try:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:time.sleep(1)
        try:
            for row in all_rows():
                for view in ['raw','normalized']:
                    folder=RESULTS/method/view/row['id'];folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
                    try:
                        with (folder/'execution.log').open('a',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):worker(method,row['id'],view)
                        status='ok'
                    except Exception:
                        status='failed';(folder/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');print(traceback.format_exc(),flush=True)
                    rows.append(dict(id=row['id'],view=view,status=status,wall_seconds=time.perf_counter()-started));jsave(ROOT/(method+'_final_batch.json'),dict(records=rows,serialized_gpu=True,model_reused_between_images=True,padding_mode=PAD_MODE))
                    gc.collect();torch.cuda.empty_cache();print('FINAL_BATCH',method,row['id'],view,status,round(rows[-1]['wall_seconds'],1),flush=True)
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='bootstrap':bootstrap()
    elif mode=='assets':assets()
    elif mode=='deps':install_deps()
    elif mode=='prepare':prepare()
    elif mode=='smoke':smoke()
    elif mode=='worker':locked_worker(*sys.argv[2:5])
    elif mode=='worker_final':
        RESULTS=ROOT/'results_edge_padding';PAD_MODE='edge';locked_worker(*sys.argv[2:5])
    elif mode=='batch':batch(sys.argv[2])
    elif mode=='baseline':baseline()
    elif mode=='evaluate':evaluate()
    elif mode=='refresh':refresh_readout()
    elif mode=='extra':extra_probes()
    elif mode=='batch_extra':batch(sys.argv[2],True)
    elif mode=='revise_padding':(ROOT/'protocol_v2_pending').write_text('Use replicated-edge padding for the final complete rerun.',encoding='utf-8')
    elif mode=='persistent':persistent(sys.argv[2])
    elif mode=='evaluate_final':
        RESULTS=ROOT/'results_edge_padding';PAD_MODE='edge';baseline();evaluate()
    elif mode=='baseline_final':
        RESULTS=ROOT/'results_edge_padding';baseline()
    elif mode=='mask_diagnostic':__import__('v30_mask_readout').run()
    elif mode=='report':
        report=__import__('v30_report');report.boards();report.interactive();report.verify_and_report()
    else:raise ValueError(mode)
