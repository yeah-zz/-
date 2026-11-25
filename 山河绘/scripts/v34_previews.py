# -*- coding: utf-8 -*-
from v34_next_experiment import *

def dhash(rgb):
    a=np.asarray(Image.fromarray(rgb).convert('L').resize((9,8)),np.int16);return (a[:,1:]>a[:,:-1]).reshape(-1)

def source_preview():
    import v28_stroke_relations as old
    from v32_stroke_model import load as load_curve
    from v32_guard import verified_paths
    folder=ROOT34/'new_sources';sources_doc=read(folder/'sources.json');boxes={'MET_CP29':(.16,.21,.61,.72),'UMMA_ROCK':(.07,.20,.49,.93),'CMA_WANG_GAI':(.70,.72,.96,.96)};out=ROOT34/'new_source_runs';out.mkdir(exist_ok=True)
    if not (out/'protocol.json').exists():jsave(out/'protocol.json',dict(crops_selected_before_viewing_predictions=boxes,input_long_edge=512,convert_to_grayscale=True,models_frozen=True,pipeline='frozen_v28_machine_centerlines_then_frozen_v33_filter_and_endpoint_control',new_source_qualitative_only=True,no_expert_reference_or_accuracy_claim=True,not_used_for_training_or_tuning=True,source_family_overlap_not_ruled_out=True))
    training=read(ROOT/'protocol.json')['records'];old_images=[]
    for r in training:old_images.append((r['id'],dhash(np.load(r['data'])['rgb'])))
    m=old.exp.load_model('base');cfg=read(CALIBRATION)['selected']['config'];records=[];images=[]
    for src in sources_doc['records']:
        if src['status']!='downloaded':continue
        key=src['id'];im=Image.open(src['file']).convert('RGB');w,h=im.size;b=boxes[key];im=im.crop(tuple(round(v*s) for v,s in zip(b,[w,h,w,h])));im.thumbnail((512,512));color=np.asarray(im);rgb=np.asarray(im.convert('L').convert('RGB'));dest=out/key;dest.mkdir(exist_ok=True);Image.fromarray(color).save(dest/'color_input.png');Image.fromarray(rgb).save(dest/'gray_input.png')
        nearest=min([(int((dhash(color)!=v).sum()),sid) for sid,v in old_images]);pred=old.exp.prev.prior.predict(m,rgb)[:5];g=old.add_visual(old.build_graph(pred,rgb),m);choice=read(old.ROOT/'calibration.json')['selected'];pr=old.probabilities(g,old.models('visual'));accepted,_=old.select(g,pr,choice['threshold'],choice['margin']);base=old.assemble(g,accepted);base=[p.tolist() for p in base];outrows=[dict(points=np.asarray(p,np.float32),ids=[i],core=True) for i,p in enumerate(base)];joined,audit=merge_endpoints(Evidence(rgb),outrows,cfg);paths=[r['points'].tolist() for r in joined];jsave(dest/'trajectory.json',dict(paths=paths,source_page=src['page'],crop_fraction=b,grayscale_preprocessing=True,reference_labels_used=False,physical_z=None,ready_for_robot=False,actually_drawn=False,accuracy_not_measured=True,audit=audit));line=line_image(rgb,paths);ov=line_image(rgb,paths,rgb);Image.fromarray(line).save(dest/'centerlines.png');Image.fromarray(ov).save(dest/'overlay.png');records.append(dict(id=key,page=src['page'],source_sha256=digest(src['file']),input_sha256=digest(dest/'gray_input.png'),paths=len(paths),closest_training_dhash_distance=nearest[0],closest_training_id=nearest[1],dhash_is_not_proof_of_independent_artwork=True,expert_gold=False));images.append((key,color,line,ov));del pred,g;torch.cuda.empty_cache();print('V34_NEW_SOURCE_PREDICT',key,len(paths),flush=True)
    del m;torch.cuda.empty_cache();jsave(out/'manifest.json',dict(records=records,qualitative_only=True,no_expert_accuracy=True))
    board=Image.new('RGB',(1500,len(images)*340+40),'#f3f0e8');draw=ImageDraw.Draw(board);draw.text((6,5),'v34 新馆藏来源外观检查：原图局部 / 冻结模型的灰度输入线稿 / 原图灰度叠加（无参考准确率）',font=FONT,fill='black')
    for i,(key,color,line,ov) in enumerate(images):
        for j,a in enumerate([color,line,ov]):draw.text((j*500+6,37+i*340),key,font=SMALL,fill='black');paste(board,a,j*500+5,62+i*340,490,300)
    board.save(ROOT34/'NEW_SOURCE_COMPARISON.jpg',quality=97)

def sweep_fixed(shape,p,radius,alpha):
    out=np.zeros(shape,np.float32)
    for i in range(len(p)-1):
        a,b=p[i:i+2];pad=max(radius[i:i+2])+3;lo=np.maximum(0,np.floor(np.minimum(a,b)-pad)).astype(int);hi=np.minimum(np.array(shape[::-1]),np.ceil(np.maximum(a,b)+pad+1)).astype(int)
        if np.any(hi<=lo):continue
        yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];q=np.stack([xx,yy],-1);d=b-a;t=np.clip(((q-a)*d).sum(-1)/max(1e-5,d@d),0,1);dist=np.linalg.norm(q-a-t[...,None]*d,axis=-1);rr=radius[i]*(1-t)+radius[i+1]*t;aa=alpha[i]*(1-t)+alpha[i+1]*t;ink=aa/(1+np.exp(np.clip((dist-rr)/.45,-30,30)));out[lo[1]:hi[1],lo[0]:hi[0]]=np.maximum(out[lo[1]:hi[1],lo[0]:hi[0]],ink)
    return np.uint8(np.clip(255*(1-out),0,255))

def brush_preview():
    from v31_profiles import load as load_band,features as band_features
    from v31_render import profiles
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');model=load_band();cfg=read(CALIBRATION)['selected']['config'];mf={r['id']:r for r in read(V31/'manifest.json')['development']};records=[];views=[]
    for key in ['R04','H01']:
        r=mf[key];rgb=np.asarray(Image.open(r['raw']).convert('RGB'));paths=read(RESULTS/cfg['name']/'pooled'/key/'trajectory.json')['paths'];e=Evidence(rgb);options=[(len(arc(p)),i) for i,p in enumerate(paths) if len(p)>3 and len(arc(p))>35 and e.support(p).mean()>=.65]
        if not options:raise RuntimeError('No image-supported machine curve for width preview')
        _,idx=max(options);p=arc(paths[idx],.75);u=tangent(p);n=np.c_[-u[:,1],u[:,0]];prof=profiles(observed(rgb),p,n)
        with torch.no_grad():param=model(band_features(prof)).cpu().numpy()
        radius=np.clip(gaussian_filter1d(param[:,1],2),.45,12);tone=np.clip(gaussian_filter1d(param[:,2],2),.005,.995);render=sweep_fixed(rgb.shape[:2],p,radius,tone);line=line_image(rgb,[p.tolist()],rgb);dest=ROOT34/'brush_parameter_preview'/key;dest.mkdir(parents=True,exist_ok=True);Image.fromarray(render).save(dest/'geometric_width_tone.png');Image.fromarray(line).save(dest/'fixed_centerline_overlay.png');jsave(dest/'image_parameters.json',dict(source_path_id=idx,paths=[p.tolist()],apparent_width_px=(2*radius).tolist(),apparent_ink_alpha=tone.tolist(),center_geometry_optimized=False,center_offset_head_ignored=True,pressure=None,ink_concentration=None,wetness=None,physical_z=None,ready_for_robot=False,actually_drawn=False,parameters_are_not_robot_controls=True,expert_brush_parameters=False))
        fig,axes=plt.subplots(1,2,figsize=(11,2.3),dpi=140);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];axes[0].plot(s,2*radius);axes[0].set_title('图像域笔宽 / px',fontproperties=font);axes[1].plot(s,tone,color='#444');axes[1].set_title('表观墨迹灰度参数 / 0–1',fontproperties=font);axes[1].set_ylim(0,1)
        for ax in axes:ax.set_xlabel('沿线距离 / px',fontproperties=font);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(dest/'parameter_curves.png');plt.close(fig);records.append(dict(id=key,path_id=idx,width_range_px=[float((2*radius).min()),float((2*radius).max())],alpha_range=[float(tone.min()),float(tone.max())],fixed_machine_geometry=True,not_physical_brush_simulation=True));views.append((key,rgb,line,np.repeat(render[:,:,None],3,2),np.asarray(Image.open(dest/'parameter_curves.png').convert('RGB'))));print('V34_BRUSH_PREVIEW',key,records[-1],flush=True)
    jsave(ROOT34/'brush_parameter_preview'/'manifest.json',dict(records=records,profile_network=str(LAB/'inverse_brush_curves_v31_20261027'/'profile_model.pt'),new_training_steps=0,reference_labels_used=False,actual_device='cuda:0',physical_calibration_done=False))
    board=Image.new('RGB',(1500,len(views)*600+40),'#f3f0e8');draw=ImageDraw.Draw(board);draw.text((6,5),'v34 固定机器轨迹的笔宽/表观灰度参数预览：非实机、非物理毛笔仿真，不是完整画作',font=FONT,fill='black')
    for i,(key,rgb,line,render,plot) in enumerate(views):
        for j,(title,a) in enumerate([('原图',rgb),('选定机器中心线',line),('图像参数几何回放',render)]):draw.text((j*500+5,35+i*600),key+' '+title,font=SMALL,fill='black');paste(board,a,j*500+5,60+i*600,490,310)
        paste(board,plot,10,380+i*600,1480,210)
    board.save(ROOT34/'BRUSH_PARAMETER_PREVIEW.jpg',quality=97)

if __name__=='__main__':
    import sys
    {'source_preview':source_preview,'brush_preview':brush_preview}[sys.argv[1]]()
