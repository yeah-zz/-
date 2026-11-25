from structure_train_v20 import *
from evaluate_clean_strokes_v12 import path_metrics,line_view
from ink_eval_v16 import normal_diagnostics,old_model,old_infer,load_links,frozen_paths,connect
from protected_refine_v16 import center_ribbons,recover_faint
from scipy.spatial import cKDTree


def metrics5(reference,paths,valid):
    allp=np.concatenate([dense(p) for p in paths]) if paths else np.zeros((0,2));tree=cKDTree(allp) if len(allp) else None;best=[];hits=[]
    for ref in reference:
        q=dense(ref);xy=np.rint(q).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<valid.shape[1])&(xy[:,1]>=0)&(xy[:,1]<valid.shape[0]);q=q[inside];xy=xy[inside];q=q[valid[xy[:,1],xy[:,0]]>0]
        if not len(q):continue
        dist=tree.query(q)[0] if tree else np.full(len(q),1e5);hits.extend((dist<=5).tolist());best.append(max([float((cKDTree(dense(p)).query(q)[0]<=5).mean()) for p in paths] or [0.]))
    return dict(point_coverage_5px=float(np.mean(hits)) if hits else 0.,mean_best_single_path_coverage_5px=float(np.mean(best)) if best else 0.)


def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);manifest=read(ROOT/'external_manifest.json');folder=ROOT/'external_results';folder.mkdir(exist_ok=True);signature=digest(ROOT/'external_manifest.json')
    if (folder/'frozen_manifest_sha.json').exists():
        if read(folder/'frozen_manifest_sha.json')['sha256']!=signature:raise RuntimeError('External references changed after scoring')
    else:jsave(folder/'frozen_manifest_sha.json',dict(sha256=signature,external_images_not_used_for_parameter_selection=True))
    parent=old_model().cuda();parent.load_state_dict(torch.load(PARENT,map_location='cuda',weights_only=False)['model']);parent.eval();link,lt=load_links(4);special=v16.RibbonNet().cuda();special.load_state_dict(torch.load(legacy.V16/'stage_03'/'final.pt',map_location='cuda',weights_only=False)['model']);special.eval();settings16=read(legacy.V16/'protected_refinement'/'calibration.json')['selected'];models={};policies={}
    for name,path,policy in [('v17',OLD/'aligned'/'final.pt',OLD/'aligned'/'calibration.json'),('v19',V19/'reviewed_v2'/'final.pt',V19/'reviewed_v2'/'calibration.json'),('v20_control',ROOT/'control'/'final.pt',ROOT/'control'/'calibration.json'),('v20_structured',ROOT/'structured'/'final.pt',ROOT/'structured'/'calibration.json')]:
        m=v16.RibbonNet().cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);models[name]=m.eval();policies[name]=read(policy)['selected']
    jsave(folder/'policies_frozen_from_old_calibration.json',policies);rows=[];figures=[]
    for r in manifest['records']:
        source=np.asarray(Image.open(r['image']).convert('RGB'));rgb=np.repeat(cv2.cvtColor(source,cv2.COLOR_RGB2GRAY)[:,:,None],3,axis=2) if r['input_mode']=='grayscale' else source;h,w=rgb.shape[:2];route=v16.monochrome_route(rgb);out=folder/r['id'];out.mkdir(exist_ok=True);p=old_infer(parent,rgb);base=frozen_paths(p);base,_,_=connect(base,p[0],rgb,link,lt);base=[q for q in base if np.linalg.norm(np.diff(q,axis=0),axis=1).sum()>=12]
        if route:
            pp=v16.predict(special,rgb);base,info=center_ribbons(base,rgb,pp,settings16['strength']);base,weak=recover_faint(base,rgb,p,pp,parent,link,lt)
        predpaths={'v16_baseline':base};row=dict(id=r['id'],split=r['split'],source_group=r['publication_group'],work_group=r['work_group'],input_mode=r['input_mode'],monochrome_route=route,reference_is_assistant_partial=True,complete_artwork_accuracy_unavailable=True,models={})
        for name,m in models.items():
            if route:
                pred=v16.predict(m,rgb);s=policies[name];paths,sk,info=prior_eval.decode(pred,s['kind'],s['high'],s['low'])
                if name=='v20_structured':Image.fromarray(np.uint8(255*(1-pred[0]))).save(out/'structured_center_probability.png');Image.fromarray(np.uint8(255*(1-pred[1]))).save(out/'structured_ink_probability.png')
            else:paths=[np.asarray(q) for q in base]
            predpaths[name]=paths
        for name,paths in predpaths.items():
            jsave(out/(name+'_trajectory.json'),dict(paths=[np.asarray(q).round(2).tolist() for q in paths],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,source_sha256=r['image_sha256'],reference_at_inference=False))
        if 'labels' in r:
            doc=read(r['labels']);ref=[s['points'] for s in doc['strokes']];_,_,mask,_=fields(doc)
            for name,paths in predpaths.items():
                sk=np.zeros((h,w),np.uint8)
                for q in paths:cv2.polylines(sk,[np.rint(q).astype(np.int32)],False,1,1)
                row['models'][name]=dict(**path_metrics(ref,paths,mask[3]),**metrics5(ref,paths,mask[3]),background_line_density=float(sk[mask[1]>0].mean()) if mask[1].any() else None,normal_diagnostics=normal_diagnostics(ref,sk.astype(bool),rgb,valid=mask[3]))
        jsave(out/'metrics.json',row);rows.append(row);page=Image.new('RGB',(1800,450),'#f5f3ec');d=ImageDraw.Draw(page)
        for j,(title,a) in enumerate([('原图',source),('v16 保留基线',line_view((h,w),predpaths['v16_baseline'],True)),('v17 对齐监督',line_view((h,w),predpaths['v17'],True)),('v20 连续/分离约束',line_view((h,w),predpaths['v20_structured'],True))]):d.text((450*j+6,5),r['id']+' '+title,font=SMALL,fill='black');paste(page,a,450*j+6,38,438,403)
        page.save(out/'comparison.jpg',quality=95);figures.append((r['id'],page));print('V20_EXTERNAL',r['id'],json.dumps({k:round(v['mean_best_single_path_coverage'],3) for k,v in row['models'].items()}),flush=True)
    aggregate={}
    for name in ['v16_baseline',*models]:
        test=[r['models'][name] for r in rows if r['models']];aggregate[name]={key:float(np.mean([r[key] for r in test])) for key in ['point_coverage_3px','mean_best_single_path_coverage','single_path_90pct_fraction','mean_reference_fragments','mean_best_single_path_coverage_5px','background_line_density','mixed_reference_paths_flagged']}
    jsave(folder/'summary.json',dict(records=rows,aggregate=aggregate,test_source_images_never_trained_this_round=True,reference_labels_partial_assistant_not_expert=True,heldout_is_by_source_not_fully_independent_artwork_guarantee=True))
    for name,ids in [('EXTERNAL_COMPARISON',['X01','X02','X03','X04']),('CHALLENGE_COMPARISON',['Q01','Q02','Q03','Q04','Q05','Q06','Q07'])]:
        board=Image.new('RGB',(1800,450*len(ids)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,4),'v20 新来源对照；颜色表示轨迹，复杂整图没有完整参考线',font=FONT,fill='black')
        for i,k in enumerate(ids):board.paste(dict(figures)[k],(0,35+450*i))
        board.save(ROOT/(name+'.jpg'),quality=94)
    print('V20_EXTERNAL_AGGREGATE',json.dumps(aggregate),flush=True)
if __name__=='__main__':main()
