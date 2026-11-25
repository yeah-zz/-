from v20_external_evaluate import *
from skimage.morphology import skeletonize
from rock_pairs_v15 import connect as learned_connect


def decode_trial(pred,parent_pred,rgb,link,threshold,cfg):
    if cfg['mode']=='center':paths,sk,info=prior_eval.decode(pred,'center',.65,.3)
    else:
        candidate=pred[1]>cfg['low']
        if cfg['mode']=='hybrid':candidate|=(parent_pred[0]>.45)&(pred[1]>.15)
        n,labels,stats,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8);seed=(pred[0]>.6)|((parent_pred[0]>.7)&(pred[1]>.3));good=np.zeros(n,bool);good[np.unique(labels[seed])]=True;good[stats[:,cv2.CC_STAT_AREA]<12]=False;good[0]=False;binary=good[labels]
        # Fill tiny enclosed white holes, never close a white channel between neighboring strokes.
        n,cc,ss,_=cv2.connectedComponentsWithStats((~binary).astype(np.uint8),8);fill=np.zeros(n,bool)
        for k in range(1,n):
            x,y,w,h,area=ss[k]
            if area<=12 and x>0 and y>0 and x+w<binary.shape[1] and y+h<binary.shape[0]:fill[k]=True
        binary|=fill[cc];fake=pred.copy();fake[1]=binary.astype(np.float32)*.99;fake[0]=np.maximum(pred[0],parent_pred[0]);paths,sk,info=prior_eval.decode(fake,'ribbon',.8,.5);info.update(small_enclosed_holes_filled=int(fill.sum()),hybrid_parent_support=cfg['mode']=='hybrid')
    if cfg['join']:
        paths,events,uncertain=learned_connect(paths,np.maximum(pred[0],pred[1]*.8),rgb,link,threshold,max_gap=8);info.update(learned_endpoint_joins=events,competing_join_candidates=uncertain)
    else:info.update(learned_endpoint_joins=[])
    return paths,info


def raster(paths,shape):
    sk=np.zeros(shape,np.uint8)
    for p in paths:cv2.polylines(sk,[np.rint(p).astype(np.int32)],False,1,1)
    return sk


def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);folder=ROOT/'decoder_trial';folder.mkdir(exist_ok=True)
    # Choose only with old calibration scores, even though new-source development outputs now exist.
    scores={k:read(ROOT/k/'calibration.json')['selected']['score'] for k in ['control','structured']};chosen=max(scores,key=scores.get);model=load_model(chosen);parent=old_model().cuda();parent.load_state_dict(torch.load(PARENT,map_location='cuda',weights_only=False)['model']);parent.eval();link,lt=load_links(4);configs=[dict(mode='center',join=False,low=.3),dict(mode='center',join=True,low=.3),dict(mode='ribbon',join=False,low=.35),dict(mode='hybrid',join=True,low=.30),dict(mode='hybrid',join=False,low=.45)];cases=[]
    for r in legacy.records():
        if r['id'] not in ['N07','N08']:continue
        rgb,doc,bg,valid=legacy.source(r);cases.append((r,rgb,doc,bg,valid,v16.predict(model,rgb),old_infer(parent,rgb)))
    choices=[]
    for cfg in configs:
        rr=[]
        for r,rgb,doc,bg,valid,p,pp in cases:
            paths,info=decode_trial(p,pp,rgb,link,lt,cfg);met=path_metrics([s['points'] for s in doc['strokes']],paths,valid);sk=raster(paths,rgb.shape[:2]);density=float(sk[bg>0].mean()) if bg.any() else 0.;rr.append(dict(id=r['id'],metric=met,background_density=density,joins=len(info['learned_endpoint_joins'])))
        score=float(np.mean([.4*r['metric']['point_coverage_3px']+.6*r['metric']['mean_best_single_path_coverage']-min(.3,r['background_density']*100)-.03*r['metric']['mixed_reference_paths_flagged'] for r in rr]));choices.append(dict(config=cfg,score=score,records=rr))
    selected=max(choices,key=lambda r:r['score']);jsave(folder/'calibration.json',dict(line_model=chosen,model_selection_old_calibration_scores=scores,selected=selected,candidates=choices,selection_ids=['N07','N08'],external_results_already_seen_during_development=True,not_new_blind_test=True));print('V20_DECODER_SELECTED',chosen,json.dumps(selected),flush=True)
    rows=[];pictures=[];manifest=read(ROOT/'external_manifest.json');sourcecases=manifest['records'][:];old_lookup={r['id']:r for r in prior_eval.evaluation_records()}
    for k in ['R03','R04']:
        r=old_lookup[k];sourcecases.append(dict(r,input_mode='grayscale',split='old_development',image_sha256=digest(r['image'])))
    for r in sourcecases:
        full=np.asarray(Image.open(r['image']).convert('RGB'));h,w=full.shape[:2];rgb=np.repeat(cv2.cvtColor(full,cv2.COLOR_RGB2GRAY)[:,:,None],3,2) if r['input_mode']=='grayscale' else full;left,top,cw,ch=r.get('native',[0,0,w,h]);crop=rgb[top:top+ch,left:left+cw];route=v16.monochrome_route(crop);out=folder/r['id'];out.mkdir(exist_ok=True)
        if route:
            pred=v16.predict(model,crop);pp=old_infer(parent,crop);paths,info=decode_trial(pred,pp,crop,link,lt,selected['config']);paths=[p+np.array([left,top]) for p in paths]
        else:
            paths=[np.asarray(p) for p in read(ROOT/'external_results'/r['id']/'v16_baseline_trajectory.json')['paths']];info=dict(color_branch_preserved=True);pred=None
        jsave(out/'trajectory.json',dict(paths=[p.round(2).tolist() for p in paths],ready_for_robot=False,actually_drawn=False,diagnostics=info));row=dict(id=r['id'],monochrome_route=route,split=r['split'])
        if 'labels' in r:
            doc=read(r['labels']);_,_,mask,_=fields(doc);native=np.zeros((h,w),np.float32);native[top:top+ch,left:left+cw]=1;mask*=native[None];refs=[s['points'] for s in doc['strokes']];sk=raster(paths,(h,w));row.update(**path_metrics(refs,paths,mask[3]),**metrics5(refs,paths,mask[3]),background_line_density=float(sk[mask[1]>0].mean()) if mask[1].any() else None)
        rows.append(row);page=Image.new('RGB',(1350,430),'#f5f3ec');d=ImageDraw.Draw(page);oldpath=(ROOT/'external_results'/r['id']/'v16_baseline_trajectory.json') if r['id'].startswith(('X','Q')) else (legacy.V16/'protected_refinement'/r['id']/'trajectory.json');baseline=read(oldpath)['paths'];raw=read(ROOT/chosen/'evaluation'/r['id']/'trajectory.json')['paths'] if r['id'].startswith('R') else read(ROOT/'external_results'/r['id']/('v20_'+chosen+'_trajectory.json'))['paths']
        for j,(title,a) in enumerate([('原图',full),('v20 原轨迹',line_view((h,w),raw,True)),('v20 路径解码试验',line_view((h,w),paths,True))]):d.text((450*j+5,5),r['id']+' '+title,font=SMALL,fill='black');paste(page,a,450*j+5,38,440,385)
        page.save(out/'comparison.jpg',quality=94)
        if r['id'] in ['X01','X04','Q06','R04'] and pred is not None:
            diag=Image.new('RGB',(2000,420),'#f5f3ec');dd=ImageDraw.Draw(diag)
            for j,(title,a) in enumerate([('输入图像',crop),('墨带概率',np.uint8(255*(1-pred[1]))),('中心概率',np.uint8(255*(1-pred[0]))),('原轨迹',line_view((h,w),raw,True)),('解码试验',line_view((h,w),paths,True))]):dd.text((400*j+5,5),r['id']+' '+title,font=SMALL,fill='black');paste(diag,a,400*j+5,35,390,375)
            pictures.append((r['id'],diag))
        print('V20_DECODER',r['id'],round(row.get('mean_best_single_path_coverage',0),3),flush=True)
    rr=[r for r in rows if r['split']=='external_test'];aggregate={k:float(np.mean([r[k] for r in rr])) for k in ['point_coverage_3px','mean_best_single_path_coverage','single_path_90pct_fraction','mean_reference_fragments','mean_best_single_path_coverage_5px','mixed_reference_paths_flagged','background_line_density']};jsave(folder/'summary.json',dict(records=rows,aggregate=aggregate,old_development=[r for r in rows if r['split']=='old_development'],weights_unchanged=True,not_blind_secondary_development_trial=True));board=Image.new('RGB',(2000,420*len(pictures)+35),'#f5f3ec');dd=ImageDraw.Draw(board);dd.text((5,4),'v20 概率与轨迹诊断（解码试验，不是实画）',font=FONT,fill='black')
    for i,(k,im) in enumerate(pictures):board.paste(im,(0,35+420*i))
    board.save(ROOT/'PIPELINE_DIAGNOSTIC.jpg',quality=94);print('V20_DECODER_AGGREGATE',json.dumps(aggregate),flush=True)
if __name__=='__main__':main()
