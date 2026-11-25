from v20_pair_train import *
from v20_external_evaluate import metrics5
from evaluate_clean_strokes_v12 import path_metrics,line_view
from ink_eval_v16 import normal_diagnostics


def role_view(rgb,p):
    out=np.asarray(rgb).copy();ink=p[1]>.5;positive=ink&(p[5]>.65);negative=ink&(p[5]<.35);uncertain=ink&~(positive|negative);out[negative]=(out[negative]*.45+np.array([25,130,225])*.55).astype(np.uint8);out[positive]=(out[positive]*.45+np.array([225,125,30])*.55).astype(np.uint8);out[uncertain]=(out[uncertain]*.6+np.array([130,130,130])*.4).astype(np.uint8);return out


def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);models={k:load_role(k) for k in ['replay','paired']};policies={k:read(CURR/k/'calibration.json')['selected'] for k in models};external=read(BASE/'external_manifest.json');folder=CURR/'external_results';folder.mkdir(exist_ok=True);rows=[];fig=[]
    for r in external['records']:
        original=np.asarray(Image.open(r['image']).convert('RGB'));rgb=np.repeat(cv2.cvtColor(original,cv2.COLOR_RGB2GRAY)[:,:,None],3,2) if r['input_mode']=='grayscale' else original;h,w=rgb.shape[:2];route=v16.monochrome_route(rgb);out=folder/r['id'];out.mkdir(exist_ok=True);row=dict(id=r['id'],split=r['split'],models={},work_group=r['work_group'],publication_group=r['publication_group'],input_mode=r['input_mode']);paths={};predictions={}
        for name,model in models.items():
            if route:
                p=predict_role(model,rgb);settings=policies[name];q,sk,info=prior_eval.decode(p[:5],settings['kind'],settings['high'],settings['low']);predictions[name]=p
            else:q=[np.asarray(p) for p in read(BASE/'external_results'/r['id']/'v16_baseline_trajectory.json')['paths']]
            paths[name]=q;jsave(out/(name+'_trajectory.json'),dict(paths=[p.round(2).tolist() for p in q],ready_for_robot=False,actually_drawn=False,role_predictions_not_used_for_path_decoding=True,source_sha256=r['image_sha256']))
            if 'labels' in r:
                doc=read(r['labels']);ref=[s['points'] for s in doc['strokes']];_,_,mask,_=fields(doc);sk=np.zeros((h,w),np.uint8)
                for p in q:cv2.polylines(sk,[np.rint(p).astype(np.int32)],False,1,1)
                row['models'][name]=dict(**path_metrics(ref,q,mask[3]),**metrics5(ref,q,mask[3]),background_line_density=float(sk[mask[1]>0].mean()) if mask[1].any() else None,normal_diagnostics=normal_diagnostics(ref,sk.astype(bool),rgb,valid=mask[3]))
        rows.append(row);jsave(out/'metrics.json',row);old=read(BASE/'external_results'/r['id']/'v20_control_trajectory.json')['paths'];page=Image.new('RGB',(1800,450),'#f5f3ec');d=ImageDraw.Draw(page)
        for j,(title,a) in enumerate([('原图',original),('v20 初始对照轨迹',line_view((h,w),old,True)),('配对训练候选轨迹',line_view((h,w),paths['paired'],True)),('内部纹理/轮廓候选（非真值）',role_view(rgb,predictions['paired']) if route else original)]):d.text((450*j+5,5),r['id']+' '+title,font=SMALL,fill='black');paste(page,a,450*j+5,38,440,405)
        page.save(out/'comparison.jpg',quality=94);fig.append((r['id'],page));print('PAIR_EXTERNAL',r['id'],{k:round(v['mean_best_single_path_coverage'],3) for k,v in row['models'].items()},flush=True)
    aggregate={}
    for name in models:
        test=[r['models'][name] for r in rows if r['models']];aggregate[name]={k:float(np.mean([r[k] for r in test])) for k in ['point_coverage_3px','mean_best_single_path_coverage','mean_best_single_path_coverage_5px','mean_reference_fragments','single_path_90pct_fraction','background_line_density','mixed_reference_paths_flagged']}
    jsave(folder/'summary.json',dict(records=rows,aggregate=aggregate,reused_external_development_set_not_new_blind_test=True,external_images_not_trained_by_line_model=True,partial_references_not_expert_complete_truth=True))
    for title,ids in [('PAIR_EXTERNAL_COMPARISON',['X01','X02','X03','X04']),('PAIR_CHALLENGE_COMPARISON',['Q01','Q02','Q03','Q04','Q05','Q06','Q07'])]:
        board=Image.new('RGB',(1800,450*len(ids)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,4),'配对训练开发对照：蓝＝轮廓候选，橙＝内部纹理候选，灰＝待定；不代表具体皴法或专家真值',font=FONT,fill='black')
        for i,k in enumerate(ids):board.paste(dict(fig)[k],(0,35+450*i))
        board.save(BASE/(title+'.jpg'),quality=94)
    validation=[];panels=[]
    for r in read(CURR/'paired_targets_manifest.json')['records']:
        if r['split']=='train':continue
        z=np.load(r['data']);rgb=z['rgb'];t=z['target'];known=t[12]>.01;expected=t[11]>.5;values={};preds={}
        for name,model in models.items():
            p=predict_role(model,rgb);preds[name]=p;found=p[5]>.5;tp=int((found&expected&known).sum());fp=int((found&~expected&known).sum());fn=int((~found&expected&known).sum());tn=int((~found&~expected&known).sum());values[name]=dict(tp=tp,fp=fp,fn=fn,tn=tn,known_pixels=int(known.sum()),uncertainty_fraction=float((((p[5]>=.35)&(p[5]<=.65))&known).sum()/max(1,known.sum())))
        validation.append(dict(pair=r['id'],group=r['group'],added_role=r['added_role'],models=values,derived_temporal_targets_not_expert=True));page=Image.new('RGB',(1600,380),'#f5f3ec');d=ImageDraw.Draw(page);target=rgb.copy();target[expected&known]=[220,125,30];target[~expected&known]=[30,135,220]
        for j,(title,a) in enumerate([('未训练示范输入',rgb),('差分派生局部提示（非真值）',target),('模型纹理/轮廓候选',role_view(rgb,preds['paired'])),('中心线概率',np.uint8(255*(1-preds['paired'][0])))]):d.text((400*j+5,5),r['id']+' '+title,font=SMALL,fill='black');paste(page,a,400*j+5,35,390,335)
        panels.append(page)
    role_aggregate={}
    for name in models:
        s={k:sum(r['models'][name][k] for r in validation) for k in ['tp','fp','fn','tn']};precision=s['tp']/max(1,s['tp']+s['fp']);recall=s['tp']/max(1,s['tp']+s['fn']);role_aggregate[name]=dict(**s,precision_against_derived_targets=precision,recall_against_derived_targets=recall,f1_against_derived_targets=2*precision*recall/max(1e-9,precision+recall))
    jsave(CURR/'role_validation.json',dict(records=validation,aggregate=role_aggregate,heldout_work_groups=sorted({r['group'] for r in validation}),not_art_expert_accuracy=True,same_tutorial_source_as_training=True,role_head_not_named_cunfa_classifier=True));board=Image.new('RGB',(1600,380*len(panels)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((5,4),'内部纹理候选的整组留出示范检查：只与差分派生目标比较，不是专家准确率',font=FONT,fill='black')
    for i,page in enumerate(panels):board.paste(page,(0,35+380*i))
    board.save(BASE/'ROLE_WORK_HOLDOUT.jpg',quality=94);print('PAIR_EXTERNAL_AGG',json.dumps(aggregate),flush=True);print('ROLE_DERIVED_AGREEMENT',json.dumps(role_aggregate),flush=True)
if __name__=='__main__':main()
