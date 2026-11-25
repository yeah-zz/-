# -*- coding: utf-8 -*-
"""Offline new-image comparison: stable baseline + temporal-supervision candidate; no labels."""
from v20_pair_evaluate import *
from auto_extract_v19 import review_paths
import autonomous_review_v19 as reviewer19
from datetime import datetime,timezone,timedelta


def run(input_path,output_path=None,grayscale=False):
    import ink_predict_v16 as stable
    source=Path(input_path);output=Path(output_path) if output_path else BASE/'batch_inference'/datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    if not output.is_absolute() or output.drive.upper() not in ['E:','F:']:raise ValueError('Use an absolute E: or F: output path')
    output.mkdir(parents=True,exist_ok=True);original_paths={};baseline_source=source
    if grayscale:
        files=[source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.webp','.tif']);prepared=output/'preprocessed_inputs';prepared.mkdir(exist_ok=True)
        for i,path in enumerate(files):
            image=Image.open(path).convert('L').convert('RGB');target=prepared/(str(i+1).zfill(3)+'_'+path.stem+'.png');image.save(target);original_paths[str(target)]=str(path)
        baseline_source=prepared
    stable.run(str(baseline_source),str(output/'baseline'));old=read(output/'baseline'/'summary.json');torch.set_num_threads(4);cv2.setNumThreads(2);model=load_role('paired');qcmodel=reviewer19.load_reviewer();policy=read(CURR/'paired'/'calibration.json')['selected'];rows=[]
    for index,row in enumerate(old['records']):
        path=Path(original_paths.get(row['source'],row['source']));original=np.asarray(Image.open(path).convert('RGB'));rgb=np.repeat(cv2.cvtColor(original,cv2.COLOR_RGB2GRAY)[:,:,None],3,2) if grayscale else original;h,w=rgb.shape[:2];baseline=read(Path(row['comparison']).parent/'candidate_trajectory.json')['paths'];route=v16.monochrome_route(rgb);folder=output/(str(index+1).zfill(3)+'_'+path.stem);folder.mkdir(exist_ok=True)
        if route:
            pred=predict_role(model,rgb);paths,sk,info=prior_eval.decode(pred[:5],policy['kind'],policy['high'],policy['low']);Image.fromarray(np.uint8(255*pred[5])).save(folder/'internal_texture_candidate_probability.png');roleimg=role_view(rgb,pred)
        else:paths=[np.asarray(p) for p in baseline];info=dict(color_branch_preserved=True);roleimg=original
        qc=review_paths(rgb,paths,qcmodel);jsave(folder/'automatic_review.json',qc);jsave(folder/'candidate_trajectory.json',dict(paths=[p.round(2).tolist() for p in paths],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,experimental_only=True,official_baseline_not_replaced=True,role_head_did_not_change_paths=True,source_sha256=digest(path)));board=Image.new('RGB',(1800,500),'#f5f3ec');d=ImageDraw.Draw(board);d.text((6,4),'v20 新图离线对照：新模型仅为研究候选；蓝/橙为轮廓/内部纹理提示，不代表具体皴法',font=FONT,fill='black')
        for j,(title,a) in enumerate([('原图',original),('保留的 v16 基线',line_view((h,w),baseline,True)),('配对监督候选轨迹',line_view((h,w),paths,True)),('辅助类别候选（非专家真值）',roleimg)]):d.text((450*j+5,36),title,font=SMALL,fill='black');paste(board,a,450*j+5,65,440,425)
        board.save(folder/'comparison.jpg',quality=95);rows.append(dict(source=str(path),source_sha256=digest(path),comparison=str(folder/'comparison.jpg'),monochrome_route=route,automatic_review=str(folder/'automatic_review.json'),ready_for_robot=False));print('V20_NEW_INPUT',path.name,flush=True)
    jsave(output/'summary.json',dict(records=rows,source_labels_loaded=False,grayscale_candidate_input=grayscale,baseline_and_candidate_use_same_preprocessing=True,official_model_unchanged=True,role_not_named_cunfa=True,robot_connected=False));print('V20_BATCH_OUTPUT',str(output),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output');p.add_argument('--grayscale',action='store_true');a=p.parse_args();run(a.input,a.output,a.grayscale)

