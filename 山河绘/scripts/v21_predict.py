# -*- coding: utf-8 -*-
"""Reference-free offline single image prediction for the experimental V21 model."""
from v21_stroke_learning import *

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--grayscale',action='store_true');a=p.parse_args();source=Path(a.input);out=Path(a.output)
    if not out.is_absolute() or out.drive.upper() not in ['E:','F:']:raise ValueError('Use absolute E:/F: output path')
    out.mkdir(parents=True,exist_ok=True);rgb=np.asarray(Image.open(source).convert('RGB'))
    if a.grayscale:rgb=np.repeat(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)[:,:,None],3,2)
    if not v16.monochrome_route(rgb):raise ValueError('V21 is a monochrome research model; colored landscape extraction has not been validated')
    model=load_model('ribbon_improved');cfg=read(ROOT/'ribbon_improved'/'calibration.json')['selected'];pred=v16.predict(model,rgb);paths,_,info=prior_eval.decode(pred,cfg['kind'],cfg['high'],cfg['low']);jsave(out/'trajectory.json',dict(paths=[q.round(3).tolist() for q in paths],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,experimental_model='v21_ribbon_improved',official_model_replaced=False,source_sha256=digest(source)));Image.fromarray(np.uint8(255*(1-pred[1]))).save(out/'ink_probability.png');Image.fromarray(np.uint8(255*(1-pred[0]))).save(out/'center_probability.png');board=Image.new('RGB',(1600,450),'#f5f3ec');d=ImageDraw.Draw(board)
    for j,(title,im) in enumerate([('原图',rgb),('预测墨迹',np.uint8(255*(1-pred[1]))),('预测中心',np.uint8(255*(1-pred[0]))),('实际候选路径',line_view(rgb.shape[:2],paths,True))]):d.text((400*j+5,5),title,font=SMALL,fill='black');paste(board,im,400*j+5,36,390,400)
    board.save(out/'comparison.jpg',quality=95);jsave(out/'summary.json',dict(source=str(source),reference_labels_used=False,monochrome_research_only=True,endpoint_policy_enabled=False,robot_connected=False,weights_sha256=digest(ROOT/'ribbon_improved'/'final.pt'),path_count=len(paths)));print('V21_REFERENCE_FREE_OUTPUT',str(out),flush=True)

if __name__=='__main__':main()
