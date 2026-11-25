# -*- coding: utf-8 -*-
"""Native-resolution full-scene demonstration. No training or robot imports."""
import sys,json,hashlib,time,shutil
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import torch,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.model import GongbiLineNet
from gongbi_training.infer import predict
from gongbi_training.stroke_train_v08 import model_new,infer
from gongbi_training.stroke_streamline_v08 import trace
ROOT=LAB/'full_landscape_probe_20261024';ROOT.mkdir(exist_ok=True)
INK=LAB/'runs'/'mobilenet_unet_rock_cunfa_v06_expanded'/'best.pt';GEOM=LAB/'rock_strokes_v08_20261023'/'best.pt'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
def paste(board,im,x,y,w,h):
    im=im.convert('RGB');im.thumbnail((w,h));board.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))
def main():
    torch.set_num_threads(4);cv2.setNumThreads(2)
    ink_model=GongbiLineNet().cuda();ink_model.load_state_dict(torch.load(INK,map_location='cuda',weights_only=False)['model']);ink_model.eval()
    geom=model_new().cuda();geom.load_state_dict(torch.load(GEOM,map_location='cuda',weights_only=False)['model']);geom.eval()
    records=[]
    for identity in ('user_qianli_06','user_qianli_07'):
        folder=ROOT/identity;folder.mkdir(exist_ok=True);source=LAB/'data'/'processed'/identity/'image.png';rgb=np.asarray(Image.open(source).convert('RGB'));h,w=rgb.shape[:2];start=time.perf_counter()
        print('NATIVE_INFERENCE',identity,w,h,flush=True)
        padded=np.pad(rgb,((32,32),(32,32),(0,0)),mode='edge');ink=predict(ink_model,padded,'cuda',tile=320)[:,32:32+h,32:32+w];p=infer(geom,rgb)
        np.savez_compressed(folder/'raw_probabilities.npz',ink=ink,geometry=p);Image.fromarray(np.uint8(np.clip(255*(1-ink[1]),0,255))).save(folder/'ink_prediction.png');Image.fromarray(np.uint8(np.clip(255*(1-p[0]),0,255))).save(folder/'center_probability.png')
        print('TRACING',identity,flush=True)
        # Same strict core direction tracer as the retained v08 baseline. We
        # omit gap merging, pressure estimation and stroke order for this probe.
        paths,stats=trace(ink,p,strict=True);line=np.full((h,w,3),255,np.uint8)
        for points in paths:cv2.polylines(line,[np.rint(points).astype(np.int32)],False,(0,0,0),1,cv2.LINE_AA)
        overlay=rgb.copy();mask=cv2.dilate((line[:,:,0]<220).astype(np.uint8),np.ones((2,2),np.uint8))>0;overlay[mask]=np.clip(overlay[mask]*.4+np.array([235,30,30])*.6,0,255).astype(np.uint8)
        Image.fromarray(rgb).save(folder/'original.png');Image.fromarray(line).save(folder/'centerline_candidates.png');Image.fromarray(overlay).save(folder/'overlay.png')
        board=Image.new('RGB',(1800,750),'#f4f4ed');d=ImageDraw.Draw(board);d.text((10,6),identity+' | 原尺寸推理；仅缩放展示；早期参考图演示，非独立测试',font=FONT,fill='black')
        for j,(label,a) in enumerate([('原图',rgb),('当前基线中心线候选',line),('候选叠加原图（红色）',overlay)]):
            d.text((j*600+10,40),label,font=FONT,fill='black');paste(board,Image.fromarray(a),j*600+9,75,582,665)
        board.save(folder/'full_comparison.jpg',quality=95)
        # Fixed representative regions: main mountain, foreground texture and
        # architecture/trees. Crops come from the native full-scene prediction.
        boxes=[(.28,.08,.63,.53),(.05,.55,.42,.96),(.60,.47,.98,.91)] if identity.endswith('07') else [(.20,.18,.61,.65),(.30,.57,.74,.99),(.72,.55,.99,.96)]
        detail=Image.new('RGB',(1500,340*len(boxes)),'#f4f4ed');dd=ImageDraw.Draw(detail)
        for i,b in enumerate(boxes):
            box=tuple(round(v*(w if k%2==0 else h)) for k,v in enumerate(b));dd.text((9,i*340+3),f'局部{i+1}：原图 / 中心线候选 / 红色叠加（均来自整图预测）',font=FONT,fill='black')
            for j,a in enumerate((rgb,line,overlay)):paste(detail,Image.fromarray(a).crop(box),j*500+8,i*340+36,482,295)
        detail.save(folder/'detail_comparison.jpg',quality=95)
        records.append(dict(id=identity,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),native_size=[w,h],paths=len(paths),diagnostics=stats,elapsed_seconds=time.perf_counter()-start,model_and_parameters_unchanged=True,manual_prediction_cleanup=False,independent_test=False,actual_drawing=False,ready_for_robot=False))
        (folder/'candidate_paths.json').write_text(json.dumps(dict(coordinate_units='pixels',physical_z=None,physical_speed=None,ready_for_robot=False,actually_drawn=False,paths=[a.round(2).tolist() for a in paths]),ensure_ascii=False),encoding='utf-8')
        print('PROBE_DONE',identity,len(paths),flush=True)
    report=dict(records=records,ink_model=str(INK),ink_sha256=hashlib.sha256(INK.read_bytes()).hexdigest(),geometry_model=str(GEOM),geometry_sha256=hashlib.sha256(GEOM.read_bytes()).hexdigest(),method='same native-resolution overlapping neural inference plus retained strict v08 core direction tracer; no gap merge, brush-width replay or motion planning',expert_ground_truth=False,accuracy_not_available=True,robot_connected=False)
    (ROOT/'probe_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(Path(__file__),ROOT/'source_snapshot.py')
    for f in ROOT.glob('*/*.png'):
        with Image.open(f) as im:im.verify()
    print('FULL_PROBE_COMPLETE',flush=True)
if __name__=='__main__':main()
