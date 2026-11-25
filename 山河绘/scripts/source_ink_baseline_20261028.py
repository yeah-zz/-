# -*- coding: utf-8 -*-
"""Untrained source-pixel baseline; deliberately no curve continuation."""
import os,sys
from pathlib import Path
sys.dont_write_bytecode=True
LAB=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022')
for k,v in {'TEMP':'temp','TMP':'temp','PIP_CACHE_DIR':'cache/pip','TORCH_HOME':'cache/torch','HF_HOME':'cache/huggingface','XDG_CACHE_HOME':'cache','CUDA_CACHE_PATH':'cache/cuda'}.items():os.environ[k]=str(LAB/v)
from rock_v15_common import jsave
import cv2,numpy as np
from scipy.ndimage import binary_propagation
from skimage.morphology import skeletonize
from PIL import Image,ImageDraw,ImageFont

ROOT=LAB/'generated_simple_landscapes_20261028'
OUT=ROOT/'loss_diagnosis'/'source_pixel_baseline'

def extract(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    smooth=cv2.GaussianBlur(gray,(0,0),.45)
    # Local dark valleys suppress broad wash interiors, without model directions.
    local=cv2.GaussianBlur(smooth,(0,0),3)-smooth
    valley=cv2.morphologyEx(smooth,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
    weak=(local>.018)&(valley>.045)
    strong=(local>.045)&(valley>.09)
    ink=binary_propagation(strong,mask=weak)
    n,cc,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8),8)
    keep=stats[:,cv2.CC_STAT_AREA]>=5;keep[0]=False
    mask=keep[cc]
    sk=skeletonize(mask)
    return mask,sk

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2)
    records=[];font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',22)
    for sid in ['01_rocks','02_pine','03_pavilion']:
        rgb=np.asarray(Image.open(ROOT/'originals'/(sid+'.png')).convert('RGB'))
        mask,sk=extract(rgb)
        folder=OUT/sid;folder.mkdir(exist_ok=True)
        ink=np.repeat(np.where(mask,0,255).astype(np.uint8)[:,:,None],3,2)
        line=np.repeat(np.where(sk,0,255).astype(np.uint8)[:,:,None],3,2)
        Image.fromarray(ink).save(folder/'ink.png');Image.fromarray(line).save(folder/'skeleton.png')
        overlay=rgb.copy();overlay[sk]=[20,105,220];Image.fromarray(overlay).save(folder/'overlay.png')
        old=np.asarray(Image.open(ROOT/'loss_diagnosis'/sid/'final_1254.png').convert('RGB'))
        board=Image.new('RGB',(1896,752),'#f3f1eb');d=ImageDraw.Draw(board)
        for i,(label,arr) in enumerate([('原图',rgb),('原模型：原分辨率',old),('原图像素基线：未识别笔顺',line)]):
            d.text((i*632+6,8),label,font=font,fill='black');im=Image.fromarray(arr);im.thumbnail((620,620),Image.Resampling.LANCZOS);board.paste(im,(i*632+6,52))
        d.text((6,695),'右图为传统图像处理对照，不是新训练成果；保留碎线，未自动连接，尚不可直接作为机械臂轨迹。',font=font,fill='#333333')
        board.save(folder/'comparison.jpg',quality=95)
        if sid=='02_pine':
            box=(130,240,705,565);detail=Image.new('RGB',(1725,410),'#f3f1eb');dd=ImageDraw.Draw(detail)
            for i,(label,arr) in enumerate([('原图：松树局部',rgb),('模型：原分辨率轨迹',old),('原图像素基线：未续接',line)]):
                dd.text((i*575+4,7),label,font=font,fill='black');detail.paste(Image.fromarray(arr).crop(box),(i*575,48))
            detail.save(OUT/'pine_comparison.jpg',quality=95)
        records.append({'id':sid,'mask_pixels':int(mask.sum()),'skeleton_pixels':int(sk.sum())})
    jsave(OUT/'protocol.json',{'method':'untrained local grayscale valley plus hysteresis and skeletonization','model_weights_updated':False,'stroke_identity_inferred':False,'semantic_segmentation':False,'references_used':False,'roi_specific_parameters':False,'same_parameters_for_all_images':True,'robot_connected':False,'frozen_parameters':{'gray_sigma':.45,'local_sigma':3,'blackhat_diameter':17,'weak_local':.018,'weak_valley':.045,'strong_local':.045,'strong_valley':.09,'min_component_area':5},'records':records})
    print('SOURCE_PIXEL_BASELINE_READY',str(OUT),flush=True)
