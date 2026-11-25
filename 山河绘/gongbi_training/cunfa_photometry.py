"""Appearance-preserving views and contrast measurements, never ink labels.

Reference brightness is estimated from observable pixels only, without any
paper, role, style or test masks. This does NOT recover physical ink density.
"""
import cv2
import numpy as np
from PIL import Image
from .cunfa_input_scale import MEAN,STD

def reference_white(image):
    return max(1.,float(np.percentile(np.asarray(image.convert('L')),90)))

def appearance_canvas(image,reference_image,mode='raw_gray',size=224):
    gray=np.asarray(image.convert('L'))
    if mode=='raw_gray':out=gray
    elif mode=='reference_linear':
        out=np.clip(gray.astype('float32')*240/reference_white(reference_image),0,255).round().astype('uint8')
    else:raise ValueError('Unknown appearance preprocessing')
    native=Image.fromarray(out).convert('RGB');scale=size/max(native.size)
    scaled=(max(1,round(native.width*scale)),max(1,round(native.height*scale)))
    native=native.resize(scaled,Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(size,size),(int(np.median(out)),)*3)
    canvas.paste(native,((size-native.width)//2,(size-native.height)//2))
    return canvas

def appearance_tensor(image,reference_image,mode):
    canvas=appearance_canvas(image,reference_image,mode)
    return ((np.asarray(canvas).astype('float32')/255-MEAN)/STD).transpose(2,0,1)

CONTRAST_NAMES=('relative_dark_mean','relative_dark_median','relative_dark_q90','relative_dark_q99',
 'gray_std','gray_q90_q10','smooth2_std','smooth5_std','fine_residual_mean','fine_residual_q90',
 'coarse_residual_mean','coarse_residual_q90','gradient_q90','gradient_anisotropy')

def contrast_features(image,reference_image):
    gray=np.asarray(image.convert('L')).astype('float32');scale=128/max(gray.shape)
    gray=cv2.resize(gray,(max(2,round(gray.shape[1]*scale)),max(2,round(gray.shape[0]*scale))),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    anchor=reference_white(reference_image);g=gray/anchor;dark=np.clip(1-g,-.3,1)
    s2=cv2.GaussianBlur(g,(0,0),2);s5=cv2.GaussianBlur(g,(0,0),5)
    r2=abs(g-s2);r5=abs(g-s5);dx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3);dy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    xx=float(np.mean(dx*dx));yy=float(np.mean(dy*dy));xy=float(np.mean(dx*dy));trace=xx+yy
    anisotropy=np.sqrt((xx-yy)**2+4*xy**2)/(trace+1e-8)
    return np.array([dark.mean(),np.median(dark),np.percentile(dark,90),np.percentile(dark,99),
        g.std(),np.percentile(g,90)-np.percentile(g,10),s2.std(),s5.std(),
        r2.mean(),np.percentile(r2,90),r5.mean(),np.percentile(r5,90),
        np.percentile(np.sqrt(dx*dx+dy*dy),90),anisotropy],dtype='float64')

def substrate_variant(reference_image,seed):
    """One fixed training augmentation; not another work or real paper capture.

    One full-reference transform is cropped for local/context views, so both
    observe the SAME simulated grain. No strokes are added, linked, or erased.
    """
    rng=np.random.default_rng(seed);x=np.asarray(reference_image.convert('RGB')).astype('float32')
    tint=np.array([1.,.94,.84],dtype='float32')
    grain=rng.normal(0,1.5,x.shape[:2])[...,None]
    transformed=.82*x*tint+grain
    return Image.fromarray(np.clip(transformed,0,255).round().astype('uint8'))
