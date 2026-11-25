"""Consistent aspect-preserving input extent; resizing adds no native detail."""
import numpy as np,cv2
from PIL import Image
MEAN=np.array([.485,.456,.406],'float32');STD=np.array([.229,.224,.225],'float32')
def input_canvas(image,size=224,fill_extent=False):
    gray=cv2.equalizeHist(np.asarray(image.convert('L')));rgb=Image.fromarray(gray).convert('RGB');native=rgb.size
    if fill_extent:
        scale=size/max(native);scaled=(max(1,round(native[0]*scale)),max(1,round(native[1]*scale)));rgb=rgb.resize(scaled,Image.Resampling.BILINEAR)
    else:rgb.thumbnail((size,size),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(size,size),(int(np.median(gray)),)*3);canvas.paste(rgb,((size-rgb.width)//2,(size-rgb.height)//2))
    return canvas,dict(native_size=list(native),content_size=list(rgb.size),content_rectangle_fraction=rgb.width*rgb.height/(size*size),resized_not_new_optical_detail=True)
def preprocess(image,fill_extent=False):
    canvas,_=input_canvas(image,fill_extent=fill_extent);x=np.asarray(canvas).astype('float32')/255.;return ((x-MEAN)/STD).transpose(2,0,1)
