from faint_joint_model_20261101 import *
from scipy.ndimage import binary_propagation,binary_fill_holes
def mountain_context(rgb):
    a=cv2.GaussianBlur(rgb.astype(np.float32)/255,(0,0),5);blue=(a[:,:,2]-a[:,:,0]>.012)&(a[:,:,1]-a[:,:,0]>.006);blue=cv2.morphologyEx(blue.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)));n,cc,stats,_=cv2.connectedComponentsWithStats(blue,8);seed=np.zeros(blue.shape,bool)
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        if h>=50 and area>=400 and w/max(h,1)<4.5:seed|=cc==i
    luminance=a@np.array([.2126,.7152,.0722]);paper=np.median(a[luminance>=np.quantile(luminance,.98)],axis=0);material=np.linalg.norm(a-paper,axis=2)>.055;current=seed.copy()
    for _ in range(60):
        grown=(cv2.dilate(current.astype(np.uint8),np.ones((3,3),np.uint8))>0)&(material|seed)
        if np.array_equal(grown,current):break
        current=grown
    current=cv2.morphologyEx(current.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)));current=binary_fill_holes(current>0);return cv2.dilate(current.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))>0
