"""Spatial and hierarchical probes on GIVEN regions, never stroke ownership."""
import cv2,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .cunfa_photometry import reference_white

def classifier():
    return make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',random_state=0,max_iter=2000))

def normalized_roi(box,view_box):
    x0,y0,x1,y1=box;a,b,c,d=view_box
    return np.array([(x0-a)/(c-a),(y0-b)/(d-b),(x1-a)/(c-a),(y1-b)/(d-b)],dtype='float64')

def rotate_roi_ccw(box):
    x0,y0,x1,y1=box
    return np.array([y0,1-x1,y1,1-x0])

def spatial_features(far_image,reference_image,roi):
    # Retain where grey/texture occurs, not only its whole-window average.
    g=np.asarray(far_image.convert('L')).astype('float32')/reference_white(reference_image)
    g=cv2.resize(g,(192,192),interpolation=cv2.INTER_AREA)
    values=[]
    for y in range(3):
        for x in range(3):
            cell=g[y*64:(y+1)*64,x*64:(x+1)*64]
            low=cv2.GaussianBlur(cell,(0,0),3)
            values.extend([float(np.median(1-cell)),float(cell.std()),float(low.std())])
    values.extend(np.asarray(roi).tolist())
    return np.array(values,dtype='float64')

def fit_role_probe(deep,appearance,targets,hierarchical=False,weights=None):
    targets=np.asarray(targets)
    if set(targets)!={0,1,2,3}:raise ValueError('Role fit must have every supported class')
    kwargs={} if weights is None else {'logisticregression__sample_weight':np.asarray(weights)}
    if not hierarchical:return dict(flat=classifier().fit(deep,targets,**kwargs),hierarchical=False)
    app=np.where(targets<2,0,targets-1)
    ink=targets<2
    ink_kwargs={} if weights is None else {'logisticregression__sample_weight':np.asarray(weights)[ink]}
    return dict(appearance=classifier().fit(appearance,app,**kwargs),
                ink_role=classifier().fit(deep[ink],targets[ink],**ink_kwargs),hierarchical=True)

def predict_role_probe(model,deep,appearance):
    if not model['hierarchical']:return model['flat'].predict(deep).astype('int64')
    pa=model['appearance'].predict_proba(appearance)
    pi=model['ink_role'].predict_proba(deep)
    # App class0=marked,1=diffuse tone,2=unmarked. Conditional class0=structure,
    # class1=interior marks. Probabilities are exploratory, not calibrated.
    posterior=np.column_stack([pa[:,0]*pi[:,0],pa[:,0]*pi[:,1],pa[:,1],pa[:,2]])
    return posterior.argmax(1).astype('int64')
