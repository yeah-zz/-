"""Spatial-pyramid region probes. No pixel labels or brush-stroke instances."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def pyramid_vector(feature_map,levels=(1,2)):
    if feature_map.ndim!=4:raise ValueError('Expected N,C,H,W')
    blocks=[]
    for n in levels:
        z=F.adaptive_avg_pool2d(feature_map,(n,n)).flatten(1)
        blocks.append(F.normalize(z,dim=1)/np.sqrt(len(levels)))
    return torch.cat(blocks,1)

def make_probe(n_train):
    # Small data: fixed low-rank projection, fitted separately in each fold.
    return make_pipeline(StandardScaler(),PCA(n_components=min(12,n_train-1),svd_solver='full'),
       LogisticRegression(C=.1,class_weight='balanced',random_state=0,max_iter=2000))

def verify_group_separation(rows,train_mask,test_mask):
    groups=np.array([r['role_fold_group'] for r in rows])
    if set(groups[train_mask])&set(groups[test_mask]):raise ValueError('Work overlaps train/check')
    return True
