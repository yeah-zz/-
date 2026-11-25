"""Rasterize reviewed nested cross-sections without treating unknown as paper."""
import numpy as np


def rasterize_band(shape, sections, role, uncertain_rows=(), paper_gap=2, paper_width=3):
    """Sections: y, certain-left/right, possible-left/right (native coordinates).

    Bounds describe assistant uncertainty, not measured physical brush width.
    Only reviewed row range is populated. End caps and all other pixels stay unknown.
    """
    s=np.asarray(sections,dtype=float)
    if s.ndim!=2 or s.shape[1]!=5 or len(s)<2 or not np.isfinite(s).all():
        raise ValueError('Need finite ordered cross-sections')
    if not np.all(np.diff(s[:,0])>0) or role not in (1,2):
        raise ValueError('Sections must be ordered and role checked')
    h,w=shape
    if s[0,0]<0 or s[-1,0]>=h or (s[:,3]<0).any() or (s[:,4]>=w).any():
        raise ValueError('Bounds outside source image')
    if not np.all((s[:,3]<=s[:,1]) & (s[:,1]<=s[:,2]) & (s[:,2]<=s[:,4])):
        raise ValueError('Certain core must lie inside possible band')
    core=np.zeros(shape,bool);possible=np.zeros(shape,bool);paper=np.zeros(shape,bool)
    yy=np.arange(int(np.ceil(s[0,0])),int(np.floor(s[-1,0]))+1)
    limits=np.stack([np.interp(yy,s[:,0],s[:,k]) for k in range(1,5)],axis=1)
    for y,(cl,cr,pl,pr) in zip(yy,limits):
        if any(a<=y<=b for a,b in uncertain_rows):continue
        core[y,int(np.ceil(cl)):int(np.floor(cr))+1]=True
        possible[y,int(np.floor(pl)):int(np.ceil(pr))+1]=True
        for a,b in [(int(np.floor(pl))-paper_gap-paper_width,int(np.floor(pl))-paper_gap),
                    (int(np.ceil(pr))+paper_gap+1,int(np.ceil(pr))+paper_gap+paper_width+1)]:
            if a<0 or b>w:raise ValueError('Paper interval outside source image')
            paper[y,a:b]=True
    ink=np.full(shape,255,np.uint8);ink[core]=1;ink[paper]=0
    roles=np.full(shape,255,np.uint8);roles[core]=role
    return dict(ink_targets=ink,role_targets=roles,core=core,possible=possible,paper=paper)


def merge_bands(bands):
    if not bands:raise ValueError('No reviewed bands')
    shape=bands[0]['core'].shape
    possible=np.logical_or.reduce([b['possible'] for b in bands])
    core=np.logical_or.reduce([b['core'] for b in bands])
    paper=np.logical_or.reduce([b['paper'] for b in bands]) & ~possible
    votes=np.zeros((2,*shape),bool)
    for b in bands:
        votes[0]|=b['role_targets']==1;votes[1]|=b['role_targets']==2
    conflicts=votes.all(0)
    ink=np.full(shape,255,np.uint8);ink[core & ~conflicts]=1;ink[paper]=0
    roles=np.full(shape,255,np.uint8);roles[votes[0]&~conflicts]=1;roles[votes[1]&~conflicts]=2
    return dict(ink_targets=ink,role_targets=roles,possible=possible,core=core,
                paper=paper,role_conflicts=conflicts,fringe=possible & ~core)
