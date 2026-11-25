# -*- coding: utf-8 -*-
"""Trace a sparse skeleton graph; pair junction branches without reference IDs."""
import sys,math
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
import numpy as np,cv2
from skimage.morphology import skeletonize
from stroke_link_v12 import tangent

def adapt_old(p):
    out=np.zeros((8,*p.shape[1:]),np.float32);out[:3]=p[:3];out[3:5]=p[1:3];return out

def center_skeleton(pred):
    p=pred[0];smooth=cv2.GaussianBlur(p.astype(np.float32),(0,0),.65);h,w=p.shape;yy,xx=np.mgrid[:h,:w].astype(np.float32);ridge=np.zeros((h,w),bool)
    for start in (1,3):
        theta=.5*np.arctan2(pred[start+1],pred[start]);nx=-np.sin(theta);ny=np.cos(theta)
        a=cv2.remap(smooth,xx+nx*1.5,yy+ny*1.5,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);b=cv2.remap(smooth,xx-nx*1.5,yy-ny*1.5,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        ridge|=(smooth>=a)&(smooth>=b)&(smooth-(a+b)*.5>.001)
    joint=(pred[6]>.45)&(p>.3);candidates=((p>=.3)&ridge)|joint;n,labels,stats,_=cv2.connectedComponentsWithStats(candidates.astype(np.uint8),8);good=np.zeros(n,bool);good[np.unique(labels[p>=.5])]=True;good[0]=False;good[stats[:,cv2.CC_STAT_AREA]<12]=False
    sk=skeletonize(good[labels]);sk[[0,-1],:]=False;sk[:,[0,-1]]=False;return sk

def graph_paths(sk,pred,pair=True,min_length=12):
    h,w=sk.shape;pixels=set(map(tuple,np.argwhere(sk)));neighbors={}
    for y,x in pixels:
        near=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                q=(y+dy,x+dx)
                if (not dx and not dy) or q not in pixels:continue
                if dx and dy and ((y,x+dx) in pixels or (y+dy,x) in pixels):continue
                near.append(q)
        neighbors[(y,x)]=near
    high=np.zeros((h,w),np.uint8)
    for p,near in neighbors.items():
        if len(near)>2:high[p]=1
    expanded=cv2.dilate(high,np.ones((5,5),np.uint8))*sk.astype(np.uint8);n,component=cv2.connectedComponents(expanded,8);node_of={};centers={};next_node=0
    for i in range(1,n):
        yy,xx=np.nonzero(component==i);centers[next_node]=np.array([xx.mean(),yy.mean()]);node_of.update({(y,x):next_node for y,x in zip(yy.tolist(),xx.tolist())});next_node+=1
    for p,near in neighbors.items():
        if len(near)!=2 and p not in node_of:node_of[p]=next_node;centers[next_node]=np.array([p[1],p[0]],float);next_node+=1
    visited=set();chains=[];nodes=[]
    def edge(a,b):return tuple(sorted((a,b)))
    for pixel,node in sorted(node_of.items()):
        for q in neighbors[pixel]:
            if node_of.get(q)==node or edge(pixel,q) in visited:continue
            visited.add(edge(pixel,q));points=[centers[node],np.array([pixel[1],pixel[0]],float)];previous=pixel;position=q;end=None
            for _ in range(len(pixels)+1):
                if position in node_of:
                    end=node_of[position];points.append(centers[end]);break
                points.append(np.array([position[1],position[0]],float));options=[z for z in neighbors[position] if z!=previous]
                if not options:break
                following=options[0];visited.add(edge(position,following));previous,position=position,following
            if len(points)>2:chains.append(np.asarray(points));nodes.append((node,end))
    # Components with no endpoints/junctions are valid closed curves.
    for a in sorted(pixels):
        for b in neighbors[a]:
            if edge(a,b) in visited or (a in node_of and b in node_of):continue
            p=[np.array([a[1],a[0]],float)];previous=a;position=b;visited.add(edge(a,b))
            for _ in range(len(pixels)+1):
                p.append(np.array([position[1],position[0]],float))
                if position==a:break
                options=[z for z in neighbors[position] if z!=previous and edge(position,z) not in visited]
                if not options:break
                following=options[0];visited.add(edge(position,following));previous,position=position,following
            if len(p)>2:chains.append(np.asarray(p));nodes.append((None,None))
    incident={}
    for i,ends in enumerate(nodes):
        for side,node in enumerate(ends):
            if node is not None:incident.setdefault(node,[]).append((i,side))
    paired={};events=[]
    if pair:
        for node,ends in incident.items():
            if len(ends)<2:continue
            x,y=np.rint(centers[node]).astype(int);x=int(np.clip(x,0,w-1));y=int(np.clip(y,0,h-1));axes=[]
            for start in (1,3):theta=.5*math.atan2(pred[start+1,y,x],pred[start,y,x]);axes.append(np.array([math.cos(theta),math.sin(theta)]))
            candidates=[];by_end={}
            for i in range(len(ends)):
                for j in range(i+1,len(ends)):
                    a,ea=ends[i];b,eb=ends[j]
                    if a==b:continue
                    va=tangent(chains[a],ea==0);vb=tangent(chains[b],eb==0);alignment=float(va@(-vb));field=max(abs(float(va@ax))*abs(float(vb@ax)) for ax in axes);score=.8*alignment+.2*field
                    if score<.72:continue
                    k=len(candidates);candidates.append((score,ends[i],ends[j]));by_end.setdefault(ends[i],[]).append((score,k));by_end.setdefault(ends[j],[]).append((score,k))
            selected=[]
            for k,(score,a,b) in enumerate(candidates):
                aa=sorted(by_end[a],reverse=True);bb=sorted(by_end[b],reverse=True)
                if aa[0][1]!=k or bb[0][1]!=k:continue
                if (len(aa)>1 and aa[0][0]-aa[1][0]<.08) or (len(bb)>1 and bb[0][0]-bb[1][0]<.08):continue
                selected.append((score,a,b))
            for score,a,b in sorted(selected,reverse=True):
                if a in paired or b in paired:continue
                paired[a]=b;paired[b]=a;events.append(dict(node_xy=centers[node].round(2).tolist(),score=score,status='junction_continuation_hypothesis'))
    used=set();paths=[]
    starts=[(i,side) for i in range(len(chains)) for side in (0,1) if (i,side) not in paired]+[(i,0) for i in range(len(chains))]
    for index,side in starts:
        if index in used:continue
        segments=[];first=(index,side)
        while index not in used:
            used.add(index);segments.append(chains[index] if side==0 else chains[index][::-1]);next_end=paired.get((index,1-side))
            if next_end is None:break
            index,side=next_end
            if (index,side)==first:break
        points=np.concatenate(segments)
        if np.linalg.norm(np.diff(points,axis=0),axis=1).sum()>=min_length:paths.append(points)
    return paths,dict(junction_pairings=events,raw_chains=len(chains),paths=len(paths))

def extract(pred,pair=True):
    sk=center_skeleton(pred);paths,info=graph_paths(sk,pred,pair);return paths,info,sk
