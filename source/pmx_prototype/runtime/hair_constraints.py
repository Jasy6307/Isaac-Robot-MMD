"""Contact-aware hair constraint projection after native PhysX integration."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import numpy as np
from realtime_xpbd import njit


@njit(cache=True)
def project_hair(p, target, weights, links, triplets, hair, cloth, edges,
                 radii, ca, cb, cr, allowed, pairs, ratio, iterations, bend):
    skirt_low=1e10;skirt_high=-1e10
    for i in cloth:
        skirt_low=min(skirt_low,p[i,2]-radii[i]-.02)
        skirt_high=max(skirt_high,p[i,2]+radii[i]+.02)
    for a,b in edges:
        skirt_low=min(skirt_low,min(p[a,2],p[b,2])-.02)
        skirt_high=max(skirt_high,max(p[a,2],p[b,2])+.02)
    for iteration in range(iterations):
        for t in range(len(triplets)):
            a,b,c=triplets[t]
            if weights[b]==0:continue
            error=(p[a]-2*p[b]+p[c])-(target[a]-2*target[b]+target[c])
            correction=.5*bend*error
            size=np.linalg.norm(correction)
            if size>.004:correction*=.004/size
            p[b]+=correction
        for k in range(len(links)):
            idx=k if iteration%2==0 else len(links)-1-k
            a,b=int(links[idx,0]),int(links[idx,1]);maximum=links[idx,2]
            delta=p[b]-p[a];distance=np.linalg.norm(delta)
            minimum=maximum/ratio*.985
            desired=min(maximum,max(minimum,distance))
            total=weights[a]+weights[b]
            if distance>1e-10 and total>0:
                correction=delta*((distance-desired)/distance)
                p[a]+=correction*(weights[a]/total)
                p[b]-=correction*(weights[b]/total)
        for k in range(len(pairs)):
            a,b=pairs[k];minimum=radii[a]+radii[b]
            if abs(p[b,0]-p[a,0])>minimum or abs(p[b,1]-p[a,1])>minimum or abs(p[b,2]-p[a,2])>minimum:continue
            delta=p[b]-p[a];distance=np.linalg.norm(delta);total=weights[a]+weights[b]
            if distance>1e-10 and distance<minimum and total>0:
                correction=delta*((minimum-distance)/distance)
                p[a]-=correction*(weights[a]/total);p[b]+=correction*(weights[b]/total)
        for slot in range(len(hair)):
            node=hair[slot]
            for j in range(len(cr)):
                if not allowed[slot,j]:continue
                margin=cr[j]+radii[node]+.001
                if p[node,0]<min(ca[j,0],cb[j,0])-margin or p[node,0]>max(ca[j,0],cb[j,0])+margin:continue
                if p[node,1]<min(ca[j,1],cb[j,1])-margin or p[node,1]>max(ca[j,1],cb[j,1])+margin:continue
                if p[node,2]<min(ca[j,2],cb[j,2])-margin or p[node,2]>max(ca[j,2],cb[j,2])+margin:continue
                axis=cb[j]-ca[j];length2=np.dot(axis,axis)
                u=min(1.,max(0.,np.dot(p[node]-ca[j],axis)/max(length2,1e-12)))
                closest=ca[j]+u*axis;delta=p[node]-closest;distance=np.linalg.norm(delta)
                minimum=cr[j]+radii[node]+.001
                if distance>1e-10 and distance<minimum:p[node]+=delta*((minimum-distance)/distance)
            # Only nearby hair tips enter this part of the skirt envelope.
            if p[node,2]<skirt_low or p[node,2]>skirt_high:continue
            for k in range(len(cloth)):
                other=cloth[k];minimum=radii[node]+radii[other]
                if abs(p[node,2]-p[other,2])>minimum or abs(p[node,0]-p[other,0])>minimum or abs(p[node,1]-p[other,1])>minimum:continue
                delta=p[node]-p[other]
                distance=np.linalg.norm(delta)
                if distance>1e-10 and distance<minimum:p[node]+=delta*((minimum-distance)/distance)
            for k in range(len(edges)):
                a,b=edges[k];minimum=radii[node]+.006
                if p[node,2]>max(p[a,2],p[b,2])+minimum or p[node,2]<min(p[a,2],p[b,2])-minimum:continue
                axis=p[b]-p[a];u=min(1.,max(0.,np.dot(p[node]-p[a],axis)/max(np.dot(axis,axis),1e-12)))
                delta=p[node]-(p[a]+u*axis);distance=np.linalg.norm(delta)
                if distance>1e-10 and distance<minimum:p[node]+=delta*((minimum-distance)/distance)
    # Keep the published chain bound exact after contact iterations converge.
    for k in range(len(links)):
        a,b=int(links[k,0]),int(links[k,1]);delta=p[b]-p[a];distance=np.linalg.norm(delta)
        if distance>links[k,2]:p[b]=p[a]+delta*(links[k,2]/distance)
    return p
