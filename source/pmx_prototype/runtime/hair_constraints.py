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
def bound_hair_lengths(particles, changed, links):
    """Apply the existing ordered length cap without per-link Python overhead."""
    for k in range(len(links)):
        parent=int(links[k,0]);child=int(links[k,1]);length=links[k,2]
        delta=particles[child]-particles[parent]
        distance=np.linalg.norm(delta)
        if distance>length:
            particles[child]=particles[parent]+delta*(length/distance)
            changed[child]=True
    return particles,changed


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


# Exact broad-phase rejection and zero-coefficient elimination; same solve order.
@njit(cache=True)
def project_hair_fast(p, target, weights, links, triplets, hair, cloth, edges,
                      radii, ca, cb, cr, allowed, pairs, ratio, iterations, bend):
    """Ordered V10 solve without temporary 3-vector allocations or fast-math."""
    # Static skirt geometry is required for cached bounds. Fall back if an
    # unusual rig shares skirt nodes with any constraint that moves hair.
    moving=np.zeros(len(p),dtype=np.bool_)
    for node in hair:moving[node]=True
    for link in links:
        a=int(link[0]);b=int(link[1])
        if weights[a]!=0:moving[a]=True
        if weights[b]!=0:moving[b]=True
    for a,b,c in triplets:
        if weights[b]!=0:moving[b]=True
    for a,b in pairs:
        if weights[a]!=0:moving[a]=True
        if weights[b]!=0:moving[b]=True
    for node in cloth:
        if moving[node]:
            return project_hair(p,target,weights,links,triplets,hair,cloth,edges,radii,ca,cb,cr,allowed,pairs,ratio,iterations,bend)
    for a,b in edges:
        if moving[a] or moving[b]:
            return project_hair(p,target,weights,links,triplets,hair,cloth,edges,radii,ca,cb,cr,allowed,pairs,ratio,iterations,bend)
    skirt_low=1e10;skirt_high=-1e10
    xmin=1e10;xmax=-1e10;ymin=1e10;ymax=-1e10
    edge_bounds=np.empty((len(edges),6))
    for i in cloth:
        skirt_low=min(skirt_low,p[i,2]-radii[i]-.02)
        skirt_high=max(skirt_high,p[i,2]+radii[i]+.02)
        xmin=min(xmin,p[i,0]-radii[i]);xmax=max(xmax,p[i,0]+radii[i])
        ymin=min(ymin,p[i,1]-radii[i]);ymax=max(ymax,p[i,1]+radii[i])
    for edge_index in range(len(edges)):
        a,b=edges[edge_index]
        for axis in range(3):
            edge_bounds[edge_index,2*axis]=min(p[a,axis],p[b,axis])
            edge_bounds[edge_index,2*axis+1]=max(p[a,axis],p[b,axis])
        skirt_low=min(skirt_low,min(p[a,2],p[b,2])-.02)
        skirt_high=max(skirt_high,max(p[a,2],p[b,2])+.02)
        xmin=min(xmin,p[a,0]-.006,p[b,0]-.006);xmax=max(xmax,p[a,0]+.006,p[b,0]+.006)
        ymin=min(ymin,p[a,1]-.006,p[b,1]-.006);ymax=max(ymax,p[a,1]+.006,p[b,1]+.006)
    edge_block_count=(len(edges)+7)//8
    edge_blocks=np.empty((edge_block_count,6))
    for block in range(edge_block_count):
        for axis in range(3):
            lo=1e10;hi=-1e10
            for j in range(block*8,min((block+1)*8,len(edges))):
                lo=min(lo,edge_bounds[j,axis*2]);hi=max(hi,edge_bounds[j,axis*2+1])
            edge_blocks[block,axis*2]=lo;edge_blocks[block,axis*2+1]=hi
    for iteration in range(iterations):
        if bend != 0.:
            for a,b,c in triplets:
                if weights[b]==0:continue
                x=.5*bend*((p[a,0]-2*p[b,0]+p[c,0])-(target[a,0]-2*target[b,0]+target[c,0]))
                y=.5*bend*((p[a,1]-2*p[b,1]+p[c,1])-(target[a,1]-2*target[b,1]+target[c,1]))
                z=.5*bend*((p[a,2]-2*p[b,2]+p[c,2])-(target[a,2]-2*target[b,2]+target[c,2]))
                size=np.sqrt(x*x+y*y+z*z)
                if size>.004:
                    scale=.004/size;x*=scale;y*=scale;z*=scale
                p[b,0]+=x;p[b,1]+=y;p[b,2]+=z
        for k in range(len(links)):
            idx=k if iteration%2==0 else len(links)-1-k
            a,b=int(links[idx,0]),int(links[idx,1]);maximum=links[idx,2]
            x=p[b,0]-p[a,0];y=p[b,1]-p[a,1];z=p[b,2]-p[a,2]
            distance=np.sqrt(x*x+y*y+z*z)
            minimum=maximum/ratio*.985;desired=min(maximum,max(minimum,distance))
            total=weights[a]+weights[b]
            if distance>1e-10 and total>0:
                scale=(distance-desired)/distance;x*=scale;y*=scale;z*=scale
                wa=weights[a]/total;wb=weights[b]/total
                p[a,0]+=x*wa;p[a,1]+=y*wa;p[a,2]+=z*wa
                p[b,0]-=x*wb;p[b,1]-=y*wb;p[b,2]-=z*wb
        for a,b in pairs:
            minimum=radii[a]+radii[b]
            if abs(p[b,0]-p[a,0])>minimum or abs(p[b,1]-p[a,1])>minimum or abs(p[b,2]-p[a,2])>minimum:continue
            x=p[b,0]-p[a,0];y=p[b,1]-p[a,1];z=p[b,2]-p[a,2]
            distance=np.sqrt(x*x+y*y+z*z);total=weights[a]+weights[b]
            if distance>1e-10 and distance<minimum and total>0:
                scale=(minimum-distance)/distance;x*=scale;y*=scale;z*=scale
                wa=weights[a]/total;wb=weights[b]/total
                p[a,0]-=x*wa;p[a,1]-=y*wa;p[a,2]-=z*wa
                p[b,0]+=x*wb;p[b,1]+=y*wb;p[b,2]+=z*wb
        for slot in range(len(hair)):
            node=hair[slot]
            for j in range(len(cr)):
                if not allowed[slot,j]:continue
                margin=cr[j]+radii[node]+.001
                if p[node,0]<min(ca[j,0],cb[j,0])-margin or p[node,0]>max(ca[j,0],cb[j,0])+margin:continue
                if p[node,1]<min(ca[j,1],cb[j,1])-margin or p[node,1]>max(ca[j,1],cb[j,1])+margin:continue
                if p[node,2]<min(ca[j,2],cb[j,2])-margin or p[node,2]>max(ca[j,2],cb[j,2])+margin:continue
                ax=cb[j,0]-ca[j,0];ay=cb[j,1]-ca[j,1];az=cb[j,2]-ca[j,2]
                x=p[node,0]-ca[j,0];y=p[node,1]-ca[j,1];z=p[node,2]-ca[j,2]
                u=min(1.,max(0.,(x*ax+y*ay+z*az)/max(ax*ax+ay*ay+az*az,1e-12)))
                x=p[node,0]-(ca[j,0]+u*ax);y=p[node,1]-(ca[j,1]+u*ay);z=p[node,2]-(ca[j,2]+u*az)
                distance=np.sqrt(x*x+y*y+z*z)
                if distance>1e-10 and distance<margin:
                    scale=(margin-distance)/distance
                    p[node,0]+=x*scale;p[node,1]+=y*scale;p[node,2]+=z*scale
            if p[node,2]<skirt_low or p[node,2]>skirt_high:continue
            if p[node,0]<xmin-radii[node] or p[node,0]>xmax+radii[node] or p[node,1]<ymin-radii[node] or p[node,1]>ymax+radii[node]:continue
            for other in cloth:
                minimum=radii[node]+radii[other]
                if abs(p[node,2]-p[other,2])>minimum or abs(p[node,0]-p[other,0])>minimum or abs(p[node,1]-p[other,1])>minimum:continue
                x=p[node,0]-p[other,0];y=p[node,1]-p[other,1];z=p[node,2]-p[other,2]
                distance=np.sqrt(x*x+y*y+z*z)
                if distance>1e-10 and distance<minimum:
                    scale=(minimum-distance)/distance
                    p[node,0]+=x*scale;p[node,1]+=y*scale;p[node,2]+=z*scale
            for block in range(edge_block_count):
                minimum=radii[node]+.006
                if p[node,2]>edge_blocks[block,5]+minimum or p[node,2]<edge_blocks[block,4]-minimum:continue
                if p[node,0]>edge_blocks[block,1]+minimum or p[node,0]<edge_blocks[block,0]-minimum:continue
                if p[node,1]>edge_blocks[block,3]+minimum or p[node,1]<edge_blocks[block,2]-minimum:continue
                for edge_index in range(block*8,min((block+1)*8,len(edges))):
                    a,b=edges[edge_index]
                    minimum=radii[node]+.006
                    if p[node,2]>edge_bounds[edge_index,5]+minimum or p[node,2]<edge_bounds[edge_index,4]-minimum:continue
                    if p[node,0]>edge_bounds[edge_index,1]+minimum or p[node,0]<edge_bounds[edge_index,0]-minimum:continue
                    if p[node,1]>edge_bounds[edge_index,3]+minimum or p[node,1]<edge_bounds[edge_index,2]-minimum:continue
                    ax=p[b,0]-p[a,0];ay=p[b,1]-p[a,1];az=p[b,2]-p[a,2]
                    x=p[node,0]-p[a,0];y=p[node,1]-p[a,1];z=p[node,2]-p[a,2]
                    u=min(1.,max(0.,(x*ax+y*ay+z*az)/max(ax*ax+ay*ay+az*az,1e-12)))
                    x=p[node,0]-(p[a,0]+u*ax);y=p[node,1]-(p[a,1]+u*ay);z=p[node,2]-(p[a,2]+u*az)
                    distance=np.sqrt(x*x+y*y+z*z)
                    if distance>1e-10 and distance<minimum:
                        scale=(minimum-distance)/distance
                        p[node,0]+=x*scale;p[node,1]+=y*scale;p[node,2]+=z*scale
    for k in range(len(links)):
        a,b=int(links[k,0]),int(links[k,1])
        x=p[b,0]-p[a,0];y=p[b,1]-p[a,1];z=p[b,2]-p[a,2]
        distance=np.sqrt(x*x+y*y+z*z)
        if distance>links[k,2]:
            scale=links[k,2]/distance
            p[b,0]=p[a,0]+x*scale;p[b,1]=p[a,1]+y*scale;p[b,2]=p[a,2]+z*scale
    return p
