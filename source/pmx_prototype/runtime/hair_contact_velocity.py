"""Non-penetrating hair contact velocity without position-to-impulse inflation."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import numpy as np
from realtime_xpbd import njit


@njit(cache=True)
def correct_velocity(velocity,p,hair,radii,ca,cb,cr,allowed,body_velocity,dt):
    for slot in range(len(hair)):
        node=hair[slot]
        for j in range(len(cr)):
            if not allowed[slot,j]:continue
            bound=cr[j]+radii[node]+.005
            if p[node,0]<min(ca[j,0],cb[j,0])-bound or p[node,0]>max(ca[j,0],cb[j,0])+bound:continue
            if p[node,1]<min(ca[j,1],cb[j,1])-bound or p[node,1]>max(ca[j,1],cb[j,1])+bound:continue
            if p[node,2]<min(ca[j,2],cb[j,2])-bound or p[node,2]>max(ca[j,2],cb[j,2])+bound:continue
            axis=cb[j]-ca[j];u=min(1.,max(0.,np.dot(p[node]-ca[j],axis)/max(np.dot(axis,axis),1e-12)))
            closest=ca[j]+u*axis;delta=p[node]-closest;distance=np.linalg.norm(delta)
            minimum=cr[j]+radii[node]+.001
            if distance<1e-10 or distance>minimum+.004:continue
            normal=delta/distance;point=closest+normal*cr[j]
            surface=body_velocity[j,:3]+np.cross(body_velocity[j,3:],point-(ca[j]+cb[j])*.5)
            speed=np.dot(velocity[node]-surface,normal)
            desired=min(.08,max(0.,.1*(minimum-distance)/dt))
            # Remove inward motion and suppress only the contact-normal rebound;
            # preserve tangential sliding and the moving collider's velocity.
            adjusted=min(.08,max(desired,speed))
            velocity[node]+=(adjusted-speed)*normal
    return velocity
