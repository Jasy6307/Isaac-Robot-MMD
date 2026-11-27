"""Fixed, close replay framing fitted to the complete recorded motion."""
from __future__ import annotations
from itertools import product
import math
import numpy as np


def replica_offsets(count,spacing):
    if count<1 or not math.isfinite(spacing) or spacing<=0:raise ValueError("Invalid replay spacing")
    offsets=np.zeros((count,3),np.float32)
    offsets[:,0]=(np.arange(count)[::-1]-(count-1)/2)*spacing
    return offsets


def fit_camera(bounds,*,width,height,header_pixels,footer_pixels,focal_length=26.,aperture=20.955):
    """Solve perspective fit, reserving the actual text area above/below robots."""
    bounds=np.asarray(bounds,float)
    if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or (bounds[1]<=bounds[0]).any():
        raise ValueError("Invalid replay bounds")
    if width<1 or height<1:raise ValueError("Invalid camera resolution")
    corners=np.array(list(product(*zip(bounds[0],bounds[1]))))
    target=bounds.mean(0)
    # More room is needed below the optical axis because labels occupy the top.
    target[2]+=.08*(bounds[1,2]-bounds[0,2])
    slope=.16;scale=math.sqrt(1+slope*slope)
    relative=corners-target
    x=-relative[:,0];y=(-slope*relative[:,1]+relative[:,2])/scale
    depth=(-relative[:,1]-slope*relative[:,2])/scale
    tan_x=aperture/(2*focal_length);tan_y=tan_x*height/width
    up=1-2*(header_pixels/height+.02);down=1-2*(footer_pixels/height+.02)
    if min(up,down)<=0 or width<1 or height<1:raise ValueError("No usable camera image area")
    horizontal=(np.abs(x)/(.94*tan_x)-depth)/scale
    vertical=(np.abs(y)/(np.where(y>=0,up,down)*tan_y)-depth)/scale
    distance=max(1.,float(horizontal.max()),float(vertical.max()),float(((.25-depth)/scale).max()))*1.02
    eye=target+np.array([0.,distance,distance*slope])
    return dict(eye=eye.tolist(),target=target.tolist(),bounds=bounds.tolist(),distance_m=distance,
                focal_length_mm=focal_length,horizontal_aperture_mm=aperture,
                header_pixels=int(header_pixels),footer_pixels=int(footer_pixels),fixed_for_entire_recording=True)


def reference_camera_track(reference,frames):
    """Follow only intended XY translation; never follow actual jitter or jump Z."""
    table=reference.get("tables",{}).get("_root_pos_delta")
    if table is None:raise ValueError("Reference camera requires packaged root translation")
    if hasattr(table,"detach"):table=table.detach().cpu().numpy()
    table=np.asarray(table,float)
    if table.ndim!=2 or table.shape[1]!=3 or not len(table) or not np.isfinite(table).all():
        raise ValueError("Invalid reference camera track")
    track=np.zeros((frames,3))
    indices=np.minimum(np.arange(frames),len(table)-1)
    kernel=np.array([1,2,3,4,3,2,1],float)/16
    for axis in (0,1):
        values=table[indices,axis]
        track[:,axis]=np.convolve(np.pad(values,(3,3),mode="edge"),kernel,mode="valid")
    return track


def motion_bounds(recordings,usd_path,offsets,ground_z,camera_track=None):
    """USD joint FK gives full-motion arm/head/foot bounds, not just root bounds."""
    from scipy.spatial.transform import Rotation
    from source.train_workflow.utils.motion.robot_fk import robot_fk
    names=recordings[0][1]["joint_names"]
    low=np.full(3,np.inf);high=np.full(3,-np.inf)
    for (poses,meta),offset in zip(recordings,offsets):
        if meta["joint_names"]!=names:raise ValueError("Replay joint ordering differs")
        _,positions=robot_fk(usd_path,names,poses[:,7:])
        local=np.stack(list(positions.values()),axis=1)
        base=Rotation.from_quat(poses[:,3:7]).as_matrix()
        points=np.einsum("nij,nkj->nki",base,local)+poses[:,None,:3]+offset
        if camera_track is not None:points-=camera_track[:len(poses),None,:]
        low=np.minimum(low,points.min(axis=(0,1)));high=np.maximum(high,points.max(axis=(0,1)))
    # Link origins do not include the visible head cap, hand tips or sole mesh.
    padding=np.array([.14,.12,.13])
    low-=padding;high+=padding;low[2]=min(low[2],ground_z-.02)
    return np.stack((low,high))
