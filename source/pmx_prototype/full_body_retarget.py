"""Anatomical trunk-axis alignment and world-space neutral head calibration."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares


def head_calibration(rotations,ids,frames):
    root=R.from_matrix(rotations[frames,ids['Hips']]).as_euler('ZYX')[:,0]
    head=R.from_euler('Z',-root[:,None]).as_matrix()@rotations[frames,ids['Head']]
    return R.from_matrix(head).mean().inv().as_matrix()


def retarget_upper(model,positions,rotations,ids,seed,head_offset):
    q=seed.copy();root=rotations[ids['Hips']]
    direction=model.origins['neck_yaw']-model.origins['waist_pitch']
    correction=R.align_vectors([[0.,0.,1.]],[(direction/np.linalg.norm(direction))])[0].as_matrix()
    trunk=rotations[ids['Chest4']]@correction
    q[:3]=R.from_matrix(root.T@trunk).as_euler('ZYX')
    desired_head=rotations[ids['Head']]@head_offset
    relative=R.from_matrix(trunk.T@desired_head).as_euler('ZYX')
    q[3:5]=relative[[0,2]]
    q=np.clip(q,model.limits[:,0]+.005,model.limits[:,1]-.005)
    for side in ['left','right']:
        cap=side.title();names=[side+'_'+s for s in ['shoulder_yaw','elbow_flex','hand']]
        joints=[model.names.index(side+'_'+s+'_joint') for s in ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_flex']]
        source=[ids[cap+s] for s in ['Shoulder','Elbow','Wrist']]
        targets=[positions[b]-positions[a] for a,b in zip(source[:-1],source[1:])]
        targets=[v/np.linalg.norm(v) for v in targets]
        arm_rest=model.origins[names[1]]-model.origins[names[0]];arm_rest/=np.linalg.norm(arm_rest)
        align=R.align_vectors([[1. if side=='left' else -1.,0,0]],[arm_rest])[0].as_matrix()
        target_rotation=rotations[source[0]]@align
        def error(x):
            temp=q.copy();temp[joints]=x;p,r=model.fk(temp,np.zeros(3),root)
            result=[]
            for a,b,target in zip(names[:-1],names[1:],targets):
                v=p[b]-p[a];result.extend(3*(v/np.linalg.norm(v)-target))
            result.extend((.8*(r[names[0]]-target_rotation)).ravel())
            return result
        fit=least_squares(error,q[joints],bounds=(model.limits[joints,0]+.005,model.limits[joints,1]-.005),max_nfev=18)
        q[joints]=fit.x
    return q
