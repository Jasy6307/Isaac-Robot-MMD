"""USD-authored position/rate limits including coupled (mimic) joints.

PhysX relation: follower + gearing * leader + offset = 0. Angular USD
limits and offsets use degrees; H5 trajectories use radians. Passive-joint
speed bounds follow from this relation rather than an invented motor limit.
"""
from pathlib import Path
import numpy as np
from source.train_workflow.utils.retarget.upper_body_ik import read_joints


def read_constraints(usd_path, names):
    from pxr import Usd
    joints={j.name:j for j in read_joints(usd_path) if j.axis is not None}
    stage=Usd.Stage.Open(str(usd_path));mimic={}
    for prim in stage.Traverse():
        name=prim.GetName()
        if name not in names:continue
        for rel in prim.GetRelationships():
            if not (rel.GetName().startswith('physxMimicJoint:') and rel.GetName().endswith(':referenceJoint')):continue
            targets=rel.GetTargets()
            if not targets:continue
            prefix=rel.GetName().rsplit(':',1)[0]
            gearing=prim.GetAttribute(prefix+':gearing').Get()
            offset=prim.GetAttribute(prefix+':offset').Get()
            mimic[name]=(targets[0].name,-float(gearing),-np.radians(float(offset or 0)))
    groups={};mapping={}
    def resolve(name,seen=()):
        if name in seen:raise ValueError('Cyclic mimic relation')
        if name not in names or name not in joints:raise ValueError(f'Missing USD joint: {name}')
        if name not in mimic:return name,1.,0.
        leader,a,b=mimic[name];root,c,d=resolve(leader,seen+(name,))
        return root,a*c,a*d+b
    for name in names:
        root,a,b=resolve(name);mapping[name]=(root,a,b)
        groups.setdefault(root,[]).append((name,a,b))
    constraints={}
    for root,members in groups.items():
        lo,hi,speed=-np.inf,np.inf,np.inf
        for name,a,b in members:
            j=joints[name]
            if abs(a)<1e-12:
                if not j.lower<=b<=j.upper:raise ValueError('Infeasible fixed follower')
                continue
            ends=sorted(((j.lower-b)/a,(j.upper-b)/a))
            lo=max(lo,ends[0]);hi=min(hi,ends[1]);speed=min(speed,j.velocity/abs(a))
        if lo>hi or not np.isfinite([lo,hi,speed]).all() or speed<=0:
            raise ValueError(f'Incomplete/infeasible authored constraints: {root}')
        constraints[root]=(lo,hi,speed)
    return joints,mapping,groups,constraints


def validate_joint_positions(q,names,fps,usd_path):
    q=np.asarray(q,dtype=float)
    if (q.ndim!=2 or not len(q) or q.shape[1]!=len(names) or
        not np.isfinite(q).all() or not np.isfinite(fps) or fps<=0):
        raise ValueError('Invalid finite joint trajectory or FPS')
    joints,mapping,groups,constraints=read_constraints(usd_path,names)
    report={'joint_count':len(names),'independent_joints':len(groups),'derived_mimic_speed_limits':0,
            'angle_violations':0,'velocity_violations':0,'mimic_violations':0,'details':[]}
    for col,name in enumerate(names):
        root,a,b=mapping[name];speed=min(joints[name].velocity,abs(a)*constraints[root][2])
        derived=not np.isfinite(joints[name].velocity)
        report['derived_mimic_speed_limits']+=int(derived)
        value=q[:,col];angle=int(np.sum((value<joints[name].lower-1e-6)|(value>joints[name].upper+1e-6)))
        rate=int(np.sum(abs(np.diff(value))*fps>speed+1e-4))
        coupling=int(np.sum(abs(value-(a*q[:,names.index(root)]+b))>1e-6)) if root!=name else 0
        report['angle_violations']+=angle;report['velocity_violations']+=rate;report['mimic_violations']+=coupling
        if angle or rate or coupling:report['details'].append(dict(joint=name,angle=angle,velocity=rate,mimic=coupling))
    report['passed']=not any(report[k] for k in ('angle_violations','velocity_violations','mimic_violations'))
    return report


def constrain_joint_positions(q,names,fps,usd_path):
    if not np.isfinite(fps) or fps<=0:raise ValueError('Invalid FPS')
    q=np.asarray(q,dtype=float)
    if q.ndim!=2 or q.shape[1]!=len(names) or not np.isfinite(q).all():raise ValueError('Invalid joint trajectory')
    before=validate_joint_positions(q,names,fps,usd_path)
    if before['passed']:return q.copy(),dict(before=before,after=before,changed_joints=[])
    _,_,groups,constraints=read_constraints(usd_path,names);result=q.copy()
    for root,members in groups.items():
        cols=[names.index(n) for n,a,b in members];coeff=np.array([a for n,a,b in members]);offset=np.array([b for n,a,b in members])
        lo,hi,speed=constraints[root]
        # Least-squares flexion target respects all coupled source joints.
        value=((q[:,cols]-offset)*coeff).sum(1)/np.dot(coeff,coeff)
        current=result[:,names.index(root)]
        valid=(current.min()>=lo-1e-6 and current.max()<=hi+1e-6 and
               (len(current)<2 or np.max(abs(np.diff(current)))*fps<=speed+1e-4) and
               np.max(abs(q[:,cols]-(current[:,None]*coeff+offset)))<=1e-6)
        if valid:continue
        value=np.clip(value,lo,hi);step=speed/fps*.999
        for _ in range(1000):
            for parity in (0,1):
                i=np.arange(parity,len(value)-1,2);d=value[i+1]-value[i]
                e=np.sign(d)*np.maximum(abs(d)-step,0)/2
                value[i]+=e;value[i+1]-=e
            if len(value)<2 or np.max(abs(np.diff(value)))<=step+1e-10:break
        # Final causal projection removes any residual solver tolerance.
        for i in range(1,len(value)):value[i]=np.clip(value[i],value[i-1]-step,value[i-1]+step)
        result[:,cols]=value[:,None]*coeff+offset
    after=validate_joint_positions(result,names,fps,usd_path)
    if not after['passed']:raise RuntimeError(f'USD joint projection failed: {after}')
    return result,dict(before=before,after=after,changed_joints=[n for i,n in enumerate(names) if np.max(abs(result[:,i]-q[:,i]))>1e-7])
