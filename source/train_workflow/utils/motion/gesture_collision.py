"""Cartesian gesture preservation under collision, floor and rate constraints.

The reference is an arbitrary robot motion, not a dance-specific pose list.
Hand/wrist targets are expressed relative to torso and head; a continuous
distance weight emphasizes near-head gestures. A collision-repaired trajectory
is the seed, and only independently validated feasible candidates are retained.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import numpy as np
import torch
from source.train_workflow.utils.retarget.upper_body_ik import read_joints, RobotArm
from source.train_workflow.utils.retarget.lower_body_ik import RobotLeg


@dataclass(frozen=True)
class GestureWeights:
    hand_position: float = 8000.
    wrist_position: float = 4000.
    palm_orientation: float = 25.
    elbow_position: float = 150.
    inter_hand: float = 1000.
    foot_position: float = 2000.
    foot_orientation: float = 10.
    foot_velocity: float = 40000.
    foot_acceleration: float = 40000.
    collision: float = 10000000.
    floor: float = 10000000.
    rate: float = 10000000.
    arm_dynamics: float = 100000.
    arm_velocity: float = 2.
    arm_acceleration: float = .05
    joint_acceleration: float = .02


@dataclass(frozen=True)
class ArmMotionBudget:
    """Playback quality budgets, not actuator torque/acceleration guarantees.

    Linear budgets scale with USD arm reach and tighten continuously near the
    head. Derivatives use seconds, so changing sample rate preserves meaning.
    """
    joint_speed_rad_s: float = 6.
    joint_acceleration_rad_s2: float = 60.
    hand_speed_reaches_s: float = 6.
    hand_acceleration_reaches_s2: float = 60.


def temporal_derivatives(points, fps):
    velocity=(points[1:]-points[:-1])*fps
    return velocity,(velocity[1:]-velocity[:-1])*fps


def arm_temporal_terms(q, hand, arm_columns, fps, reach, proximity, budget, peak_weight=.1):
    """Differentiable budgets for actual motion, including IK branch jumps."""
    jv,ja=temporal_derivatives(q[:,arm_columns],fps)
    hv,ha=temporal_derivatives(hand,fps)
    speed=torch.linalg.vector_norm(hv,dim=-1)/reach[None]
    acceleration=torch.linalg.vector_norm(ha,dim=-1)/reach[None]
    near_speed=torch.maximum(proximity[1:],proximity[:-1]).sqrt()
    near_acc=torch.maximum(near_speed[1:],near_speed[:-1])
    ratios=[abs(jv)/budget.joint_speed_rad_s,
            speed*near_speed/budget.hand_speed_reaches_s]
    if len(ja):
        ratios += [abs(ja)/budget.joint_acceleration_rad_s2,
                   acceleration*near_acc/budget.hand_acceleration_reaches_s2]
    # A mean alone dilutes a one-frame spike over long clips and many joints.
    # Retain a peak term so a single route switch cannot hide in that average.
    penalty=sum(torch.relu(x-.95).square().mean()+peak_weight*torch.relu(x-.95).square().amax() for x in ratios)
    energy=(jv.square().mean(),ja.square().mean() if len(ja) else q.sum()*0,
            speed.square().mean(),acceleration.square().mean() if len(ha) else q.sum()*0)
    return penalty,energy,ratios


def validate_arm_motion(model, q, robot_usd, fps, reference=None, budget=ArmMotionBudget()):
    """Check the encoded trajectory with the same reference-relative budget."""
    q=model._tensor(q)
    if fps<=0 or not torch.isfinite(q).all():raise ValueError('Invalid arm trajectory')
    if len(q)<2:return {'passed':True,'maximum_budget_ratio':0.}
    joints=read_joints(robot_usd)
    hands=['lh_hand_base_link','rh_hand_base_link']
    shapes={s.name:s for s in model.shapes}
    centers=model._tensor(np.stack([shapes[n].a for n in hands]))
    reach=model._tensor([RobotArm(joints,s).reach for s in ('left','right')])
    arms=[i for i in model.adjustable if any(s in model.names[i] for s in ('shoulder','elbow','wrist'))]
    with torch.no_grad():
        rr,pp=model.fk(q if reference is None else model._tensor(reference))
        head=relative_points(rr,pp,hands,centers,'head_link')
        distance=torch.linalg.vector_norm(head-model._tensor(shapes['head_link'].a)[None,None],dim=-1)/reach[None]
        proximity=1+3*torch.exp(-(distance/.6).square())
        r,p=model.fk(q);hand=relative_points(r,p,hands,centers,'torso_link')
        _,_,ratios=arm_temporal_terms(q,hand,arms,fps,reach,proximity,budget)
        maximum=max(float(x.max()) for x in ratios)
        v,a=temporal_derivatives(hand,fps)
    return {'passed':maximum<=1.001,'maximum_budget_ratio':maximum,
            'hand_speed_max_m_s':float(torch.linalg.vector_norm(v,dim=-1).max()),
            'hand_acceleration_max_m_s2':float(torch.linalg.vector_norm(a,dim=-1).max()) if len(a) else 0.,
            'budget':asdict(budget),'validation':'torso-relative hand centres; sample derivatives in seconds'}


def relative_points(rotations, positions, links, offsets, anchor):
    """Landmarks in an anatomical anchor frame; invariant to rigid world pose."""
    points=torch.stack([positions[n]+(rotations[n]@offsets[i,:,None]).squeeze(-1)
                        for i,n in enumerate(links)],dim=1)
    return torch.einsum('nij,nfj->nfi',rotations[anchor].transpose(1,2),points-positions[anchor][:,None])


def project_rates(q, columns, step, sweeps=300):
    result=np.asarray(q).copy()
    for _ in range(sweeps):
        for parity in (0,1):
            i=np.arange(parity,len(result)-1,2)
            d=result[np.ix_(i+1,columns)]-result[np.ix_(i,columns)]
            correction=np.sign(d)*np.maximum(abs(d)-step,0)/2
            result[np.ix_(i,columns)]+=correction
            result[np.ix_(i+1,columns)]-=correction
    return result


def refine_gestures(model, reference, seed, robot_usd, fps, root_positions, root_rotations,
                    iterations=3000, weights=GestureWeights(), progress=None,
                    motion_budget=ArmMotionBudget(),polish_iterations=200,repair_only=False):
    """Return the best *feasible* Cartesian fit, never an unchecked last iterate."""
    if fps<=0 or iterations<1 or len(reference)<2:raise ValueError('Invalid gesture trajectory')
    if polish_iterations<0:raise ValueError('Invalid constrained-refinement iteration budget')
    if polish_iterations:
        import clarabel  # Dependency preflight before the coarse solve.
    reference=np.asarray(reference,dtype=np.float32);seed=np.asarray(seed,dtype=np.float32)
    if seed.shape!=reference.shape or not np.isfinite([reference,seed]).all():raise ValueError('Invalid finite reference/seed')
    count=len(reference);ix=model.adjustable
    joints=read_joints(robot_usd);by_name={j.name:j for j in joints if j.axis is not None}
    rw=model._tensor(root_rotations);rp=model._tensor(root_positions)
    hands=['lh_hand_base_link','rh_hand_base_link']
    wrists=['left_wrist_roll_link','right_wrist_roll_link']
    elbows=['left_elbow_link','right_elbow_link']
    feet=['left_ankle_roll_link','right_ankle_roll_link']
    shapes={s.name:s for s in model.shapes}
    centers=model._tensor(np.stack([shapes[n].a for n in hands]))
    # An orientation discrepancy moves the outline of a large hand farther
    # than that of a small one. Use the USD box radius as a Cartesian lever,
    # instead of a clip-tuned palm-angle weight.
    palm_leverage=model._tensor([float(np.dot(shapes[n].b,shapes[n].b)) for n in hands])
    palm_weight=weights.palm_orientation+weights.hand_position*palm_leverage
    zero=model._tensor(np.zeros((2,3)))
    palm=model._tensor(np.stack([RobotArm(joints,s).palm_local for s in ('left','right')]))
    reach=model._tensor([RobotArm(joints,s).reach for s in ('left','right')])
    sole=model._tensor(np.stack([RobotLeg(robot_usd,s).sole for s in ('left','right')]))
    arms=[i for i in ix if any(s in model.names[i] for s in ('shoulder','elbow','wrist'))]
    step=np.array([min(by_name[model.names[i]].velocity,
                       motion_budget.joint_speed_rad_s if i in arms else float('inf'))/fps*.99 for i in ix])
    limit=model._tensor(step)
    ref=model._tensor(reference);start=model._tensor(seed)
    with torch.no_grad():
        rr,pp=model.fk(ref)
        target={anchor:relative_points(rr,pp,hands,centers,anchor) for anchor in ('head_link','torso_link')}
        wrist_target=relative_points(rr,pp,wrists,zero,'torso_link')
        elbow_target=relative_points(rr,pp,elbows,zero,'torso_link')
        palm_target=torch.stack([rr['torso_link'].transpose(1,2)@rr[n]@palm[i] for i,n in enumerate(hands)],dim=1)
        # A USD head-link origin is often at the neck, not the head centre.
        # Use the geometry centroid for proximity; target coordinates still
        # remain in the head frame, with no pose- or clip-specific thresholds.
        head_center=model._tensor(shapes['head_link'].a)
        distance=torch.linalg.vector_norm(target['head_link']-head_center[None,None],dim=-1)/reach[None]
        proximity=1+3*torch.exp(-(distance/.6).square())
        importance=proximity.square()
        near=distance<.7
        sr,sp=model.fk(start)
        foot_target=torch.stack([sp[n] for n in feet],dim=1)
        foot_rot_target=torch.stack([sr[n] for n in feet],dim=1)
        raw_feet=torch.stack([pp[n] for n in feet],dim=1)
        raw_speed=torch.linalg.vector_norm(raw_feet[1:]-raw_feet[:-1],dim=-1).amax(-1)
        raw_acceleration=(torch.linalg.vector_norm(raw_feet[2:]-2*raw_feet[1:-1]+raw_feet[:-2],dim=-1).amax(-1)
                          if count>2 else None)
    def cartesian(r,p):
        hp={a:relative_points(r,p,hands,centers,a) for a in target}
        wp=relative_points(r,p,wrists,zero,'torso_link')
        ep=relative_points(r,p,elbows,zero,'torso_link')
        orientation=torch.stack([r['torso_link'].transpose(1,2)@r[n]@palm[i] for i,n in enumerate(hands)],dim=1)
        # Retain the established coarse feasibility objective. The stronger,
        # geometry-scaled gesture priority is applied only inside the feasible
        # refinement, where it cannot trade away the motion constraints.
        hand=sum((proximity*(hp[a]-target[a]).square().sum(-1)).mean() for a in target)/len(target)
        wrist=(proximity*(wp-wrist_target).square().sum(-1)).mean()
        orient=(proximity*(orientation-palm_target).square().sum((-1,-2))).mean()
        elbow=(ep-elbow_target).square().sum(-1).mean()
        pair=((hp['torso_link'][:,0]-hp['torso_link'][:,1])-(target['torso_link'][:,0]-target['torso_link'][:,1])).square().sum(-1).mean()
        objective=weights.hand_position*hand+weights.wrist_position*wrist+weights.palm_orientation*orient+weights.elbow_position*elbow+weights.inter_hand*pair
        return objective,hp,wp,orientation
    def floor_heights(r,p):
        fr=torch.stack([r[n] for n in feet],1);fp=torch.stack([p[n] for n in feet],1)
        corners=torch.einsum('nfij,fkj->nfki',fr,sole)+fp[:,:,None]
        world=torch.einsum('nij,nfkj->nfki',rw,corners)+rp[:,None,None]
        return world[...,2],fp,fr
    def evaluate(q):
        with torch.no_grad():
            qt=model._tensor(q);r,p=model.fk(qt);obj,hp,wp,ori=cartesian(r,p)
            dynamics,energy,ratios=arm_temporal_terms(qt,hp['torso_link'],arms,fps,reach,proximity,motion_budget)
            # Select on the same temporal objective used during optimization;
            # pose-only selection otherwise rewards rapid route switching.
            obj=obj+weights.arm_velocity*energy[2]+weights.arm_acceleration*energy[3]+weights.joint_acceleration*energy[1]
            temporal_ratio=max(float(x.max()) for x in ratios)
            z,fp,_=floor_heights(r,p);dist=model.distances(r,p);mid=model.distances(*model.fk((qt[:-1]+qt[1:])/2))
            rate_ok=np.max(abs(np.diff(q[:,ix],axis=0))-step)<=1e-5
            angle_ok=bool(np.all(q[:,ix]>=model.lower[ix].numpy()-1e-6) and
                          np.all(q[:,ix]<=model.upper[ix].numpy()+1e-6))
            speed=torch.linalg.vector_norm(fp[1:]-fp[:-1],dim=-1).amax(-1)
            spikes=((speed>.12*30/fps)&(speed>2.5*raw_speed)).sum()
            if count>2:
                acceleration=torch.linalg.vector_norm(fp[2:]-2*fp[1:-1]+fp[:-2],dim=-1).amax(-1)
                spikes+=((acceleration>.15*(30/fps)**2)&(acceleration>2.5*raw_acceleration)).sum()
            kinematic_feasible=float(z.min())>=-.0009 and rate_ok and angle_ok and int(spikes)==0 and temporal_ratio<=1.001
            feasible=min(float(dist.min()),float(mid.min()))>=.0005 and kinematic_feasible
            handerr=torch.linalg.vector_norm(hp['head_link']-target['head_link'],dim=-1)
            wristerr=torch.linalg.vector_norm(wp-wrist_target,dim=-1)
            rel=palm_target.transpose(-1,-2)@ori
            angle=torch.rad2deg(torch.acos(((rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)))
            metrics={'objective':float(obj),'hand_head_error_mean_m':float(handerr.mean()),
                     'near_head_error_mean_m':float(handerr[near].mean()) if near.any() else None,
                     'wrist_error_mean_m':float(wristerr.mean()),'palm_error_mean_deg':float(angle.mean()),
                     'minimum_clearance_m':float(dist.min()),'midpoint_minimum_clearance_m':float(mid.min()),
                     'minimum_sole_height_m':float(z.min()),'foot_spike_count':int(spikes),
                     'arm_motion_budget_max_ratio':temporal_ratio,'arm_joint_speed_max_rad_s':float(abs(temporal_derivatives(qt[:,arms],fps)[0]).max()),
                     'arm_motion_component_ratios':[float(x.max()) for x in ratios],
                     'maximum_joint_step_excess_rad':float(np.max(abs(np.diff(q[:,ix],axis=0))-step)),
                     'angle_ok':angle_ok,'rate_ok':bool(rate_ok),
                     'arm_joint_acceleration_max_rad_s2':float(abs(temporal_derivatives(qt[:,arms],fps)[1]).max()) if count>2 else 0.,
                     'kinematic_feasible':bool(kinematic_feasible),'feasible':bool(feasible)}
            return float(obj),metrics
    best=None;best_score=float('inf');best_iteration=None
    seed_score,seed_metrics=evaluate(seed)
    can_restore=bool(polish_iterations and seed_metrics['kinematic_feasible'])
    if seed_metrics['feasible'] or can_restore:best=seed.copy();best_score=seed_score;best_iteration=-1
    # A previous frame-wise repair may already have changed IK branches. Begin
    # from a time-coherent arm path rather than spending iterations reproducing
    # those jumps. This is only a seed: every result is rechecked for collision.
    if seed_metrics['arm_motion_budget_max_ratio']>2.:
        from scipy.ndimage import gaussian_filter1d
        smooth=seed.copy()
        smooth[:,arms]=gaussian_filter1d(seed[:,arms],sigma=max(1.,fps*.10),axis=0,mode='nearest')
        start=model._tensor(smooth)
    variable=torch.nn.Parameter(start[:,ix].clone());optim=torch.optim.Adam([variable],lr=.004)
    for iteration in range(0 if seed_metrics['feasible'] or can_restore else iterations):
        optim.param_groups[0]['lr']=.004*(.02+.98*(1+math.cos(math.pi*iteration/max(1,iterations-1)))/2)
        optim.zero_grad(set_to_none=True);q=ref.clone();q[:,ix]=variable
        r,p=model.fk(q);task,hp,_,_=cartesian(r,p)
        # First find a coherent route around the body; then tighten isolated
        # derivative peaks without making an initially intersecting seed rigid.
        phase=max(0.,2*iteration/max(1,iterations-1)-1.)
        dynamics,energy,_=arm_temporal_terms(q,hp['torso_link'],arms,fps,reach,proximity,motion_budget,peak_weight=.1*phase*phase*(3-2*phase))
        dist=model.distances(r,p);mid=model.distances(*model.fk((q[:-1]+q[1:])/2))
        collision=torch.relu(.003-dist).square().sum(-1).mean()+torch.relu(.003-mid).square().sum(-1).mean()
        z,fp,fr=floor_heights(r,p)
        floor=torch.relu(.0008-z).square().sum((1,2)).mean()
        rate=torch.relu(abs(variable[1:]-variable[:-1])-limit).square().sum(-1).mean()
        change=variable-ref[:,ix]
        foot_change=fp-foot_target
        foot_velocity=(foot_change[1:]-foot_change[:-1]).square().sum(-1).mean()
        foot_acceleration=((foot_change[2:]-2*foot_change[1:-1]+foot_change[:-2]).square().sum(-1).mean()
                           if count>2 else torch.zeros((),device=q.device))
        loss=task+weights.arm_dynamics*dynamics+weights.arm_velocity*energy[2]+weights.arm_acceleration*energy[3]+weights.joint_acceleration*energy[1]+weights.collision*collision+weights.floor*floor+weights.rate*rate+.2*change.square().sum(-1).mean()+2*(change[1:]-change[:-1]).square().sum(-1).mean()+weights.foot_position*foot_change.square().sum(-1).mean()+weights.foot_orientation*(fr-foot_rot_target).square().sum((-1,-2)).mean()+weights.foot_velocity*foot_velocity+weights.foot_acceleration*foot_acceleration
        loss.backward();optim.step()
        with torch.no_grad():variable.clamp_(model.lower[ix],model.upper[ix])
        if iteration%100==0 or iteration==iterations-1:
            candidate=reference.copy();candidate[:,ix]=variable.detach().numpy()
            candidate=project_rates(candidate,ix,step)
            score,metrics=evaluate(candidate)
            if metrics['feasible'] and score<best_score:
                best,best_score,best_iteration=candidate.copy(),score,iteration
            print('GESTURE',iteration,metrics,flush=True)
            if progress:progress((.7 if polish_iterations else 1.)*(iteration+1)/iterations,metrics)
    if best is None:
        error=RuntimeError('No collision/rate/floor/arm-motion-feasible gesture candidate')
        error.candidate=candidate
        error.metrics=metrics
        raise error
    refinement=None
    if polish_iterations:
        from .constrained_trajectory import refine_feasible_windows
        def sample(q,selector=slice(None)):
            with torch.no_grad():
                qt=model._tensor(q);r,p=model.fk(qt)
                local_count=len(q);local_importance=importance[selector]
                hp={a:relative_points(r,p,hands,centers,a) for a in target}
                wp=relative_points(r,p,wrists,zero,'torso_link')
                ori=torch.stack([r['torso_link'].transpose(1,2)@r[n]@palm[i] for i,n in enumerate(hands)],dim=1)
                ep=relative_points(r,p,elbows,zero,'torso_link')
                blocks=[(hp[a]-target[a][selector])*torch.sqrt(weights.hand_position*local_importance[:,:,None]/(2*len(target))) for a in target]
                blocks += [(wp-wrist_target[selector])*torch.sqrt(weights.wrist_position*local_importance[:,:,None]/2),
                           (ori-palm_target[selector])*torch.sqrt(palm_weight[None,:,None,None]*local_importance[:,:,None,None]/2),
                           (ep-elbow_target[selector])*math.sqrt(weights.elbow_position/2),
                           ((hp['torso_link'][:,0]-hp['torso_link'][:,1])-(target['torso_link'][selector,0]-target['torso_link'][selector,1]))*math.sqrt(weights.inter_hand)]
                residual=torch.cat([b.reshape(local_count,-1) for b in blocks],dim=-1)
                distances=torch.cat([model.distances(r,p),*[model.distances(*model.fk(qt[:-1]*(1-a)+qt[1:]*a)) for a in (.25,.5,.75)]])
                return residual.numpy().astype(float),hp['torso_link'].numpy().astype(float),distances.numpy().astype(float)
        near_speed=torch.maximum(proximity[1:],proximity[:-1]).sqrt().numpy()
        speed_caps=reach.numpy()[None]*motion_budget.hand_speed_reaches_s/near_speed
        acceleration_caps=reach.numpy()[None]*motion_budget.hand_acceleration_reaches_s2/np.maximum(near_speed[1:],near_speed[:-1])
        arm_step=np.array([step[list(ix).index(i)] for i in arms])
        best,refinement=refine_feasible_windows(best,arms,model.lower[arms].numpy(),model.upper[arms].numpy(),
            arm_step,np.full(len(arms),motion_budget.joint_acceleration_rad_s2*.99/fps**2),fps,
            speed_caps,acceleration_caps,sample,lambda q:evaluate(q)[1]['kinematic_feasible'],iterations=polish_iterations,
            progress=lambda f,m:progress((0. if seed_metrics['feasible'] else .7)+(1. if seed_metrics['feasible'] else .3)*f,m) if progress else None,
            collision_fractions=(.25,.5,.75),repair_only=repair_only)
    _,after=evaluate(best)
    if not after['feasible']:raise RuntimeError('Articulated-hand result failed final collision or motion validation')
    report={'method':'anatomical-cartesian-articulated-hand-gesture-collision-v4','weights':asdict(weights),'arm_motion_budget':asdict(motion_budget),
            'hand_geometry_mode':model.hand_geometry_mode,'collision_restoration_seed':can_restore and not seed_metrics['feasible'],
            'fidelity_weighting':'squared anatomical proximity; USD hand-box orientation leverage','palm_weights':palm_weight.tolist(),
            'best_iteration':best_iteration,'iterations':iterations,'before':seed_metrics,'after':after,
            'near_head_samples':int(near.sum()),'frames':count,'fps':fps,'constrained_refinement':refinement}
    return best,report
