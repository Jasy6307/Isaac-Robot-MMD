"""Sparse sequential quadratic refinement of an already feasible trajectory.

Joint derivatives are exact linear constraints. Cartesian derivatives and
collision distances are linearized together; backtracking checks the actual
nonlinear trajectory before accepting any step. Nonselected channels are fixed.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def difference_matrix(count, columns, order=1):
    coefficients=(-1.,1.) if order==1 else (1.,-2.,1.)
    if order not in (1,2):raise ValueError('Expected first or second difference')
    if count<=order:return sparse.csr_matrix((0,count*columns))
    return sparse.kron(sparse.diags(coefficients,range(order+1),shape=(max(0,count-order),count)),
                       sparse.eye(columns),format='csr')


def local_rows(blocks, sample_indices, component_indices, count, columns,fractions=(.5,)):
    """Per-sample derivatives, including interpolation chain-rule factors."""
    sample_indices=np.asarray(sample_indices);component_indices=np.asarray(component_indices)
    values=blocks[sample_indices,component_indices]
    rows=np.repeat(np.arange(len(values)),columns)
    frames=sample_indices.copy()
    midpoint=sample_indices>=count
    block=(sample_indices[midpoint]-count)//(count-1)
    frames[midpoint]=(sample_indices[midpoint]-count)%(count-1)
    alpha=np.zeros(len(values));alpha[midpoint]=np.asarray(fractions)[block]
    cols=(frames[:,None]*columns+np.arange(columns)).ravel()
    scale=1-alpha
    matrix=sparse.coo_matrix(((values*scale[:,None]).ravel(),(rows,cols)),shape=(len(values),count*columns)).tocsr()
    if midpoint.any():
        take=np.flatnonzero(midpoint)
        matrix+=sparse.coo_matrix(((values[take]*alpha[take,None]).ravel(),
            (np.repeat(take,columns),((frames[take,None]+1)*columns+np.arange(columns)).ravel())),shape=matrix.shape).tocsr()
    matrix.eliminate_zeros()
    return matrix


def soc_rows(jacobian, vectors, caps):
    """A*x+s=b with s=(cap, vector+J*x) in a 4D Lorentz cone."""
    vectors=np.asarray(vectors).reshape(-1,3);caps=np.asarray(caps).ravel()
    entries=jacobian.tocoo()
    rows=(entries.row//3)*4+entries.row%3+1
    matrix=sparse.coo_matrix((-entries.data,(rows,entries.col)),shape=(len(vectors)*4,jacobian.shape[1])).tocsc()
    rhs=np.column_stack([caps,vectors]).ravel()
    return matrix,rhs


def refine_feasible_trajectory(seed, columns, lower, upper, joint_step, joint_acceleration_step,
                               fps, hand_speed_caps, hand_acceleration_caps, sample, feasible,
                               iterations=100, progress=None,fixed_endpoints=(0,0),
                               collision_fractions=(.5,),restore_collisions=False):
    """Minimize weighted pose residuals inside unchanged nonlinear constraints.

    sample(q) returns per-frame weighted residuals, torso-relative hand points,
    and frame/interpolation distances in collision_fractions order. feasible(q) is an
    independent nonlinear gate. A failed QP or line search returns the last
    accepted trajectory; it cannot replace it with an unchecked candidate.
    """
    import clarabel
    current=np.asarray(seed,dtype=np.float32).copy();columns=np.asarray(columns,dtype=int)
    joint_step=np.asarray(joint_step,dtype=float)
    joint_acceleration_step=np.asarray(joint_acceleration_step,dtype=float)
    count=len(current);width=len(columns);size=count*width
    if count<2 or not width or not feasible(current):raise ValueError('Refinement requires a feasible seed')
    d1=difference_matrix(count,width);d2=difference_matrix(count,width,2)
    hd1=difference_matrix(count,6);hd2=difference_matrix(count,6,2)
    identity=sparse.eye(size,format='csr')
    left,right=fixed_endpoints
    free_frame=np.ones(count,dtype=bool)
    if left:free_frame[:left]=False
    if right:free_frame[-right:]=False
    free_columns=sparse.diags(np.repeat(free_frame,width).astype(float),format='csr')
    residual,hand,distances=sample(current)
    violation=max(0.,.0005-float(distances.min()))
    collision_merit=float(np.maximum(.0005-distances,0).ravel()@np.maximum(.0005-distances,0).ravel())
    score=float(np.sum(residual**2)/count);initial_score=score
    trust=.08;history=[];accepted=0
    for iteration in range(iterations):
        # Simultaneous perturbation of one joint column in every frame exposes
        # block-diagonal local derivatives; midpoint rows use the chain rule.
        rjac=[];hjac=[];cjac=[];epsilon=.0005
        for column in columns:
            plus=current.copy();minus=current.copy()
            plus[:,column]+=epsilon;minus[:,column]-=epsilon
            a=sample(plus);b=sample(minus)
            rjac.append((a[0]-b[0])/(2*epsilon))
            hjac.append((a[1]-b[1])/(2*epsilon))
            cjac.append((a[2]-b[2])/(2*epsilon))
        rjac=np.stack(rjac,-1);hjac=np.stack(hjac,-1);cjac=np.stack(cjac,-1)
        jr=sparse.block_diag(list(rjac.reshape(count,-1,width)),format='csr')
        jh=sparse.block_diag(list(hjac.reshape(count,6,width)),format='csr')
        jv=hd1@jh*fps;ja=hd2@jh*fps**2
        velocity=np.diff(hand,axis=0)*fps
        acceleration=np.diff(hand,n=2,axis=0)*fps**2
        selected=np.where(distances<.15)
        ac=local_rows(cjac,*selected,count,width,collision_fractions)
        # The nonlinear acceptance clearance remains 0.5 mm; linear subproblems
        # retain an extra small reserve instead of relaxing that requirement.
        mutable=np.asarray(abs(ac@free_columns).sum(axis=1)).ravel()>1e-7
        collision_target=np.where(mutable,.0015,np.minimum(.0015,distances[selected])-1e-7)
        bc=collision_target-distances[selected]
        x=current[:,columns].astype(float).ravel()
        limits1=np.tile(joint_step*.999,count-1)
        limits2=np.tile(joint_acceleration_step*.999,max(0,count-2))
        # Frozen boundary derivatives are already independently validated. They
        # cannot satisfy an additional optimization reserve by moving, so keep
        # their actual values; never change the final acceptance budgets.
        fixed_velocity=np.repeat(~(free_frame[1:]|free_frame[:-1]),width)
        limits1[fixed_velocity]=np.maximum(limits1[fixed_velocity],abs(d1@x)[fixed_velocity]+1e-8)
        fixed_acceleration=np.repeat(~(free_frame[2:]|free_frame[1:-1]|free_frame[:-2]),width)
        limits2[fixed_acceleration]=np.maximum(limits2[fixed_acceleration],abs(d2@x)[fixed_acceleration]+1e-8)
        a=sparse.vstack([identity,d1,d2,ac],format='csc')
        lower_delta=np.maximum(np.tile(lower,count)-x,-trust)
        upper_delta=np.minimum(np.tile(upper,count)-x,trust)
        if left:lower_delta[:left*width]=upper_delta[:left*width]=0.
        if right:lower_delta[-right*width:]=upper_delta[-right*width:]=0.
        lo=np.concatenate([lower_delta,-limits1-d1@x,-limits2-d2@x,bc])
        hi=np.concatenate([upper_delta,limits1-d1@x,limits2-d2@x,
                           np.full(ac.shape[0],np.inf)])
        p=sparse.triu(2*(jr.T@jr)/count+.1*identity,format='csc')
        gradient=2*np.asarray(jr.T@residual.ravel()).ravel()/count
        restoration=restore_collisions and violation>1e-8
        slack_count=ac.shape[0] if restoration else 0
        if slack_count:
            # Elastic geometric constraints make restoration possible even when
            # the full new clearance cannot be reached inside one trust step.
            # Motion constraints remain hard; slacks are never an output waiver.
            slack=sparse.vstack([sparse.csr_matrix((a.shape[0]-slack_count,slack_count)),sparse.eye(slack_count)],format='csc')
            a=sparse.hstack([a,slack],format='csc')
            a=sparse.vstack([a,sparse.hstack([sparse.csr_matrix((slack_count,size)),sparse.eye(slack_count)])],format='csc')
            lo=np.r_[lo,np.zeros(slack_count)];hi=np.r_[hi,np.full(slack_count,np.inf)]
            p=sparse.block_diag([p,sparse.csc_matrix((slack_count,slack_count))],format='csc')
            gradient=np.r_[np.zeros(size),np.full(slack_count,1e4/count)]
        finite_hi=np.isfinite(hi);finite_lo=np.isfinite(lo)
        linear=sparse.vstack([a[finite_hi],-a[finite_lo]],format='csc')
        rhs=np.r_[hi[finite_hi],-lo[finite_lo]]
        def caps_with_fixed_reserve(jac,base,caps):
            caps=np.asarray(caps).ravel()*.999
            movable=np.asarray(abs(jac@free_columns).sum(axis=1)).reshape(-1,3).sum(-1)>1e-7
            caps[~movable]=np.maximum(caps[~movable],np.linalg.norm(base.reshape(-1,3),axis=-1)[~movable]+1e-8)
            return caps
        av,bv=soc_rows(jv,velocity,caps_with_fixed_reserve(jv,velocity,hand_speed_caps))
        aa,ba=soc_rows(ja,acceleration,caps_with_fixed_reserve(ja,acceleration,hand_acceleration_caps))
        if slack_count:
            av=sparse.hstack([av,sparse.csr_matrix((av.shape[0],slack_count))],format='csc')
            aa=sparse.hstack([aa,sparse.csr_matrix((aa.shape[0],slack_count))],format='csc')
        a=sparse.vstack([linear,av,aa],format='csc');rhs=np.r_[rhs,bv,ba]
        cones=[clarabel.NonnegativeConeT(linear.shape[0])]+[clarabel.SecondOrderConeT(4) for _ in range((av.shape[0]+aa.shape[0])//4)]
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=80
        settings.tol_gap_abs=1e-6;settings.tol_gap_rel=1e-6;settings.tol_feas=1e-7;settings.max_threads=4
        solution=clarabel.DefaultSolver(p,gradient,a,rhs,cones,settings).solve()
        found=False;alpha=1.;new_score=score
        if str(solution.status) in ('Solved','AlmostSolved') and solution.x is not None and np.isfinite(solution.x).all():
            direction=np.asarray(solution.x[:size]).reshape(count,width)
            for _ in range(16):
                candidate=current.copy();candidate[:,columns]+=alpha*direction
                if left:candidate[:left]=current[:left]
                if right:candidate[-right:]=current[-right:]
                values=sample(candidate)
                new_score=float(np.sum(values[0]**2)/count)
                new_violation=max(0.,.0005-float(values[2].min()))
                excess=np.maximum(.0005-values[2],0).ravel();new_merit=float(excess@excess)
                improving=(new_score<score-1e-6)
                if restore_collisions:
                    improving=(new_merit<=collision_merit+1e-14 and
                               (improving or new_merit<collision_merit-1e-14))
                if improving and feasible(candidate):
                    current=candidate;residual,hand,distances=values
                    score=new_score;violation=new_violation;collision_merit=new_merit;accepted+=1;found=True
                    break
                alpha*=.5
        entry={'iteration':iteration,'objective':score,'accepted':found,'step_fraction':alpha if found else 0.,
               'trust_rad':trust,'qp_status':str(solution.status),'cartesian_norm_cones':len(cones)-1,
               'subframe_clearance_deficit_m':violation,'collision_restoration_merit':collision_merit}
        history.append(entry)
        print('GESTURE_SQP',entry,flush=True)
        if progress:progress((iteration+1)/iterations,entry)
        if found:
            trust=min(.12,trust*1.2) if alpha>=.5 else max(.002,trust*.7)
        else:
            trust*=.5
            if trust<.0002:break
    if not feasible(current):raise RuntimeError('Feasible refinement lost its validated trajectory')
    return current,{'method':'sparse-feasible-trajectory-socp-v1','initial_objective':initial_score,
                    'final_objective':score,'accepted_steps':accepted,'iterations':len(history),'history':history}


def refine_feasible_windows(seed,columns,lower,upper,joint_step,joint_acceleration_step,fps,
                           hand_speed_caps,hand_acceleration_caps,sample,feasible,iterations=100,progress=None,
                           collision_fractions=(.5,),repair_only=False):
    """Overlapping, time-scaled blocks; every splice passes the full-motion gate."""
    current=np.asarray(seed).copy();count=len(current)
    length=min(count,max(12,round(2*fps)));stride=max(1,length//2)
    starts=list(range(0,max(1,count-length+1),stride))
    if starts[-1]!=count-length:starts.append(count-length)
    if repair_only:
        distances=sample(current,slice(None))[2]
        rows=np.where(distances.min(axis=1)<.0005-1e-7)[0]
        frames=np.unique(np.where(rows<count,rows,(rows-count)%(count-1)))
        starts=[]
        for frame in frames:
            if starts and starts[-1]+2<=frame<starts[-1]+length-2:continue
            starts.append(int(np.clip(frame-length//2,0,count-length)))
        starts=sorted(set(starts))
    windows=[(lo,lo+length) for lo in starts]
    schedules=[windows,list(reversed(windows))] if len(windows)>1 else [windows]
    inner_iterations=max(20,min(iterations,40)) if repair_only else (min(20,max(6,iterations//10)) if len(windows)>1 else iterations)
    before=float(np.sum(sample(current,slice(None))[0]**2)/count);reports=[]
    def improve_window(lo,hi,steps,sweep):
        def local_sample(q):return sample(q,slice(lo,hi))
        def local_feasible(q):
            full=current.copy();full[lo:hi]=q
            return feasible(full)
        result,report=refine_feasible_trajectory(current[lo:hi],columns,lower,upper,joint_step,joint_acceleration_step,
            fps,hand_speed_caps[lo:hi-1],hand_acceleration_caps[lo:hi-2],local_sample,local_feasible,
            iterations=steps,fixed_endpoints=(2 if lo else 0,2 if hi<count else 0),
            collision_fractions=collision_fractions,restore_collisions=True)
        current[lo:hi]=result
        report['frames']=[lo,hi];report['sweep']=sweep;reports.append(report)
        if progress:progress(min(1.,len(reports)/max(1,sum(map(len,schedules)))),report)
        print('GESTURE_WINDOW',sweep,lo,hi,'objective',report['initial_objective'],report['final_objective'],flush=True)
    for sweep,schedule in enumerate(schedules):
        for lo,hi in schedule:improve_window(lo,hi,inner_iterations,sweep)
    # Restore remaining geometric violations by measured state, including small
    # numerical residuals, rather than accepting them or naming special frames.
    recovery_passes=0
    for recovery in range(2):
        distances=sample(current,slice(None))[2]
        rows=np.where(distances.min(axis=1)<.0005-1e-7)[0]
        if not len(rows):break
        frames=np.unique(np.where(rows<count,rows,(rows-count)%(count-1)))
        repair_starts=[]
        for frame in frames:
            if repair_starts and repair_starts[-1]+2<=frame<repair_starts[-1]+length-2:continue
            repair_starts.append(int(np.clip(frame-length//2,0,count-length)))
        for lo in sorted(set(repair_starts)):
            improve_window(lo,lo+length,max(20,inner_iterations),len(schedules)+recovery)
        recovery_passes+=1
    if not feasible(current):raise RuntimeError('Temporal-window splice failed full-motion validation')
    if float(sample(current,slice(None))[2].min())<.0005-1e-7:
        error=RuntimeError('Temporal-window refinement did not clear every interpolation sample')
        error.candidate=current
        error.metrics={'minimum_clearance_m':float(sample(current,slice(None))[2].min()),'completed_windows':len(reports)}
        raise error
    return current,{'method':'overlapping-feasible-trajectory-socp-v1','window_seconds':length/fps,'sweeps':len(schedules),'recovery_passes':recovery_passes,
        'initial_objective':before,'final_objective':float(np.sum(sample(current,slice(None))[0]**2)/count),
        'accepted_steps':sum(r['accepted_steps'] for r in reports),'windows':reports,'collision_fractions':list(collision_fractions)}
