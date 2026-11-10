"""Reference-aware motion candidates, bounded rehearsal feedback, auxiliary ranking.

Foot-link kinematics are diagnostic proxies, not measured sole slip/friction.
No new motion label, reward, action filter or reference edit is introduced.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

PROTOCOL="reference-motion-v1"
CHATTER_RAD=.035
EXCESS_SPEED=.15
PIVOT_RADIUS_M=.18
CRITERIA=dict(chatter_rad=CHATTER_RAD,excess_speed_m_s=EXCESS_SPEED,pivot_radius_m=PIVOT_RADIUS_M,
    base_velocity_tolerance_m_s=.08,missing_reference_contact_fraction=.15,window_s=2.,stride_s=.5,
    jump_margin_s=.3,chatter_actual_contact_fraction=.9,chatter_reference_support_fraction=.2,
    candidate_sampling_share=.1,priority_bin_fraction=.2)


def band_rms(values,hz):
    x=np.asarray(values,float)
    if x.ndim==1:x=x[:,None]
    n=len(x);t=np.arange(n);window=np.hanning(n)
    trend=np.column_stack((t,np.ones(n)))@np.linalg.lstsq(np.column_stack((t,np.ones(n))),x,rcond=None)[0]
    fft=np.fft.rfft((x-trend)*window[:,None],axis=0)
    freq=np.fft.rfftfreq(n,1/hz);weight=np.full(len(freq),2.);weight[0]=1.
    if n%2==0:weight[-1]=1.
    power=np.abs(fft)**2*weight[:,None]/(n*np.sum(window**2))
    return np.sqrt(power[(freq>=4)&(freq<=min(12,.45*hz))].sum(0))


def assess_motion(seconds,joint_error,command_residual,actual_feet,reference_feet,reference_yaw,
                  expected,contacts,joint_sides,schedules,*,duration,hz):
    t=np.asarray(seconds,float);n=len(t)
    command_observed=command_residual is not None
    if not command_observed:command_residual=np.zeros_like(joint_error)
    result=dict(protocol=PROTOCOL,complete=False,score=None,assessed_windows=0,candidates=[],windows=[],
        interpretation="Foot-origin motion/angle-error proxies; candidates require visual review")
    result["command_observed"]=command_observed
    result["contact_observed"]=True
    result["criteria"]=dict(CRITERIA)
    arrays=[np.asarray(a,float) for a in (joint_error,command_residual,actual_feet,reference_feet,reference_yaw,expected,contacts)]
    if n<3 or any(len(a)!=n or not np.isfinite(a).all() for a in arrays) or not np.isfinite(t).all():
        result["error"]="missing_or_nonfinite_motion_trace";return result
    if np.any(np.abs(np.diff(t)-1/hz)>1.e-4):result["error"]="motion_clock_gap";return result
    error,command,feet,ref,yaw,expected,contacts=arrays
    if error.shape!=command.shape or error.shape[1]!=len(joint_sides) or feet.shape!=(n,2,3) or ref.shape!=feet.shape or yaw.shape!=(n,2) or expected.shape!=(n,2) or contacts.shape!=(n,2):
        result["error"]="motion_shape_mismatch";return result
    result["complete"]=bool(t[0]<=1/hz+1.e-5 and t[-1]>=duration-1/hz-1.e-5)
    velocity=np.gradient(feet[:,:,:2],1/hz,axis=0);ref_velocity=np.gradient(ref[:,:,:2],1/hz,axis=0)
    # One control step of timing tolerance for held targets and physical lag.
    padded=np.pad(ref_velocity,((1,1),(0,0),(0,0)),mode="edge")
    deviation=np.min(np.stack([np.linalg.norm(velocity-padded[i:i+n],axis=-1) for i in range(3)]),axis=0)
    yaw_rate=np.abs(np.gradient(np.unwrap(yaw,axis=0),1/hz,axis=0))
    excess=np.maximum(deviation-(.08+PIVOT_RADIUS_M*yaw_rate),0.)
    protected=np.zeros(n,bool)
    for schedule in schedules:protected|=(t>=schedule[0][0]-.3-1.e-6)&(t<=schedule[-1][1]+.3+1.e-6)
    supported=(expected>.5)&(contacts>.5)
    sides=np.asarray(joint_sides)
    for start in np.arange(math.ceil(t[0]*2)/2,max(t[0],t[-1]-1.+1.e-5),.5):
        mask=(t>=start-1.e-5)&(t<start+2.-1.e-5)
        if mask.sum()<max(20,round(hz)) or protected[mask].any():continue
        row=dict(start_s=float(start),end_s=float(min(start+2,t[-1])),assessed=False,
                 chatter_rad=0.,command_chatter_rad=0. if command_observed else None,excess_speed_m_s=0.,missing_contact_fraction=0.)
        fractions=supported[mask].mean(0)
        for side in range(2):
            selected=sides==side
            reference_supported=expected[mask,side]>.5
            if reference_supported.mean()>=.65:
                missing=float((contacts[mask,side][reference_supported]<.5).mean())
                row["missing_contact_fraction"]=max(row["missing_contact_fraction"],missing)
                row["assessed"]=True
            if (contacts[mask,side]>.5).mean()>=.9 and reference_supported.mean()>=.2 and selected.any():
                row["chatter_rad"]=max(row["chatter_rad"],float(np.sqrt(np.mean(band_rms(error[mask][:,selected],hz)**2))))
                if command_observed:row["command_chatter_rad"]=max(row["command_chatter_rad"],float(np.sqrt(np.mean(band_rms(command[mask][:,selected],hz)**2))))
                row["assessed"]=True
            if fractions[side]>=.65:
                selected_steps=mask&supported[:,side]
                row["excess_speed_m_s"]=max(row["excess_speed_m_s"],float(excess[selected_steps,side].mean()))
                row["assessed"]=True
        if not row["assessed"]:continue
        row["severity"]=max(row["chatter_rad"]/CHATTER_RAD,row["excess_speed_m_s"]/EXCESS_SPEED,row["missing_contact_fraction"]/.15)
        result["windows"].append(row)
        if row["severity"]>1.:
            reasons=[]
            if row["chatter_rad"]>CHATTER_RAD:reasons.append("extra_joint_oscillation")
            if row["excess_speed_m_s"]>EXCESS_SPEED:reasons.append("excess_foot_origin_motion")
            if row["missing_contact_fraction"]>.15:reasons.append("intermittent_reference_support")
            result["candidates"].append(dict(row,reasons=reasons))
    result["assessed_windows"]=len(result["windows"])
    result["score"]=max((w["severity"] for w in result["windows"]),default=0.)
    result["status"]="assessed" if result["windows"] else "no_positive_support_evidence"
    return result


class KinematicMotionAnalyzer:
    def __init__(self,env,report,buf):
        import torch
        from scipy.spatial.transform import Rotation
        from source.train_workflow.utils.motion.self_collision import G1CollisionModel
        robot=env.scene["robot"]
        self.names=list(robot.joint_names);self.report=report;self.hz=1/env.step_dt
        self.joints=[i for i,n in enumerate(self.names) if any(p in n for p in ("hip_yaw","hip_roll","ankle_pitch","ankle_roll"))]
        self.sides=[0 if self.names[i].startswith("left_") else 1 for i in self.joints]
        self.model=G1CollisionModel(Path(env.cfg.scene.robot.spawn.usd_path),self.names,"cpu")
        steps=torch.arange(buf.num_steps+1,device=env.device)
        self.reference_q=buf.q_ref_abs(steps).detach().cpu().numpy()
        default=robot.data.default_root_state.torch[0,:7].detach().cpu().numpy()
        delta=buf.root_quat_wxyz(steps).detach().cpu().numpy()
        rotation=Rotation.from_quat(delta[:,[1,2,3,0]])*Rotation.from_quat(default[3:7])
        root=np.column_stack((default[:3]+buf.root_pos_delta(steps).detach().cpu().numpy(),rotation.as_quat()))
        self.reference_feet,self.reference_yaw=self.feet(self.reference_q,root)

    def feet(self,q,root):
        import torch
        from scipy.spatial.transform import Rotation
        with torch.no_grad():rotations,positions=self.model.fk(torch.as_tensor(q,dtype=torch.float32))
        base=Rotation.from_quat(root[:,3:7]).as_matrix()
        feet=[];yaw=[]
        for side in ("left","right"):
            name=side+"_ankle_roll_link"
            feet.append(np.einsum("nij,nj->ni",base,positions[name].numpy())+root[:,:3])
            orientation=base@rotations[name].numpy()
            yaw.append(np.arctan2(orientation[:,1,0],orientation[:,0,0]))
        return np.stack(feet,axis=1),np.stack(yaw,axis=1)

    def assess(self,poses,commands,trace):
        from source.train_workflow.utils.quality_selection import INDEX
        n=len(poses);seconds=np.arange(n)/self.hz;indices=np.minimum(np.arange(n),len(self.reference_q)-1)
        if (commands is not None and len(commands)!=n-1) or len(trace)!=n-1:raise ValueError("Motion captures and control trace disagree")
        actual_feet,_=self.feet(poses[:,7:],poses[:,:7])
        ref=self.reference_q[indices]
        # Commands at post-step n were generated from reference n-1.
        residual=None
        if commands is not None:
            residual=np.zeros((n,len(self.joints)))
            residual[1:]=np.asarray(commands)-self.reference_q[indices[:-1]][:,self.joints]
        expected=np.zeros((n,2),bool)
        table=np.asarray(self.report.get("support_feet",[]),bool)
        if len(table):expected=table[np.minimum((seconds*self.report["fps"]).astype(int),len(table)-1)]
        contacts=np.zeros((n,2),bool)
        contacts[1:]=trace[:,[INDEX["contact_left"],INDEX["contact_right"]]]>.5
        contacts[0]=contacts[1]
        schedules=[e["phases"] for e in self.report["events"] if e["flight_reward"]]
        return assess_motion(seconds,poses[:,7:][:,self.joints]-ref[:,self.joints],residual,actual_feet,
            self.reference_feet[indices],self.reference_yaw[indices],expected,contacts,self.sides,schedules,
            duration=self.report.get("episode_duration_s",self.report["duration_s"]),hz=self.hz)


class MotionFeedback:
    def __init__(self,bins,state=None):
        self.priority=np.zeros(bins);self.latest=None
        if state:
            if state.get("protocol")!=PROTOCOL:raise ValueError("Motion feedback protocol differs from checkpoint")
            if state.get("criteria")!=CRITERIA:raise ValueError("Motion feedback criteria differ from checkpoint")
            p=np.asarray(state["priority"],float)
            if p.shape!=(bins,) or not np.isfinite(p).all() or (p<0).any():raise ValueError("Invalid motion feedback priority")
            self.priority=p.copy();self.latest=state.get("latest")

    def update(self,records):
        if len(records)!=3 or len({r.get("seed") for r in records})!=3 or any(r.get("kind","full")!="full" for r in records):raise ValueError("Three distinct full-validation seeds required")
        votes=[];observed=[];all_candidates=[];errors=[]
        for record in records:
            scores=np.zeros(len(self.priority));coverage=np.zeros(len(self.priority),bool)
            for trial in record["results"]:
                motion=trial.get("motion_diagnostics",{})
                if motion.get("error") or motion.get("protocol")!=PROTOCOL:
                    errors.append(dict(seed=record["seed"],error=motion.get("error","missing_motion_diagnostics")))
                windows=motion.get("windows")
                if windows is None and motion.get("complete"):coverage[:]=True
                for row in (windows or []):
                    a=max(0,int(row["start_s"]));b=min(len(scores),int(math.ceil(row["end_s"])))
                    coverage[a:b]=True
                for row in motion.get("candidates",[]):
                    a=max(0,int(row["start_s"]));b=min(len(scores),int(math.ceil(row["end_s"])))
                    coverage[a:b]=True
                    scores[a:b]=np.maximum(scores[a:b],min(3.,row["severity"]-1.))
                    all_candidates.append(dict(row,seed=record["seed"]))
            votes.append(scores);observed.append(coverage)
        if len(votes)<3:raise ValueError("Motion feedback needs all three independent validation seeds")
        if errors:
            self.latest=dict(status="incomplete_diagnostics",errors=errors,candidate_windows=len(all_candidates),
                consensus_seconds=[],boosted_seconds=np.flatnonzero(self.priority>0).tolist(),candidates=[])
            return self.latest
        consensus=np.median(np.stack(votes),axis=0)
        known=np.sum(observed,axis=0)>=2
        # Bound the number of boosted seconds as well as their sampling share.
        cap=max(1,int(math.ceil(.2*len(consensus))))
        keep=np.argsort(consensus)[-cap:];bounded=np.zeros_like(consensus);bounded[keep]=consensus[keep]
        self.priority=np.where(known,.5*self.priority+.5*bounded,self.priority)
        keep=np.argsort(self.priority)[-cap:]
        selected=np.zeros_like(self.priority);selected[keep]=self.priority[keep]
        self.priority=selected
        self.priority[self.priority<.02]=0.
        self.latest=dict(status="assessed",candidate_windows=len(all_candidates),consensus_seconds=np.flatnonzero(consensus>0).tolist(),
            boosted_seconds=np.flatnonzero(self.priority>0).tolist(),candidates=sorted(all_candidates,key=lambda r:r["severity"],reverse=True)[:12])
        return self.latest

    def state_dict(self):return dict(protocol=PROTOCOL,criteria=dict(CRITERIA),priority=self.priority.tolist(),latest=self.latest)
