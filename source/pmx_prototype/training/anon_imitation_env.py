"""Reference-state-initialized full-body mocap imitation, without root assistance.

The entire captured motion drives the nominal PD targets and imitation rewards.
There is no analytical foot-placement gait in this environment.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import torch
import numpy as np
from tensordict import TensorDict
from anon_locomotion_env import AnonLocomotionEnv,quat_rotate_inverse
from full_motion_reference import FullMotionReference


def quat_multiply(a,b):
    return torch.cat([a[...,3:]*b[...,:3]+b[...,3:]*a[...,:3]+torch.cross(a[...,:3],b[...,:3],dim=-1),
                      a[...,3:]*b[...,3:]-(a[...,:3]*b[...,:3]).sum(-1,keepdim=True)],-1)


class AnonImitationEnv(AnonLocomotionEnv):
    def __init__(self,num_envs=2048,seed=42,app=None,evaluation=False,config=None,shared_physics=None):
        device=config.get('simulation_device','cuda:0')
        self.motion=FullMotionReference(config['full_motion_path'],device)
        if config.get('unified_walk_run'):
            from unified_walk_run_reference import prepare
            config['unified_contact_offsets']=prepare(self.motion)
        if config.get('continuous_phase_alignment'):
            from continuous_motion_reference import align_contacts
            config['contact_phase_offsets']=align_contacts(self.motion)
        self.clip_ids=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.previous_clip_ids=self.clip_ids.clone()
        self.blend_age=torch.ones(num_envs,device=device)
        self.reference_heading=torch.zeros(num_envs,device=device)
        self.continuous_ref_commands=torch.zeros((num_envs,3),device=device)
        self.continuous_ref_rate=torch.zeros_like(self.continuous_ref_commands)
        self.preferred_clip_ids=torch.arange(num_envs,device=device)%len(self.motion.clip_names)
        self.imitation_ready=False
        super().__init__(num_envs,seed,app,evaluation,config,shared_physics=shared_physics)
        self.q_order=[self.motion.names.index(n) for n in self.names]
        self.link_order=[self.motion.links.index(n) for n in self.links]
        self.base_id=0
        self.base_reference_id=self.motion.links.index(self.links[0])
        if self.cfg.get('reference_velocity_at_com'):
            self.root_com_local=self.robot.get_coms()[0,0,:3].to(self.device).clone()
            expected=self.cfg.get('reference_root_com_local_m')
            if expected is not None and not torch.allclose(self.root_com_local,torch.tensor(expected,device=self.device),atol=1e-4):
                raise ValueError('Root COM differs from the trained physical model')
            self.cfg['reference_root_com_local_m']=self.root_com_local.cpu().tolist()
            self.cfg['reference_root_link_name']=self.links[0]
        self.pelvis_reference_id=self.motion.links.index('pelvis')
        self.foot_reference_ids=[self.motion.links.index(s+'_ankle_roll') for s in ['left','right']]
        if self.cfg.get('leg_acceleration_from_positions',False):
            self.leg_difference_velocity=torch.zeros_like(self.q[:,self.leg_ids])
        if self.cfg.get('stance_slip_weight',0.):
            self.foot_com_local=self.robot.get_coms()[0,self.foot_ids,:3].to(self.device).clone()
        self.key_names=['waist_pitch','head','left_knee_pitch','right_knee_pitch',
                        'left_ankle_roll','right_ankle_roll','left_hand','right_hand']
        self.key_ids=[self.links.index(n) for n in self.key_names]
        self.key_ref_ids=[self.motion.links.index(n) for n in self.key_names]
        self.upper_ids=[i for i,n in enumerate(self.names) if any(k in n for k in ['waist','neck','shoulder','elbow','forearm','wrist'])]
        if self.cfg.get('amp_enabled'):
            from amp_motion_prior import AMPMotionPrior
            if self.cfg.get('amp_reward_diagnostics',True):self.amp=AMPMotionPrior(self)
            self.amp_neutral=self.motion.data['q'][2].mean(0)[self.q_order] if self.motion.bank else self.motion.data['q'].mean(0)[self.q_order]
            reference_q=self.motion.data['q'][2] if self.motion.bank else self.motion.data['q']
            if self.cfg.get('unified_walk_run'):reference_q=self.motion.data['q'].flatten(0,1)
            self.amp_reference_std=reference_q[:,self.q_order].std(0).clamp_min(self.cfg.get('amp_phase_std_floor',.04))
            if self.cfg.get('amp_mirror_augmentation',False):
                from lateral_symmetry import AnatomicalMirror,paired_reference_std
                mirror=self.amp.mirror if hasattr(self,'amp') else AnatomicalMirror.from_env(self)
                self.amp_reference_std=paired_reference_std(reference_q[:,self.q_order],mirror,self.cfg.get('amp_phase_std_floor',.04))
        self.imitation_ready=True
        self.reset(self.ids);self.refresh()

    def reference(self,phase,commands):
        q=self.motion.sample(phase)['q']
        return q[:,[self.motion.names.index(n) for n in self.leg_names]]

    def reference_root_velocity(self,state):
        velocity=state['velocities'][:,self.base_reference_id]
        if not self.cfg.get('reference_velocity_at_com'):return velocity
        from unified_walk_run_reference import velocity_at_offset
        return velocity_at_offset(velocity,state['rotations'][:,self.base_reference_id],self.root_com_local)

    def upper_reference(self,phase,commands):
        from natural_walk_reference import UPPER_NAMES
        return self.motion.sample(phase)['q'][:,[self.motion.names.index(n) for n in UPPER_NAMES]]

    def sample_commands(self,ids,on_reset=False):
        if not len(ids):return
        if self.cfg.get('unified_walk_run'):
            from unified_walk_run_reference import sample_commands
            return sample_commands(self,ids,on_reset)
        if self.cfg.get('continuous_commands'):
            n=len(ids);angle=torch.rand(n,device=self.device)*2*torch.pi
            speed=torch.rand(n,device=self.device)*.4
            if self.cfg.get('continuous_boundary_fraction',0.):
                speed=torch.where(torch.rand(n,device=self.device)<self.cfg['continuous_boundary_fraction'],torch.full_like(speed,.4),speed)
            cmd=torch.stack([speed*torch.cos(angle),speed*torch.sin(angle),(torch.rand(n,device=self.device)*2-1)*.6],-1)
            if not self.evaluation and self.cfg.get('continuous_curriculum_start') is not None:
                elapsed=max(0,getattr(self,'iteration',0)-self.cfg['continuous_curriculum_start'])
                cmd[:,:2]*=.5+.5*min(1.,elapsed/100.)
                cmd[:,2]*=max(0.,min(1.,(elapsed-60)/120.))
            choose=torch.rand(n,device=self.device)
            cmd[choose<.15]=0
            cmd[(choose>=.15)&(choose<.45),2]=0
            anchor=(choose>=.75)
            bank=torch.tensor([2,3,4,5,6,7,8,9,10],device=self.device)
            cmd[anchor]=self.motion.commands[bank[torch.randint(len(bank),(int(anchor.sum()),),device=self.device)]]
            if not self.evaluation and self.cfg.get('continuous_targeted_turns',False):
                choose_turn=torch.rand(n,device=self.device)
                pure_fraction=self.cfg.get('continuous_pure_turn_fraction',.25)
                boundary_fraction=self.cfg.get('continuous_boundary_turn_fraction',.25)
                pure=choose_turn<pure_fraction;boundary=(choose_turn>=pure_fraction)&(choose_turn<pure_fraction+boundary_fraction)
                cmd[pure,:2]=0.
                signs=torch.where(torch.rand(n,device=self.device)<.5,-torch.ones(n,device=self.device),torch.ones(n,device=self.device))
                cmd[pure|boundary,2]=.6*signs[pure|boundary]
                boundary_angle=angle
                if self.cfg.get('continuous_boundary_grid',False):boundary_angle=torch.randint(8,(n,),device=self.device)*torch.pi/4
                cmd[boundary,0]=.4*torch.cos(boundary_angle[boundary]);cmd[boundary,1]=.4*torch.sin(boundary_angle[boundary])
                low=(choose_turn>=pure_fraction+boundary_fraction)&(choose_turn<pure_fraction+boundary_fraction+self.cfg.get('continuous_low_speed_fraction',0.))
                low_speed=torch.randint(1,4,(n,),device=self.device)*.05
                low_angle=torch.where(torch.rand(n,device=self.device)<.5,torch.zeros_like(angle),angle)
                cmd[low,0]=low_speed[low]*torch.cos(low_angle[low]);cmd[low,1]=low_speed[low]*torch.sin(low_angle[low]);cmd[low,2]=0.
            if not self.evaluation and on_reset and self.cfg.get('continuous_reset_idle_fraction',0.):
                cmd[torch.rand(n,device=self.device)<self.cfg['continuous_reset_idle_fraction']]=0.
            if self.evaluation:cmd.zero_()
            self.commands[ids]=cmd;self.command_age[ids]=0
            if on_reset:self.continuous_ref_commands[ids]=cmd
            return
        if self.motion.bank:
            probabilities=torch.tensor(self.cfg.get('imitation_motion_probabilities',[1.]*len(self.motion.clip_names)),device=self.device,dtype=torch.float32)
            self.previous_clip_ids[ids]=self.clip_ids[ids]
            if self.evaluation:self.clip_ids[ids]=self.cfg.get('imitation_evaluation_initial_clip',0)
            elif on_reset and self.cfg.get('imitation_persistent_motion',False):self.clip_ids[ids]=self.preferred_clip_ids[ids]
            else:self.clip_ids[ids]=torch.multinomial(probabilities,len(ids),replacement=True)
            self.blend_age[ids]=0.
            self.commands[ids]=self.motion.commands[self.clip_ids[ids]]
        else:self.commands[ids]=torch.tensor([self.motion.speed,0.,0.],device=self.device)
        self.command_age[ids]=0.

    def set_command(self,command):
        if not self.motion.bank:return
        if self.cfg.get('amp_forward_only'):command=[self.cfg.get('amp_forward_speed',.4) if float(command[0])>.05 else 0.,0.,0.]
        if self.cfg.get('continuous_commands'):
            cmd=torch.as_tensor(command,device=self.device,dtype=torch.float32).clone()
            cmd[:2]*=torch.minimum(torch.ones((),device=self.device),self.cfg.get('max_speed',.4)/cmd[:2].norm().clamp_min(1e-6))
            cmd[2]=cmd[2].clamp(-self.cfg.get('max_yaw',.6),self.cfg.get('max_yaw',.6));self.commands[:]=cmd
            return
        command=torch.tensor(command,device=self.device,dtype=torch.float32)
        clip=((self.motion.commands-command).square().sum(-1)).argmin()
        changed=(self.clip_ids!=clip)
        self.previous_clip_ids[changed]=self.clip_ids[changed];self.clip_ids[changed]=clip
        self.blend_age[changed]=0.
        self.commands[:]=self.motion.commands[clip]

    def frequency(self):
        if self.cfg.get('unified_walk_run'):
            from unified_walk_run_reference import components
            return components(self.motion,self.continuous_ref_commands,self.cfg)[2]
        if self.cfg.get('continuous_commands'):
            from continuous_motion_reference import components
            return components(self.motion,self.continuous_ref_commands)[2]
        if not self.motion.bank:return torch.full_like(self.phase,1/self.motion.period)
        s=self.blend_age.clamp(0,1);weight=s*s*(3-2*s)
        return (1-weight)/self.motion.periods[self.previous_clip_ids]+weight/self.motion.periods[self.clip_ids]

    def yaw_rate(self):
        if self.cfg.get('continuous_commands'):return self.continuous_ref_commands[:,2]
        if not self.motion.bank:return torch.zeros_like(self.phase)
        s=self.blend_age.clamp(0,1);weight=s*s*(3-2*s)
        return (1-weight)*self.motion.commands[self.previous_clip_ids,2]+weight*self.motion.commands[self.clip_ids,2]

    def apply_heading(self,result,indices=None,future=0.):
        if not self.cfg.get('imitation_steering',False):return result
        theta=self.reference_heading if indices is None else self.reference_heading[indices]
        rate=self.yaw_rate() if indices is None else self.yaw_rate()[indices]
        theta=theta+future*rate
        quat=torch.zeros((len(theta),4),device=self.device);quat[:,2]=torch.sin(theta/2);quat[:,3]=torch.cos(theta/2)
        result['local_pelvis_rotation']=result['rotations'][:,self.pelvis_reference_id].clone()
        result['heading_quaternion']=quat
        expanded=quat[:,None,:].expand(-1,len(self.motion.links),-1)
        conjugate=torch.cat([-expanded[...,:3],expanded[...,3:]],-1)
        result['positions']=quat_rotate_inverse(conjugate,result['positions'])
        result['rotations']=quat_multiply(expanded,result['rotations'])
        result['velocities']=torch.cat([quat_rotate_inverse(conjugate,result['velocities'][...,:3]),
                                       quat_rotate_inverse(conjugate,result['velocities'][...,3:])],-1)
        return result

    def sample_reference(self,phase,indices=None,future=0.):
        if self.cfg.get('unified_walk_run'):
            from unified_walk_run_reference import sample
            cmd=self.continuous_ref_commands if indices is None else self.continuous_ref_commands[indices]
            rate=self.continuous_ref_rate if indices is None else self.continuous_ref_rate[indices]
            if self.cfg.get('unified_reference_derivatives'):cmd=cmd+future*rate
            return self.apply_heading(sample(self.motion,phase,cmd,self.cfg,rate),indices,future)
        if self.cfg.get('continuous_commands'):
            from continuous_motion_reference import sample_continuous
            cmd=self.continuous_ref_commands if indices is None else self.continuous_ref_commands[indices]
            return self.apply_heading(sample_continuous(self.motion,phase,cmd),indices,future)
        if not self.motion.bank:return self.apply_heading(self.motion.sample(phase),indices,future)
        ids=self.clip_ids if indices is None else self.clip_ids[indices]
        old_ids=self.previous_clip_ids if indices is None else self.previous_clip_ids[indices]
        age=self.blend_age if indices is None else self.blend_age[indices]
        duration=self.cfg.get('imitation_blend_seconds',.6)
        s=(age+future/duration).clamp(0,1);weight=s*s*(3-2*s)
        old=self.motion.sample(phase,old_ids);new=self.motion.sample(phase,ids)
        result={}
        for key,value in new.items():
            w=weight.reshape((len(weight),)+(1,)*(value.ndim-1));previous=old[key]
            if key=='rotations':value=torch.where(((previous*value).sum(-1)<0)[...,None],-value,value)
            result[key]=previous*(1-w)+value*w
        result['rotations']=torch.nn.functional.normalize(result['rotations'],dim=-1)
        result['qd']+=((6*s*(1-s)/duration)[:,None])*(new['q']-old['q'])
        return self.apply_heading(result,indices,future)

    def reset(self,ids,*,heading=None,root_xy=None,command=None,phase=None):
        if not self.imitation_ready:return super().reset(ids)
        if not len(ids):return
        self.sample_commands(ids,on_reset=True)
        if command is not None:
            if len(ids)!=self.num_envs:raise ValueError('Explicit reset command requires all environments')
            self.set_command(command)
            self.continuous_ref_commands[ids]=self.commands[ids]
        self.continuous_ref_rate[ids]=0.
        if hasattr(self,'amp'):self.amp.reset(ids)
        self.previous_clip_ids[ids]=self.clip_ids[ids];self.blend_age[ids]=1.
        self.reference_heading[ids]=0.
        if not self.evaluation and self.cfg.get('imitation_random_heading',False):
            self.reference_heading[ids]=(torch.rand(len(ids),device=self.device)*2-1)*torch.pi
        if heading is not None:self.reference_heading[ids]=heading
        self.phase[ids]=0. if self.evaluation else torch.rand(len(ids),device=self.device)*(4 if self.cfg.get('unified_walk_run') else 1)
        if phase is not None:self.phase[ids]=phase
        if not self.evaluation and self.cfg.get('unified_reset_from_idle_fraction',0.):
            start_idle=torch.rand(len(ids),device=self.device)<self.cfg['unified_reset_from_idle_fraction']
            self.continuous_ref_commands[ids[start_idle]]=0.
        ref=self.sample_reference(self.phase[ids],ids)
        q=ref['q'][:,self.q_order];qd=ref['qd'][:,self.q_order]
        state=self.initial_root.clone();velocity=self.zero_vel.clone()
        state[ids,:3]=self.origins[ids]+ref['positions'][:,self.base_reference_id]
        if root_xy is not None:state[ids,:2]=root_xy
        state[ids,3:]=ref['rotations'][:,self.base_reference_id]
        velocity[ids]=self.reference_root_velocity(ref)
        self.robot.set_root_transforms(state,ids);self.robot.set_root_velocities(velocity,ids)
        all_q=self.default.clone();all_qd=self.zero_qvel.clone();all_q[ids]=q;all_qd[ids]=qd
        self.robot.set_dof_positions(all_q,ids);self.robot.set_dof_velocities(all_qd,ids)
        self.robot.set_dof_position_targets(all_q,ids)
        self.actions[ids]=0;self.previous_actions[ids]=0;self.filtered_leg_actions[ids]=0
        self.episode_length_buf[ids]=0;self.episode_returns[ids]=0;self.episode_errors[ids]=0

    def get_observations(self):
        self.refresh()
        ref=self.sample_reference(self.phase)
        future=self.sample_reference(self.phase+.1*self.frequency(),future=.1)
        qref=ref['q'][:,self.q_order];qdref=ref['qd'][:,self.q_order]
        pelvis_q=ref['rotations'][:,self.pelvis_reference_id]
        gravity=quat_rotate_inverse(pelvis_q,torch.tensor([0.,0.,-1.],device=self.device).expand(self.num_envs,-1))
        obs=torch.cat([self.vel_b*2,self.omega_b*.25,self.gravity,self.commands,
                       self.q,self.qd*.05,self.actions,qref,qdref*.05,self.q-qref,
                       ref['positions'][:,self.pelvis_reference_id,2:3],gravity,ref['contacts'],
                       ref.get('gait_sin',torch.sin(2*torch.pi*self.phase))[:,None],ref.get('gait_cos',torch.cos(2*torch.pi*self.phase))[:,None],
                       future['q'][:,self.q_order]],-1)
        if self.cfg.get('imitation_heading_observation',False):
            actual=self.poses[:,self.pelvis_id,3:]
            velocity_delta=self.root_vel[:,:3]-self.reference_root_velocity(ref)[:,:3]
            if self.cfg.get('imitation_invariant_observation',False):
                inverse=torch.cat([-pelvis_q[:,:3],pelvis_q[:,3:]],-1)
                actual=quat_multiply(inverse,actual);actual=torch.where(actual[:,3:]<0,-actual,actual)
                pelvis_q=ref['local_pelvis_rotation']
                velocity_delta=quat_rotate_inverse(ref['heading_quaternion'],velocity_delta)
            obs=torch.cat([obs,actual,pelvis_q,velocity_delta],-1)
        observations={'policy':obs.clamp(-20,20)}
        if self.cfg.get('train_lateral_mirror',False) and not self.evaluation:
            from lateral_symmetry import eligibility
            # Rollout metadata only: the actor/critic still consume exactly248
            # policy channels. Preserve planner eligibility through shuffling.
            observations['mirror_eligible']=eligibility(self)
        return TensorDict(observations,batch_size=[self.num_envs],device=self.device)

    def step(self,actions):
        reward_diagnostics=not (self.evaluation and self.cfg.get('runtime_inference_only',False))
        leg_smooth_enabled=bool(self.cfg.get('leg_acceleration_weight',0.)) and reward_diagnostics
        if leg_smooth_enabled:
            if self.cfg.get('leg_acceleration_from_positions',False):
                previous_leg_q=self.q[:,self.leg_ids].clone()
            else:
                previous_leg_qd=self.qd[:,self.leg_ids].clone()
        geometric_slip_enabled=bool(self.cfg.get('stance_displacement_weight',0.) or self.cfg.get('stance_displacement_speed_weight',0.)) and reward_diagnostics
        if geometric_slip_enabled:
            old_feet=self.poses[:,self.foot_ids]
            old_quat=old_feet[:,:,None,3:].expand(-1,-1,8,-1)
            old_conjugate=torch.cat([-old_quat[...,:3],old_quat[...,3:]],-1)
            old_offset=quat_rotate_inverse(old_conjugate,self.foot_corners[None].expand(self.num_envs,-1,-1,-1))
            old_sole=old_feet[:,:,None,:3]+old_offset
            old_heights=old_sole[:,:,:,2];old_bottom=old_heights.amin(-1)
            old_near=(old_heights<old_bottom[:,:,None]+.003)&(old_bottom[:,:,None]<self.cfg.get('contact_height_threshold_m',.012))
        self.previous_actions.copy_(self.actions);self.actions.copy_(actions.clamp(-2,2))
        if self.cfg.get('continuous_commands'):
            from command_plan import command_increment
            delta=command_increment(self.commands,self.continuous_ref_commands,self.dt,self.cfg)
            self.continuous_ref_rate=delta/self.dt
            self.continuous_ref_commands+=delta
        self.blend_age+=self.dt/self.cfg.get('imitation_blend_seconds',.6)
        if self.cfg.get('imitation_steering',False):self.reference_heading+=self.dt*self.yaw_rate()
        self.phase=(self.phase+self.dt*self.frequency())%(4096. if self.cfg.get('unified_walk_run') else 1.)
        if not self.motion.bank:self.commands[:]=torch.tensor([self.motion.speed,0.,0.],device=self.device)
        ref=self.sample_reference(self.phase)
        target=ref['q'][:,self.q_order]+self.cfg.get('imitation_residual_scale',.3)*self.actions
        if self.cfg.get('amp_enabled'):
            mix=self.cfg.get('amp_upper_reference_mix',.35)
            neutral=self.amp_neutral
            target[:,self.upper_ids]=neutral[self.upper_ids]+mix*(ref['q'][:,self.q_order][:,self.upper_ids]-neutral[self.upper_ids])+.45*self.actions[:,self.upper_ids]
        target=target.clamp(self.limits[:,0]+.01,self.limits[:,1]-.01)
        if self.cfg.get('diagnostic_controller_trace',False):
            self.diagnostic_target=target.clone()
            self.diagnostic_reference=ref['q'][:,self.q_order].clone()
        self.robot.set_dof_position_targets(target,self.ids)
        if self.cfg.get('reference_velocity_feedforward',False):
            velocity_target=torch.zeros_like(self.default)
            if self.cfg.get('reference_full_body_velocity_feedforward',False):velocity_target[:]=ref['qd'][:,self.q_order]
            velocity_target[:,self.leg_ids]=ref['qd'][:,self.q_order][:,self.leg_ids]*self.cfg.get('velocity_feedforward_scale',1.)
            if self.cfg.get('reference_distal_arm_velocity_feedforward',False):
                distal=[i for i,n in enumerate(self.names) if any(k in n for k in ['forearm','wrist'])]
                velocity_target[:,distal]=ref['qd'][:,self.q_order][:,distal]
            self.robot.set_dof_velocity_targets(velocity_target.clamp(-10,10),self.ids)
        measure_forces=self.evaluation and self.cfg.get('unified_substep_support_forces',False)
        if measure_forces and not hasattr(self,'ground_contact_view'):raise ValueError('Substep force audit requires contact sensors')
        substep_forces=[]
        for _ in range(self.cfg['decimation']):
            self._simulate()
            if measure_forces:
                substep_forces.append(self.ground_contact_view.get_contact_force_matrix(1/self.cfg['physics_hz'])[:,0,:].clone())
        if measure_forces:self.last_substep_support_forces=torch.stack(substep_forces)
        self.refresh();self.steps+=1;self.episode_length_buf+=1
        if self.evaluation and self.cfg.get('runtime_inference_only',False):
            # The controller above is identical. Rewards and AMP diagnostics do
            # not feed back into inference; retain the original fall/reset gate.
            fallen=(self.poses[:,self.pelvis_id,2]<.65)|(self.gravity[:,2]>-.5)|~torch.isfinite(self.q).all(-1)
            timeout=self.episode_length_buf>=self.max_episode_length
            done=fallen|timeout
            ids=done.nonzero(as_tuple=False).flatten()
            if len(ids):self.reset(ids)
            self.command_age+=self.dt
            # The interactive worker requests fresh observations after the
            # next command, so a second reference sample here is unused.
            return None,torch.zeros_like(self.phase),done,{'time_outs':timeout}
        qref=ref['q'][:,self.q_order];qdref=ref['qd'][:,self.q_order]
        joint_weights=torch.ones(self.num_actions,device=self.device)
        if self.cfg.get('amp_enabled'):joint_weights[self.upper_ids]=self.cfg.get('amp_upper_pose_weight',.2)
        pose_error=((self.q-qref).square()*joint_weights).sum(-1)/joint_weights.sum()
        velocity_error=((self.qd-qdref).square()*joint_weights).sum(-1)/joint_weights.sum()
        pelvis=self.poses[:,self.pelvis_id]
        relative=self.poses[:,self.key_ids,:3]-pelvis[:,None,:3]
        target_relative=ref['positions'][:,self.key_ref_ids]-ref['positions'][:,self.pelvis_reference_id,None,:]
        key_weights=torch.ones(len(self.key_ids),device=self.device)
        if self.cfg.get('amp_enabled'):key_weights[[0,1,6,7]]=self.cfg.get('amp_upper_keypoint_weight',.25)
        key_error=((relative-target_relative).square().sum(-1)*key_weights).sum(-1)/key_weights.sum()
        height_error=(pelvis[:,2]-ref['positions'][:,self.pelvis_reference_id,2]).square()
        dot=(pelvis[:,3:]*ref['rotations'][:,self.pelvis_reference_id]).sum(-1).abs().clamp(0,1)
        orientation_error=(2*torch.acos(dot.clamp(max=1.-1e-7))).square()
        base_velocity_error=(self.root_vel[:,:3]-self.reference_root_velocity(ref)[:,:3]).square().sum(-1)
        feet=self.poses[:,self.foot_ids]
        quat=feet[:,:,None,3:].expand(-1,-1,8,-1)
        conjugate=torch.cat([-quat[...,:3],quat[...,3:]],-1)
        offset=quat_rotate_inverse(conjugate,self.foot_corners[None].expand(self.num_envs,-1,-1,-1))
        bottom=(feet[:,:,None,2]+offset[:,:,:,2]).amin(-1)
        contact=(bottom<self.cfg.get('contact_height_threshold_m',.012)).float()
        contact_error=(contact-ref['contacts']).abs().mean(-1)
        if geometric_slip_enabled:
            from stance_motion import displacement_slip_square
            sole=feet[:,:,None,:3]+offset
            near_geometry=(sole[:,:,:,2]<bottom[:,:,None]+.003)&(contact[:,:,None]>.5)
            geometric_slip_square=displacement_slip_square(sole,old_sole,near_geometry,old_near,self.dt)
            if self.cfg.get('stance_displacement_speed_weight',0.):
                from stance_motion import displacement_slip_speed
                geometric_slip_speed=displacement_slip_speed(sole,old_sole,near_geometry,old_near,self.dt)
        if self.cfg.get('stance_slip_weight',0.):
            from stance_motion import point_velocity
            local_com=self.foot_com_local[None,:,None,:].expand(self.num_envs,-1,8,-1)
            world_com=quat_rotate_inverse(conjugate,local_com)[:,:,0,:]
            foot_velocity=self.link_vel[:,self.foot_ids]
            velocity=point_velocity(foot_velocity[:,:,:3],foot_velocity[:,:,3:],offset,world_com)
            heights=feet[:,:,None,2]+offset[:,:,:,2]
            near=(heights<bottom[:,:,None]+.003)&(contact[:,:,None]>.5)
            weight=near.float()*ref['contacts'][:,:,None].clamp(0,1)
            slip_square=(velocity[...,:2].square().sum(-1).clamp(max=1.)*weight).sum((1,2))/weight.sum((1,2)).clamp_min(1)
        reward=(3.*torch.exp(-pose_error/.065)+.7*torch.exp(-velocity_error/12.)
                +self.cfg.get('imitation_keypoint_weight',2.)*torch.exp(-key_error/.008)+self.cfg.get('imitation_height_weight',1.)*torch.exp(-height_error/self.cfg.get('imitation_height_error_scale',.0025))
                +.7*torch.exp(-orientation_error/self.cfg.get('imitation_orientation_error_scale',.12))+self.cfg.get('imitation_velocity_weight',1.)*torch.exp(-base_velocity_error/self.cfg.get('imitation_velocity_error_scale',.12))
                +self.cfg.get('imitation_contact_weight',.8)*(1-contact_error)-.003*(self.actions-self.previous_actions).square().sum(-1)
                -.005*self.actions.square().mean(-1))
        if self.cfg.get('imitation_angular_velocity_weight',0.):
            angular_error=(self.root_vel[:,3:]-ref['velocities'][:,self.base_reference_id,3:]).square().sum(-1)
            reward+=self.cfg['imitation_angular_velocity_weight']*torch.exp(-angular_error/.25)
        if self.cfg.get('stance_slip_weight',0.):
            speed=self.continuous_ref_commands[:,:2].norm(dim=-1)
            reward+=self.cfg['stance_slip_weight']*torch.exp(-slip_square/.04)*(speed>.2)
        if self.cfg.get('foot_orientation_weight',0.):
            alignment=(feet[:,:,3:]*ref['rotations'][:,self.foot_reference_ids]).sum(-1).square().clamp(0,1)
            reward+=self.cfg['foot_orientation_weight']*torch.exp(-(1-alignment).mean(-1)/.015)
        if self.cfg.get('swing_clearance_weight',0.):
            ref_rotation=ref['rotations'][:,self.foot_reference_ids,None,:].expand(-1,-1,8,-1)
            ref_inverse=torch.cat([-ref_rotation[...,:3],ref_rotation[...,3:]],-1)
            ref_offset=quat_rotate_inverse(ref_inverse,self.foot_corners[None].expand(self.num_envs,-1,-1,-1))
            ref_bottom=(ref['positions'][:,self.foot_reference_ids,None,2]+ref_offset[:,:,:,2]).amin(-1)
            desired=(ref_bottom+self.cfg.get('swing_clearance_extra_m',.012)).clamp(.012,.05)
            swing=(1-ref['contacts']).clamp(0,1).square()
            swing_deficit=(desired-bottom).clamp_min(0)
            cmd=self.continuous_ref_commands
            difficult=((-cmd[:,0]).clamp_min(0)+cmd[:,1].abs())/cmd[:,:2].abs().sum(-1).clamp_min(.001)
            running=((cmd[:,:2].norm(dim=-1)-.55)/.35).clamp(0,1)
            turning=(cmd[:,2].abs()/self.cfg.get('max_yaw',.6)).clamp(0,1)
            turn_weight=1-(1-self.cfg.get('stance_displacement_turn_scale',1.))*turning
            reward-=self.cfg['swing_clearance_weight']*(swing*(swing_deficit/.025).square()).mean(-1)*difficult*running*turn_weight
        if geometric_slip_enabled:
            moving=self.continuous_ref_commands[:,:2].norm(dim=-1)>.2
            turn=(self.continuous_ref_commands[:,2].abs()/self.cfg.get('max_yaw',.6)).clamp(0,1)
            turn_weight=1-(1-self.cfg.get('stance_displacement_turn_scale',1.))*turn
            reward-=self.cfg.get('stance_displacement_weight',0.)*geometric_slip_square*moving*turn_weight
            if self.cfg.get('stance_displacement_speed_weight',0.):
                cmd=self.continuous_ref_commands[:,:2]
                difficult=((-cmd[:,0]).clamp_min(0)+cmd[:,1].abs())/cmd.abs().sum(-1).clamp_min(.001)
                reward-=self.cfg['stance_displacement_speed_weight']*geometric_slip_speed*moving*turn_weight*difficult
        if leg_smooth_enabled:
            if self.cfg.get('leg_acceleration_from_positions',False):
                leg_velocity=(self.q[:,self.leg_ids]-previous_leg_q)/self.dt
                leg_acceleration=(leg_velocity-self.leg_difference_velocity)/self.dt
                leg_acceleration*= (self.episode_length_buf>1)[:,None]
                self.leg_difference_velocity.copy_(leg_velocity)
            else:
                leg_acceleration=(self.qd[:,self.leg_ids]-previous_leg_qd)/self.dt
            if self.cfg.get('leg_acceleration_cost')=='pseudo_huber':
                smooth_cost=(torch.sqrt(1+(leg_acceleration/30.).square())-1).mean(-1)
            else:
                smooth_cost=(leg_acceleration/30.).square().clamp(max=9.).mean(-1)
            reward-=self.cfg['leg_acceleration_weight']*smooth_cost
        if self.cfg.get('unified_backward_height_weight',0.):
            speed=self.continuous_ref_commands[:,:2].norm(dim=-1)
            backward=(-self.continuous_ref_commands[:,0]/speed.clamp_min(.001)).clamp(0,1)
            run=((speed-.7)/.3).clamp(0,1);run=run*run*(3-2*run)
            height_quality=torch.exp(-height_error/self.cfg.get('unified_backward_height_error_scale',.0003))
            reward+=self.cfg['unified_backward_height_weight']*backward*run*height_quality
        if self.cfg.get('unified_heading_weight',0.):
            target_rotation=ref['rotations'][:,self.pelvis_reference_id]
            inverse=torch.cat([-target_rotation[:,:3],target_rotation[:,3:]],-1)
            relative_rotation=quat_multiply(inverse,pelvis[:,3:])
            x,y,z,w=relative_rotation.unbind(-1)
            heading_error=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            reward+=self.cfg['unified_heading_weight']*torch.exp(-heading_error.square()/self.cfg.get('unified_heading_error_scale',.0225))
        if self.cfg.get('continuous_tracking_weight',0.):
            speed=self.continuous_ref_commands[:,:2].norm(dim=-1)
            scale=self.cfg.get('continuous_tracking_error_base',.003)+self.cfg.get('continuous_tracking_error_speed_scale',.05)*speed.square()
            reward+=self.cfg['continuous_tracking_weight']*torch.exp(-base_velocity_error/scale)
            if self.cfg.get('continuous_low_speed_pose_weight',0.):
                low=(speed>.02)&(speed<.16)
                reward+=self.cfg['continuous_low_speed_pose_weight']*(torch.exp(-key_error/.004)+torch.exp(-orientation_error/.03))*low
        pivot_terms=None
        if self.cfg.get('unified_pivot_rewards',self.cfg.get('unified_pivot_training',False)):
            from pivot_stability import pivot_stability_terms
            pivot_terms=pivot_stability_terms(self.commands,self.continuous_ref_commands,bottom,
                                              ref['contacts'],self.root_vel,self.cfg)
            reward+=pivot_terms['reward_rate']
        if self.cfg.get('learn_arm_clearance',False):
            # Reward only: never alter PD targets, references or policy outputs.
            deficits=[];arm_errors=[];gates=[]
            for side,sign in [('left',1.),('right',-1.)]:
                wrist=self.poses[:,self.links.index(side+'_hand'),:3]-pelvis[:,:3]
                local=quat_rotate_inverse(pelvis[:,3:],wrist)
                s=((local[:,1].abs()-.035)/.13).clamp(0,1);gate=1-s*s*(3-2*s)
                if self.cfg.get('learn_clearance_height_gate',False):
                    h=((local[:,2]-.02)/.12).clamp(0,1);gate*=1-h*h*(3-2*h)
                desired_lateral=self.cfg.get('learn_wrist_lateral_m',.185)
                if self.cfg.get('clearance_radial_envelope',False):
                    radius=.12+.2*(-local[:,2]).clamp(0,.25)
                    desired_lateral=(radius.square()-local[:,1].square()).clamp_min(0).sqrt()
                deficit=(desired_lateral-sign*local[:,0]).clamp_min(0)*gate
                deficits.append(deficit)
                gates.append(gate)
                k=self.names.index(side+'_shoulder_roll_joint')
                desired=qref[:,k]-sign*.08*gate
                arm_errors.append(self.q[:,k]-desired)
            deficit=torch.stack(deficits,-1);error=torch.stack(arm_errors,-1)
            reward+=3.*torch.exp(-deficit.square().mean(-1)/.0004)
            if not self.cfg.get('spatial_clearance_only'):reward+=1.5*torch.exp(-error.square().mean(-1)/.01)
            self.learned_clearance_deficit=deficit.sum()/torch.stack(gates,-1).sum().clamp_min(1e-6)
        if hasattr(self,'amp'):
            style=self.amp.observe()
            reward+=self.cfg.get('amp_style_weight',5.)*style*(self.commands[:,:2].norm(dim=-1)>.05)
        if self.cfg.get('hand_orientation_weight',0.):
            hands=[self.links.index(s+'_hand') for s in ['left','right']]
            refs=[self.motion.links.index(s+'_hand') for s in ['left','right']]
            dot=(self.poses[:,hands,3:]*ref['rotations'][:,refs]).sum(-1).abs().clamp_max(1.)
            reward+=self.cfg['hand_orientation_weight']*torch.exp(-(1-dot.square()).mean(-1)/.03)
        if self.cfg.get('running_arm_spacing_weight',0.):
            torso=self.poses[:,self.links.index('waist_pitch')]
            deficits=[]
            for side,sign in [('left',1.),('right',-1.)]:
                for suffix,minimum in [('elbow_flex',.19),('hand',.10)]:
                    local=quat_rotate_inverse(torso[:,3:],self.poses[:,self.links.index(side+'_'+suffix),:3]-torso[:,:3])
                    deficits.append((minimum-sign*local[:,0]).clamp_min(0))
            error=torch.stack(deficits,-1).square().mean(-1)
            reward+=self.cfg['running_arm_spacing_weight']*torch.exp(-error/.0004)*(self.commands[:,:2].norm(dim=-1)>.1)
        if self.cfg.get('amp_phase_upper_weight',0.):
            normalized=(self.q[:,self.upper_ids]-qref[:,self.upper_ids])/self.amp_reference_std[self.upper_ids]
            phase_pose=torch.exp(-normalized.square().mean(-1))
            phase_velocity=torch.exp(-(self.qd[:,self.upper_ids]-qdref[:,self.upper_ids]).square().mean(-1)/.3)
            reward+=(self.cfg['amp_phase_upper_weight']*phase_pose+self.cfg.get('amp_phase_velocity_weight',0.)*phase_velocity)*(self.commands[:,:2].norm(dim=-1)>.05)
        if self.cfg.get('idle_yaw_weight',0.):
            idle=self.commands.abs().sum(-1)<.01
            quiet_yaw=torch.exp(-self.root_vel[:,5].square()/.0036)
            idle_pose=torch.exp(-((self.q[:,self.upper_ids]-qref[:,self.upper_ids])/.06).square().mean(-1))
            reward+=(self.cfg['idle_yaw_weight']*quiet_yaw+self.cfg.get('idle_upper_pose_weight',0.)*idle_pose)*idle
        if self.cfg.get('unified_idle_velocity_weight',0.):
            idle=self.commands.abs().sum(-1)<.001
            reward+=self.cfg['unified_idle_velocity_weight']*torch.exp(-self.root_vel[:,:2].square().sum(-1)/.0025)*idle
        fallen=(pelvis[:,2]<.65)|(self.gravity[:,2]>-.5)|~torch.isfinite(self.q).all(-1)
        # Early termination teaches recovery around the reference manifold.
        if not self.evaluation:
            fallen|=(key_error>.09)|(height_error>.035)
        timeout=self.episode_length_buf>=self.max_episode_length
        reward=(reward-5*fallen)*self.dt
        self.episode_returns+=reward;self.episode_errors+=torch.sqrt(pose_error)
        done=fallen|timeout
        extras={'time_outs':timeout,'log':{
            'Imitation/joint_rmse_rad':pose_error.sqrt().mean(),
            'Imitation/keypoint_rmse_m':key_error.sqrt().mean(),
            'Imitation/pelvis_height_rmse_m':height_error.sqrt().mean(),
            'Imitation/pelvis_orientation_rmse_rad':orientation_error.sqrt().mean(),
            'Imitation/contact_accuracy':(1-contact_error).mean(),
            'Imitation/base_velocity_rmse':base_velocity_error.sqrt().mean(),
            'Stability/fall_rate':fallen.float().mean()}}
        if pivot_terms is not None:
            active=pivot_terms['active'];count=active.sum().clamp_min(1)
            extras['log']['Pivot/occupancy']=active.float().mean()
            for key in ('support','landing','drift','reward_rate'):
                extras['log']['Pivot/'+key]=(pivot_terms[key].sum()/count)
        if self.cfg.get('unified_high_speed_focus'):
            fast=self.commands[:,:2].norm(dim=-1)>.8
            extras['log']['Unified/high_speed_occupancy']=fast.float().mean()
            extras['log']['Unified/high_speed_fall_rate']=fallen[fast].float().mean() if fast.any() else torch.zeros((),device=self.device)
            extras['log']['Unified/high_speed_velocity_error']=base_velocity_error[fast].sqrt().mean() if fast.any() else torch.zeros((),device=self.device)
        if self.cfg.get('unified_heading_weight',0.):
            extras['log']['Unified/pelvis_heading_error_abs_rad']=heading_error.abs().mean()
        if self.cfg.get('stance_slip_weight',0.):
            extras['log']['Natural/stance_point_velocity_rms']=slip_square.mean().sqrt()
        if geometric_slip_enabled:
            extras['log']['Natural/stance_displacement_velocity_rms']=geometric_slip_square.mean().sqrt()
            if self.cfg.get('stance_displacement_speed_weight',0.):
                extras['log']['Natural/stance_displacement_velocity_mean']=geometric_slip_speed.mean()
        if leg_smooth_enabled:
            extras['log']['Natural/leg_acceleration_rms_rad_s2']=leg_acceleration.square().mean().sqrt()
        if self.cfg.get('swing_clearance_weight',0.):
            extras['log']['Natural/swing_clearance_deficit_m']=(swing_deficit*swing).sum()/swing.sum().clamp_min(1)
        if self.cfg.get('unified_backward_env_fraction',0.):
            assigned=self.ids.remainder(100)<round(100*self.cfg['unified_backward_env_fraction'])
            moving=assigned&(self.commands[:,:2].norm(dim=-1)>.1)
            extras['log']['Unified/backward_rollout_fraction']=moving.float().mean()
            extras['log']['Unified/backward_reference_velocity_error']=base_velocity_error[moving].sqrt().mean() if moving.any() else torch.zeros((),device=self.device)
            extras['log']['Unified/backward_termination_rate']=fallen[moving].float().mean() if moving.any() else torch.zeros((),device=self.device)
            height_delta=pelvis[:,2]-ref['positions'][:,self.pelvis_reference_id,2]
            extras['log']['Unified/backward_height_bias_m']=height_delta[moving].mean() if moving.any() else torch.zeros((),device=self.device)
            extras['log']['Unified/backward_height_rmse_m']=height_error[moving].mean().sqrt() if moving.any() else torch.zeros((),device=self.device)
        if self.cfg.get('unified_right_env_fraction',0.):
            first=round(100*self.cfg.get('unified_backward_env_fraction',0.));last=first+round(100*self.cfg['unified_right_env_fraction'])
            assigned=(self.ids.remainder(100)>=first)&(self.ids.remainder(100)<last)
            moving=assigned&(self.commands[:,:2].norm(dim=-1)>.1)
            extras['log']['Unified/right_rollout_fraction']=moving.float().mean()
            extras['log']['Unified/right_reference_velocity_error']=base_velocity_error[moving].sqrt().mean() if moving.any() else torch.zeros((),device=self.device)
            extras['log']['Unified/right_termination_rate']=fallen[moving].float().mean() if moving.any() else torch.zeros((),device=self.device)
        if self.evaluation:self.recorded_state=(self.poses.clone(),self.q.clone(),self.logical_vel.clone(),self.phase.clone())
        ids=done.nonzero(as_tuple=False).flatten()
        if hasattr(self,'amp'):
            extras['log']['Imitation/amp_style_reward']=self.amp.last_reward
            extras['log']['Imitation/amp_policy_score']=self.amp.last_score
        if self.cfg.get('learn_arm_clearance',False):extras['log']['Imitation/near_hip_clearance_deficit_m']=self.learned_clearance_deficit
        if self.cfg.get('hand_orientation_weight',0.):extras['log']['Imitation/hand_orientation_error_rad']=(2*torch.acos(dot.clamp_max(1.))).mean()
        if self.cfg.get('running_arm_spacing_weight',0.):extras['log']['Imitation/arm_spacing_deficit_m']=torch.stack(deficits,-1).mean()
        if len(ids):
            extras['log']['Episode/seconds']=self.episode_length_buf[ids].float().mean()*self.dt
            extras['log']['Episode/return']=self.episode_returns[ids].mean()
            self.reset(ids)
        self.command_age+=self.dt
        if self.motion.bank and not self.evaluation and self.cfg.get('imitation_transition_training',False):
            hold=self.unified_command_hold if self.cfg.get('unified_walk_run') else self.cfg.get('imitation_command_hold_seconds',4.)
            self.sample_commands((self.command_age>hold).nonzero(as_tuple=False).flatten())
        return self.get_observations(),reward,done,extras
