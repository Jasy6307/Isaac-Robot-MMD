"""GPU PhysX vector environment for Anon's velocity-conditioned residual PPO.

Commands are (forward=-body Y, left=body X, CCW yaw), in m/s and rad/s.
Only resets write the floating root; policy actions drive twelve leg joints.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
import omni.usd
import omni.physx
import omni.physics.tensors as tensors
from isaacsim.core.cloner import GridCloner
from natural_walk_reference import UPPER_NAMES, upper_reference, cadence
from anon_config_compat import clean_legacy_config

ROOT = Path(__file__).resolve().parents[2]


def quat_rotate_inverse(q, v):
    xyz, w = -q[..., :3], q[..., 3:4]
    t = 2*torch.cross(xyz, v, dim=-1)
    return v+w*t+torch.cross(xyz, t, dim=-1)


class AnonLocomotionEnv:
    def __init__(self, num_envs=2048, seed=42, app=None, evaluation=False, config=None):
        self.cfg = dict(asset=str(ROOT/'output/anon_physical/anon_physical.usda'),
                        physics_hz=120, decimation=4, episode_seconds=12., max_speed=.4,
                        max_yaw=.6, residual_scale=.18, gait_hz=1.25, swing_height=.045,
                        crouch=.015, curriculum_iterations=700, seed=seed)
        if config: self.cfg.update(config)
        self.cfg=clean_legacy_config(self.cfg)
        self.full_body = self.cfg.get('full_body_reference', False)
        if self.full_body:
            if self.cfg.get('articulated_wrists'):
                from build_anon_v4 import ensure_asset
            else:
                from build_anon_v3 import ensure_asset
            self.cfg['asset'] = str(ensure_asset())
        self.num_envs, self.num_actions = num_envs, (31 if self.cfg.get('articulated_wrists') else 25) if self.full_body else 12
        self.app, self.evaluation = app, evaluation
        self.device = self.cfg.get('simulation_device','cuda:0')
        if self.device not in ('cpu','cuda:0'):
            raise ValueError('simulation_device must be cpu or cuda:0')
        if self.device == 'cpu' and not evaluation:
            raise ValueError('CPU backend is for single-character evaluation, not training')
        self.dt = self.cfg['decimation']/self.cfg['physics_hz']
        self.max_episode_length = round(self.cfg['episode_seconds']/self.dt)
        self.iteration, self.physics_time, self.steps = 0, 0., 0
        torch.manual_seed(seed); np.random.seed(seed)
        context = omni.usd.get_context(); context.new_stage()
        stage = self.stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, 'Z'); UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
        scene = UsdPhysics.Scene.Define(stage, '/World/Physics')
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); scene.CreateGravityMagnitudeAttr(9.81)
        px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        px.CreateEnableGPUDynamicsAttr(self.device != 'cpu'); px.CreateBroadphaseTypeAttr('MBP' if self.device == 'cpu' else 'GPU')
        px.CreateTimeStepsPerSecondAttr(self.cfg['physics_hz'])
        # One-character playback does not need the contact buffers reserved for
        # 8192 clones. Capacity changes do not change solver/contact parameters.
        small=num_envs<=16
        px.CreateGpuMaxRigidContactCountAttr(2**16 if small else 2**23)
        px.CreateGpuMaxRigidPatchCountAttr(2**14 if small else 2**19)
        px.CreateGpuFoundLostPairsCapacityAttr(2**16 if small else 2**23)
        px.CreateGpuFoundLostAggregatePairsCapacityAttr(2**14 if small else 2**22)
        px.CreateGpuTotalAggregatePairsCapacityAttr(2**16 if small else 2**23)
        floor = UsdGeom.Cube.Define(stage, '/World/Ground')
        floor.CreateSizeAttr(1.)
        floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.05))
        floor.AddScaleOp().Set(Gf.Vec3f(1000, 1000, .1))
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        floor.CreateDisplayColorAttr([(.22, .25, .28)])
        cloner = self.cloner = GridCloner(spacing=4.)
        cloner.define_base_env('/World/envs')
        paths = cloner.generate_paths('/World/envs/env', num_envs)
        UsdGeom.Xform.Define(stage, paths[0])
        robot_path = paths[0]+'/Robot'
        prim = stage.DefinePrim(robot_path)
        prim.GetReferences().AddReference(self.cfg['asset'])
        if self.cfg.get('upper_drive_scale'):
            for item in Usd.PrimRange(prim):
                if any(key in item.GetName() for key in ['waist','shoulder','elbow','neck','forearm','wrist']):
                    drive=UsdPhysics.DriveAPI.Get(item,'angular')
                    if drive and drive.GetStiffnessAttr().Get() is not None:
                        drive.GetStiffnessAttr().Set(drive.GetStiffnessAttr().Get()*self.cfg['upper_drive_scale'])
                        drive.GetDampingAttr().Set(drive.GetDampingAttr().Get()*self.cfg.get('upper_damping_scale',1.))
        # Keep learning lightweight; a separate renderer replays actual trajectories.
        if self.cfg.get('measure_ground_contact',False):
            for side in ['left','right']:
                body_prim=stage.GetPrimAtPath(robot_path+'/Bodies/'+side+'_ankle_roll')
                PhysxSchema.PhysxContactReportAPI.Apply(body_prim).CreateThresholdAttr(0.)
        stage.GetPrimAtPath(robot_path+'/Visual').SetActive(False)
        stage.GetPrimAtPath(robot_path+'/LivePose').SetActive(False)
        self.origins_np = cloner.clone(paths[0], paths, replicate_physics=True, enable_env_ids=True)
        import carb
        from omni.physx.bindings._physx import SETTING_UPDATE_TO_USD
        carb.settings.get_settings().set(SETTING_UPDATE_TO_USD, False)
        carb.settings.get_settings().set('/physics/suppressReadback', self.device != 'cpu')
        self.interface = omni.physx.get_physx_simulation_interface()
        self.interface.attach_stage(context.get_stage_id())
        self._simulate()
        self.view = tensors.create_simulation_view('torch', context.get_stage_id(), backend='physx')
        self.robot = self.view.create_articulation_view('/World/envs/env_*/Robot')
        if self.cfg.get('measure_ground_contact',False):
            patterns=['/World/envs/env_0/Robot/Bodies/'+side+'_ankle_roll' for side in ['left','right']]
            self.ground_contact_view=self.view.create_rigid_contact_view(patterns,[['/World/Ground'],['/World/Ground']],max_contact_data_count=2048)
        assert self.robot.count == num_envs and not self.robot.shared_metatype.fixed_base
        self.names = list(self.robot.shared_metatype.dof_names)
        self.links = list(self.robot.shared_metatype.link_names)
        self.num_dofs = len(self.names)
        assert self.num_dofs == (self.num_actions if self.full_body else 23)
        self.leg_names = [s+'_'+n+'_joint' for s in ('left', 'right') for n in
                          ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch', 'ankle_roll')]
        self.leg_ids = torch.tensor([self.names.index(n) for n in self.leg_names], device=self.device)
        self.upper_ids = torch.tensor([self.names.index(n) for n in UPPER_NAMES], device=self.device) if self.full_body else None
        self.mocap = None
        if self.cfg.get('mocap_library'):
            from mocap_walk_reference import MocapReference
            self.mocap = MocapReference(self.cfg['mocap_library'], self.device)
        self.foot_ids = [self.links.index(s+'_ankle_roll') for s in ('left', 'right')]
        self.pelvis_id = self.links.index('pelvis')
        self.ids = torch.arange(num_envs, dtype=torch.int32, device=self.device)
        # Cloner order and tensor order need not match. Use observed root positions.
        self.initial_root = self.robot.get_root_transforms().clone()
        self.origins = self.initial_root[:, :3].clone(); self.origins[:, 2] = 0
        self.initial_root[:, 2] -= self.cfg['crouch']
        self.zero_vel = torch.zeros((num_envs, 6), device=self.device)
        self.zero_qvel = torch.zeros((num_envs, self.num_dofs), device=self.device)
        self.default = torch.zeros((num_envs, self.num_dofs), device=self.device)
        for side, sign in [('left', 1), ('right', -1)]:
            self.default[:, self.names.index(side+'_shoulder_roll_joint')] = sign*.65
        self.commands = torch.zeros((num_envs, 3), device=self.device)
        self.phase = torch.zeros(num_envs, device=self.device)
        self.command_age = torch.zeros(num_envs, device=self.device)
        self.motion_commands = torch.zeros_like(self.commands)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.actions = torch.zeros((num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.filtered_leg_actions = torch.zeros((num_envs,12), device=self.device)
        self.episode_returns = torch.zeros(num_envs, device=self.device)
        self.episode_errors = torch.zeros(num_envs, device=self.device)
        self.target = self.default.clone()
        self.limits = self.robot.get_dof_limits()[0].to(self.device).clone()
        manifest_path = Path(self.cfg['asset']).parent/'manifest.json' if self.full_body else ROOT/'output/anon_physical/stand/manifest.json'
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        foot_corners=[]
        for side,jp in [('left','左'),('right','右')]:
            ankle=np.array(self.manifest['links'][side+'_ankle_roll']['origin'])
            toe=np.array(self.manifest['rest_positions'][self.manifest['bone_names'].index(jp+'つま先')])
            center=np.array([ankle[0],(ankle[1]+toe[1])/2,.035])-ankle
            foot_corners.append(np.array([[x,y,z] for x in [-.0475,.0475] for y in [-.115,.115] for z in [-.035,.035]])+center)
        self.foot_corners=torch.tensor(np.array(foot_corners),device=self.device,dtype=torch.float32)
        self.geometry = []
        for side in ('left', 'right'):
            o = self.manifest['links']
            hip = np.array(o[side+'_hip_pitch']['origin'])
            knee = np.array(o[side+'_knee_pitch']['origin'])
            ankle = np.array(o[side+'_ankle_roll']['origin'])
            a, b = np.linalg.norm(knee-hip), np.linalg.norm(ankle-knee)
            angle = math.atan2(knee[1]-hip[1], hip[2]-knee[2])
            knee_angle = math.atan2(ankle[1]-knee[1], knee[2]-ankle[2])-angle
            self.geometry.append((hip, ankle-hip, a, b, angle, knee_angle))
        self.default[:, self.leg_ids] = self.reference(torch.zeros_like(self.phase), self.commands)
        if self.full_body:
            self.default[:, self.upper_ids] = self.upper_reference(torch.zeros_like(self.phase), self.commands)
            self.style_target = self.default.clone()
            self.previous_style_target = self.default.clone()
            self.upper_scale = torch.tensor([.07, .05, .05, .05, .05]+[.10, .08, .08, .08]*2, device=self.device)
        self.reset(self.ids)
        self.refresh()
        print(f'ANON_ENV_READY envs={num_envs} device={self.robot.get_dof_positions().device} actions={self.num_actions} dofs={self.num_dofs}', flush=True)

    def _simulate(self):
        dt = 1/self.cfg['physics_hz']
        self.interface.simulate(dt, self.physics_time)
        self.interface.fetch_results()
        self.physics_time += dt

    def reference(self, phase, commands):
        speed = torch.linalg.vector_norm(commands[:, :2], dim=-1)
        moving = (speed > .025) | (commands[:, 2].abs() > .04)
        target = torch.zeros((self.num_envs, 12), device=self.device)
        period = 1/(cadence(commands) if self.full_body else self.cfg['gait_hz'])
        for side in range(2):
            p = (phase+side*.5) % 1.
            swing = ((p-.6)/.4).clamp(0, 1)
            wave = torch.where(p < .6, .5-p/.6, -.5*torch.cos(math.pi*swing))
            lift_wave = torch.sin(math.pi*swing).clamp_min(0)
            if self.full_body: lift_wave = lift_wave.square()
            if self.cfg.get('smooth_foot_trajectory',False):
                # Match stance position, velocity and acceleration at both swing boundaries.
                m=-.4/.6
                swing_wave=-.5+m*swing+(1-m)*(10*swing**3-15*swing**4+6*swing**5)
                wave=torch.where(p<.6,.5-p/.6,swing_wave)
                lift_wave=64*swing**3*(1-swing)**3
                activity=torch.maximum((speed-.025)/.095,(commands[:,2].abs()-.04)/.26).clamp(0,1)
                activity=activity.square()*(3-2*activity)
                lift_wave=lift_wave*activity
            rise = lift_wave*self.cfg['swing_height']*moving
            hip, delta, a, b, theta0, knee0 = self.geometry[side]
            dx = delta[0]+(commands[:, 1]-commands[:, 2]*hip[1])*period*.6*wave
            if self.cfg.get('natural_gait', False):
                # Side-stepping needs room for the trailing foot to pass without
                # crossing the support foot at the enlarged speed boundary.
                dx = dx+(1. if side == 0 else -1.)*self.cfg.get('lateral_stance_gain', 0.)*commands[:, 1].abs()
            dy = delta[1]+(-commands[:, 0]+commands[:, 2]*hip[0])*period*.6*wave
            dz = delta[2]+self.cfg['crouch']+rise
            lateral = -torch.atan2(dx, torch.sqrt(dy*dy+dz*dz))
            dist2 = dx*dx+dy*dy+dz*dz
            knee = torch.acos(((dist2-a*a-b*b)/(2*a*b)).clamp(-.99, .995))
            pitch = torch.atan2(dy, -dz)-torch.atan2(b*torch.sin(knee), a+b*torch.cos(knee))-theta0
            k = side*6
            yaw = commands[:, 2]*period*.6*wave
            target[:, k] = yaw
            target[:, k+1] = lateral
            target[:, k+2] = pitch
            target[:, k+3] = knee-knee0
            target[:, k+4] = -pitch-(knee-knee0)
            target[:, k+5] = -lateral
        return target

    def upper_reference(self, phase, commands):
        if self.mocap:
            return self.mocap.sample(phase, commands, UPPER_NAMES)
        return upper_reference(phase, commands)

    def sample_commands(self, ids):
        n = len(ids)
        progress = 1. if self.evaluation else min(1., self.iteration/self.cfg['curriculum_iterations'])
        max_speed = .12+(self.cfg['max_speed']-.12)*progress
        angle = torch.rand(n, device=self.device)*2*math.pi
        speed = .05+torch.rand(n, device=self.device)*(max_speed-.05)
        cmd = torch.stack([speed*torch.cos(angle), speed*torch.sin(angle),
                           (torch.rand(n, device=self.device)*2-1)*self.cfg['max_yaw']*(.3+.7*progress)], dim=-1)
        choice = torch.rand(n, device=self.device)
        cmd[choice < .20] = 0.
        cmd[(choice >= .20) & (choice < .30), :2] = 0.
        boundary = (torch.rand(n, device=self.device) < self.cfg.get('boundary_command_probability', 0.)) & (choice >= .30)
        cardinal = torch.randint(0, 4, (n,), device=self.device)*math.pi/2
        cmd[boundary, 0] = max_speed*torch.cos(cardinal[boundary])
        cmd[boundary, 1] = max_speed*torch.sin(cardinal[boundary])
        cmd[boundary, 2] = 0.
        self.commands[ids] = cmd
        self.command_age[ids] = 0.

    def reset(self, ids):
        if len(ids) == 0: return
        self.sample_commands(ids)
        self.motion_commands[ids] = 0.
        self.phase[ids] = 0. if self.evaluation else torch.rand(len(ids), device=self.device)
        self.actions[ids] = 0.; self.previous_actions[ids] = 0.
        self.filtered_leg_actions[ids] = 0.
        self.episode_length_buf[ids] = 0
        self.episode_returns[ids] = 0.; self.episode_errors[ids] = 0.
        pose = self.initial_root.clone()
        self.robot.set_root_transforms(pose, ids)
        self.robot.set_root_velocities(self.zero_vel, ids)
        self.robot.set_dof_positions(self.default, ids)
        self.robot.set_dof_velocities(self.zero_qvel, ids)
        self.robot.set_dof_position_targets(self.default, ids)
        if self.full_body:
            self.style_target[ids] = self.default[ids]
            self.previous_style_target[ids] = self.default[ids]

    def refresh(self):
        self.root = self.robot.get_root_transforms().clone()
        self.root_vel = self.robot.get_root_velocities().clone()
        self.q = self.robot.get_dof_positions().clone()
        self.qd = self.robot.get_dof_velocities().clone()
        self.poses = self.robot.get_link_transforms().clone()
        self.link_vel = self.robot.get_link_velocities().clone()
        quat = self.root[:, 3:]
        self.vel_b = quat_rotate_inverse(quat, self.root_vel[:, :3])
        self.omega_b = quat_rotate_inverse(quat, self.root_vel[:, 3:])
        gravity = torch.zeros((self.num_envs, 3), device=self.device); gravity[:, 2] = -1.
        self.gravity = quat_rotate_inverse(quat, gravity)
        self.logical_vel = torch.stack([-self.vel_b[:, 1], self.vel_b[:, 0], self.root_vel[:, 5]], dim=-1)

    def get_observations(self):
        self.refresh()
        obs = torch.cat([self.vel_b*2, self.omega_b*.25, self.gravity, self.commands*2,
                         (self.q-self.default)[:, self.leg_ids], self.qd[:, self.leg_ids]*.05,
                         self.actions[:, :12], torch.sin(2*math.pi*self.phase)[:, None],
                         torch.cos(2*math.pi*self.phase)[:, None]], dim=-1)
        if self.full_body:
            obs = torch.cat([obs, (self.q-self.default)[:, self.upper_ids],
                             self.qd[:, self.upper_ids]*.05, self.actions[:, 12:]], dim=-1)
        return TensorDict({'policy': obs.clamp(-20, 20)}, batch_size=[self.num_envs], device=self.device)

    def step(self, actions):
        previous_qd = self.qd[:,self.leg_ids].clone()
        previous_feet = self.poses[:,self.foot_ids].clone()
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-2., 2.))
        moving = (torch.linalg.vector_norm(self.commands[:, :2], dim=-1) > .025) | (self.commands[:, 2].abs() > .04)
        if self.cfg.get('arm_swing', False):
            # Smooth the commanded gait/arm reference, while the policy sees the
            # actual requested command and learns its transient response.
            self.motion_commands.lerp_(self.commands, .15)
            gait_commands = self.motion_commands
        else:
            gait_commands = self.commands
        frequency = cadence(gait_commands) if self.full_body else self.cfg['gait_hz']
        if self.cfg.get('smooth_foot_trajectory',False):
            moving = (torch.linalg.vector_norm(gait_commands[:,:2],dim=-1)>.025) | (gait_commands[:,2].abs()>.04)
        self.phase = (self.phase+self.dt*frequency*moving) % 1.
        self.target.copy_(self.default)
        self.filtered_leg_actions.lerp_(self.actions[:,:12],self.cfg.get('leg_action_filter_alpha',1.))
        self.target[:, self.leg_ids] = self.reference(self.phase, gait_commands)+self.cfg['residual_scale']*self.filtered_leg_actions
        if self.full_body:
            self.previous_style_target.copy_(self.style_target)
            self.style_target.copy_(self.default)
            self.style_target[:, self.leg_ids] = self.reference(self.phase, gait_commands)
            self.style_target[:, self.upper_ids] = self.upper_reference(self.phase, gait_commands)
            self.target[:, self.upper_ids] = self.style_target[:, self.upper_ids]+self.upper_scale*self.actions[:, 12:]
        elif self.cfg.get('arm_swing', False):
            # Contralateral arm swing is a physical shoulder/elbow PD target,
            # present during training as well as inference, not a render edit.
            gain = self.cfg.get('arm_swing_gain', 1.)
            wave = torch.cos(2*math.pi*self.phase)
            forward = gait_commands[:, 0]
            turn = gait_commands[:, 2]
            for side, sign in [('left', 1.), ('right', -1.)]:
                swing = gain*(sign*.65*forward+.08*turn)*wave
                self.target[:, self.names.index(side+'_shoulder_pitch_joint')] = swing
                self.target[:, self.names.index(side+'_elbow_flex_joint')] = -sign*(.12+.10*gain*torch.abs(forward)/.4)
        self.target = torch.maximum(torch.minimum(self.target, self.limits[:, 1]-.015), self.limits[:, 0]+.015)
        self.robot.set_dof_position_targets(self.target, self.ids)
        for _ in range(self.cfg['decimation']): self._simulate()
        self.refresh()
        self.steps += 1; self.episode_length_buf += 1; self.command_age += self.dt
        error = (self.logical_vel[:, :2]-self.commands[:, :2]).square().sum(-1)
        yaw_error = (self.logical_vel[:, 2]-self.commands[:, 2]).square()
        height = self.root[:, 2]
        if self.cfg.get('pelvis_height_control', False): height = self.poses[:, self.pelvis_id, 2]
        tilt = self.gravity[:, :2].square().sum(-1)
        feet = self.poses[:, self.foot_ids]
        foot_vel = self.link_vel[:, self.foot_ids, :3]
        foot_height = feet[:, :, 2]-.087
        contact = foot_height < .025
        phases = (self.phase[:, None]+torch.tensor([0., .5], device=self.device)) % 1.
        desired_stance = (phases < .6) | ~moving[:, None]
        swing_p = ((phases-.6)/.4).clamp(0, 1)
        desired_height = torch.sin(math.pi*swing_p).clamp_min(0)*self.cfg['swing_height']*moving[:, None]
        if self.cfg.get('smooth_foot_trajectory',False):
            activity=torch.maximum((torch.linalg.vector_norm(gait_commands[:,:2],dim=-1)-.025)/.095,
                                   (gait_commands[:,2].abs()-.04)/.26).clamp(0,1)
            activity=activity.square()*(3-2*activity)
            desired_height=64*swing_p**3*(1-swing_p)**3*self.cfg['swing_height']*activity[:,None]
        contact_match = (contact == desired_stance).float().mean(-1)
        clearance = torch.exp(-((foot_height-desired_height)/.035).square()).mean(-1)
        slide = (foot_vel[:, :, :2].square().sum(-1)*contact).sum(-1)
        action_rate = (self.actions-self.previous_actions).square().sum(-1)
        residual = self.actions.square().sum(-1)
        natural_penalty = torch.zeros_like(error)
        if self.cfg.get('natural_gait', False):
            foot_gravity = quat_rotate_inverse(feet[:, :, 3:].reshape(-1, 4),
                                                torch.tensor([0., 0., -1.], device=self.device).expand(self.num_envs*2, -1))
            foot_tilt = foot_gravity[:, :2].square().sum(-1).reshape(self.num_envs, 2)
            # Discourage toe/edge support and fast foot motion in stance.
            natural_penalty = .5*(foot_tilt*desired_stance).mean(-1)+.5*slide+.008*action_rate
        reward = (3.*torch.exp(-error/.035)+1.5*torch.exp(-yaw_error/.12)+.5
                  +.7*torch.exp(-tilt/.05)+.5*torch.exp(-((height-(.90 if self.cfg.get('pelvis_height_control', False) else .91))/.08).square())
                  +.65*contact_match+.5*clearance-.5*slide-.018*action_rate-.012*residual)
        reward -= natural_penalty
        if self.cfg.get('contact_point_slip_weight',0.):
            # Finite-difference actual lowest sole corners, matching the offline audit.
            # PhysX link velocities refer to COM, so do not treat them as ankle velocities.
            quat=feet[:,:,None,3:].expand(-1,-1,8,-1)
            conjugate=torch.cat([-quat[...,:3],quat[...,3:]],-1)
            offset=quat_rotate_inverse(conjugate,self.foot_corners[None].expand(self.num_envs,-1,-1,-1))
            points=feet[:,:,None,:3]+offset
            bottom=points[:,:,:,2].amin(-1)
            near=(points[:,:,:,2]<bottom[:,:,None]+.002).float()
            old_quat=previous_feet[:,:,None,3:].expand(-1,-1,8,-1)
            old_conjugate=torch.cat([-old_quat[...,:3],old_quat[...,3:]],-1)
            old_points=previous_feet[:,:,None,:3]+quat_rotate_inverse(old_conjugate,self.foot_corners[None].expand(self.num_envs,-1,-1,-1))
            corner_vel=(points-old_points)/self.dt
            sole_speed2=(corner_vel[...,:2].square().sum(-1)*near).sum(-1)/near.sum(-1).clamp_min(1)
            on_ground=((bottom<.008)&(old_points[:,:,:,2].amin(-1)<.008)).float()
            slip_cost=((sole_speed2/.01).clamp(max=4)*on_ground).sum(-1)/on_ground.sum(-1).clamp_min(1)
            reward-=self.cfg['contact_point_slip_weight']*slip_cost
            leg_acc=(self.qd[:,self.leg_ids]-previous_qd)/self.dt
            smooth_cost=(leg_acc/30).square().clamp(max=9).mean(-1)
            reward-=self.cfg.get('leg_acceleration_weight',0.)*smooth_cost
        if self.full_body:
            pose_error = ((self.q[:, self.upper_ids]-self.style_target[:, self.upper_ids])/.18).square().mean(-1)
            ref_velocity = ((self.style_target-self.previous_style_target)/self.dt).clamp(-5, 5)
            vel_error = ((self.qd[:, self.upper_ids]-ref_velocity[:, self.upper_ids])/2.).square().mean(-1)
            head_omega = self.link_vel[:, self.links.index('head'), 3:5].square().sum(-1)
            reward += .8*torch.exp(-pose_error)+.25*torch.exp(-vel_error)-.025*head_omega
            if self.mocap:
                captured = self.mocap.sample(self.phase, gait_commands, self.leg_names)
                leg_style = ((self.q[:, self.leg_ids]-captured)/.45).square().mean(-1)
                reward += .35*torch.exp(-leg_style)
            head_q = self.poses[:, self.links.index('head'), 3:]
            head_gravity = quat_rotate_inverse(head_q, torch.tensor([0., 0., -1.], device=self.device).expand(self.num_envs, -1))
            head_tilt = head_gravity[:, :2].square().sum(-1)
            reward += self.cfg.get('style_head_upright_weight', 0.)*torch.exp(-head_tilt/.02)
            extras_style = (pose_error.mean(), vel_error.mean())
        fallen = (height < .65) | (self.gravity[:, 2] > -.65) | ~torch.isfinite(self.q).all(-1)
        timeout = self.episode_length_buf >= self.max_episode_length
        reward -= 5.*fallen
        reward *= self.dt
        self.episode_returns += reward; self.episode_errors += torch.sqrt(error)
        done = fallen | timeout
        extras = {'time_outs': timeout, 'log': {
            'Tracking/linear_error': torch.sqrt(error).mean(), 'Tracking/yaw_error': torch.sqrt(yaw_error).mean(),
            'Stability/height': height.mean(), 'Stability/fall_rate': fallen.float().mean(),
            'Gait/contact_match': contact_match.mean(), 'Gait/foot_slide': slide.mean()}}
        if self.full_body:
            extras['log']['Style/upper_pose_error'] = extras_style[0]
            extras['log']['Style/upper_velocity_error'] = extras_style[1]
        if self.evaluation:
            self.recorded_state = (self.poses.clone(), self.q.clone(), self.logical_vel.clone(), self.phase.clone())
        ids = done.nonzero(as_tuple=False).flatten()
        if len(ids):
            extras['log']['Episode/seconds'] = self.episode_length_buf[ids].float().mean()*self.dt
            extras['log']['Episode/return'] = self.episode_returns[ids].mean()
            self.reset(ids)
        if not self.evaluation:
            self.sample_commands((self.command_age > 4.).nonzero(as_tuple=False).flatten())
        return self.get_observations(), reward, done, extras

    def close(self):
        self.interface.detach_stage()

    def reference_step(self):
        """Kinematic reference preview only. Never used for policy rollouts."""
        self.motion_commands.lerp_(self.commands, .15)
        moving = (self.commands[:, :2].norm(dim=-1) > .025) | (self.commands[:, 2].abs() > .04)
        self.phase = (self.phase+self.dt*cadence(self.motion_commands)*moving) % 1.
        target = self.default.clone()
        target[:, self.leg_ids] = self.reference(self.phase, self.motion_commands)
        target[:, self.upper_ids] = self.upper_reference(self.phase, self.motion_commands)
        self.preview_yaw = getattr(self, 'preview_yaw', torch.zeros(self.num_envs, device=self.device))
        root = getattr(self, 'preview_root', self.initial_root.clone()).clone()
        self.preview_yaw += self.dt*self.motion_commands[:, 2]
        yaw = self.preview_yaw
        f, l = self.motion_commands[:, 0], self.motion_commands[:, 1]
        root[:, 0] += self.dt*(l*torch.cos(yaw)+f*torch.sin(yaw))
        root[:, 1] += self.dt*(l*torch.sin(yaw)-f*torch.cos(yaw))
        root[:, 3:5] = 0.; root[:, 5] = torch.sin(yaw/2); root[:, 6] = torch.cos(yaw/2)
        self.preview_root = root
        self.robot.set_root_transforms(root, self.ids)
        self.robot.set_dof_positions(target, self.ids)
        self.view.update_articulations_kinematic()
        self.refresh()
        return self.poses.clone(), target.clone(), self.motion_commands.clone(), self.phase.clone()
