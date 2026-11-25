"""Isaac Lab Direct lifecycle for the existing Anon imitation controller.

Lab owns SimulationContext, scene cloning, substeps, termination and reset
dispatch. The task retains its tested reference, XYZW PhysX tensor convention,
residual PD targets, rewards and AMP state.
"""
import copy
from importlib.metadata import version

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
import omni.physics.tensors as tensors
import omni.usd

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
try:
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
except ImportError:
    from isaaclab.sim import PhysxCfg

from anon_imitation_env import AnonImitationEnv
from anon_locomotion_env import AnonLocomotionEnv
from reference_asset_builder import resolve_body_asset
from source_layout import PROJECT_ROOT


def physics_config(num_envs):
    """Keep the authored USD solver and contact settings."""
    stage=Usd.Stage.CreateInMemory()
    api=PhysxSchema.PhysxSceneAPI.Apply(UsdPhysics.Scene.Define(stage,'/Physics').GetPrim())
    settings={}
    mapping={
        'enable_ccd':'GetEnableCCDAttr',
        'enable_stabilization':'GetEnableStabilizationAttr',
        'enable_enhanced_determinism':'GetEnableEnhancedDeterminismAttr',
        'enable_external_forces_every_iteration':'GetEnableExternalForcesEveryIterationAttr',
        'bounce_threshold_velocity':'GetBounceThresholdAttr',
        'friction_offset_threshold':'GetFrictionOffsetThresholdAttr',
        'friction_correlation_distance':'GetFrictionCorrelationDistanceAttr',
    }
    for key,getter in mapping.items():
        if hasattr(api,getter):
            value=getattr(api,getter)().Get()
            if value is not None:settings[key]=value
    settings['solver_type']=1 if api.GetSolverTypeAttr().Get()=='TGS' else 0
    small=num_envs<=16
    settings.update(gpu_max_rigid_contact_count=2**16 if small else 2**23,
                    gpu_max_rigid_patch_count=2**14 if small else 2**19,
                    gpu_found_lost_pairs_capacity=2**16 if small else 2**23,
                    gpu_found_lost_aggregate_pairs_capacity=2**14 if small else 2**22,
                    gpu_total_aggregate_pairs_capacity=2**16 if small else 2**23)
    return PhysxCfg(**settings)


class _LabSceneBinding:
    """Initialize task buffers against the scene already owned by Lab."""
    def __init__(self,owner,*args,**kwargs):
        self.lab_owner=owner
        super().__init__(*args,**kwargs)

    def _create_physics(self,num_envs,seed):
        torch.manual_seed(seed);np.random.seed(seed)
        self.stage=self.lab_owner.sim.stage
        stage_id=omni.usd.get_context().get_stage_id()
        self.view=tensors.create_simulation_view('torch',stage_id,backend='physx')
        self.robot=self.view.create_articulation_view('/World/envs/env_*/Robot')
        print('ANON_LAB_PHYSICS',self.robot.get_dof_positions().device,
              self.lab_owner.sim.physics_manager.get_device(),
              PhysxSchema.PhysxSceneAPI.Get(self.stage,'/World/Physics').GetEnableGPUDynamicsAttr().Get(),flush=True)
        actual_device=self.robot.get_dof_positions().device
        if actual_device!=torch.device(self.device):
            raise RuntimeError(f'Lab physics device mismatch: requested {self.device}, tensor view is on {actual_device}')
        self.origins_np=self.lab_owner.scene.env_origins.cpu().numpy()
        if self.cfg.get('measure_ground_contact',False):
            paths=self.lab_owner.scene.env_prim_paths
            contact_paths=paths if self.evaluation and self.cfg.get('unified_eval_parallel_envs')==4 else paths[:1]
            patterns=[f'{path}/Robot/Bodies/{side}_ankle_roll' for path in contact_paths for side in ('left','right')]
            self.ground_contact_view=self.view.create_rigid_contact_view(
                patterns,[['/World/Ground'] for _ in patterns],max_contact_data_count=2048)


    def _environment_origins(self, initial_root):
        if self.device != 'cpu':
            return super()._environment_origins(initial_root)
        # The startup simulation can move the root before the task is bound.
        # Use the authored transform as the CPU scene origin. Reading
        # the authored root-link transform also preserves PhysX articulation
        # ordering, unlike indexing the cloner's environment origins directly.
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        positions = [cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(
            path + '/Bodies/' + self.links[0])).ExtractTranslation()
            for path in self.robot.prim_paths]
        origins = torch.tensor(np.asarray(positions, dtype=np.float32), device=self.device)
        origins[:, 2] = 0
        return origins


class _BoundLocomotionTask(_LabSceneBinding, AnonLocomotionEnv):
    pass


class _BoundImitationTask(_LabSceneBinding, AnonImitationEnv):
    pass


class AnonDirectEnv(DirectRLEnv):
    """Real DirectRLEnv; base step() drives the migrated controller stages."""
    def __init__(self,num_envs,seed,app,*,evaluation,config):
        self.controller_config=copy.deepcopy(config)
        self.controller_config['asset']=str(resolve_body_asset(PROJECT_ROOT,self.controller_config))
        self.controller_config['environment_backend']='lab-direct'
        self.task=None
        self._step_context=None
        self._reset_reference=None
        self._reset_ids=None
        device=config.get('simulation_device','cuda:0')
        if device=='cpu' and not evaluation:raise ValueError('CPU training is not supported; use cuda:0')
        full_body=config.get('full_body_reference',False)
        imitation=bool(config.get('full_motion_path'))
        if imitation and not full_body:
            raise ValueError('Full-motion imitation requires the full-body articulation')
        actions=(31 if config.get('articulated_wrists') else 25) if full_body else 12
        if imitation:
            observations=7*actions+20+(11 if config.get('imitation_heading_observation') else 0)
        else:
            observations=50+3*(actions-12)
        sim=SimulationCfg(device=device,dt=1/config.get('physics_hz',120),
                          render_interval=config.get('decimation',4),physics_prim_path='/World/Physics',
                          use_fabric=True,enable_scene_query_support=True,physics=physics_config(num_envs))
        cfg=DirectRLEnvCfg(seed=seed,decimation=config.get('decimation',4),
                          episode_length_s=config.get('episode_seconds',12.),sim=sim,
                          scene=InteractiveSceneCfg(num_envs=num_envs,env_spacing=4.,replicate_physics=True),
                          action_space=actions,observation_space=observations,state_space=0,
                          events=None,compute_final_obs=False,ui_window_class_type=None)
        super().__init__(cfg)
        if self._physics_handles_decimation:
            raise ValueError('Anon requires individual substeps to preserve 120 Hz support-force sampling')
        task_class=_BoundImitationTask if imitation else _BoundLocomotionTask
        self.task=task_class(self,num_envs,seed,app,evaluation=evaluation,config=self.controller_config)
        self.episode_length_buf=self.task.episode_length_buf
        actual=self.task.get_observations()['policy'].shape[-1]
        if actual!=observations:raise ValueError(f'Observation layout mismatch: {actual} != {observations}')
        self.task.cfg['lab_direct_runtime']=dict(isaaclab_version=version('isaaclab'),
                                               environment='DirectRLEnv',physics='physx',
                                               physics_dt=self.physics_dt,control_dt=self.step_dt,
                                               actions=actions,observations=actual)
        self.obs_buf=self._get_observations()
        self.render_enabled=bool(self.sim.has_gui)
        print(f'ANON_LAB_DIRECT_READY envs={num_envs} actions={actions} observations={actual}',flush=True)

    def _setup_scene(self):
        stage=self.sim.stage
        UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
        floor=UsdGeom.Cube.Define(stage,'/World/Ground');floor.CreateSizeAttr(1.)
        floor.AddTranslateOp().Set(Gf.Vec3d(0,0,-.05));floor.AddScaleOp().Set(Gf.Vec3f(1000,1000,.1))
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim());floor.CreateDisplayColorAttr([(.22,.25,.28)])
        config=self.controller_config
        path=self.scene.env_prim_paths[0]+'/Robot'
        prim=stage.DefinePrim(path);prim.GetReferences().AddReference(config['asset'])
        if config.get('upper_drive_scale'):
            for item in Usd.PrimRange(prim):
                if any(key in item.GetName() for key in ('waist','shoulder','elbow','neck','forearm','wrist')):
                    drive=UsdPhysics.DriveAPI.Get(item,'angular')
                    if drive and drive.GetStiffnessAttr().Get() is not None:
                        drive.GetStiffnessAttr().Set(drive.GetStiffnessAttr().Get()*config['upper_drive_scale'])
                        drive.GetDampingAttr().Set(drive.GetDampingAttr().Get()*config.get('upper_damping_scale',1.))
        if config.get('measure_ground_contact',False):
            # Match Lab ContactSensor: CPU contact reports require processing.
            # A raw tensor contact view does not enable this setting itself.
            self.sim.set_setting('/physics/disableContactProcessing',False)
            for side in ('left','right'):
                PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path+'/Bodies/'+side+'_ankle_roll')).CreateThresholdAttr(0.)
        stage.GetPrimAtPath(path+'/Visual').SetActive(False)
        stage.GetPrimAtPath(path+'/LivePose').SetActive(False)
        if hasattr(self.scene,'clone_environments'):
            self.scene.clone_environments(copy_from_source=False)
        else:
            from isaaclab import cloner
            positions,_=cloner.grid_transforms(self.num_envs,self.cfg.scene.env_spacing,device=self.device)
            # Raw USD assets are not in Lab 3's AssetBase replication queue.
            # Replicate their USD subtree explicitly before publishing the plan.
            cloner.usd_replicate(stage,[self.scene.env_prim_paths[0]],['/World/envs/env_{}'],
                                 torch.arange(self.num_envs,device=self.device),positions=positions)
            plan=cloner.clone_plan_from_env_0(self.scene.env_prim_paths[0],'/World/envs/env_{}',
                                             self.num_envs,self.device,positions,global_paths=('/World/Ground',))
            cloner.replicate(plan,stage=stage)
            from isaaclab_physx.cloner import physx_replicate
            physx_replicate(stage,plan.sources,plan.destinations,plan.env_ids,plan.clone_mask,
                            positions=plan.positions,device=self.device)
            # Lab 3 PhysX replication uses USD groups, not native env IDs.
            self.scene.filter_collisions(global_prim_paths=['/World/Ground'])
        if self.device=='cpu':self.scene.filter_collisions(global_prim_paths=['/World/Ground'])

    def _pre_physics_step(self,actions):
        self._step_context=self.task._prepare_step(actions)
        self.task._begin_substeps()
        self._applied_substeps=0
        self._reset_reference=None
        self._reset_ids=self.task.ids[:0]

    def _apply_action(self):
        if self._applied_substeps:
            self.task._record_substep()
        else:
            self.task._apply_step_targets()
        self._applied_substeps+=1

    def _get_dones(self):
        self.task._record_substep()
        self.task.physics_time+=self.step_dt
        self._reward,self._done,self.extras=self.task._evaluate_step(self._step_context)
        timeout=self.extras['time_outs']
        return self._done & ~timeout,timeout

    def _get_rewards(self):
        # The shared task already applies dt exactly once.
        return self._reward

    def _reset_envs_from_buffer(self):
        ids=self.task._automatic_reset_ids(self.reset_buf)
        if len(ids):self._reset_idx(ids)
        return ids

    def _reset_idx(self,env_ids):
        if self.task is not None:
            ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.int32)
            self._reset_reference=self.task._reset_step(ids,self.extras)
            self._reset_ids=ids
        super()._reset_idx(env_ids)

    def _get_observations(self):
        if self.task is None:return {}
        if self._step_context is not None:
            observations=self.task._complete_step(self._step_context,self._reset_ids,self._reset_reference)
            self._step_context=None
            if observations is not None:return dict(observations.items())
        return dict(self.task.get_observations().items())


class AnonLabRunnerEnv(RslRlVecEnvWrapper):
    """Official Lab/RSL wrapper plus the existing evaluation task interface."""
    @property
    def cfg(self):return self.unwrapped.task.cfg

    @property
    def max_episode_length(self):return self.unwrapped.task.max_episode_length

    @max_episode_length.setter
    def max_episode_length(self,value):self.unwrapped.task.max_episode_length=value

    @property
    def episode_length_buf(self):return self.unwrapped.task.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self,value):
        self.unwrapped.episode_length_buf=value
        self.unwrapped.task.episode_length_buf=value

    def __getattr__(self,name):
        return getattr(self.unwrapped.task,name)

    def __setattr__(self,name,value):
        env=self.__dict__.get('env')
        task=getattr(env,'task',None) if env is not None else None
        if task is not None and name in task.__dict__ and name not in ('num_envs','device','num_actions','episode_length_buf','max_episode_length'):
            setattr(task,name,value)
        else:object.__setattr__(self,name,value)

    def get_observations(self):
        # Evaluators can change commands between steps, so cached obs are stale.
        return self.unwrapped.task.get_observations()

    def reset(self,ids=None,**kwargs):
        if ids is None:return super().reset()
        lab=self.unwrapped
        lab.scene.reset(ids)
        result=lab.task.reset(ids,**kwargs)
        lab._step_context=None
        lab.obs_buf=dict(lab.task.get_observations().items())
        return result
