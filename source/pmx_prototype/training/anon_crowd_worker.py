"""One Lab Direct physics scene and one batched policy for all crowd actors."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
import argparse,json,traceback,os
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--characters',type=int,required=True)
p.add_argument('--body-device',choices=['cpu','cuda:0'],default='cpu')
p.add_argument('--seed',type=int,default=42)
p.add_argument('--address',required=True);p.add_argument('--authkey',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--inter-character-collisions',action='store_true')
p.add_argument('--manual-control',action='store_true')
p.add_argument('--controlled-actor',type=int,default=0)
a=p.parse_args()
from anon_startup_progress import StartupJournal
journal=StartupJournal(a.output,'policy');journal.mark('engine')
from isaaclab.app import AppLauncher
app=AppLauncher(dict(headless=True,device=a.body_device,disable_viewport_updates=True,
                     limit_cpu_threads=2,kit_args='--/persistent/physics/numThreads=0')).app
env=connection=None
try:
    journal.mark('imports')
    import torch
    torch.set_num_threads(1)
    from pxr import UsdPhysics
    from anon_lab_direct_env import AnonDirectEnv,AnonLabRunnerEnv
    from anon_policy_transport import connect_worker
    from anon_config_compat import clean_legacy_config
    from locomotion_acceptance import inspector_planar_speed_limit
    from command_plan import yaw_speed_limit
    from rsl_rl.runners import OnPolicyRunner

    class CrowdDirectEnv(AnonDirectEnv):
        def __init__(self,*args,**kwargs):
            self.controlled_actor=a.controlled_actor
            super().__init__(*args,**kwargs)

        def _setup_scene(self):
            super()._setup_scene()
            # Lab uses inverted filters: listed groups ARE allowed to collide.
            # Preserve isolation by default; optionally allow other actors,
            # retaining the asset-level self exclusions in either mode.
            groups=[]
            for prim in self.sim.stage.Traverse():
                if prim.IsA(UsdPhysics.CollisionGroup):
                    group=UsdPhysics.CollisionGroup(prim)
                    included=group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()
                    if any(str(path) in self.scene.env_prim_paths for path in included):groups.append(group)
            if len(groups)!=self.num_envs:raise RuntimeError('Cannot identify all Lab actor collision groups')
            if not self.sim.stage.GetPrimAtPath('/World/Physics').GetAttribute('physxScene:invertCollisionGroupFilter').Get():
                raise RuntimeError('Unexpected Lab collision filter convention')
            paths=[g.GetPath() for g in groups]
            if a.inter_character_collisions:
                for group in groups:
                    rel=group.GetFilteredGroupsRel()
                    rel.SetTargets(list(dict.fromkeys([*rel.GetTargets(),*paths])))
            self.crowd_collision_groups=[dict(path=str(g.GetPath()),allowed=[str(t) for t in g.GetFilteredGroupsRel().GetTargets()]) for g in groups]

        def _reset_envs_from_buffer(self):
            if not a.manual_control:return super()._reset_envs_from_buffer()
            ids=self.reset_buf.nonzero(as_tuple=False).flatten()
            ids=ids[ids!=self.controlled_actor]
            if len(ids):self._reset_idx(ids)
            return ids

    journal.mark('config')
    config=clean_legacy_config(json.loads((a.checkpoint.parent/'env_config.json').read_text(encoding='utf-8')))
    if not config.get('continuous_commands') or not config.get('unified_walk_run'):
        raise ValueError('Crowd viewer requires the current continuous unified walk/run policy')
    config.update(simulation_device=a.body_device,runtime_inference_only=True,runtime_manual_reset_only=a.manual_control)
    if config.get('amp_enabled'):config['amp_reward_diagnostics']=False
    with np.load(config['full_motion_path']) as bank:
        config['imitation_evaluation_initial_clip']=int(np.linalg.norm(bank['commands'],axis=-1).argmin())
    journal.mark('environment')
    env=AnonLabRunnerEnv(CrowdDirectEnv(a.characters,a.seed,app,evaluation=True,config=config))
    env.max_episode_length=round(3600/env.dt)
    journal.mark('network')
    agent=json.loads((a.checkpoint.parent/'agent_config.json').read_text(encoding='utf-8'))
    runner=OnPolicyRunner(env,agent,log_dir=None,device=a.body_device)
    runner.load(str(a.checkpoint),map_location=a.body_device)
    policy=runner.get_inference_policy()
    journal.mark('reset');env.reset(env.ids);env.refresh()
    connection=connect_worker(a.address,a.authkey)
    import omni.usd
    stage_id=int(omni.usd.get_context().get_stage_id())
    controlled=a.controlled_actor;previous_fallen=np.zeros(a.characters,bool)
    if not 0<=controlled<a.characters:raise ValueError('Controlled actor out of range')
    from anon_joint_telemetry import JointTelemetry
    telemetry=JointTelemetry(env) if a.manual_control else None
    limits=np.array([inspector_planar_speed_limit(row,env.cfg) for row in env.commands.cpu().numpy()])
    def publish(seq,falls=None,resets=None,fallen=None,selection_changed=False):
        applied=env.commands.cpu().numpy().copy()
        planned=env.continuous_ref_commands
        if env.cfg.get('command_yaw_limit_by_speed'):applied[:,2]=planned[:,2].cpu().numpy()
        state=dict(seq=seq,poses=env.poses.cpu().numpy(),commands=applied,
            yaw_limits=yaw_speed_limit(planned[:,:2].norm(dim=-1),env.cfg).cpu().numpy(),planar_speed_limits=limits,
            velocity=env.logical_vel.cpu().numpy(),headings=env.reference_heading.cpu().numpy(),
            falls=np.zeros(a.characters,bool) if falls is None else falls,
            resets=np.zeros(a.characters,bool) if resets is None else resets,
            roots=env.root.cpu().numpy(),joint_positions=env.q.cpu().numpy(),controlled_actor=controlled,
            fallen=np.zeros(a.characters,bool) if fallen is None else fallen,worker_pid=os.getpid(),physics_stage_id=stage_id)
        if telemetry is not None:
            state.update(telemetry.read(env,index=controlled,reset=seq==-1 or selection_changed or bool(resets[controlled])),
                checkpoint=str(a.checkpoint),generation=f'actor_{controlled}')
        if seq==-1:state.update(config=env.cfg,links=env.links,joint_names=env.names,physics_stage_id=stage_id,
            collision_groups=env.unwrapped.crowd_collision_groups,actor_paths=env.robot.prim_paths)
        connection.send(state)
    publish(-1);journal.mark('ready')
    while True:
        request=connection.recv()
        if request.get('stop'):break
        chosen=int(request.get('controlled_actor',controlled))
        if not 0<=chosen<a.characters:raise ValueError('Controlled actor out of range')
        selection_changed=chosen!=controlled;controlled=chosen;env.unwrapped.controlled_actor=controlled
        resets=np.zeros(a.characters,bool)
        if request.get('reset'):
            with torch.inference_mode():env.reset(env.ids);env.refresh()
            resets[:]=True
        else:
            ids=np.asarray(request.get('reset_ids',[]),dtype=np.int64)
            if ids.ndim!=1 or np.any((ids<0)|(ids>=a.characters)):raise ValueError('Reset actor out of range')
            if len(ids):
                ids=np.unique(ids)
                with torch.inference_mode():env.reset(torch.as_tensor(ids,device=env.device,dtype=torch.int32));env.refresh()
                resets[ids]=True
        previous_fallen[resets]=False
        requested=np.asarray(request['commands'],dtype=np.float32)
        if requested.shape!=(a.characters,3) or not np.isfinite(requested).all():raise ValueError('Invalid crowd commands')
        limits=np.array([env.cfg.get('max_speed',1.2) if i==controlled and request.get('acceptance_inputs')
                         else inspector_planar_speed_limit(row,env.cfg) for i,row in enumerate(requested)])
        requested=requested.copy()
        requested[:,:2]*=np.minimum(1.,limits/np.maximum(np.linalg.norm(requested[:,:2],axis=1),1e-6))[:,None]
        requested[:,2]=np.clip(requested[:,2],-env.cfg.get('max_yaw',.6),env.cfg.get('max_yaw',.6))
        env.commands.copy_(torch.as_tensor(requested,device=env.device));env._pending_command_delta=None
        with torch.inference_mode():
            if request.get('push_character') and controlled>=0:
                from anon_character_push import push_character
                print('CHARACTER_PUSH',json.dumps(push_character(env,controlled)),flush=True)
            _,_,done,extras=env.step(policy(env.get_observations()))
        env.refresh()
        ended=done.cpu().numpy().astype(bool)
        fallen=(done & ~extras['time_outs']).cpu().numpy().astype(bool)
        falls=fallen & ~previous_fallen
        automatic=ended.copy()
        if a.manual_control:automatic[controlled]=False
        previous_fallen=fallen & ~automatic
        publish(int(request['seq']),falls,resets|automatic,fallen=fallen & ~automatic,selection_changed=selection_changed)
except BaseException:
    journal.mark('failed','See policy_worker.log');traceback.print_exc();raise
finally:
    if connection is not None:connection.close()
    if env is not None:env.close()
    app.close()
