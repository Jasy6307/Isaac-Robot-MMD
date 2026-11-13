"""Check that the new observations are invariant under global yaw rotation."""
import json
from pathlib import Path
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'fast_shutdown':True})
import torch
from anon_imitation_env import AnonImitationEnv
ROOT=Path(__file__).resolve().parents[2];env=None;result=0
try:
    cfg=json.loads((ROOT/'logs/rsl_rl/anon_locomotion/v6_steering_steady/env_config.json').read_text())
    env=AnonImitationEnv(4,42,app,True,cfg)
    env.phase[:]=.3;env.clip_ids[:]=0;env.previous_clip_ids[:]=0;env.blend_age[:]=1
    env.commands[:]=env.motion.commands[0]
    env.reference_heading[:]=torch.tensor([0.,.7,1.7,-2.4],device=env.device)
    ref=env.sample_reference(env.phase)
    pose=env.initial_root.clone();pose[:,:3]=env.origins+ref['positions'][:,env.base_reference_id]
    pose[:,3:]=ref['rotations'][:,env.base_reference_id]
    env.robot.set_root_transforms(pose,env.ids);env.robot.set_root_velocities(ref['velocities'][:,env.base_reference_id].contiguous(),env.ids)
    env.robot.set_dof_positions(ref['q'][:,env.q_order].contiguous(),env.ids);env.robot.set_dof_velocities(ref['qd'][:,env.q_order].contiguous(),env.ids)
    env.view.update_articulations_kinematic();env.refresh()
    obs=env.get_observations()['policy'];delta=(obs-obs[0]).abs()
    print('HEADING_INVARIANCE',json.dumps(dict(max_error=float(delta.max()),max_index=int(delta.max(0).values.argmax()),
          per_yaw_max=delta.max(-1).values.cpu().tolist())),flush=True)
    result=int(float(delta.max())>.001)
except Exception:
    import traceback
    traceback.print_exc();result=1
finally:
    if env:env.close()
    app.close(exit_code=result)
