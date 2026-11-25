"""Compare complete reference FK with the physical articulation at four phases."""

# Allow both the canonical tool path and the historical root entry point.
import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import json
from pathlib import Path
from isaaclab.app import AppLauncher
app=AppLauncher(dict(headless=True,device='cuda:0')).app
import torch
from anon_environment import create_environment
root=PROJECT_ROOT
env=None
try:
    config=json.loads((root/'source/pmx_prototype/configs/training/v6/anon_v6_config.json').read_text())
    env=create_environment(1,42,app,True,config)
    for phase in [0.,.25,.5,.75]:
        env.phase[:]=phase
        env.evaluation=False
        # Use the same state mapping as reset, with a specified phase.
        ref=env.motion.sample(env.phase)
        pose=env.initial_root.clone();pose[:,:3]=env.origins+ref['positions'][:,env.base_reference_id];pose[:,3:]=ref['rotations'][:,env.base_reference_id]
        env.robot.set_root_transforms(pose,env.ids)
        env.robot.set_dof_positions(ref['q'][:,env.q_order],env.ids)
        env.view.update_articulations_kinematic();env.refresh()
        delta=(env.poses[:,:,:3]-env.origins[:,None]-ref['positions'][:,env.link_order]).norm(dim=-1)[0]
        print('FK_CHECK',phase,'root_name',env.links[0],'max_m',float(delta.max()),'worst',env.links[int(delta.argmax())],flush=True)
        print([(n,round(float(delta[env.links.index(n)]),5)) for n in ['pelvis','waist_pitch','head','left_ankle_roll','right_ankle_roll']],flush=True)
finally:
    if env:env.close()
    app.close()
