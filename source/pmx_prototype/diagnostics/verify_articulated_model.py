"""Check actual PhysX forward kinematics against the arbitrary-axis retargeter."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from isaaclab.app import AppLauncher
app=AppLauncher(dict(headless=True,device='cuda:0')).app
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from anon_environment import create_environment
from retarget_100style import Retarget
root=PROJECT_ROOT;out=root/'output/anon_locomotion/running_wrists'
env=None;code=0
try:
    cfg=json.loads((out/'training.json').read_text());cfg.update(amp_reward_diagnostics=False,imitation_evaluation_initial_clip=2)
    env=create_environment(1,42,app,evaluation=True,config=cfg)
    for _ in range(12):env.step(torch.zeros_like(env.actions))
    model=Retarget(root/'output/anon_physical_v4/manifest.json')
    actual=env.poses[0].cpu().numpy();q=env.q[0].cpu().numpy();pelvis=actual[env.links.index('pelvis')]
    qm=q[[env.names.index(n) for n in model.names]]
    p,r=model.fk(qm,pelvis[:3],R.from_quat(pelvis[3:]).as_matrix())
    pe=[];re=[]
    for i,name in enumerate(env.links):
        pe.append(float(np.linalg.norm(actual[i,:3]-p[name])))
        re.append(float(R.from_matrix(r[name].T@R.from_quat(actual[i,3:]).as_matrix()).magnitude()))
    report=dict(dofs=env.num_dofs,observations=env.get_observations()['policy'].shape[-1],max_position_error_m=max(pe),max_rotation_error_rad=max(re),passed=max(pe)<.002 and max(re)<.01)
    (out/'articulation_verification.json').write_text(json.dumps(report,indent=2));print('ARTICULATION_CHECK',report,flush=True)
    assert report['passed']
except Exception:
    import traceback;traceback.print_exc();code=1
finally:
    if env:env.close()
    app.close(exit_code=code)
