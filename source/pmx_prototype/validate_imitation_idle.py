"""Check captured standing reference with PD only, no policy/root assistance."""
import json
from pathlib import Path
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'fast_shutdown':True})
import torch
from anon_imitation_env import AnonImitationEnv
ROOT=Path(__file__).resolve().parents[2]
env=None
result=0
try:
    config=json.loads((ROOT/'logs/rsl_rl/anon_locomotion/v6_anatomical_speeds/env_config.json').read_text())
    config['imitation_motion_probabilities']=[0,0,0,1]
    env=AnonImitationEnv(1,42,app,False,config)
    env.evaluation=True;env.max_episode_length=10000;env.refresh()
    falls=0;min_height=2.
    for k in range(300):
        _,_,done,_=env.step(torch.zeros_like(env.actions));falls+=int(done.sum())
        min_height=min(min_height,float(env.poses[:,env.pelvis_id,2].min()))
    print('IDLE_PD_CHECK',json.dumps(dict(seconds=10,falls=falls,min_pelvis_height=min_height)),flush=True)
    result=int(falls>0)
except Exception:
    import traceback
    traceback.print_exc();result=1
finally:
    if env:env.close()
    app.close(exit_code=result)
