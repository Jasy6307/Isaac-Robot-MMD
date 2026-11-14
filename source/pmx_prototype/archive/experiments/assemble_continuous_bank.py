"""Replace forward reference with an unaveraged passage and preserve idle."""

# Allow both the canonical tool path and the historical root entry point.
import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[2]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--motion',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--source-report',type=Path)
a=p.parse_args();old=np.load(a.base/'full_motion.npz');long=np.load(a.motion);count=len(long['q']);n=old['q'].shape[1]
phi=np.arange(count)/count;x=phi*n;i=x.astype(int)%n;j=(i+1)%n;w=x-i
data={k:old[k].copy() for k in old.files}
for key in ['q','qd','positions','rotations','velocities','contacts']:
    value=old[key];u=value[:,i];v=value[:,j].copy()
    if key=='rotations':v[(u*v).sum(-1)<0]*=-1
    weight=w.reshape((1,count)+(1,)*(u.ndim-2));sample=u*(1-weight)+v*weight
    if key=='rotations':sample/=np.linalg.norm(sample,axis=-1,keepdims=True)
    sample[2]=long[key];data[key]=sample.astype(value.dtype)
data['periods'][2]=float(long['period']);data['speeds'][2]=float(long['speed'])
data['commands'][2]=long['command']
# Exactly six chronological cycles, with unequal intervals inside the passage.
meta=json.loads((a.source_report or a.motion.with_name('long_raw.json')).read_text());cycles=meta['consecutive_gait_cycles']
data['periods'][3]=float(long['period'])
data['gait_sin']=np.tile(np.sin(2*np.pi*phi),(len(data['periods']),1)).astype('float32')
data['gait_cos']=np.tile(np.cos(2*np.pi*phi),(len(data['periods']),1)).astype('float32')
data['gait_sin'][2]=long['gait_sin'];data['gait_cos'][2]=long['gait_cos']
data['gait_sin'][3]=np.sin(2*np.pi*(cycles*phi+.9921875));data['gait_cos'][3]=np.cos(2*np.pi*(cycles*phi+.9921875))
a.output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output/'full_motion.npz',**data)
cfg=dict(full_motion_path=str((a.output/'full_motion.npz').resolve()),amp_expert_paths=[str(a.motion.resolve())],amp_reset_discriminator=True,
         episode_seconds=26.,imitation_command_hold_seconds=10.,diagnostic_controller_trace=False,
         train_upper_std_cap=.08,train_entropy_coef=.001,amp_evaluation_forward_seconds=30.)
(a.output/'training_override.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
report=dict(frames_per_clip=count,forward_period_seconds=float(data['periods'][2]),gait_cycles=cycles,cycle_averaging=False,idle_geometry='Preserved from last accepted model; static clip only gets a synchronized phase clock',source_motion=str(a.motion.resolve()),base_model=str(a.base.resolve()))
(a.output/'bank_processing.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
