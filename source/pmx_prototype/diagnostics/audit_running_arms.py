"""Measure actual arm spacing and torso-relative hand tracking in a rollout."""

# Allow both the canonical tool path and the historical root entry point.
import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from full_motion_reference import FullMotionReference
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--rollout',type=Path,required=True);p.add_argument('--reference',type=Path)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=np.load(a.rollout);poses=d['poses'][:,0];links=list(d['link_names']);time=np.arange(len(poses))*float(d['dt'])
mask=(time>=8)&(d['commands'][:,0,0]>.5)
poses=poses[mask];torso=poses[:,links.index('waist_pitch')];inv=R.from_quat(torso[:,3:]).inv()
result=dict(samples=int(mask.sum()),method='Actual measured links in the torso frame; running samples after 8s',arms={})
ref=None
if a.reference:
    motion=FullMotionReference(str(a.reference),'cpu')
    phase=torch.tensor(d['phases'][mask,0],dtype=torch.float32)
    ref=motion.sample(phase)
for side,sign in [('left',1.),('right',-1.)]:
    values={}
    for suffix,label in [('elbow_flex','elbow'),('hand','hand')]:
        local=inv.apply(poses[:,links.index(side+'_'+suffix),:3]-torso[:,:3])
        values[label+'_lateral_cm_10_50_90']=np.percentile(sign*local[:,0]*100,[10,50,90]).tolist()
        values[label+'_foreaft_range_cm_10_90']=float(np.ptp(np.percentile(local[:,1]*100,[10,90])))
    if ref is not None:
        ri=motion.links.index(side+'_hand');ti=motion.links.index('waist_pitch')
        expected=R.from_quat(ref['rotations'][:,ti].numpy()).inv()*R.from_quat(ref['rotations'][:,ri].numpy())
        actual=inv*R.from_quat(poses[:,links.index(side+'_hand'),3:])
        error=np.degrees((expected.inv()*actual).magnitude())
        values['hand_orientation_error_deg_mean']=float(error.mean())
        values['hand_orientation_error_deg_p95']=float(np.percentile(error,95))
    result['arms'][side]=values
a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
