"""Comparable geometric sole-slip estimate; contact forces are not available."""

# Allow both the canonical tool path and the historical root entry point.
import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--motion',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();corners=np.load(a.motion)['foot_corners']
def measure(path):
    d=np.load(path);links=list(d['link_names']);ids=[links.index(s+'_ankle_roll') for s in ['left','right']]
    pose=d['poses'][:,0,ids];rotation=R.from_quat(pose[:,:,3:].reshape(-1,4)).as_matrix().reshape(len(pose),2,3,3)
    sole=np.einsum('nfij,fkj->nfki',rotation,corners)+pose[:,:,:3,None].transpose(0,1,3,2)
    dt=float(d['dt']);velocity=np.gradient(sole,dt,axis=0);bottom=sole[:,:,:,2].min(-1)
    near=sole[:,:,:,2]<bottom[:,:,None]+.003
    speed=(np.linalg.norm(velocity[:,:,:,:2],axis=-1)*near).sum(-1)/near.sum(-1)
    mask=(np.arange(len(pose))*dt>=8)&(np.arange(len(pose))*dt<30)
    contact=(bottom<.012)&mask[:,None]
    return dict(geometric_ground_slip_mean_m_s=float(speed[contact].mean()),geometric_ground_slip_p95_m_s=float(np.quantile(speed[contact],.95)))
out=dict(baseline=measure(a.baseline),candidate=measure(a.candidate),method='Sole corner world velocities, lowest 3mm corners, foot bottom below 12mm; t=8..30s; geometric proxy, not force-confirmed contact')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
