"""Measure where reference motion is attenuated; plot phase-aligned joint curves."""

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trace',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
d=np.load(args.trace);names=list(d['dof_names']);dt=float(d['dt']);t=np.arange(len(d['joints']))*dt
mask=(t>=8)&(t<30)&(d['commands'][:,0,0]>.1)
span=lambda x:float(np.degrees(np.ptp(np.quantile(x,[.05,.95]))))
rows={}
for i,n in enumerate(names):
    if not any(k in n for k in ['waist','shoulder','elbow']):continue
    q,ref,target=[d[k][mask,0,i] for k in ['joints','references','targets']]
    rows[n]=dict(reference_range_deg=span(ref),target_range_deg=span(target),actual_range_deg=span(q),
                 actual_reference_rmse_deg=float(np.degrees(np.sqrt(np.mean((q-ref)**2)))),
                 actual_target_rmse_deg=float(np.degrees(np.sqrt(np.mean((q-target)**2)))))
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'controller_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
fig,axes=plt.subplots(3,2,figsize=(13,10))
phase=d['phases'][mask,0];bins=np.minimum((phase*48).astype(int),47);x=(np.arange(48)+.5)/48
for ax,name in zip(axes.ravel(),['waist_yaw_joint','waist_pitch_joint','left_shoulder_roll_joint','right_shoulder_roll_joint','left_elbow_flex_joint','right_elbow_flex_joint']):
    i=names.index(name)
    for key,label in [('references','Mocap reference'),('targets','Policy PD target'),('joints','Physical joint')]:
        values=np.degrees(d[key][mask,0,i]);mean=np.array([values[bins==b].mean() if np.any(bins==b) else np.nan for b in range(48)])
        ax.plot(x,mean,label=label)
    ax.set(title=name,xlabel='Gait phase',ylabel='Degrees');ax.grid(alpha=.25)
axes[0,0].legend();fig.suptitle('Actual controller chain: steady forward walking, t=8..30 s')
fig.tight_layout();fig.savefig(args.output/'controller_phase_curves.png',dpi=160);plt.close(fig)
print(args.output/'controller_audit.json')
