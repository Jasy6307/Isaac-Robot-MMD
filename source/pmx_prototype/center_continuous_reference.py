"""Canonicalize motion origins about the physical root used for velocity control.

Pure coordinate/pivot correction: q, qd, contacts and body-relative poses remain
unchanged. Remove each clip's root XY center and its rotational velocity term.
"""
from pathlib import Path
import json,numpy as np,torch
from full_motion_reference import FullMotionReference
from continuous_motion_reference import sample_continuous,align_contacts
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'output/anon_locomotion/continuous_wrists';out=work/'centered';out.mkdir(exist_ok=True)
z=np.load(work/'full_motion.npz');d={k:z[k].copy() for k in z.files};root=list(d['link_names']).index('waist_pitch')
centers=d['positions'][:,:,root,:3].mean(1);centers[:,2]=0
for i,center in enumerate(centers):
    d['positions'][i]-=center
    d['velocities'][i,:,:,:3]-=np.cross([0.,0.,d['commands'][i,2]],center)
assert all(np.array_equal(d[k],z[k]) for k in ['q','qd','contacts'])
np.savez_compressed(out/'full_motion.npz',**d)
paths=[]
for c in [2,4,5,6,9,10]:
    ex={k:d[k][c] for k in ['q','qd','positions','rotations','velocities','contacts']}
    ex.update(joint_names=d['joint_names'],link_names=d['link_names'],idle=d['idle'],period=d['periods'][c],speed=d['speeds'][c],command=d['commands'][c])
    path=out/f'expert_{c}.npz';np.savez_compressed(path,**ex);paths.append(str(path))
m=FullMotionReference(str(out/'full_motion.npz'),'cpu');align_contacts(m);tests=[]
for i in range(8):
    for yaw in [-.6,0.,.6]:
        cmd=np.array([.4*np.cos(i*np.pi/4),.4*np.sin(i*np.pi/4),yaw],dtype='float32')
        s=sample_continuous(m,torch.arange(1024)/1024,torch.tensor(cmd).repeat(1024,1));v=s['velocities'][:,root,:3].mean(0).numpy();actual=np.array([-v[1],v[0]])
        error=float(np.linalg.norm(actual-cmd[:2]));tests.append(dict(command=cmd.tolist(),mean_reference_velocity=actual.tolist(),error=error))
assert max(t['error'] for t in tests)<1e-5
(out/'verification.json').write_text(json.dumps(dict(method=__doc__,root_link='waist_pitch',clip_centers=centers.tolist(),q_qd_contacts_unchanged=True,max_reference_velocity_error=max(t['error'] for t in tests),tests=tests),indent=2))
(work/'centered_training.json').write_text(json.dumps(dict(full_motion_path=str(out/'full_motion.npz'),amp_expert_paths=paths,amp_reset_discriminator=True),indent=2))
print('Root-coordinate velocity audit passed',max(t['error'] for t in tests))
