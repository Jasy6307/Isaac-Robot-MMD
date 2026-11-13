"""Export the exact complete imitation reference as a kinematic preview."""
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from full_motion_reference import FullMotionReference

ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'output/anon_locomotion/full_motion'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--motion',type=Path,default=folder/'forward_aligned.npz')
parser.add_argument('--output',type=Path,default=folder/'preview_aligned')
args=parser.parse_args()
motion=FullMotionReference(args.motion,'cpu')
time=np.arange(480)/30
state=motion.sample(torch.tensor(time/motion.period,dtype=torch.float32))
p=state['positions'].numpy();p[:,:,1]-=motion.speed*time[:,None]
rot=state['rotations'].numpy()
poses=np.concatenate([p,rot],-1)[:,None]
root=motion.links.index('waist_yaw');v=state['velocities'][:,root].numpy()
local=Rotation.from_quat(rot[:,root]).inv().apply(v[:,:3])
logical=np.stack([-local[:,1],local[:,0],v[:,5]],-1)[:,None]
out=args.output;out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'rollout.npz',poses=poses,joints=state['q'].numpy()[:,None],
                    dof_names=motion.names,link_names=motion.links,commands=np.tile([motion.speed,0.,0.],(480,1,1)),
                    velocities=logical,phases=(time/motion.period%1)[:,None],labels=['forward']*480,
                    dones=np.zeros((480,1),bool),dt=1/30,kind='mocap_reference')
(out/'report.json').write_text(json.dumps(dict(kind='complete_mocap_reference',
    source='100STYLE Neutral / CC BY 4.0',env_config=dict(asset=str(ROOT/'output/anon_physical_v3/anon_physical.usda'))),indent=2),encoding='utf-8')
print(out)
