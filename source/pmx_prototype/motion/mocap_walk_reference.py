"""Periodic 100STYLE mocap style targets, with velocity and direction blending."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from pathlib import Path
import json
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

ROOT = PROJECT_ROOT


def build_library():
    folder=ROOT/'output/anon_locomotion/100style_reference'
    tables=[]; info=[]
    for kind in ['FW','BW','SW']:
        d=np.load(folder/(kind+'.npz')); q=d['q']; names=list(d['joint_names'])
        peaks=find_peaks(q[:,names.index('left_knee_pitch_joint')],distance=18,prominence=.15)[0]
        yaw=np.unwrap(Rotation.from_matrix(d['root_rotations']).as_euler('ZYX')[:,0])
        candidates=[]
        for a,b in zip(peaks[:-1],peaks[1:]):
            period=(b-a)*float(d['dt'])
            if .9<period<1.8 and np.ptp(yaw[a:b]) < .6:
                candidates.append((float(np.ptp(yaw[a:b])),int(a),int(b)))
        candidates=sorted(candidates)[:2]
        assert candidates, kind
        cycles=[]
        for turn,a,b in candidates:
            # Captured left knee flexion peaks at phase .8, the left swing midpoint.
            t=((np.arange(128)/128-.8)%1)*(b-a)+a
            cycles.append(np.stack([np.interp(t,np.arange(len(q)),q[:,j]) for j in range(25)],-1))
        table=np.mean(cycles,axis=0)
        # Suppress discontinuities at the stitched cycle boundary using measured low harmonics.
        spectrum=np.fft.rfft(table,axis=0); spectrum[5:]=0
        table=np.fft.irfft(spectrum,n=128,axis=0)
        tables.append(table)
        info.append(dict(clip=kind,cycles=[dict(first=a,last=b,yaw_range_rad=turn) for turn,a,b in candidates]))
    idle=np.load(folder/'ID.npz')['q'].mean(0)
    # Neutral-capture sensor/rest offsets: zero neck gaze, retain a slight torso inclination.
    bias=np.zeros(25); bias[:5]=idle[:5]; bias[2]-=.04
    tables=np.array(tables)-bias
    idle=idle-bias
    np.savez_compressed(folder/'library.npz',tables=tables.astype('float32'),idle=idle.astype('float32'),joint_names=names)
    (folder/'library.json').write_text(json.dumps(dict(source='100STYLE Neutral CC BY 4.0',cycles=info,neutral_calibration_rad=bias.tolist(),harmonics=4),indent=2),encoding='utf-8')


class MocapReference:
    def __init__(self,path,device):
        d=np.load(path)
        self.tables=torch.tensor(d['tables'],device=device)
        self.idle=torch.tensor(d['idle'],device=device)
        self.names=list(d['joint_names'])
        self.mirror_index=[]; self.mirror_sign=[]
        for n in self.names:
            opposite=n.replace('left_','SIDE_').replace('right_','left_').replace('SIDE_','right_')
            self.mirror_index.append(self.names.index(opposite))
            self.mirror_sign.append(-1 if any(x in n for x in ['yaw','roll','elbow']) else 1)
        self.mirror_sign=torch.tensor(self.mirror_sign,device=device)

    def sample(self,phase,commands,names):
        x=phase*128; a=x.long()%128; b=(a+1)%128; fraction=(x%1)[:,None]
        values=self.tables[:,a,:]*(1-fraction)+self.tables[:,b,:]*fraction
        forward=commands[:,0]; lateral=commands[:,1]
        side=values[2]
        mirrored=side[:,self.mirror_index]*self.mirror_sign
        side=torch.where((lateral<0)[:,None],mirrored,side)
        fb=torch.where((forward<0)[:,None],values[1],values[0])
        side_weight=(lateral.abs()/(forward.abs()+lateral.abs()+1e-6))[:,None]
        moving=fb*(1-side_weight)+side*side_weight
        activity=torch.maximum(torch.linalg.vector_norm(commands[:,:2],dim=-1)/.3,commands[:,2].abs()/.6*.35).clamp(0,1)[:,None]
        result=self.idle+activity*(moving-self.idle)
        return result[:,[self.names.index(n) for n in names]]


if __name__=='__main__': build_library()
