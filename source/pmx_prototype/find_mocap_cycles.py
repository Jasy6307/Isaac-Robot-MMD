"""Rank complete source side-walk cycles by heading and endpoint consistency."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from retarget_100style import read_bvh

ROOT=Path(__file__).resolve().parents[2]
nodes,p,r,dt=read_bvh(ROOT/'media/mocap/100style/Neutral_SW.bvh')
ids={n['name']:i for i,n in enumerate(nodes)}
hip,knee,ankle=[ids['Left'+n] for n in ['Hip','Knee','Ankle']]
u=p[:,knee]-p[:,hip];v=p[:,ankle]-p[:,knee]
flex=np.arccos(np.clip((u*v).sum(-1)/(np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)),-1,1))
peaks=find_peaks(flex,distance=30,prominence=.12)[0]
yaw=np.unwrap(Rotation.from_matrix(r[:,0]).as_euler('ZYX')[:,0])
speed=np.gradient(p[:,0],dt,axis=0)
local=np.einsum('nij,nj->ni',r[:,0].transpose(0,2,1),speed)
keys=[ids[n] for n in ['LeftKnee','LeftAnkle','RightKnee','RightAnkle']]
relative=np.einsum('nij,nkj->nki',r[:,0].transpose(0,2,1),p[:,keys]-p[:,0,None,:])
candidates=[]
for a,b in zip(peaks[:-1],peaks[1:]):
    period=(b-a)*dt
    if a<354 or b>5730 or not .9<period<2.5:continue
    f,l=-local[a:b,1].mean(),local[a:b,0].mean()
    turn=float(np.ptp(yaw[a:b]));endpoint=float(np.sqrt(np.mean((relative[b]-relative[a])**2)))
    if l<.2 or abs(f)>.2 or turn>.35:continue
    cv=float(np.std(local[a:b,0])/max(l,.01))
    score=turn+endpoint*8+cv*.15
    candidates.append(dict(first=int(a),last=int(b),period=period,forward=float(f),left=float(l),yaw_range=turn,endpoint_rmse=endpoint,score=score))
candidates=sorted(candidates,key=lambda d:d['score'])
path=ROOT/'output/anon_locomotion/full_motion/side_cycle_candidates.json'
path.write_text(json.dumps(candidates,indent=2),encoding='utf-8')
print(json.dumps(candidates[:8],indent=2))
