"""Retarget original 100STYLE directions onto Anon's physical joint chains.

Source: Ian Mason, Sebastian Starke, Taku Komura, CC BY 4.0.
This exports kinematic reference data, never a simulated policy result.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import csv
import json
from pathlib import Path
import re
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

ROOT = PROJECT_ROOT
B = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])


def read_bvh(path):
    header, motion = path.read_text().split('MOTION')
    tokens = re.findall(r'[{}]|[^\s{}]+', header)
    nodes = []
    channel_count = 0
    def node(i, parent):
        nonlocal channel_count
        kind = tokens[i]; name = tokens[i+1]; i += 2
        if kind == 'End': name = nodes[parent]['name']+'_End'
        index = len(nodes)
        item = dict(name=name, parent=parent, channels=[], start=channel_count)
        nodes.append(item)
        assert tokens[i] == '{'; i += 1
        while tokens[i] != '}':
            if tokens[i] == 'OFFSET':
                item['offset'] = np.array(list(map(float, tokens[i+1:i+4])))/100; i += 4
            elif tokens[i] == 'CHANNELS':
                count = int(tokens[i+1]); item['channels'] = tokens[i+2:i+2+count]
                item['start'] = channel_count; channel_count += count; i += 2+count
            else:
                i = node(i, index)
        return i+1
    node(1, -1)
    lines = motion.strip().splitlines()
    dt = float(lines[1].split(':')[1])
    data = np.loadtxt(lines[2:])
    assert data.shape[1] == channel_count
    positions = np.zeros((len(data), len(nodes), 3)); rotations = np.zeros((len(data), len(nodes), 3, 3))
    for k, item in enumerate(nodes):
        p = np.broadcast_to(item['offset'], (len(data), 3)).copy()
        angles, axes = [], ''
        for c, name in enumerate(item['channels']):
            v = data[:, item['start']+c]
            if name.endswith('position'): p[:, 'XYZ'.index(name[0])] += v/100
            else: axes += name[0]; angles.append(v)
        local = R.from_euler(axes, np.stack(angles, -1), degrees=True).as_matrix() if axes else np.broadcast_to(np.eye(3), (len(data), 3, 3))
        parent = item['parent']
        if parent < 0: positions[:, k] = p; rotations[:, k] = local
        else:
            rotations[:, k] = rotations[:, parent]@local
            positions[:, k] = positions[:, parent]+np.einsum('nij,nj->ni', rotations[:, parent], p)
    return nodes, positions@B.T, B@rotations@B.T, dt


class Retarget:
    def __init__(self, manifest=None):
        self.manifest = json.loads(Path(manifest or ROOT/'output/anon_physical_v3/manifest.json').read_text(encoding='utf-8'))
        self.links = list(self.manifest['links'])
        self.origins = {k: np.array(v['origin']) for k,v in self.manifest['links'].items()}
        self.joints = self.manifest['joints']; self.names = [j['name'] for j in self.joints]
        self.limits = np.deg2rad([j['limits_deg'] for j in self.joints])
        self.parents = {j['child']: j['parent'] for j in self.joints}
        self.parents.update(head='neck_pitch', left_hand='left_elbow_flex', right_hand='right_elbow_flex')
        self.parents.update(self.manifest.get('fixed_parents', {}))
        self.by_child = {j['child']: i for i,j in enumerate(self.joints)}

    def fk(self, q, root_pos, root_rot):
        positions, rotations = {'pelvis': root_pos}, {'pelvis': root_rot}
        def solve(name):
            if name in positions: return
            parent = self.parents[name]; solve(parent)
            positions[name] = positions[parent]+rotations[parent]@(self.origins[name]-self.origins[parent])
            local = np.eye(3)
            if name in self.by_child:
                k = self.by_child[name]
                axis=np.array(self.joints[k].get('axis_vector',np.eye(3)['XYZ'.index(self.joints[k]['axis'])]))
                local = R.from_rotvec(axis*q[k]).as_matrix()
            rotations[name] = rotations[parent]@local
        for name in self.links: solve(name)
        return positions, rotations

    def frame(self, source_p, source_r, source_ids, previous):
        q = previous.copy(); root_r = source_r[source_ids['Hips']]
        trunk = root_r.T@source_r[source_ids['Chest4']]
        q[:3] = R.from_matrix(trunk).as_euler('ZYX')
        head = source_r[source_ids['Chest4']].T@source_r[source_ids['Head']]
        h = R.from_matrix(head).as_euler('ZYX'); q[3:5] = h[[0,2]]
        q = np.clip(q, self.limits[:,0]+1e-5, self.limits[:,1]-1e-5)
        for side in ('left','right'):
            cap = side.title()
            for group, source_chain, dest_chain in [
                ('leg', [cap+'Hip',cap+'Knee',cap+'Ankle'], [side+'_hip_pitch',side+'_knee_pitch',side+'_ankle_roll']),
                ('arm', [cap+'Shoulder',cap+'Elbow',cap+'Wrist'], [side+'_shoulder_yaw',side+'_elbow_flex',side+'_hand'])]:
                prefix = ['hip_yaw','hip_roll','hip_pitch','knee_pitch','ankle_pitch','ankle_roll'] if group=='leg' else ['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_flex']
                ids = [self.names.index(side+'_'+x+'_joint') for x in prefix]
                vectors = [source_p[source_ids[b]]-source_p[source_ids[a]] for a,b in zip(source_chain[:-1],source_chain[1:])]
                vectors = [root_r.T@v/np.linalg.norm(v) for v in vectors]
                foot = root_r.T@source_r[source_ids[cap+'Ankle']] if group=='leg' else None
                rest_vector = self.origins[dest_chain[1]]-self.origins[dest_chain[0]]
                rest_vector /= np.linalg.norm(rest_vector)
                canonical = np.array([0.,0,-1]) if group=='leg' else np.array([1. if side=='left' else -1.,0,0])
                correction = R.align_vectors(canonical[None],rest_vector[None])[0].as_matrix()
                proximal = root_r.T@source_r[source_ids[source_chain[0]]]@correction
                def error(values):
                    temp=q.copy(); temp[ids]=values
                    p,r=self.fk(temp,np.zeros(3),np.eye(3))
                    e=[]
                    for a,b,target in zip(dest_chain[:-1],dest_chain[1:],vectors):
                        v=p[b]-p[a]; e.extend((v/np.linalg.norm(v)-target)*3)
                    if foot is not None: e.extend(((r[dest_chain[-1]]-foot)*.55).ravel())
                    e.extend(((r[dest_chain[0]]-proximal)*.8).ravel())
                    e.extend(.08*(values-previous[ids]))
                    return e
                fit=least_squares(error,q[ids],bounds=(self.limits[ids,0]+1e-5,self.limits[ids,1]-1e-5),max_nfev=18,ftol=1e-5)
                q[ids]=fit.x
        return q


def main():
    out=ROOT/'output/anon_locomotion/100style_reference'; out.mkdir(parents=True,exist_ok=True)
    raw=ROOT/'media/mocap/100style'
    cuts=next(row for row in csv.DictReader((raw/'Frame_Cuts.csv').open()) if row['STYLE_NAME']=='Neutral')
    model=Retarget(); poses=[]; joints=[]; labels=[]; velocities=[]; reports=[]
    for kind,label in [('FW','forward'),('BW','backward'),('SW','left'),('ID','stand'),('TR1','turn_left')]:
        nodes,p,r,dt=read_bvh(raw/f'Neutral_{kind}.bvh'); ids={n['name']:i for i,n in enumerate(nodes)}
        start=int(cuts[kind+'_START']); stop=int(cuts[kind+'_STOP'])
        # Skip the initial acceleration, then preview eight seconds of actual capture.
        start+=120 if kind!='ID' else 0
        frames=np.arange(start,min(start+480,stop),2)
        source_speed=np.linalg.norm(np.gradient(p[:,0,:2],dt,axis=0),axis=1)
        q=np.zeros(25)
        q[model.names.index('left_shoulder_roll_joint')]=.9
        q[model.names.index('right_shoulder_roll_joint')]=-.9
        clip_poses=[]; clip_q=[]
        for n,frame in enumerate(frames):
            q=model.frame(p[frame],r[frame],ids,q)
            # Scale translation by measured leg length, preserving captured pelvis bob and sway.
            scale=(np.linalg.norm(model.origins['left_knee_pitch']-model.origins['left_hip_pitch'])+np.linalg.norm(model.origins['left_ankle_roll']-model.origins['left_knee_pitch']))/.869581
            root=p[frame,0]*scale; root[:2]-=p[frames[0],0,:2]*scale
            root[2]+=.025
            pp,rr=model.fk(q,root,r[frame,0])
            clip_poses.append(np.array([np.r_[pp[name],R.from_matrix(rr[name]).as_quat()] for name in model.links]))
            clip_q.append(q.copy())
            if n%60==0: print(kind,n,len(frames),flush=True)
        clip_poses=np.array(clip_poses)
        # One constant vertical alignment per clip: retain the mocap's vertical trajectory.
        bottoms=[]
        for name in ['left_ankle_roll','right_ankle_roll']:
            idx=model.links.index(name); local_center=np.array([0,-.06,.035])-model.origins[name]+np.array([model.origins[name][0],0,0])
            corners=np.array([[x,y,z] for x in [-.0475,.0475] for y in [-.115,.115] for z in [-.035,.035]])+local_center
            pts=np.einsum('nij,kj->nki',R.from_quat(clip_poses[:,idx,3:]).as_matrix(),corners)+clip_poses[:,idx,None,:3]
            bottoms.append(pts[:,:,2].min(1))
        floor_offset=-np.percentile(np.minimum(*bottoms),5)
        clip_poses[:,:,2]+=floor_offset
        vel=np.gradient(clip_poses[:,0,:3],2*dt,axis=0)
        local=np.einsum('nij,nj->ni',r[frames,0].transpose(0,2,1),vel)
        logical=np.stack([-local[:,1],local[:,0],np.zeros(len(frames))],-1)
        logical[:,2]=np.gradient(np.unwrap(R.from_matrix(r[frames,0]).as_euler('ZYX')[:,0]),2*dt)
        poses.extend(clip_poses); joints.extend(clip_q); labels.extend(['Neutral_'+kind]*len(frames)); velocities.extend(logical)
        np.savez_compressed(out/f'{kind}.npz',q=clip_q,joint_names=model.names,source_frames=frames,dt=2*dt,root_rotations=r[frames,0],root_positions=clip_poses[:,0,:3])
        reports.append(dict(clip=kind,source_start=int(frames[0]),source_stop=int(frames[-1]),frames=len(frames),source_speed_median=float(np.median(source_speed[frames])),scale=scale,floor_offset=float(floor_offset)))
    v=np.array(velocities)[:,None]
    poses,joints,bias=calibrate_preview(model,np.array(poses),np.array(joints),np.array(labels))
    np.savez_compressed(out/'rollout.npz',kind='mocap_reference',poses=poses[:,None],joints=joints[:,None],link_names=model.links,joint_names=model.names,dt=2*dt,labels=labels,velocities=v,commands=v,dones=np.zeros((len(poses),1),bool),neutral_calibration_rad=bias)
    (out/'report.json').write_text(json.dumps(dict(kind='mocap_reference',source='100STYLE Neutral, Ian Mason et al., CC BY 4.0',clips=reports,env_config=dict(asset=str(ROOT/'output/anon_physical_v3/anon_physical.usda'))),indent=2),encoding='utf-8')


def calibrate_preview(model, poses, joints, labels):
    """Apply the same neutral sensor/rest calibration as the training library."""
    idle=joints[labels=='Neutral_ID'].mean(0)
    bias=np.zeros(25); bias[:5]=idle[:5]; bias[2]-=.04
    calibrated=np.clip(joints-bias,model.limits[:,0],model.limits[:,1])
    result=[]
    for old,q in zip(poses,calibrated):
        p,r=model.fk(q,old[0,:3],R.from_quat(old[0,3:]).as_matrix())
        result.append([np.r_[p[name],R.from_matrix(r[name]).as_quat()] for name in model.links])
    return np.array(result),calibrated,bias


if __name__=='__main__': main()
