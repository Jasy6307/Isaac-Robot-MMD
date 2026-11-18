"""Compare blended reference positions with FK of its actual joint targets."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
sys.path[:0]=[str(SOURCE),str(SOURCE/'motion')]
from full_motion_reference import FullMotionReference
from retarget_100style import Retarget
from unified_walk_run_reference import components,sample,velocity_at_offset

OUT=ROOT/'output/anon_locomotion/low_speed_diagnosis_20261117'


def fk(model,q,root_pos,root_rot):
    p={'pelvis':root_pos};r={'pelvis':root_rot}
    def solve(name):
        if name in p:return
        parent=model.parents[name];solve(parent)
        p[name]=p[parent]+np.einsum('nij,j->ni',r[parent],model.origins[name]-model.origins[parent])
        local=np.broadcast_to(np.eye(3),r[parent].shape)
        if name in model.by_child:
            k=model.by_child[name];joint=model.joints[k]
            axis=np.asarray(joint.get('axis_vector',np.eye(3)['XYZ'.index(joint['axis'])]))
            local=Rotation.from_rotvec(q[:,k,None]*axis).as_matrix()
        r[name]=r[parent]@local
    for name in model.links:solve(name)
    return p,r


def main():
    torch.set_num_threads(2)
    meta=json.loads((OUT/'matched/metadata.json').read_text());cfg=meta['config']
    motion=FullMotionReference(cfg['full_motion_path'],'cpu')
    motion.gait_cycles=torch.tensor(motion.metadata_cycles,dtype=torch.float32)
    motion.gait_offsets=torch.tensor(cfg['unified_contact_offsets'],dtype=torch.float32)
    model=Retarget(Path(cfg['asset']).parent/'manifest.json')
    assert model.names==motion.names
    pelvis=motion.links.index('pelvis');base=meta['base_reference_id'];feet=meta['foot_reference_ids']
    names=[motion.links[i] for i in feet];corners=np.asarray(meta['foot_corners'])
    offset=np.asarray(meta['root_com_local']);base_name=motion.links[base]
    def geometry(ref,use_fk):
        if use_fk:
            p,r=fk(model,ref['q'].numpy(),ref['positions'][:,pelvis].numpy(),Rotation.from_quat(ref['rotations'][:,pelvis].numpy()).as_matrix())
            positions=np.stack([p[name] for name in names],1);rotations=np.stack([r[name] for name in names],1)
            com=p[base_name]+np.einsum('nij,j->ni',r[base_name],offset)
        else:
            positions=ref['positions'][:,feet].numpy();rotations=Rotation.from_quat(ref['rotations'][:,feet].numpy().reshape(-1,4)).as_matrix().reshape(-1,2,3,3)
            rotation=Rotation.from_quat(ref['rotations'][:,base].numpy()).as_matrix()
            com=ref['positions'][:,base].numpy()+np.einsum('nij,j->ni',rotation,offset)
        sole=positions[:,:,None,:]+np.einsum('nfij,fkj->nfki',rotations,corners)
        return positions,sole,com
    records=[]
    for speed in [.15,.2,.25,.3,.35,.4,.55]:
        commands=torch.tensor([[speed,0.,0.]],dtype=torch.float32)
        frequency=float(components(motion,commands,cfg)[2][0]);dt=.001
        phase=torch.arange(0,8,1/120)*frequency+.137
        commands=commands.expand(len(phase),-1)
        refs=[sample(motion,phase+shift*frequency,commands,cfg) for shift in [-dt,0.,dt]]
        for use_fk in [False,True]:
            geom=[geometry(ref,use_fk) for ref in refs]
            position,sole,com=geom[1]
            contact=(refs[1]['contacts'].numpy()>.8)&(refs[0]['contacts'].numpy()>.8)&(refs[2]['contacts'].numpy()>.8)
            near=[g[1][:,:,:,2]<g[1][:,:,:,2].min(-1,keepdims=True)+.003 for g in geom]
            stable=contact[:,:,None]&near[0]&near[1]&near[2]
            derivative=(geom[2][1]-geom[0][1])/(2*dt)
            com_derivative=(geom[2][2]-geom[0][2])/(2*dt)
            implied=derivative[:,:,:,1]-com_derivative[:,None,None,1]
            target=velocity_at_offset(refs[1]['velocities'][:,base],refs[1]['rotations'][:,base],torch.tensor(offset,dtype=torch.float32))
            desired=-target[:,1].numpy()[:,None,None]
            row=dict(speed=speed,geometry='FK_of_blended_joints' if use_fk else 'linearly_blended_positions',
                gait_frequency_hz=frequency,stance_samples=int(stable.sum()),
                implied_no_slip_root_forward_mean=float(implied[stable].mean()),
                target_root_forward_stance_mean=float(np.broadcast_to(desired,stable.shape)[stable].mean()),
                implied_forward_mismatch_mean=float((desired-implied)[stable].mean()),
                implied_forward_mismatch_abs_mean=float(abs(desired-implied)[stable].mean()),
                stance_sole_height_mean=float(sole[:,:,:,2][stable].mean()))
            if use_fk:
                blended=refs[1]['positions'][:,feet].numpy()
                row['foot_position_fk_blend_rmse_m']=float(np.sqrt(np.mean((position-blended)**2)))
            records.append(row)
    paths={name:Path(path).parent/'env_config.json' for name,path in json.loads((OUT/'models.json').read_text()).items()}
    hashes={name:hashlib.sha256(Path(json.loads(path.read_text())['full_motion_path']).read_bytes()).hexdigest() for name,path in paths.items()}
    result=dict(training=False,reference_data_hashes=hashes,reference_data_identical=len(set(hashes.values()))==1,records=records,
        caveat='Kinematic no-slip proxy at stable lowest sole corners, not a simulated speed prediction. No training or reference changes.')
    (OUT/'reference_kinematics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
