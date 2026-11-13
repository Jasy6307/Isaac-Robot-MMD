"""Refine an existing collision-repaired H5 toward arbitrary reference gestures."""
from pathlib import Path
import sys,argparse,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,h5py,torch
from scipy.spatial.transform import Rotation
from source.train_workflow.utils.motion.self_collision import G1CollisionModel
from source.train_workflow.utils.motion.self_collision import playback_baseline
from source.train_workflow.utils.motion.collision_reference import _audit,_spikes
from source.train_workflow.utils.motion.gesture_collision import refine_gestures,validate_arm_motion
from source.train_workflow.utils.motion.joint_constraints import validate_joint_positions

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='Uncorrected motion to preserve')
    p.add_argument('--seed',type=Path,required=True,help='Collision-repaired initial motion')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=3000)
    p.add_argument('--polish-iterations',type=int,default=200)
    p.add_argument('--repair-only',action='store_true',help='Project only detected collision windows; retain other verified poses')
    p.add_argument('--robot-usd',type=Path,default=ROOT/'assets/g1_29dof_o6_hand_V3.usd')
    a=p.parse_args()
    from source.paths import resolve_dance_asset
    if a.input:
        a.input = resolve_dance_asset(a.input)
    if a.seed:
        a.seed = resolve_dance_asset(a.seed)
    if a.output:
        a.output = resolve_dance_asset(a.output)
    torch.set_num_threads(4)
    if a.output.exists():raise ValueError('Output exists')
    with h5py.File(a.input) as f,h5py.File(a.seed) as g:
        names=list(f['joint_names'].asstr()[:]);base=f.attrs.get('record_baseline_joint_pos',playback_baseline(names,ROOT/'source/my_task/g1_replay_env_cfg.py'));fps=float(f.attrs['fps'])
        if names!=list(g['joint_names'].asstr()[:]):raise ValueError('Joint order differs')
        for k in ('frames','root_pos_delta','root_quat_delta_wxyz','root_valid'):
            np.testing.assert_array_equal(f[k][:],g[k][:])
        reference=f['joint_pos_delta'][:]+base;seed=g['joint_pos_delta'][:]+g.attrs.get('record_baseline_joint_pos',base)
        pos=f['root_pos_delta'][:]+f.attrs.get('record_root_anchor_pos',[0,0,.76]);dq=f['root_quat_delta_wxyz'][:];aq=f.attrs.get('record_root_anchor_quat_wxyz',[2**-.5,0,0,2**-.5])
    rw=(Rotation.from_quat(dq[:,[1,2,3,0]])*Rotation.from_quat(np.array(aq)[[1,2,3,0]])).as_matrix()
    model=G1CollisionModel(a.robot_usd,names,'cpu')
    try:
        result,report=refine_gestures(model,reference,seed,a.robot_usd,fps,pos,rw,iterations=a.iterations,polish_iterations=a.polish_iterations,repair_only=a.repair_only)
    except RuntimeError as exc:
        if hasattr(exc,'candidate'):
            a.output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(a.output.with_suffix('.failed.npz'),joint_pos=exc.candidate)
            a.output.with_suffix('.failed.json').write_text(json.dumps(exc.metrics,indent=2))
        raise
    delta=(result-base).astype(np.float32);encoded=delta+base
    audit=_audit(model,encoded,subframes=True);spikes=_spikes(model,reference,encoded,fps)
    arm=validate_arm_motion(model,encoded,a.robot_usd,fps,reference=reference)
    joints=validate_joint_positions(encoded,names,fps,a.robot_usd)
    if any(r['penetrating_frames'] for r in audit.values()) or spikes or not arm['passed'] or not joints['passed']:raise RuntimeError('Final encoded collision/spike/arm-motion/joint audit failed')
    report['collision']=audit;report['remaining_spikes']=spikes
    report['arm_motion']=arm;report['joint_constraints']=joints
    report['source_sha256']=hashlib.sha256(a.input.read_bytes()).hexdigest()
    report['seed_sha256']=hashlib.sha256(a.seed.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_suffix('.tmp.h5');shutil.copyfile(a.seed,temporary)
    with h5py.File(temporary,'r+') as f:
        f['joint_pos_delta'][:]=delta
        f.attrs['without_collision_validated']=True
        f.attrs['collision_validation']='selected USD envelopes at frames and 1/4, 1/2, 3/4 interpolation samples; tolerance 1mm'
        f.attrs['collision_reference_source_sha256']=report['source_sha256']
        f.attrs['collision_gesture_preservation_method']=report['method']
        f.attrs['collision_hand_geometry_mode']=report['hand_geometry_mode']
        f.attrs['arm_motion_validated']=True
    temporary.replace(a.output)
    a.output.with_suffix('.gesture_collision.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('GESTURE_COLLISION_PASS',report['before'],report['after'],flush=True)
if __name__=='__main__':main()
