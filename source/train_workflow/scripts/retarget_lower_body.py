"""Evaluate generic lower-limb IK on an existing H5, without overwriting input."""
from pathlib import Path
import sys,argparse,json,shutil
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import h5py,numpy as np
from scipy.spatial.transform import Rotation
from source.train_workflow.utils.retarget.lower_body_ik import solve_lower_body_arrays
from source.train_workflow.utils.motion.self_collision import playback_baseline

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion-h5',type=Path,required=True);p.add_argument('--source-animation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start-frame',type=int,default=0);p.add_argument('--frames',type=int,default=0,help='Validate this window only; no partial H5')
    a=p.parse_args()
    from source.paths import resolve_dance_asset
    a.motion_h5 = resolve_dance_asset(a.motion_h5)
    if a.output:
        a.output = resolve_dance_asset(a.output)
    if a.output.exists() or a.start_frame<0 or a.frames<0:raise ValueError('Invalid output/window')
    with h5py.File(a.motion_h5) as f:
        names=list(f['joint_names'].asstr()[:]);delta=f['joint_pos_delta'][:];fps=float(f.attrs['fps'])
        base=f.attrs.get('record_baseline_joint_pos',playback_baseline(names,ROOT/'source/my_task/g1_replay_env_cfg.py'))
        anchor=f.attrs.get('record_root_anchor_pos',[0.,0.,.76]);aq=f.attrs.get('record_root_anchor_quat_wxyz',[2**-.5,0,0,2**-.5])
        pos=f['root_pos_delta'][:]+anchor;dq=f['root_quat_delta_wxyz'][:]
        rot=(Rotation.from_quat(dq[:,[1,2,3,0]])*Rotation.from_quat(np.array(aq)[[1,2,3,0]])).as_quat()[:,[3,0,1,2]]
    window=slice(a.start_frame,a.start_frame+a.frames) if a.frames else slice(None)
    if a.start_frame and not a.frames:raise ValueError('Window start requires --frames')
    result,report=solve_lower_body_arrays(delta[window],names,base,fps,a.source_animation,
        ROOT/'assets/g1_29dof_o6_hand_V3.usd',pos[window],rot[window],start_frame=a.start_frame)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if not a.frames:
        shutil.copyfile(a.motion_h5,a.output)
        with h5py.File(a.output,'r+') as f:
            f['joint_pos_delta'][:]=result
            for k in list(f.attrs):
                if k.startswith('collision_') or k=='without_collision_validated':del f.attrs[k]
            f.attrs['lower_body_retarget_method']=report['method']
    a.output.with_suffix('.lower_body.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('LOWER_IK_PASS '+json.dumps(report),flush=True)
if __name__=='__main__':main()
