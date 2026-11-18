"""Matched old/new policy inference, with a nominal-reference control."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import source_layout


def diagnose(env,runner,output,checkpoint):
    import numpy as np
    import torch
    output.mkdir(parents=True,exist_ok=True)
    models=json.loads(args.models.read_text())
    files={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in models.items()}
    phases=torch.tensor([0.,.25,.5,.75],device=env.device).repeat(4)
    headings=torch.tensor([-np.pi,-np.pi/2,0.,np.pi/2],device=env.device).repeat_interleave(4)
    env.max_episode_length=100000
    leg_ids=env.leg_ids.detach().cpu().tolist()
    meta=dict(training=False,models=models,checkpoint_hashes=files,num_envs=env.num_envs,
        phases=phases.cpu().tolist(),headings=headings.cpu().tolist(),dt=env.dt,
        config=env.cfg,q_order=env.q_order,link_names=env.links,dof_names=env.names,
        upper_ids=env.upper_ids,leg_ids=leg_ids,base_reference_id=env.base_reference_id,
        foot_reference_ids=env.foot_reference_ids,foot_corners=env.foot_corners.cpu().tolist(),
        root_com_local=env.root_com_local.cpu().tolist())
    (output/'metadata.json').write_text(json.dumps(meta,indent=2))
    result=dict(training=False,cases=[],checkpoint_unchanged=False)
    start=time.perf_counter()
    from unified_walk_run_reference import heading_velocity
    from anon_locomotion_env import quat_rotate_inverse
    with torch.inference_mode():
        for name in [*models,*([] if args.no_nominal else ['nominal'])]:
            if name!='nominal':
                state=torch.load(models[name],map_location=env.device,weights_only=False)
                runner.load(models[name],map_location=env.device)
                if hasattr(env,'amp') and 'amp_state' in state:env.amp.load_state_dict(state['amp_state'])
                runner.alg.actor.eval();infer=runner.get_inference_policy()
            for speed in ([.3] if name=='nominal' else [.15,.2,.25,.3,.35,.4]):
                env.reset(env.ids,heading=headings,phase=phases,command=[0,0,0]);env.refresh()
                history={key:[] for key in ['commands','ref_commands','headings','phase','velocity','root_velocity','reference_velocity',
                    'q','qd','reference_q','reference_qd','poses','reference_positions','reference_rotations','reference_contacts',
                    'actions','target_q','reward','done']}
                falls=0
                for step in range(round(19/env.dt)):
                    t=step*env.dt
                    env.set_command([speed,0,0] if 2<=t<14 else [0,0,0])
                    observation=env.get_observations()
                    action=torch.zeros_like(env.actions) if name=='nominal' else infer(observation)
                    _,reward,done,_=env.step(action)
                    ref=env.sample_reference(env.phase)
                    reference_velocity=env.reference_root_velocity(ref)
                    values=dict(commands=env.commands,ref_commands=env.continuous_ref_commands,headings=env.reference_heading,
                        phase=env.phase,velocity=heading_velocity(env.root_vel[:,:3],env.reference_heading),root_velocity=env.root_vel,
                        reference_velocity=reference_velocity,q=env.q,qd=env.qd,reference_q=ref['q'][:,env.q_order],
                        reference_qd=ref['qd'][:,env.q_order],poses=env.poses,reference_positions=ref['positions'],
                        reference_rotations=ref['rotations'],reference_contacts=ref['contacts'],actions=action,
                        target_q=env.diagnostic_target,reward=reward,done=done)
                    for key,value in values.items():history[key].append(value.cpu().numpy().copy())
                    falls+=int(done.sum())
                arrays={key:np.asarray(value) for key,value in history.items()}
                tag=f'{name}_{round(speed*100):03d}'
                np.savez_compressed(output/(tag+'.npz'),**arrays)
                # Steady window is the last six seconds of the twelve-second walk.
                a,b=round(8/env.dt),round(14/env.dt)
                actual=arrays['velocity'][a:b].mean(0)
                error=np.linalg.norm(actual-np.array([speed,0]),axis=-1)
                refvel=heading_velocity(torch.tensor(arrays['reference_velocity'][a:b,:,:3]).flatten(0,1),
                    torch.tensor(arrays['headings'][a:b]).flatten()).reshape(b-a,env.num_envs,2).numpy()
                row=dict(model=name,speed=speed,falls=falls,steady_forward_by_env=actual[:,0].tolist(),
                    steady_forward_mean=float(actual[:,0].mean()),steady_forward_std=float(actual[:,0].std()),
                    steady_tracking_error_mean=float(error.mean()),reference_forward_mean=float(refvel[:,:,0].mean()),
                    leg_residual_rms=float(np.sqrt(np.mean(arrays['actions'][a:b,:,leg_ids]**2))),
                    leg_action_clip_fraction=float(np.mean(abs(arrays['actions'][a:b,:,leg_ids])>=2)),file=str(output/(tag+'.npz')))
                result['cases'].append(row);result['wall_seconds']=time.perf_counter()-start
                (output/'report.json').write_text(json.dumps(result,indent=2))
                print('LOW_SPEED_PROBE '+json.dumps(row),flush=True)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==files[name] for name,path in models.items())
    result['checkpoint_unchanged']=True
    (output/'report.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config-override',type=Path,required=True)
    parser.add_argument('--no-nominal',action='store_true',help='Validate fixed policies only; omit the zero-residual diagnostic')
    args=parser.parse_args()
    models=json.loads(args.models.read_text())
    import evaluate_unified_walk_run
    evaluate_unified_walk_run.evaluate=diagnose
    sys.argv=[str(SOURCE/'train_anon_locomotion.py'),'--mode','eval','--headless','--num-envs','16',
        '--checkpoint',next(iter(models.values())),'--output',str(args.output),'--config-override',str(args.config_override)]
    runpy.run_path(str(SOURCE/'train_anon_locomotion.py'),run_name='__main__')
