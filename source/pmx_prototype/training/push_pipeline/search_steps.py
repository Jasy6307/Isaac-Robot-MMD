"""Search physically executed waist and leg corrections on independent near-limit trials."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from measure import DIRECTIONS, run_trials, sha256 as digest
from trials import read_batch


# First and second support times, reach, clearance, leg-bracing gain, and prior-action blend.
VARIANTS = ((.24,.52,.14,.045,.06,0.),(.27,.61,.22,.06,.07,0.),(.32,.65,.22,.06,.12,0.),
            (.32,.71,.30,.065,.10,0.),(.38,.76,.29,.06,.13,0.),(.42,.84,.30,.08,.15,0.),
            (.48,.95,.35,.075,.13,0.),(.34,.72,.18,.055,0.,0.),(.30,.68,.24,.055,.10,.28),
            (.37,.78,.25,.065,.08,.50),(.38,.78,0.,0.,0.,1.),(.30,.70,0.,0.,0.,.6),
            (.42,.88,0.,0.,0.,1.35),(.40,.75,.23,.05,.04,.36))
GROUP_SIZE = len(VARIANTS)+2


def cells(seed, count, batches):
    import numpy as np
    identified = []
    for batch in batches:
        protocol, trials = read_batch(batch)
        key = lambda row: (row['case'], row['phase_bin'], row['direction'])
        recoverable = {key(row) for row in trials if row['impulse_m_s'] == .40 and row['recovered']}
        for row in trials:
            if row['impulse_m_s'] == .525 and row['fell'] and key(row) in recoverable:
                identified.append(dict(row, angle_offset=protocol['angle_offset_degrees'],
                                       phase_offset=protocol['phase_offset_cycles']))
    rng = np.random.default_rng(seed)
    result = []
    cases = ('stand','walk_forward','walk_backward','walk_turn_left','walk_left','walk_right','run_forward')
    priorities = (1.4,1.2,1.2,1.3,.55,.5,.6)
    for case, weight in zip(cases,priorities):
        matching = [row for row in identified if row['case'] == case]
        rng.shuffle(matching)
        for i,row in enumerate(matching[:round(count*weight/sum(priorities))]):
            phase = int(row['phase_bin']) / 8 + row['phase_offset']
            heading = DIRECTIONS.index(row['direction']) * 45 + row['angle_offset']
            result.append(dict(case=case, phase=(phase+rng.uniform(.019,.054))%1,
                               direction=(heading+rng.uniform(8.,17.))%360,
                               strength=(.45,.48,.50,.525,.525)[i%5]))
    return result


class StepSearch:
    def __init__(self, env, configs, prior):
        import torch
        from anon_locomotion_env import quat_rotate_inverse
        self.torch, self.env, self.prior = torch, env, prior
        self.rotate_inverse = quat_rotate_inverse
        dev = env.device
        self.instances = torch.arange(env.num_envs,device=dev)
        self.variant_ids = (self.instances % GROUP_SIZE)-2
        table = torch.tensor(VARIANTS, device=dev)
        self.params = table[self.variant_ids.clamp_min(0)]
        self.params[self.variant_ids < 0] = 0.
        self.phase = torch.tensor([cell['phase'] for cell in configs],device=dev).repeat_interleave(GROUP_SIZE)
        self.angle = torch.tensor([cell['direction'] for cell in configs],device=dev).repeat_interleave(GROUP_SIZE)*torch.pi/180
        self.push_xy = torch.zeros((env.num_envs,2),device=dev)
        self.velocity_at_impact = torch.zeros_like(self.push_xy)
        self.first_foot = torch.zeros(env.num_envs,dtype=torch.long,device=dev)
        self.pelvis_height = torch.zeros(env.num_envs,device=dev)
        self.feet = torch.tensor(env.foot_ids,device=dev)
        leg_names = ('hip_roll','hip_pitch','knee_pitch')
        self.joints = torch.tensor([[env.names.index(f'{s}_{j}_joint') for j in leg_names]
                                    for s in ('left','right')],device=dev)
        self.pivots = torch.tensor([[env.links.index(f'{s}_{j}') for j in leg_names]
                                    for s in ('left','right')],device=dev)
        self.directions = torch.tensor([[0.,1.,0.],[1.,0.,0.],[1.,0.,0.]],device=dev)
        self.ankles = torch.tensor([[env.names.index(f'{s}_ankle_pitch_joint')] for s in ('left','right')],device=dev).squeeze(-1)
        self.waist = {name:env.names.index(f'waist_{name}_joint') for name in ('pitch','yaw','roll')}
        self.lower = torch.cat((env.leg_ids,torch.tensor(list(self.waist.values()),device=dev))).sort().values
        self.contact = env.view.create_rigid_contact_view('/World/envs/env_*/Robot/Bodies/*_ankle_roll',
            ['/World/Ground'],max_contact_data_count=env.num_envs*16)
        mapping = {str(path):i for i,path in enumerate(self.contact.sensor_paths)}
        self.contact_ids = torch.tensor([[mapping[f'{path}/Bodies/{s}_ankle_roll'] for s in ('left','right')]
                                        for path in env.robot.prim_paths],device=dev)
        self.ankle_height = torch.tensor([env.manifest['links'][f'{s}_ankle_roll']['origin'][2]
                                          for s in ('left','right')],device=dev)
        mass = env.robot.get_masses().to(dev)
        self.weights = (mass / mass.sum(dim=1,keepdim=True))[...,None]
        self.com_local = env.robot.get_coms().to(dev)[...,:3]

    def schedule(self, _phase, _angle):
        return self.phase, self.angle

    def begin(self, ids, direction):
        torch,env=self.torch,self.env
        self.push_xy[ids] = direction
        self.velocity_at_impact[ids] = (env.link_vel[ids,:,:2]*self.weights[ids]).sum(dim=1)
        self.pelvis_height[ids] = env.poses[ids,env.pelvis_id,2]
        support = self.contact.get_contact_force_matrix(1/env.cfg['physics_hz'])[self.contact_ids[ids],0,2]
        side = torch.nn.functional.pad(direction, (0,1))
        local = self.rotate_inverse(env.root[ids,3:],side)
        self.first_foot[ids] = torch.where((support.min(dim=1).values > 65.), (local[:,0] < 0.).long(), support.argmin(dim=1))

    def action(self, frame, onset, observation, base):
        torch,env=self.torch,self.env
        age = (frame-onset)*env.dt
        ids = ((onset >= 0) & (age > 0.) & (age < 1.35) & (self.variant_ids >= 0)).nonzero().flatten()
        if not len(ids):
            return base
        age = age[ids]
        params = self.params[ids]
        first, second, reach, lift, bracing, blend = params.unbind(-1)
        action = base[ids].clone()
        weight = (1. - ((age - .89)/.40).clamp(0.,1.)) * blend
        previous = self.prior(observation[ids])
        action[:,self.lower] += weight[:,None]*(previous[:,self.lower]-action[:,self.lower])

        working = (reach > 0.) & (age < second + .13)
        second_step = age >= first
        foot = torch.where(second_step,1-self.first_foot[ids],self.first_foot[ids])
        swing_duration = torch.where(second_step, second-first, first)
        phase = ((age - torch.where(second_step, first, 0.))/swing_duration).clamp(0.,1.)
        # The first foot reaches along the disturbance. The other foot tracks the remaining COM motion.
        body = env.poses[ids]
        q = body[...,3:]
        com = ((body[...,:3]+self.rotate_inverse(torch.cat((-q[...,:3],q[...,3:]),dim=-1),self.com_local[ids]))
               *self.weights[ids]).sum(dim=1)
        velocity = (env.link_vel[ids,:,:2]*self.weights[ids]).sum(dim=1)
        excess = velocity-self.velocity_at_impact[ids]
        direction = self.push_xy[ids]
        projected = (excess*direction).sum(dim=-1)
        lead = reach*torch.where(second_step,.62,1.) + .10*projected.clamp(-.5,1.)
        feet = body[torch.arange(len(ids),device=env.device),self.feet[foot],:3]
        desired = com[:,:2] + lead[:,None]*direction + .15*self.velocity_at_impact[ids]
        side = self.rotate_inverse(torch.cat((-env.root[ids,3:6],env.root[ids,6:]),dim=-1),
                                   torch.tensor([1.,0.,0.],device=env.device).expand(len(ids),3))[:,:2]
        desired += .075*torch.where(foot==0,1.,-1.)[:,None]*side
        forward_shift = (desired-feet[:,:2]).clamp(-.24,.24)
        ztarget = self.ankle_height[foot]+torch.sin(torch.pi*phase)*lift
        gap = torch.cat((forward_shift,(ztarget-feet[:,2]).clamp(-.08,.08)[:,None]),dim=-1)
        pivots = env.poses[ids[:,None],self.pivots[foot]]
        r = pivots[...,3:]
        joint_axes = self.rotate_inverse(torch.cat((-r[...,:3],r[...,3:]),dim=-1),
                                         self.directions[None].expand(len(ids),-1,-1))
        jac = torch.cross(joint_axes,feet[:,None]-pivots[...,:3],dim=-1).transpose(1,2)
        target, _ = torch.linalg.solve_ex(jac@jac.transpose(1,2)+.035*torch.eye(3,device=env.device),gap[...,None])
        correction = (.65*(jac.transpose(1,2)@target).squeeze(-1)).clamp(-.24,.24)
        amount = working*(phase*5.).clamp(max=1.)*(1.-((age-second)/.13).clamp(0.,1.))
        ii=torch.arange(len(ids),device=env.device)
        action[ii[:,None],self.joints[foot]] += amount[:,None] * correction/env.cfg['imitation_residual_scale']
        # Support-leg extension and waist counter-lean are evaluated by the same fall/velocity criterion.
        plant = self.joints[1-foot]
        drop = ((self.pelvis_height[ids]-env.poses[ids,env.pelvis_id,2])/.12).clamp(0.,1.)
        hold = amount*bracing*(.35+.65*drop) / env.cfg['imitation_residual_scale']
        action[ii,plant[:,2]] -= hold
        action[ii,plant[:,1]] += .5*hold
        action[ii,self.ankles[1-foot]] += .5*hold
        action[:,self.waist['pitch']] += amount*(.65*env.gravity[ids,1]-.08*env.omega_b[ids,0]).clamp(-.30,.30)
        action[:,self.waist['roll']] += amount*(-.65*env.gravity[ids,0]-.08*env.omega_b[ids,1]).clamp(-.30,.30)
        result = base.clone()
        result[ids[:,None],self.lower] = action[:,self.lower].clamp(-2.,2.)
        return result


def load_actor(path, env, runner_class):
    cfg = json.loads((path.parent/'agent_config.json').read_text(encoding='utf-8'))
    runner=runner_class(env,cfg,log_dir=None,device=env.device)
    runner.load(str(path),load_cfg={'actor':True},map_location=env.device)
    return runner.get_inference_policy()


def run(args):
    from isaaclab.app import AppLauncher
    app=AppLauncher(dict(headless=True,device='cuda:0',disable_viewport_updates=True,
                         limit_cpu_threads=2,kit_args='--/persistent/physics/numThreads=0')).app
    env = None
    exit_code=1
    try:
        import numpy as np
        import torch
        from anon_environment import create_environment
        from rsl_rl.runners import OnPolicyRunner
        torch.set_num_threads(2)
        base=args.base.resolve()
        prior=args.prior.resolve()
        designs = args.designs
        entries = [(cell['case'],0,direction,level) for cell in designs
                   for direction,level in [(-1,0.)]+[(0,cell['strength'])]*(GROUP_SIZE-1)]
        config=json.loads((base.parent/'env_config.json').read_text(encoding='utf-8'))
        config.update(simulation_device='cuda:0',runtime_inference_only=True,runtime_manual_reset_only=True,
                      amp_reward_diagnostics=False,measure_ground_contact=True,
                      unified_substep_support_forces=False,unified_command_frame_metrics=True)
        config.pop('unified_eval_parallel_envs',None)
        env=create_environment(len(entries),42,app,evaluation=True,config=config)
        env.max_episode_length=round(60/env.dt)
        reference = load_actor(base,env,OnPolicyRunner)
        source = load_actor(prior,env,OnPolicyRunner)
        experiment=StepSearch(env,designs,source)
        schedule=SimpleNamespace(action_experiment=experiment,phase_offset=0.,angle_offset_deg=0.,
                                 lead_seconds=3.85,rehearsal=True)
        protocol,rows,data=run_trials(env,reference,entries,8,GROUP_SIZE,np,torch,schedule)
        samples=data['rehearsal']
        retained, diagnostics = [],[]
        for index,cell in enumerate(designs):
            start=index*GROUP_SIZE
            control,baseline=rows[start:start+2]
            successful=[]
            if control['recovered'] and not baseline['recovered']:
                for row in rows[start+2:start+GROUP_SIZE]:
                    i=row['trial_id']; j=baseline['trial_id']; a,b=data['onset'][[i,j]]
                    preshock=np.abs(data['frames'][a,i,[0,1,2,5,6]]-data['frames'][b,j,[0,1,2,5,6]]).max()
                    if row['recovered'] and a==b and preshock < .025:
                        successful.append(row)
            successes=sorted(successful,key=lambda row:row['recovery_seconds'])[:3]
            retained.extend(row['trial_id'] for row in successes)
            diagnostics.append({**cell,'control_success': control['recovered'],'base_success':baseline['recovered'],
                                'successful_variants':[row['trial_id']-start-2 for row in successes]})
        with torch.inference_mode():
            observations=torch.from_numpy(samples['observations']).to(env.device)
            means=[]
            from tensordict import TensorDict
            for raw in observations.split(8192):
                means.append(reference(TensorDict({'policy':raw},batch_size=[len(raw)])).cpu())
            base_means=torch.cat(means).numpy()
        selected=np.isin(samples['trial_ids'],retained)
        uncorrected=samples['trial_ids']%GROUP_SIZE<=1
        take=selected|uncorrected
        np.savez_compressed(args.output/'steps.npz', observations=samples['observations'][take],
                            actions=samples['actions'][take], reference=base_means[take],
                            frames=samples['frames_after_push'][take],
                            corrected=selected[take],trial_ids=samples['trial_ids'][take],
                            lower=experiment.lower.cpu().numpy())
        meta=dict(protocol=protocol,seed=args.seed,cells=diagnostics,source=str(base),source_sha256=digest(base),
                  retained=len(retained),rescued_cells=sum(bool(item['successful_variants']) for item in diagnostics),
                  dataset_sha256=digest(args.output/'steps.npz'),variants=VARIANTS,
                  ordinary_successes=sum(item['base_success'] for item in diagnostics),
                  positive_frames=int(selected.sum()))
        (args.output/'report.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        print(f'STEP_SEARCH seed={args.seed} cells={len(designs)} rescued={meta["rescued_cells"]} samples={meta["positive_frames"]}',flush=True)
        exit_code=0
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        app.close(exit_code=exit_code)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--frontier', type=Path, nargs=2, required=True)
    parser.add_argument('--cells',type=int,default=110)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    args.designs = cells(args.seed, args.cells, args.frontier)
    if not args.designs:
        (args.output/'report.json').write_text(json.dumps(dict(rescued_cells=0, skipped='No paired .40/.525 frontier')),
                                                encoding='utf-8')
    else:
        run(args)
