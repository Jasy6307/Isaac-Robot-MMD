"""Prepare a fresh augmentation-only run; launch only with explicit --start.

Preparation imports no Isaac/Kit, constructs no policy, and starts no worker.
The stopped short-training comparison is never resumed by this entry.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
RECIPE=SOURCE/'configs/training/unified/lateral_mirror_augmentation_scratch.json'
DEFAULT_OUT=ROOT/'output/anon_locomotion/scratch_mirror_augmentation_v1'
DEFAULT_RUN=ROOT/'logs/rsl_rl/anon_locomotion/scratch_mirror_augmentation_v1'
LEDGER=ROOT/'output/anon_locomotion/lateral_mirror_ablation_20261117_8192/pause_state.json'
FILES=['train_anon_locomotion.py','training/anon_scratch_validation.py','training/low_speed_commands.py',
       'training/locomotion_acceptance.py','training/command_plan.py','training/anon_imitation_env.py',
       'training/anon_locomotion_env.py','training/lateral_symmetry.py','training/amp_motion_prior.py',
       'motion/full_motion_reference.py','motion/unified_walk_run_reference.py',
       'diagnostics/evaluate_unified_walk_run.py','diagnostics/probe_low_speed_tracking.py',
       'training/run_balanced_scratch.py','training/prepare_mirror_scratch.py']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data,indent=2),encoding='utf-8');temporary.replace(path)


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true',help='Prepare only (default); never launch training')
    mode.add_argument('--start',action='store_true',help='Launch only after the user explicitly requests starting')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    return parser.parse_args(argv)


def configuration():
    config=read(RECIPE)
    if not(config.get('train_lateral_mirror') and config.get('train_mirror_augmentation') and config.get('amp_mirror_augmentation')):
        raise ValueError('Fresh recipe must enable PPO/AMP mirror augmentation')
    if config.get('train_mirror_loss_coeff')!=0:
        raise ValueError('This run must not enable the weak mirror loss')
    if config.get('unified_acceptance_protocol')!='response_steady_v2':raise ValueError('Acceptance protocol changed')
    for key in ['train_retention_weight','train_retention_teacher_checkpoint','train_teacher_anchor_path',
                'train_reset_actor_head','train_reset_optimizer','train_normalization_rebase']:
        if config.get(key):raise ValueError('Unexpected pretrained/migration setting: '+key)
    if config.get('train_leg_std')!=.08:raise ValueError('Fresh exploration initialization must be explicit')
    return config


def prepare(output,run):
    if run.exists() or (output/'experiment_plan.json').exists():
        raise RuntimeError('A training run already exists; preparation cannot overwrite it')
    config=configuration()
    assets={config['full_motion_path'],*config['amp_expert_paths']}
    asset=ROOT/'output/anon_physical_v4/anon_physical.usda'
    assets.update(str(asset.parent/name) for name in ['anon_physical.usda','rig.usda','manifest.json'])
    plan=dict(status='prepared_waiting_user_start',training_started=False,
        prepared_at=datetime.now().astimezone().isoformat(),requires_explicit_start=True,
        started_from_random_weights=True,checkpoint=None,external_teacher=None,
        actor_critic_amp_optimizer_normalization='Random actor/critic/discriminator; empty optimizers and actor/critic observation statistics. AMP feature statistics come from mirrored mocap experts, never an old checkpoint; audit at launch.',
        num_envs=8192,seed=42,maximum_iterations=6000,screen_interval=500,full_interval=1000,
        recipe=str(RECIPE),recipe_sha256=digest(RECIPE),run_dir=str(run),output=str(output),
        mirror_augmentation=True,amp_mirror_augmentation=True,weak_mirror_loss=False,
        augmentation_scope='PPO pure lateral full-running samples; AMP raw expert/replay mirrors and invariant style reward.',
        appearance_and_runtime='Existing31-DOF wrists/arms/clearance retained; no drive compensation.',
        acceptance_protocol='response_steady_v2',complete_suite_seconds=355,
        original_slip_reduction_required=.2,original_speed_retention_required=.95,zero_falls=True,
        supplementary_check='After complete pass: six low speeds0.15..0.40m/s,16 initial phase/heading states,zero falls.',
        stop_rule='First complete+supplementary pass or6000 iterations; no automatic extension.',
        progress_rule='Report after acceptance rounds, with current and cumulative training time.',
        time_rule='Actual shared-GPU training time; initialization/acceptance recorded separately.',
        cumulative_training_seconds_before=read(LEDGER)['cumulative_training_seconds'],
        source_hashes={name:digest(SOURCE/name) for name in FILES},
        input_hashes={path:digest(path) for path in sorted(assets)})
    write(output/'config.json',config)
    continuation=dict(config);continuation.pop('train_leg_std')
    write(output/'continuation_config.json',continuation)
    write(output/'prepared_plan.json',plan)
    write(output/'controller_status.json',dict(stage='prepared_waiting_user_start',training=False,prepared_plan=str(output/'prepared_plan.json')))
    return plan


def launch(output,run):
    plan=read(output/'prepared_plan.json')
    if run.exists() or (output/'experiment_plan.json').exists():raise RuntimeError('Training already exists')
    if plan['output']!=str(output) or plan['run_dir']!=str(run):raise ValueError('Prepared paths differ')
    if digest(RECIPE)!=plan['recipe_sha256'] or configuration()!=read(output/'config.json'):
        raise ValueError('Prepared recipe changed; prepare again before starting')
    for name,expected in plan['source_hashes'].items():
        if digest(SOURCE/name)!=expected:raise ValueError('Prepared source changed: '+name)
    for path,expected in plan['input_hashes'].items():
        if digest(path)!=expected:raise ValueError('Prepared input changed: '+path)
    if read(LEDGER)['cumulative_training_seconds']!=plan['cumulative_training_seconds_before']:
        raise ValueError('Training ledger changed; prepare again for accurate timing')
    spec=importlib.util.spec_from_file_location('mirror_scratch_controller',SOURCE/'training/run_balanced_scratch.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    controller.OUT=output;controller.RUN=run;controller.RECIPE=RECIPE
    controller.BEFORE=plan['cumulative_training_seconds_before'];controller.MAXIMUM=plan['maximum_iterations']
    write(output/'controller_process.json',dict(pid=os.getpid(),started_at=datetime.now().astimezone().isoformat()))
    controller.main()
    if any(digest(SOURCE/name)!=value for name,value in plan['source_hashes'].items()):
        raise RuntimeError('Prepared training sources changed during execution')


def main(argv=None):
    args=arguments(argv);output=args.output.resolve();run=args.run_dir.resolve()
    if args.start:
        if not (output/'prepared_plan.json').exists():raise RuntimeError('Prepare first; this entry never auto-prepares and launches')
        os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
        try:launch(output,run)
        except Exception as error:
            write(output/'controller_status.json',dict(stage='error',training=False,error=str(error)));raise
    else:
        plan=prepare(output,run)
        print(json.dumps(dict(status=plan['status'],training_started=False,prepared_plan=str(output/'prepared_plan.json'),
            num_envs=plan['num_envs'],weak_mirror_loss=False),indent=2))


if __name__=='__main__':main()
