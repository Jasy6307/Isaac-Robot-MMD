"""Resume the user's paused scratch experiment without changing its recipe."""
from source.paths import REPO_ROOT
from source.pmx_prototype.experiments import jobs
from source.pmx_prototype.experiments.jobs import read, write
import hashlib
import json
import os
from pathlib import Path
import time

import run_split_scratch as experiment

OUT=experiment.OUT
ROOT=experiment.ROOT


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pause=experiment.read(OUT/'pause_state.json')
    checkpoint=Path(pause['checkpoint'])
    assert digest(checkpoint)==pause['checkpoint_sha256'],'Paused checkpoint has changed'
    plan=experiment.read(OUT/'experiment_plan.json')
    total=plan['maximum_iterations'];envs=plan['num_envs']
    assert envs==12288 and pause['completed_iterations']==4873
    current_config=experiment.read(OUT/'continuation_config.json')
    initial_config=experiment.read(OUT/'config.json')
    changes={k:(initial_config.get(k),current_config.get(k)) for k in initial_config.keys()|current_config.keys()
             if initial_config.get(k)!=current_config.get(k)}
    # This is an initialization-only exploration setting; keep learned stds.
    assert changes=={'train_leg_std':(.08,None)},changes
    preserved=['train_anon_locomotion.py','training/anon_imitation_env.py','training/anon_locomotion_env.py',
        'training/command_plan.py','motion/unified_walk_run_reference.py','motion/full_motion_reference.py',
        'motion/continuous_motion_reference.py','training/amp_motion_prior.py',
        'diagnostics/evaluate_unified_walk_run.py','training/continuation_normalization.py',
        'training/anon_scratch_validation.py']
    frozen=Path(pause['run_dir'])
    source_hashes={name:digest(experiment.SOURCE/name) for name in preserved}
    assert all(source_hashes[name]==digest(frozen/name) for name in preserved),'Paused implementation changed'
    index=1
    while Path(str(experiment.RUN)+f'_resume{index}').exists():index+=1
    run=Path(str(experiment.RUN)+f'_resume{index}')
    session=dict(started_unix=time.time(),source_checkpoint=str(checkpoint),source_sha256=digest(checkpoint),
        completed_before=pause['completed_iterations'],training_seconds_before=pause['training_seconds'],
        global_training_seconds_before=pause['cumulative_training_seconds'],
        acceptance_seconds_before=pause['acceptance_seconds'],remaining_iterations=total-pause['completed_iterations'],
        run_dir=str(run),num_envs=envs,configuration_unchanged=True,source_hashes=source_hashes,
        continuation_configuration_sha256=digest(OUT/'continuation_config.json'),
        initialization_only_setting_omitted='train_leg_std: retain checkpoint exploration std instead of resetting it',
        gpu_sharing='User explicitly authorized shared GPU; actual measured time is reported')
    experiment.write(OUT/f'resume_session_{index}.json',session)
    code,_=experiment.train(run,OUT/'continuation_config.json',envs,total-pause['completed_iterations'],checkpoint)
    if code:raise RuntimeError('Resume failed; inspect '+run.name+'.log')
    attempts=experiment.read(OUT/'extra_history.json') if (OUT/'extra_history.json').exists() else []
    extra_seconds=0.;accepted=False
    for continuation_index in range(20):
        timing=experiment.read(run/'training_time.json')
        history=experiment.read(OUT/'acceptance/history.json');latest=history[-1]
        completed=timing['completed_iterations']
        if latest['passed']:
            started=time.perf_counter();extra=experiment.extra_checks(run/'policy_final.pt',f'checks_{completed:06d}')
            extra_seconds+=time.perf_counter()-started;accepted=extra['passed']
            attempts.append(dict(completed_iterations=completed,full_passed=True,extra=extra))
            experiment.write(OUT/'extra_history.json',attempts)
        if accepted or completed>=total:break
        next_run=Path(session['run_dir']+f'_cont{continuation_index+1}')
        code,_=experiment.train(next_run,OUT/'continuation_config.json',envs,total-completed,run/'policy_final.pt')
        if code:raise RuntimeError('Continuation failed; inspect '+next_run.name+'.log')
        run=next_run
    active=timing['experiment_training_seconds']
    result=dict(accepted=accepted,full_passed=latest['passed'],completed_iterations=completed,num_envs=envs,
        run_dir=str(run),checkpoint=str(run/'policy_final.pt'),training_seconds=active,
        resumed_training_seconds=active-pause['training_seconds'],
        acceptance_seconds=timing['acceptance_seconds']+extra_seconds,
        resumed_acceptance_seconds=timing['acceptance_seconds']-pause['acceptance_seconds']+extra_seconds,
        cumulative_training_seconds=plan['cumulative_training_seconds_before']+active,
        started_from_random_weights=True,configuration_unchanged_during_training=True,
        resumed_from=str(checkpoint),resume_session=str(OUT/f'resume_session_{index}.json'),
        extra_attempts=attempts,initialization_audit=str(OUT/'acceptance/initialization_audit.json'),
        gpu_sharing=session['gpu_sharing'])
    experiment.write(OUT/'result.json',result)
    experiment.write(OUT/'controller_status.json',dict(stage='complete',training=False,accepted=accepted,
        **{k:result[k] for k in ['training_seconds','resumed_training_seconds','cumulative_training_seconds','checkpoint']}))


