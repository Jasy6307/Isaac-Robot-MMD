"""Enforce the user's6000-iteration limit without changing the learner recipe."""
from source.paths import REPO_ROOT
from source.pmx_prototype.experiments import jobs
from source.pmx_prototype.experiments.jobs import read, write
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import time

import run_split_scratch as experiment

OUT=experiment.OUT
ROOT=experiment.ROOT
LIMIT=6000




def process_handle(pid,terminate=False):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    handle=kernel.OpenProcess(0x100000|(1 if terminate else 0),False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    return kernel,handle


def close(kernel,handle):
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.CloseHandle(handle)


def stop_own_controller(pid):
    # Confirm the exact script, then hold its process handle to avoid PID reuse.
    kernel,handle=process_handle(pid,True)
    try:
        check=subprocess.run(['powershell.exe','-NoProfile','-Command',
            f"Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}' | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
            capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),check=True)
        row=json.loads(check.stdout)
        if row['ProcessId']!=pid or 'resume_split_scratch.py' not in row['CommandLine']:
            raise RuntimeError('Controller identity mismatch; will not terminate it')
        kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
        if not kernel.TerminateProcess(handle,0):raise ctypes.WinError(ctypes.get_last_error())
    finally:close(kernel,handle)


def wait_exit(pid):
    try:kernel,handle=process_handle(pid)
    except OSError:return
    try:
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        while kernel.WaitForSingleObject(handle,1000)==258:pass
    finally:close(kernel,handle)


def main():
    if (OUT/'budget_update.json').exists():raise RuntimeError('Limit guard already configured')
    controller=read(OUT/'controller_process.json')
    update=dict(maximum_iterations=LIMIT,previous_maximum_iterations=7500,requested_by_user=True,
        training_configuration_changed=False,acceptance_thresholds_changed=False,
        action='Stop after6000 iterations; run original complete acceptance; analyze failures before further training.',
        watched_controller_pid=controller['pid'],started_unix=time.time())
    experiment.write(OUT/'budget_update.json',update)
    while True:
        status=read(OUT/'controller_status.json') or {}
        if status.get('stage')=='complete':
            experiment.write(OUT/'budget_guard_status.json',dict(stage='complete',reason='Original controller completed before the limit'))
            return
        if status.get('stage') in ['paused','pausing','error']:
            experiment.write(OUT/'budget_guard_status.json',dict(stage='stopped',reason=status['stage']))
            return
        files=list((ROOT/'logs/rsl_rl/anon_locomotion').glob('scratch_split_20261117_*/status.json'))
        file=max(files,key=lambda p:p.stat().st_mtime)
        train_status=read(file)
        if train_status and train_status['iteration']>=LIMIT-1:break
        time.sleep(1)
    run=file.parent;active=read(OUT/'active_process.json')
    assert active['stage']==run.name,'Expected the current learner at the boundary'
    stop_own_controller(controller['pid'])
    # The callback publishes iteration5999 before its6000-iteration screen.
    # This marker is consumed after that screen and before any next update.
    (run/'stop.request').touch()
    experiment.write(OUT/'controller_status.json',dict(stage='stopping_at_user_limit',maximum_iterations=LIMIT,run_dir=str(run)))
    wait_exit(active['pid'])
    timing=experiment.read(run/'training_time.json')
    assert timing['completed_iterations']==LIMIT,timing
    checkpoint=run/'policy_final.pt'
    work=OUT/'final_006000';work.mkdir(parents=True,exist_ok=True)
    previous=experiment.read(OUT/'acceptance/history.json')[-1]
    reuse=previous['completed_iterations']==LIMIT and previous.get('full_evaluated')
    full_seconds=0.
    report_path=Path(previous['full_report']) if reuse else work/'unified/report.json'
    if not reuse:
        config=work/'full_config.json';experiment.write(config,dict(unified_eval_lite=False,unified_record_controller=True))
        code,full_seconds=experiment.launch('limit6000_full',[experiment.SOURCE/'train_anon_locomotion.py',
            '--mode','eval','--headless','--num-envs','1','--checkpoint',checkpoint,'--output',work/'unified','--config-override',config])
        if code:raise RuntimeError('6000 complete acceptance failed to execute; inspect log')
    report=experiment.read(report_path)
    old={s['label']:s for s in experiment.read(ROOT/'output/anon_locomotion/unified_walk_run/full8_final_cpu/report.json')['segments']}
    slip={}
    for segment in report['segments']:
        label=segment['label']
        if label not in ['run_backward','run_left','run_right']:continue
        reference=old[label]
        reduction=1-segment['stance_slip_geometry_m_s']/reference['stance_slip_geometry_m_s']
        retained=segment['mean_speed']>=reference['mean_speed']*.95
        slip[label]=dict(baseline_slip_m_s=reference['stance_slip_geometry_m_s'],candidate_slip_m_s=segment['stance_slip_geometry_m_s'],
            reduction_fraction=reduction,baseline_speed_m_s=reference['mean_speed'],candidate_speed_m_s=segment['mean_speed'],
            retained_speed=retained,passed=reduction>=.2 and retained)
    official=bool(report['passed_unified_gate'] and report['total_falls']==0)
    full_passed=official and all(s['passed'] for s in slip.values())
    history=experiment.read(OUT/'acceptance/history.json')
    record=dict(completed_iterations=LIMIT,active_training_seconds=timing['experiment_training_seconds'],checkpoint=str(checkpoint),
        acceptance_kind='full',full_evaluated=True,official_passed=official,lite_passed=False,lite_pass_streak=0,
        slip_checks=slip,slip_passed=all(s['passed'] for s in slip.values()),total_falls=report['total_falls'],passed=full_passed,
        evaluation_seconds_cumulative=timing['acceptance_seconds']+full_seconds,report=str(report_path),
        full_report=str(report_path),lite_report=None,forced_by_user_iteration_limit=not reuse)
    if not reuse:history.append(record);experiment.write(OUT/'acceptance/history.json',history)
    experiment.write(OUT/'acceptance/experiment_status.json',dict(phase='limit_full_acceptance',**record))
    attempts=read(OUT/'extra_history.json') or [];extra_seconds=0.;accepted=False
    if full_passed:
        start=time.perf_counter();extra=experiment.extra_checks(checkpoint,'checks_006000')
        extra_seconds=time.perf_counter()-start;accepted=extra['passed']
        attempts.append(dict(completed_iterations=LIMIT,full_passed=True,extra=extra))
        experiment.write(OUT/'extra_history.json',attempts)
    pause=experiment.read(OUT/'pause_state.json')
    active_seconds=timing['experiment_training_seconds']
    result=dict(accepted=accepted,full_passed=full_passed,completed_iterations=LIMIT,num_envs=12288,
        run_dir=str(run),checkpoint=str(checkpoint),training_seconds=active_seconds,
        resumed_training_seconds=active_seconds-pause['training_seconds'],
        acceptance_seconds=timing['acceptance_seconds']+full_seconds+extra_seconds,
        resumed_acceptance_seconds=timing['acceptance_seconds']-pause['acceptance_seconds']+full_seconds+extra_seconds,
        cumulative_training_seconds=51378.49378980009+active_seconds,started_from_random_weights=True,
        configuration_unchanged_during_training=True,acceptance_thresholds_unchanged=True,
        stopped_at_user_limit=True,failure_analysis_required=not accepted,full_report=record['full_report'],extra_attempts=attempts,
        gpu_sharing='User authorized shared GPU; actual measured training/acceptance times reported')
    experiment.write(OUT/'result.json',result)
    experiment.write(OUT/'budget_guard_status.json',dict(stage='complete',accepted=accepted,analysis_required=not accepted))
    experiment.write(OUT/'controller_status.json',dict(stage='complete',training=False,accepted=accepted,stopped_at_user_limit=True,
        failure_analysis_required=not accepted,**{k:result[k] for k in ['training_seconds','resumed_training_seconds','cumulative_training_seconds','checkpoint']}))


