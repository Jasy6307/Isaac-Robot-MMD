"""Explain original gate failures from saved rollouts; never change a gate."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'source/pmx_prototype'
sys.path[:0]=[str(SOURCE),str(SOURCE/'diagnostics')]
from evaluate_unified_walk_run import command,linear_tracking_error,yaw_tracking_error


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def failures(s,report):
    result=[]
    def upper(name,value,limit):
        if value>=limit:result.append(dict(metric=name,value=value,limit=limit,excess=value-limit,excess_fraction=(value-limit)/limit))
    if s['falls']:result.append(dict(metric='falls',value=s['falls'],limit=0))
    upper('velocity_error',s['velocity_error'],s['velocity_limit'])
    upper('geometric_velocity_error',s['geometric_velocity_error'],s['velocity_limit'])
    stationary=s['label'] in ['stand','stand_end','stop']
    upper('yaw_error',s['yaw_error'],.1 if stationary else .2)
    upper('keypoint_rmse_m',s['metrics']['Imitation/keypoint_rmse_m'],.075)
    if stationary:upper('mean_speed',s['mean_speed'],.08)
    if s['label'] in ['stand','stand_end']:
        upper('displacement_m',s['displacement_m'],.25)
        upper('pelvis_yaw_drift_rad',s['pelvis_yaw_drift_rad'],.35)
    if s['label'] in ['run_forward','run_backward','run_left','run_right']:
        force=report['env_config'].get('unified_eval_force_flight')
        value=s['force_flight_fraction'] if force else s['flight_fraction']
        if value<=.01:result.append(dict(metric='flight_fraction',value=value,limit=.01,shortfall=.01-value))
    upper('clearance_deficit_m',s['metrics'].get('Imitation/near_hip_clearance_deficit_m',1),.015)
    if report['env_config'].get('unified_command_frame_metrics'):
        upper('pelvis_orientation_rmse_rad',s['metrics']['Imitation/pelvis_orientation_rmse_rad'],.25)
    assert bool(result)==(not s['passed']),(s['label'],result,s['passed'])
    return result


def analyze(folder):
    report=read(folder/'report.json');data=np.load(folder/'rollout.npz')
    dt=float(data['dt']);start=0;rows=[]
    for index,s in enumerate(report['segments']):
        n=round(s['seconds']/dt);vel=data['velocities'][start:start+n,0]
        target=np.asarray([command(s['label'],i*dt,s['seconds'],s['command']) for i in range(n)],dtype=np.float32)
        variable=s['label'].startswith('ramp_') or s['label']=='direction_sweep'
        error=vel[:,:2]-target[:,:2]
        score=linear_tracking_error(error[60:],variable,dt)
        yaw=yaw_tracking_error((vel[:,2]-target[:,2])[60:],target[60:,2])
        assert abs(score-s['velocity_error'])<1e-4,(index,score,s['velocity_error'])
        assert abs(yaw-s['yaw_error'])<1e-4,(index,yaw,s['yaw_error'])
        reasons=failures(s,report)
        windows={}
        for label,begin in [('official',2.),('last_two_seconds',max(2.,s['seconds']-2)),('last_second',max(2.,s['seconds']-1))]:
            ix=round(begin/dt)
            windows[label]=dict(start_seconds=begin,linear_error=linear_tracking_error(error[ix:],variable,dt),
                yaw_error=yaw_tracking_error((vel[:,2]-target[:,2])[ix:],target[ix:,2]),
                mean_velocity=vel[ix:].mean(0).tolist(),mean_command=target[ix:].mean(0).tolist())
        rows.append(dict(index=index,label=s['label'],previous_label=report['segments'][index-1]['label'] if index else None,
            command=s['command'],seconds=s['seconds'],passed=s['passed'],failures=reasons,windows=windows))
        start+=n
    assert start==len(data['velocities'])
    return dict(report=str(folder/'report.json'),evaluator_sha256=report['evaluator_sha256'],
        passed=report['passed_unified_gate'],falls=report['total_falls'],passed_segments=sum(s['passed'] for s in report['segments']),
        failure_rows=[s for s in rows if not s['passed']],walk_rows=[s for s in rows if s['label']=='walk'],
        note='Later windows measure sensitivity only; the official two-second exclusion and every threshold remain unchanged.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-folder',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.report_folder)
    tests=ROOT/'output/anon_locomotion/split_command_tests_20261117'
    current=ROOT/'output/anon_locomotion/scratch_split_20261117/acceptance/checkpoint_006000/lite'
    result['matched_lite_comparison']={}
    for name,folder in [('previous_policy_same_planner',tests/'split_120_lite'),('scratch_6000_same_planner',current)]:
        if not (folder/'report.json').exists():continue
        report=read(folder/'report.json')
        assert report['evaluator_sha256']==result['evaluator_sha256']
        result['matched_lite_comparison'][name]=[dict(label=s['label'],seconds=s['seconds'],passed=s['passed'],
            velocity_error=s['velocity_error'],velocity_limit=s['velocity_limit'],mean_speed=s['mean_speed'],yaw_error=s['yaw_error'])
            for s in report['segments'] if s['label'] in ['walk','run_diagonal_135','reverse_turn']]
    baseline=read(ROOT/'output/anon_locomotion/command_plan_tests_20261116/baseline_full/report.json')
    assert baseline['evaluator_sha256']==result['evaluator_sha256']
    result['previous_default_full']=dict(passed=baseline['passed_unified_gate'],falls=baseline['total_falls'],
        caveat='This full baseline uses the previous0.6 planner. The matched lite comparison above uses the same split planner.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['walk_rows']},indent=2))


if __name__=='__main__':main()
