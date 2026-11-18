"""Create the fixed-policy audit summary and a scientific comparison figure."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/anon_locomotion/low_speed_diagnosis_20261117'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    sampling=read(OUT/'sampling_audit.json');matched=read(OUT/'matched/report.json')
    reference=read(OUT/'reference_kinematics.json');rewards=read(OUT/'reward_audit.json')
    timing=read(OUT/'response_window_audit.json');models=read(OUT/'models.json')
    assert matched['checkpoint_unchanged']
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==read(OUT/'matched/metadata.json')['checkpoint_hashes'][name]
               for name,path in models.items())
    steady=[]
    for row in matched['cases']:
        z=np.load(row['file']);valid=~z['done'][:420].any(0)
        item=dict(model=row['model'],speed=row['speed'],walking_falls=int(z['done'][60:420].sum()),
            walking_fall_envs=int(z['done'][60:420].any(0).sum()),stop_falls=int(z['done'][420:].sum()),
            stop_fall_envs=int(z['done'][420:].any(0).sum()),valid_steady_envs=int(valid.sum()),
            total_envs=len(valid),usable_for_speed_comparison=bool(valid.any()))
        if valid.any():
            per_env=z['velocity'][240:420,valid,0].mean(0)
            item.update(steady_forward_mean=float(per_env.mean()),steady_forward_std=float(per_env.std()))
        steady.append(item)
    late={name:dict(general_pool=r['remaining_general_pool'],**r['stages']['5674']['event_fraction'])
          for name,r in sampling['recipes'].items()}
    metadata=read(OUT/'matched/metadata.json');configs={name:read(Path(path).parent/'env_config.json') for name,path in models.items()}
    physics_keys=['asset','physics_hz','decimation','imitation_residual_scale','upper_drive_scale','upper_damping_scale',
        'reference_velocity_feedforward','reference_full_body_velocity_feedforward','reference_distal_arm_velocity_feedforward',
        'reference_velocity_at_com','reference_root_com_local_m','max_speed','max_yaw']
    assert all(configs['old'].get(k)==configs['new'].get(k) for k in physics_keys)
    summary=dict(training=False,new_training_seconds=0,checkpoints_unchanged=True,official_thresholds_changed=False,
        source_models=models,source_sampling=sampling['recipes'],same_physics_parameters={k:configs['old'].get(k) for k in physics_keys},
        common_planner=read(OUT/'config.json'),steady_speed_comparison=steady,late_sampling_event_fractions=late,
        legacy_low_speed_fraction_flag_effective=sampling['legacy_low_speed_fraction_flag_changes_commands'],
        reference_data_identical=reference['reference_data_identical'],reference_kinematics=reference,
        reward_audit=rewards,response_windows=timing,
        evidence_confirmed=[
            'The unified sampler bypasses the legacy continuous low-speed sampling flag.',
            'The general environment pool shrank from30 percent in the original scratch recipe to5 percent in the final continuation recipe.',
            'The new policy has a persistent low-speed tracking deficit across initial phases/headings, not just a short scoring-window artifact.',
            'Old and new policy reference data and tested physics are identical; the reference root velocity remains near the requested speed.',
            'Some new low-speed stopping states fall; these are hard stability failures, not threshold-boundary failures.',
            'The same lateral reversal can fail the2-to5-second window while tracking accurately in seconds5-to8.',
        ],
        causal_limit='Coverage deficiency is directly confirmed. Its contribution versus reward weighting has not been isolated by training ablations, which were not authorized.',
        next_changes_for_review_only=[
            'Reserve explicit sustained low-speed and low-speed-to-stop tasks in the unified sampler, rather than relying on the bypassed legacy flag.',
            'Use a from-scratch curriculum that retains foundational low-speed coverage before adding specialized pools.',
            'Report command-planning time, physical response time and steady tracking separately, retaining all original pass/fail results and explicit response-time requirements.',
        ])
    (OUT/'diagnosis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),dpi=160)
    colors={'old':'#2767a3','new':'#da7b29'}
    for name in ['old','new']:
        rows=[s for s in steady if s['model']==name and s['usable_for_speed_comparison']]
        axes[0].plot([s['speed'] for s in rows],[s['steady_forward_mean'] for s in rows],'-o',label=name.title()+' policy',color=colors[name])
        for s in rows:
            if s['valid_steady_envs']<16:
                axes[0].annotate(f"n={s['valid_steady_envs']}",(s['speed'],s['steady_forward_mean']),xytext=(3,8),textcoords='offset points',fontsize=9)
    axes[0].plot([.1,.43],[.1,.43],'--',color='#777777',label='Requested speed')
    axes[0].set(xlabel='Requested forward speed (m/s)',ylabel='Steady forward speed (m/s)',title='Persistent low-speed bias',xlim=(.13,.42),ylim=(.035,.43))
    axes[0].grid(alpha=.2);axes[0].legend(loc='upper left',fontsize=9)
    row=next(r for r in timing['rows'] if r['case']=='left_to_right_stop' and r['segment']==2)
    for offset,name in [(-.17,'old'),(.17,'new')]:
        windows=row['models'][name]['windows']
        values=[windows['prefix_5_seconds']['velocity_error_mean'],windows['last_3_seconds']['velocity_error_mean']]
        bars=axes[1].bar(np.arange(2)+offset,values,width=.32,color=colors[name],label=name.title()+' policy')
        axes[1].bar_label(bars,fmt='%.3f',padding=4,fontsize=10)
    axes[1].axhline(.264,ls='--',color='#777777',label='Original tracking tolerance')
    axes[1].set_xticks([0,1],['After change: 2-5 s','Settled: 5-8 s'])
    axes[1].set(ylabel='Mean tracking bias norm (m/s)',title='Same lateral reversal, different windows',ylim=(0,.4))
    axes[1].grid(axis='y',alpha=.2);axes[1].legend(loc='upper right',fontsize=9)
    fig.suptitle('Fixed checkpoints, same physics and planner; 16 initial phase/heading states',fontsize=12)
    fig.text(.02,.015,'Speed estimates exclude trials that fell during walking; falls are reported separately. Window comparison is diagnostic only; official gates are unchanged.',fontsize=8,color='#444444')
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(OUT/'diagnosis.png');plt.close(fig)
    print('DIAGNOSIS_COMPLETE '+str(OUT/'diagnosis.json'))


if __name__=='__main__':main()
