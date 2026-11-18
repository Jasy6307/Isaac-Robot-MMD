"""Reporting diagnostic recipes with shared workspace and explicit I/O."""
from pathlib import Path
from types import SimpleNamespace
from source.paths import REPO_ROOT

def build_audit_low_speed_rewards(workspace, legacy_argv=None):
    read = workspace.read_json
    write = workspace.write_json
    execute = workspace.execute
    import json
    from pathlib import Path
    import sys
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    ROOT = Path(REPO_ROOT)
    OUT = workspace.directory(ROOT / 'output/anon_locomotion/low_speed_diagnosis_20261117')
    sys.path[:0] = [str(ROOT / 'source/pmx_prototype'), str(ROOT / 'source/pmx_prototype/training')]
    from amp_motion_prior import features

    def main():
        torch.set_num_threads(2)
        meta = json.loads(workspace.read_text(OUT / 'matched/metadata.json'))
        cfg = meta['config']
        source = np.load(workspace.input_path(cfg['full_motion_path']))
        ref_links = list(source['link_names'])
        names = meta['dof_names']
        links = meta['link_names']
        legs = meta['leg_ids']
        upper = meta['upper_ids']
        dt = meta['dt']
        keys = ['waist_pitch', 'head', 'left_knee_pitch', 'right_knee_pitch', 'left_ankle_roll', 'right_ankle_roll', 'left_hand', 'right_hand']
        actual_ids = [links.index(n) for n in keys]
        reference_ids = [ref_links.index(n) for n in keys]
        pelvis = links.index('pelvis')
        ref_pelvis = ref_links.index('pelvis')
        jw = np.ones(len(names))
        jw[upper] = cfg.get('amp_upper_pose_weight', 0.2)
        kw = np.ones(len(keys))
        kw[[0, 1, 6, 7]] = cfg.get('amp_upper_keypoint_weight', 0.25)
        rows = {}
        for model in ['old', 'new', 'nominal']:
            path = OUT / 'matched' / f'{model}_030.npz'
            if not path.exists():
                continue
            z = np.load(workspace.input_path(path))
            a, b = (round(8 / dt), round(14 / dt))
            valid_env = ~z['done'][:b].any(0)
            if not valid_env.any():
                rows[model] = dict(valid_for_steady_comparison=False, falls=int(z['done'].sum()), reason='Repeated reference-state resets invalidate steady speed/reward comparisons.')
                continue
            e = ((z['root_velocity'][:, :, :3] - z['reference_velocity'][:, :, :3]) ** 2).sum(-1)
            delta = z['root_velocity'][:, :, :3] - z['reference_velocity'][:, :, :3]
            heading = z['headings']
            sine = np.sin(heading)
            cosine = np.cos(heading)
            heading_error = np.stack([sine * delta[:, :, 0] - cosine * delta[:, :, 1], cosine * delta[:, :, 0] + sine * delta[:, :, 1], delta[:, :, 2]], -1)
            speed = np.linalg.norm(z['ref_commands'][:, :, :2], axis=-1)
            scale = cfg['continuous_tracking_error_base'] + cfg['continuous_tracking_error_speed_scale'] * speed ** 2
            pose = ((z['q'] - z['reference_q']) ** 2 * jw).sum(-1) / jw.sum()
            qd = ((z['qd'] - z['reference_qd']) ** 2 * jw).sum(-1) / jw.sum()
            actual = z['poses'][:, :, actual_ids, :3] - z['poses'][:, :, pelvis, None, :3]
            reference = z['reference_positions'][:, :, reference_ids] - z['reference_positions'][:, :, ref_pelvis, None]
            key = (((actual - reference) ** 2).sum(-1) * kw).sum(-1) / kw.sum()
            accel = np.diff(z['q'][:, :, legs], n=2, axis=0) / dt ** 2
            smooth = (np.sqrt(1 + (accel / 30) ** 2) - 1).mean(-1)
            terms = dict(continuous_velocity=cfg['continuous_tracking_weight'] * np.exp(-e / scale), imitation_velocity=cfg['imitation_velocity_weight'] * np.exp(-e / cfg['imitation_velocity_error_scale']), joint_pose=3 * np.exp(-pose / 0.065), joint_velocity=0.7 * np.exp(-qd / 12), keypoints=cfg['imitation_keypoint_weight'] * np.exp(-key / 0.008), action_cost=-0.005 * (z['actions'] ** 2).mean(-1), action_change_cost=-0.003 * (np.diff(z['actions'], axis=0) ** 2).sum(-1))
            mean_terms = {name: float(value[a:b, valid_env].mean()) for name, value in terms.items()}
            mean_terms['leg_acceleration_cost'] = -cfg['leg_acceleration_weight'] * float(smooth[a - 2:b - 2, valid_env].mean())
            joints = {}
            for name in ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_knee_pitch_joint', 'right_knee_pitch_joint']:
                k = names.index(name)
                joints[name] = dict(reference_mean=float(z['reference_q'][a:b, :, k].mean()), actual_mean=float(z['q'][a:b, :, k].mean()), residual_mean_rad=float((z['target_q'][a:b, :, k] - z['reference_q'][a:b, :, k]).mean()), reference_rms_variation_rad=float(z['reference_q'][a:b, :, k].std(0).mean()), target_rms_variation_rad=float(z['target_q'][a:b, :, k].std(0).mean()), actual_rms_variation_rad=float(z['q'][a:b, :, k].std(0).mean()))
            rows[model] = dict(valid_for_steady_comparison=True, valid_envs=int(valid_env.sum()), walk_falls=int(z['done'][round(2 / dt):b].sum()), stop_falls=int(z['done'][b:].sum()), mean_velocity_reward_target_error_rms=float(np.sqrt(e[a:b, valid_env].mean())), heading_frame_velocity_error_mean=heading_error[a:b, valid_env].mean((0, 1)).tolist(), heading_frame_velocity_error_mse=(heading_error[a:b, valid_env] ** 2).mean((0, 1)).tolist(), mean_reward_terms_per_second=mean_terms, leg_reference_rmse_rad=float(np.sqrt(((z['q'][a:b, :, legs] - z['reference_q'][a:b, :, legs]) ** 2).mean())), leg_pd_target_rmse_rad=float(np.sqrt(((z['q'][a:b, :, legs] - z['target_q'][a:b, :, legs]) ** 2).mean())), joints=joints)
        curve = []
        for error in [0, 0.03, 0.06, 0.066, 0.098, 0.1, 0.15]:
            imitation = cfg['imitation_velocity_weight'] * np.exp(-error ** 2 / cfg['imitation_velocity_error_scale'])
            tracking = cfg['continuous_tracking_weight'] * np.exp(-error ** 2 / (cfg['continuous_tracking_error_base'] + cfg['continuous_tracking_error_speed_scale'] * 0.3 ** 2))
            curve.append(dict(horizontal_error_m_s=error, imitation_velocity=float(imitation), continuous_tracking=float(tracking), combined=float(imitation + tracking), fraction_of_zero_error=float((imitation + tracking) / (cfg['imitation_velocity_weight'] + cfg['continuous_tracking_weight']))))
        windows = {}
        models = json.loads(workspace.read_text(OUT / 'models.json'))
        for name in ['old', 'new']:
            z = np.load(workspace.input_path(OUT / 'matched' / f'{name}_030.npz'))
            t, n = z['q'].shape[:2]

            def tensor(x):
                return torch.tensor(x).flatten(0, 1)
            f = features(tensor(z['q']), tensor(z['qd']), tensor(z['poses'][:, :, pelvis, :3]), tensor(z['poses'][:, :, pelvis, 3:]), tensor(z['root_velocity']), tensor(z['poses'][:, :, actual_ids, :3])).reshape(t, n, -1)
            windows[name] = torch.stack([f[i:t - 3 + i] for i in range(4)], 2)[a - 3:b - 3]
        cross = {}
        with torch.inference_mode():
            for discriminator_name, path in models.items():
                state = torch.load(path, map_location='cpu', weights_only=False)['amp_state']
                size = len(state['mean'])
                discriminator = torch.nn.Sequential(torch.nn.Linear(size * 4, 256), torch.nn.LeakyReLU(0.2), torch.nn.Linear(256, 128), torch.nn.LeakyReLU(0.2), torch.nn.Linear(128, 1))
                discriminator.load_state_dict(state['discriminator'])
                discriminator.eval()
                cross[discriminator_name] = {}
                for trajectory, window in windows.items():
                    normalized = ((window.reshape(-1, 4, size) - state['mean']) / state['scale'] * state['feature_weights']).clamp(-8, 8).flatten(1)
                    score = discriminator(normalized).squeeze(-1)
                    style = (1 - 0.25 * (score - 1) ** 2).clamp(0, 1)
                    cross[discriminator_name][trajectory] = dict(score=float(score.mean()), style_reward=float(style.mean()))
        result = dict(training=False, reward_targets='Instantaneous3D root COM velocity from the smoothed whole-body reference, not raw command or mean horizontal error.', core_terms=rows, velocity_reward_curve_at_command_030=curve, amp_cross_frozen_discriminators=cross, lower_pose_bonus_applies_at_030=False, lower_pose_bonus_speed_range=[0.02, 0.16], limitations='Reported terms are a common subset, not total reward. AMP scores under different learned discriminators are not directly comparable. No reward settings changed.')
        workspace.write_text(OUT / 'reward_audit.json', json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

def build_summarize_low_speed_diagnosis(workspace, legacy_argv=None):
    read = workspace.read_json
    write = workspace.write_json
    execute = workspace.execute
    import hashlib
    import json
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    ROOT = Path(REPO_ROOT)
    OUT = workspace.directory(ROOT / 'output/anon_locomotion/low_speed_diagnosis_20261117')

    def main():
        sampling = read(OUT / 'sampling_audit.json')
        matched = read(OUT / 'matched/report.json')
        reference = read(OUT / 'reference_kinematics.json')
        rewards = read(OUT / 'reward_audit.json')
        timing = read(OUT / 'response_window_audit.json')
        models = read(OUT / 'models.json')
        assert matched['checkpoint_unchanged']
        assert all((hashlib.sha256(Path(path).read_bytes()).hexdigest() == read(OUT / 'matched/metadata.json')['checkpoint_hashes'][name] for name, path in models.items()))
        steady = []
        for row in matched['cases']:
            z = np.load(workspace.input_path(row['file']))
            valid = ~z['done'][:420].any(0)
            item = dict(model=row['model'], speed=row['speed'], walking_falls=int(z['done'][60:420].sum()), walking_fall_envs=int(z['done'][60:420].any(0).sum()), stop_falls=int(z['done'][420:].sum()), stop_fall_envs=int(z['done'][420:].any(0).sum()), valid_steady_envs=int(valid.sum()), total_envs=len(valid), usable_for_speed_comparison=bool(valid.any()))
            if valid.any():
                per_env = z['velocity'][240:420, valid, 0].mean(0)
                item.update(steady_forward_mean=float(per_env.mean()), steady_forward_std=float(per_env.std()))
            steady.append(item)
        late = {name: dict(general_pool=r['remaining_general_pool'], **r['stages']['5674']['event_fraction']) for name, r in sampling['recipes'].items()}
        metadata = read(OUT / 'matched/metadata.json')
        configs = {name: read(Path(path).parent / 'env_config.json') for name, path in models.items()}
        physics_keys = ['asset', 'physics_hz', 'decimation', 'imitation_residual_scale', 'upper_drive_scale', 'upper_damping_scale', 'reference_velocity_feedforward', 'reference_full_body_velocity_feedforward', 'reference_distal_arm_velocity_feedforward', 'reference_velocity_at_com', 'reference_root_com_local_m', 'max_speed', 'max_yaw']
        assert all((configs['old'].get(k) == configs['new'].get(k) for k in physics_keys))
        summary = dict(training=False, new_training_seconds=0, checkpoints_unchanged=True, official_thresholds_changed=False, source_models=models, source_sampling=sampling['recipes'], same_physics_parameters={k: configs['old'].get(k) for k in physics_keys}, common_planner=read(OUT / 'config.json'), steady_speed_comparison=steady, late_sampling_event_fractions=late, legacy_low_speed_fraction_flag_effective=sampling['legacy_low_speed_fraction_flag_changes_commands'], reference_data_identical=reference['reference_data_identical'], reference_kinematics=reference, reward_audit=rewards, response_windows=timing, evidence_confirmed=['The unified sampler bypasses the legacy continuous low-speed sampling flag.', 'The general environment pool shrank from30 percent in the original scratch recipe to5 percent in the final continuation recipe.', 'The new policy has a persistent low-speed tracking deficit across initial phases/headings, not just a short scoring-window artifact.', 'Old and new policy reference data and tested physics are identical; the reference root velocity remains near the requested speed.', 'Some new low-speed stopping states fall; these are hard stability failures, not threshold-boundary failures.', 'The same lateral reversal can fail the2-to5-second window while tracking accurately in seconds5-to8.'], causal_limit='Coverage deficiency is directly confirmed. Its contribution versus reward weighting has not been isolated by training ablations, which were not authorized.', next_changes_for_review_only=['Reserve explicit sustained low-speed and low-speed-to-stop tasks in the unified sampler, rather than relying on the bypassed legacy flag.', 'Use a from-scratch curriculum that retains foundational low-speed coverage before adding specialized pools.', 'Report command-planning time, physical response time and steady tracking separately, retaining all original pass/fail results and explicit response-time requirements.'])
        workspace.write_text(OUT / 'diagnosis.json', json.dumps(summary, indent=2), encoding='utf-8')
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=160)
        colors = {'old': '#2767a3', 'new': '#da7b29'}
        for name in ['old', 'new']:
            rows = [s for s in steady if s['model'] == name and s['usable_for_speed_comparison']]
            axes[0].plot([s['speed'] for s in rows], [s['steady_forward_mean'] for s in rows], '-o', label=name.title() + ' policy', color=colors[name])
            for s in rows:
                if s['valid_steady_envs'] < 16:
                    axes[0].annotate(f"n={s['valid_steady_envs']}", (s['speed'], s['steady_forward_mean']), xytext=(3, 8), textcoords='offset points', fontsize=9)
        axes[0].plot([0.1, 0.43], [0.1, 0.43], '--', color='#777777', label='Requested speed')
        axes[0].set(xlabel='Requested forward speed (m/s)', ylabel='Steady forward speed (m/s)', title='Persistent low-speed bias', xlim=(0.13, 0.42), ylim=(0.035, 0.43))
        axes[0].grid(alpha=0.2)
        axes[0].legend(loc='upper left', fontsize=9)
        row = next((r for r in timing['rows'] if r['case'] == 'left_to_right_stop' and r['segment'] == 2))
        for offset, name in [(-0.17, 'old'), (0.17, 'new')]:
            windows = row['models'][name]['windows']
            values = [windows['prefix_5_seconds']['velocity_error_mean'], windows['last_3_seconds']['velocity_error_mean']]
            bars = axes[1].bar(np.arange(2) + offset, values, width=0.32, color=colors[name], label=name.title() + ' policy')
            axes[1].bar_label(bars, fmt='%.3f', padding=4, fontsize=10)
        axes[1].axhline(0.264, ls='--', color='#777777', label='Original tracking tolerance')
        axes[1].set_xticks([0, 1], ['After change: 2-5 s', 'Settled: 5-8 s'])
        axes[1].set(ylabel='Mean tracking bias norm (m/s)', title='Same lateral reversal, different windows', ylim=(0, 0.4))
        axes[1].grid(axis='y', alpha=0.2)
        axes[1].legend(loc='upper right', fontsize=9)
        fig.suptitle('Fixed checkpoints, same physics and planner; 16 initial phase/heading states', fontsize=12)
        fig.text(0.02, 0.015, 'Speed estimates exclude trials that fell during walking; falls are reported separately. Window comparison is diagnostic only; official gates are unchanged.', fontsize=8, color='#444444')
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        fig.savefig(workspace.output_path(OUT / 'diagnosis.png'))
        plt.close(fig)
        print('DIAGNOSIS_COMPLETE ' + str(OUT / 'diagnosis.json'))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

