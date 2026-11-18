"""Reference diagnostic recipes with shared workspace and explicit I/O."""
from pathlib import Path
from types import SimpleNamespace
from source.paths import REPO_ROOT

def build_audit_reference_kinematics(workspace, legacy_argv=None):
    read = workspace.read_json
    write = workspace.write_json
    execute = workspace.execute
    import hashlib
    import json
    from pathlib import Path
    import sys
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    ROOT = Path(REPO_ROOT)
    SOURCE = Path(REPO_ROOT) / 'source/pmx_prototype'
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'motion')]
    from full_motion_reference import FullMotionReference
    from retarget_100style import Retarget
    from unified_walk_run_reference import components, sample, velocity_at_offset
    OUT = workspace.directory(ROOT / 'output/anon_locomotion/low_speed_diagnosis_20261117')

    def fk(model, q, root_pos, root_rot):
        p = {'pelvis': root_pos}
        r = {'pelvis': root_rot}

        def solve(name):
            if name in p:
                return
            parent = model.parents[name]
            solve(parent)
            p[name] = p[parent] + np.einsum('nij,j->ni', r[parent], model.origins[name] - model.origins[parent])
            local = np.broadcast_to(np.eye(3), r[parent].shape)
            if name in model.by_child:
                k = model.by_child[name]
                joint = model.joints[k]
                axis = np.asarray(joint.get('axis_vector', np.eye(3)['XYZ'.index(joint['axis'])]))
                local = Rotation.from_rotvec(q[:, k, None] * axis).as_matrix()
            r[name] = r[parent] @ local
        for name in model.links:
            solve(name)
        return (p, r)

    def main():
        torch.set_num_threads(2)
        meta = json.loads(workspace.read_text(OUT / 'matched/metadata.json'))
        cfg = meta['config']
        motion = FullMotionReference(cfg['full_motion_path'], 'cpu')
        motion.gait_cycles = torch.tensor(motion.metadata_cycles, dtype=torch.float32)
        motion.gait_offsets = torch.tensor(cfg['unified_contact_offsets'], dtype=torch.float32)
        model = Retarget(Path(cfg['asset']).parent / 'manifest.json')
        assert model.names == motion.names
        pelvis = motion.links.index('pelvis')
        base = meta['base_reference_id']
        feet = meta['foot_reference_ids']
        names = [motion.links[i] for i in feet]
        corners = np.asarray(meta['foot_corners'])
        offset = np.asarray(meta['root_com_local'])
        base_name = motion.links[base]

        def geometry(ref, use_fk):
            if use_fk:
                p, r = fk(model, ref['q'].numpy(), ref['positions'][:, pelvis].numpy(), Rotation.from_quat(ref['rotations'][:, pelvis].numpy()).as_matrix())
                positions = np.stack([p[name] for name in names], 1)
                rotations = np.stack([r[name] for name in names], 1)
                com = p[base_name] + np.einsum('nij,j->ni', r[base_name], offset)
            else:
                positions = ref['positions'][:, feet].numpy()
                rotations = Rotation.from_quat(ref['rotations'][:, feet].numpy().reshape(-1, 4)).as_matrix().reshape(-1, 2, 3, 3)
                rotation = Rotation.from_quat(ref['rotations'][:, base].numpy()).as_matrix()
                com = ref['positions'][:, base].numpy() + np.einsum('nij,j->ni', rotation, offset)
            sole = positions[:, :, None, :] + np.einsum('nfij,fkj->nfki', rotations, corners)
            return (positions, sole, com)
        records = []
        for speed in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.55]:
            commands = torch.tensor([[speed, 0.0, 0.0]], dtype=torch.float32)
            frequency = float(components(motion, commands, cfg)[2][0])
            dt = 0.001
            phase = torch.arange(0, 8, 1 / 120) * frequency + 0.137
            commands = commands.expand(len(phase), -1)
            refs = [sample(motion, phase + shift * frequency, commands, cfg) for shift in [-dt, 0.0, dt]]
            for use_fk in [False, True]:
                geom = [geometry(ref, use_fk) for ref in refs]
                position, sole, com = geom[1]
                contact = (refs[1]['contacts'].numpy() > 0.8) & (refs[0]['contacts'].numpy() > 0.8) & (refs[2]['contacts'].numpy() > 0.8)
                near = [g[1][:, :, :, 2] < g[1][:, :, :, 2].min(-1, keepdims=True) + 0.003 for g in geom]
                stable = contact[:, :, None] & near[0] & near[1] & near[2]
                derivative = (geom[2][1] - geom[0][1]) / (2 * dt)
                com_derivative = (geom[2][2] - geom[0][2]) / (2 * dt)
                implied = derivative[:, :, :, 1] - com_derivative[:, None, None, 1]
                target = velocity_at_offset(refs[1]['velocities'][:, base], refs[1]['rotations'][:, base], torch.tensor(offset, dtype=torch.float32))
                desired = -target[:, 1].numpy()[:, None, None]
                row = dict(speed=speed, geometry='FK_of_blended_joints' if use_fk else 'linearly_blended_positions', gait_frequency_hz=frequency, stance_samples=int(stable.sum()), implied_no_slip_root_forward_mean=float(implied[stable].mean()), target_root_forward_stance_mean=float(np.broadcast_to(desired, stable.shape)[stable].mean()), implied_forward_mismatch_mean=float((desired - implied)[stable].mean()), implied_forward_mismatch_abs_mean=float(abs(desired - implied)[stable].mean()), stance_sole_height_mean=float(sole[:, :, :, 2][stable].mean()))
                if use_fk:
                    blended = refs[1]['positions'][:, feet].numpy()
                    row['foot_position_fk_blend_rmse_m'] = float(np.sqrt(np.mean((position - blended) ** 2)))
                records.append(row)
        paths = {name: Path(path).parent / 'env_config.json' for name, path in json.loads(workspace.read_text(OUT / 'models.json')).items()}
        hashes = {name: hashlib.sha256(Path(json.loads(workspace.read_text(path))['full_motion_path']).read_bytes()).hexdigest() for name, path in paths.items()}
        result = dict(training=False, reference_data_hashes=hashes, reference_data_identical=len(set(hashes.values())) == 1, records=records, caveat='Kinematic no-slip proxy at stable lowest sole corners, not a simulated speed prediction. No training or reference changes.')
        workspace.write_text(OUT / 'reference_kinematics.json', json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

def build_audit_transition_chain(workspace, legacy_argv=None):
    read = workspace.read_json
    write = workspace.write_json
    execute = workspace.execute
    import json
    from pathlib import Path
    import sys
    import numpy as np
    import torch
    SOURCE = Path(REPO_ROOT) / 'source/pmx_prototype'
    sys.path.insert(0, str(SOURCE))
    import source_layout
    from full_motion_reference import FullMotionReference
    from unified_walk_run_reference import components, sample, velocity_at_offset
    ROOT = Path(REPO_ROOT)
    OUT = workspace.directory(ROOT / 'output/anon_locomotion/transition_chain_20261116')

    def main():
        OUT.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(8)
        folder = ROOT / 'output/anon_locomotion/retention_strength_20261116'
        result = {}
        for name in ['baseline', 'original']:
            d = np.load(workspace.input_path(folder / (name + '_lite') / 'rollout.npz'))
            cfg = json.loads(workspace.read_text(folder / (name + '_lite') / 'report.json', encoding='utf-8'))['env_config']
            motion = FullMotionReference(cfg['full_motion_path'], 'cpu')
            motion.gait_cycles = torch.tensor(motion.metadata_cycles, dtype=torch.float32)
            motion.gait_offsets = torch.tensor(cfg['unified_contact_offsets'], dtype=torch.float32)
            raw = torch.tensor(d['commands'][:, 0], dtype=torch.float32)
            dt = float(d['dt'])
            smooth = torch.zeros_like(raw)
            rate = torch.zeros_like(raw)
            current = torch.zeros(3)
            for i in range(len(raw)):
                delta = (raw[i] - current) * (1 - np.exp(-dt / 0.35))
                delta[:2] *= (0.6 * dt / delta[:2].norm().clamp_min(1e-06)).clamp(max=1)
                delta[2] = delta[2].clamp(-1.2 * dt, 1.2 * dt)
                current = current + delta
                smooth[i] = current
                rate[i] = delta / dt
            phase = torch.tensor(d['phases'][:, 0])
            ref = sample(motion, phase, smooth, cfg, rate)
            order = [motion.names.index(n) for n in d['dof_names']]
            leg = [i for i, n in enumerate(d['dof_names']) if any((k in n for k in ['hip', 'knee', 'ankle']))]
            manifest = json.loads(workspace.read_text(Path(cfg['asset']).parent / 'manifest.json', encoding='utf-8'))
            limits = {j['name']: np.deg2rad(j['limits_deg']) for j in manifest['joints']}
            lower = np.array([limits[n][0] + 0.01 for n in d['dof_names']])
            upper = np.array([limits[n][1] - 0.01 for n in d['dof_names']])
            nominal = ref['q'][:, order].numpy()
            residual = np.clip(d['actions'][:, 0], -2, 2) * cfg.get('imitation_residual_scale', 0.3)
            target = np.clip(nominal + residual, lower, upper)
            base = motion.links.index(cfg['reference_root_link_name'])
            v = velocity_at_offset(ref['velocities'][:, base], ref['rotations'][:, base], torch.tensor(cfg['reference_root_com_local_m']))
            refvel = torch.stack([-v[:, 1], v[:, 0]], -1).numpy()
            previous_phase = torch.cat([torch.zeros(1), phase[:-1]])
            previous_smooth = torch.cat([torch.zeros(1, 3), smooth[:-1]])
            previous_rate = torch.cat([torch.zeros(1, 3), rate[:-1]])
            pre = sample(motion, previous_phase, previous_smooth, cfg, previous_rate)
            recorded = d['observations'][:, 0, 105:136]
            reconstructed = pre['q'][:, order].numpy()
            identity = float(abs(recorded - reconstructed).max())
            ids, weights, frequency = components(motion, smooth, cfg)
            rows = {}
            actual = d['velocities'][:, 0, :2]
            for label in ['run_diagonal_135', 'reverse_turn']:
                ix = np.flatnonzero(d['labels'] == label)
                times = np.arange(len(ix)) * dt
                row = dict(seconds=len(ix) * dt, windows={}, timeline=[])
                for title, start in [('scored', 2.0), ('last_second', len(ix) * dt - 1.0)]:
                    chosen = ix[times >= start]
                    terms = dict(command_smoothing=(smooth.numpy()[chosen, :2] - raw.numpy()[chosen, :2]).mean(0), reference_motion=(refvel[chosen] - smooth.numpy()[chosen, :2]).mean(0), policy_physics=(actual[chosen] - refvel[chosen]).mean(0), total=(actual[chosen] - raw.numpy()[chosen, :2]).mean(0))
                    row['windows'][title] = {k: dict(vector=v.tolist(), norm=float(np.linalg.norm(v))) for k, v in terms.items()}
                    row['windows'][title]['leg_target_tracking_rmse_rad'] = float(np.sqrt(np.mean((d['joints'][chosen, 0][:, leg] - target[chosen][:, leg]) ** 2)))
                    row['windows'][title]['leg_reference_tracking_rmse_rad'] = float(np.sqrt(np.mean((d['joints'][chosen, 0][:, leg] - nominal[chosen][:, leg]) ** 2)))
                    assert np.max(abs(terms['total'] - terms['command_smoothing'] - terms['reference_motion'] - terms['policy_physics'])) < 1e-05
                for t in np.arange(0, len(ix) * dt, 0.5):
                    i = ix[min(len(ix) - 1, round(t / dt))]
                    row['timeline'].append(dict(t=float(t), smooth=smooth[i].tolist(), actual=actual[i].tolist(), reference_root_velocity=refvel[i].tolist(), gait_frequency=float(frequency[i]), run_weight=float(weights[i, 2:4].sum()), pivot_weight=float(weights[i, 5]), clips=[str(motion.clip_names[int(c)]) for c in ids[i]], weights=weights[i].tolist()))
                rows[label] = row
            result[name] = dict(reference_q_reconstruction_max_error=identity, segments=rows)
        workspace.write_text(OUT / 'reference_chain.json', json.dumps(result, indent=2), encoding='utf-8')
        for name, item in result.items():
            print(name, 'reference reconstruction error', item['reference_q_reconstruction_max_error'])
            for label, row in item['segments'].items():
                print(label, json.dumps(row['windows']))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

