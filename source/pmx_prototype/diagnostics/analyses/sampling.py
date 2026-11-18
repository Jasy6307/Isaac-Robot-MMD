"""Sampling diagnostic recipes with shared workspace and explicit I/O."""
from pathlib import Path
from types import SimpleNamespace
from source.paths import REPO_ROOT

def build_audit_low_speed_sampling(workspace, legacy_argv=None):
    read = workspace.read_json
    write = workspace.write_json
    execute = workspace.execute
    import json
    from pathlib import Path
    import sys
    from types import SimpleNamespace
    import torch
    ROOT = Path(REPO_ROOT)
    SOURCE = Path(REPO_ROOT) / 'source/pmx_prototype'
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'motion')]
    from unified_walk_run_reference import sample_commands
    OUT = workspace.directory(ROOT / 'output/anon_locomotion/low_speed_diagnosis_20261117')
    CONFIGS = workspace.config_paths or {'original_scratch_recipe': ROOT / 'output/anon_locomotion/scratch_v8_unified_20261116_12288/config_override.json', 'sampling_only_7600': ROOT / 'output/anon_locomotion/pivot_ablation_preserved_20261116/sampling_only.json', 'latest_scratch_recipe': ROOT / 'output/anon_locomotion/scratch_split_20261117/config.json'}

    def environment(cfg, iteration, n=12288):
        return SimpleNamespace(num_envs=n, device='cpu', cfg=cfg, evaluation=False, iteration=iteration, commands=torch.zeros(n, 3), continuous_ref_commands=torch.zeros(n, 3), command_age=torch.zeros(n))

    def flags(commands):
        speed = commands[:, :2].norm(dim=-1)
        return dict(idle=commands.abs().sum(-1) < 0.001, pivot=(speed < 0.001) & (commands[:, 2].abs() > 0.001), low_moving=(speed >= 0.02) & (speed < 0.4), low_020_040=(speed >= 0.2) & (speed < 0.4), benchmark_030_neighborhood=(speed >= 0.25) & (speed <= 0.35) & (commands[:, 0] > 0) & (commands[:, 1].abs() < 0.03) & (commands[:, 2].abs() < 0.05), running=speed >= 0.85)

    def main():
        torch.set_num_threads(2)
        rows = {}
        for name, path in CONFIGS.items():
            cfg = json.loads(workspace.read_text(path))
            stages = {}
            for iteration in [0, 1500, 5674]:
                torch.manual_seed(42)
                env = environment(cfg, iteration)
                ids = torch.arange(env.num_envs)
                count = {}
                weighted = {}
                total = 0
                hold_total = 0.0
                for sample in range(128):
                    sample_commands(env, ids, on_reset=sample == 0)
                    masks = flags(env.commands)
                    hold = env.unified_command_hold
                    for key, mask in masks.items():
                        count[key] = count.get(key, 0) + int(mask.sum())
                        weighted[key] = weighted.get(key, 0.0) + float(hold[mask].sum())
                    total += env.num_envs
                    hold_total += float(hold.sum())
                stages[str(iteration)] = dict(generated_commands=total, event_fraction={k: v / total for k, v in count.items()}, hold_weighted_fraction_without_falls={k: v / hold_total for k, v in weighted.items()})
            dedicated = sum((cfg.get(k, 0.0) for k in ['unified_backward_env_fraction', 'unified_right_env_fraction']))
            pivots = cfg.get('unified_pivot_sampling', cfg.get('unified_pivot_training', False))
            if pivots:
                dedicated += cfg.get('unified_pivot_env_fraction', 0.1) + cfg.get('unified_reversal_env_fraction', 0.1)
            dedicated += cfg.get('unified_transition_env_fraction', 0.0)
            rows[name] = dict(source=str(path), remaining_general_pool=1 - dedicated, stages=stages)
        cfg = json.loads(workspace.read_text(CONFIGS['latest_scratch_recipe']))
        variants = []
        for fraction in [0.0, 0.9]:
            altered = dict(cfg, continuous_low_speed_fraction=fraction)
            torch.manual_seed(123)
            env = environment(altered, 5674)
            sample_commands(env, torch.arange(env.num_envs), on_reset=True)
            variants.append(env.commands)
        result = dict(training=False, method='Production sampler on CPU,12288 environment IDs,128 resampling events per stage.', limitation='These are generated-command proportions, not historical successful rollout-frame occupancy. Hold weighting ignores falls and asynchronous episode resets.', recipes=rows, legacy_low_speed_fraction_flag_changes_commands=not torch.equal(*variants))
        OUT.mkdir(parents=True, exist_ok=True)
        workspace.write_text(OUT / 'sampling_audit.json', json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

def build_verify_low_speed_fix(workspace, legacy_argv=None):
    read = workspace.read_json
    write = workspace.write_json
    execute = workspace.execute
    import json
    from pathlib import Path
    import sys
    import torch
    ROOT = Path(REPO_ROOT)
    SOURCE = Path(REPO_ROOT) / 'source/pmx_prototype'
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'diagnostics')]
    from audit_low_speed_sampling import environment, flags
    from unified_walk_run_reference import sample_commands

    def main():
        torch.set_num_threads(2)
        out = workspace.output_dir
        out.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(workspace.read_text(SOURCE / 'configs/training/unified/low_speed_balanced_v2.json'))
        torch.manual_seed(42)
        e = environment(cfg, 5674)
        ids = torch.arange(e.num_envs)
        count = {}
        total = 0
        for step in range(128):
            sample_commands(e, ids, on_reset=step == 0)
            for name, mask in flags(e.commands).items():
                count[name] = count.get(name, 0) + int(mask.sum())
            total += e.num_envs
        before = json.loads(workspace.read_text(ROOT / 'output/anon_locomotion/low_speed_diagnosis_20261117/sampling_audit.json'))['recipes']['latest_scratch_recipe']['stages']['5674']['event_fraction']
        result = dict(training=False, generated_commands=total, before=before, after={k: v / total for k, v in count.items()}, pool_fractions=dict(backward=0.2, lateral=0.25, pivot=0.08, reversal=0.07, run_transition=0.05, sustained_low_speed=0.15, low_speed_stop=0.1, general=0.1), limitation='Generated command proportions, not historical rollout occupancy; the low_moving bin excludes the exact0.4 boundary.')
        assert abs(sum(result['pool_fractions'].values()) - 1) < 1e-08
        assert result['after']['low_moving'] > 0.15
        assert result['after']['benchmark_030_neighborhood'] > before['benchmark_030_neighborhood'] * 20
        workspace.write_text(out / 'sampling_verification.json', json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    return SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('_')})

