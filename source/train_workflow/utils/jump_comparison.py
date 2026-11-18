"""Consolidated jump comparison; historical measurement rules retained."""
from __future__ import annotations
from source.train_workflow.utils.jump_analysis import load_telemetry, require_new_files, contact_window_metrics
import argparse
import json
from pathlib import Path
import numpy as np


def compare_jump_training_summary(root):
    rows = []
    for path in sorted(root.glob("episode_*")):
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata["contact_source"] != "ground-only filtered ankle normal forces":
            raise ValueError("Comparison requires the same ground-only contact measurement")
        d = load_telemetry(path)
        metrics = contact_window_metrics(d, *JUMP_WINDOWS[0][1:])
        rows.append(dict(seed=metadata["seed"], episode=str(path), end_reason=metadata["end_reason"],
                         phase_no_contact_fraction=metrics['no_contact_fraction'],
                         phase_longest_no_contact_s=metrics['longest_air_s'],
                         phase_max_root_z_m=metrics['max_root_z_m'],
                         phase_height_rmse_m=metrics['height_rmse_m']))
    return rows


def compare_jump_training_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--before-label', default='Before: warm start')
    parser.add_argument('--after-label', default='After: 300 PPO iterations')
    args = parser.parse_args(argv)
    require_new_files(args.output, args.output.with_suffix('.png'))
    before, after = (compare_jump_training_summary(args.before), compare_jump_training_summary(args.after))
    if not before or not after or [x['seed'] for x in before] != [x['seed'] for x in after]:
        raise ValueError('Need matching nonempty seed sets')
    result = dict(before=before, after=after, note='No-contact and root height are diagnostic metrics, not proof of a visually correct jump.')
    for key in ('phase_no_contact_fraction', 'phase_longest_no_contact_s', 'phase_max_root_z_m', 'phase_height_rmse_m'):
        result[key] = dict(before=float(np.mean([x[key] for x in before])), after=float(np.mean([x[key] for x in after])))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for column, (label, rows) in enumerate(((args.before_label, before), (args.after_label, after))):
        data = load_telemetry(Path(rows[0]['episode']))
        seconds = data['time_s'] + 15.7
        for key in ('ref_root_z_m', 'root_z_m'):
            axes[0, column].plot(seconds, data[key], label=key)
        for side in ('left', 'right'):
            axes[1, column].plot(seconds, data[f'{side}_contact_N'], label=f'{side} ground contact N')
        axes[0, column].set_title(label + ' / seed42')
        for ax in axes[:, column]:
            ax.axvspan(17.08, 17.3, color='green', alpha=0.12, label='intended flight/close')
            ax.set_xlim(16.5, 17.6)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
        axes[1, column].set_xlabel('original motion seconds')
    axes[0, 0].set_ylabel('root height (m)')
    axes[1, 0].set_ylabel('ground normal force (N)')
    fig.tight_layout()
    fig.savefig(args.output.with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(json.dumps({key: value for key, value in result.items() if key not in ('before', 'after')}, indent=2))


JUMP_WINDOWS = (("17s",17.08,17.30),("21s",20.75,20.95),("48s",48.18,48.38))


def summarize_multi_jump_summary(root):
    result = {name: [] for name, _, _ in JUMP_WINDOWS}
    for path in sorted(root.glob("episode_*")):
        d = load_telemetry(path)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if meta["contact_source"] != "ground-only filtered ankle normal forces":
            raise ValueError("Ground-only telemetry required")
        for name, start, end in JUMP_WINDOWS:
            mask = (d["ref_time_s"] >= start) & (d["ref_time_s"] < end)
            if mask.sum() < 10:
                continue
            metrics = contact_window_metrics(d, start, end)
            result[name].append(dict(seed=meta["seed"], report=str(path), end_reason=meta["end_reason"],
                no_contact_fraction=metrics['no_contact_fraction'],
                longest_air_s=metrics['longest_air_s'],
                height_rmse_m=metrics['height_rmse_m']))
    return {name: dict(episodes=rows, mean={key:float(np.mean([row[key] for row in rows]))
                   for key in ("no_contact_fraction","longest_air_s","height_rmse_m")} if rows else {})
            for name,rows in result.items()}


def summarize_multi_jump_main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("reports",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    require_new_files(args.output)
    result=summarize_multi_jump_summary(args.reports)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({name:value["mean"] for name,value in result.items()},indent=2))
