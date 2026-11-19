"""Cache the full-profile display skeleton from a measured Anon policy rollout."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_layout import PROJECT_ROOT
from anon_secondary_stream import AnonSecondaryStream
from secondary_profiles import DEFAULT_SECONDARY_PROFILE, DEFAULT_SECONDARY_QUALITY


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through-label', default='walk')
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    started = perf_counter()
    rollout = args.report.with_name('rollout.npz')
    report = json.loads(args.report.read_text(encoding='utf-8'))
    manifest_path = Path(report['env_config']['asset']).with_name('manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    with np.load(rollout) as trace:
        if not np.isclose(float(trace['dt']), 1 / 30):
            raise ValueError('Secondary replay requires a 30 Hz policy trace')
        matches = np.flatnonzero(trace['labels'] == args.through_label)
        if not len(matches):
            raise ValueError(f'No segment labeled {args.through_label} in {rollout}')
        count = int(matches[-1]) + 1
        poses = trace['poses'][:count, 0]
        dones = trace['dones'][:count, 0]
        links = trace['link_names'].tolist()
    settings = SimpleNamespace(
        output=args.output, quality=DEFAULT_SECONDARY_QUALITY, physics_threads=16,
        skirt_edge_stride=1, secondary_profile=DEFAULT_SECONDARY_PROFILE,
        legacy_secondary_physics=False, reference_hair_compute=False, legacy_compute=False,
        pmx=PROJECT_ROOT / 'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx',
    )
    stream = AnonSecondaryStream(settings, poses[0], manifest, links, SimpleNamespace(update=lambda: None))
    try:
        metadata = stream.metadata
        if (metadata['physics_hz'] != 120 or not metadata['edge_nodes']
                or metadata['settings']['preview_skirt_edge_stride'] != 1):
            raise ValueError(f'Expected full 120 Hz secondary physics, got {metadata}')
        print(f'SECONDARY_READY nodes={metadata["dynamic_nodes"]} edges={metadata["edge_nodes"]}', flush=True)
        positions = np.empty((count, len(manifest['bone_paths']), 3), dtype=np.float32)
        rotations = np.empty((count, positions.shape[1], 4), dtype=np.float32)
        for frame, body in enumerate(poses):
            stream.send(frame, dict(poses=body, restart=bool(frame and dones[frame - 1])))
            state = stream.receive(frame)
            positions[frame], rotations[frame] = state['p'], state['q']
            if (frame + 1) % 300 == 0:
                print(f'SECONDARY_FRAME {frame + 1}/{count}', flush=True)
    finally:
        stream.close()
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError('Secondary replay produced nonfinite skin transforms')
    np.savez_compressed(args.output / 'secondary.npz', positions=positions, rotations=rotations)
    metadata.update(frames=count, fps=30, rollout=str(rollout.resolve()), rollout_sha256=sha256(rollout),
                    pmx=str(settings.pmx), pmx_sha256=sha256(settings.pmx),
                    secondary_profile=str(DEFAULT_SECONDARY_PROFILE),
                    secondary_profile_sha256=sha256(DEFAULT_SECONDARY_PROFILE),
                    time_seconds=perf_counter() - started)
    (args.output / 'secondary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'SECONDARY_COMPLETE {count} frames in {metadata["time_seconds"]:.2f}s', flush=True)


if __name__ == '__main__':
    main()
