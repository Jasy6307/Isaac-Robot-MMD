"""Simulation-free comparison CLI, actor layout and frame-clock contracts."""
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def parse_options(argv=None, *, default_preset='standard'):
    parser = argparse.ArgumentParser(description='Synchronized G1 H5 / Anon cache comparison and recording.')
    parser.add_argument('--preset', choices=('standard', 'girls-full', 'girls-segment'), default=default_preset)
    for flag, required in (('g1-h5', True), ('corrected-h5', False), ('fidelity-h5', False),
                           ('smooth-h5', False), ('pmx-cache', True), ('audio', False)):
        parser.add_argument('--'+flag, type=Path, required=required)
    parser.add_argument('--g1-usd', type=Path, default=ROOT/'assets/g1_29dof_o6_hand_V3.usd')
    parser.add_argument('--separation', type=float, default=1.9)
    parser.add_argument('--actor-z-offset', type=float, default=0., help='Shift all actors onto the same floor datum in meters.')
    parser.add_argument('--width', type=int, default=1600)
    parser.add_argument('--height', type=int, default=900)
    parser.add_argument('--auto-play', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--preview-seconds', type=float, default=0)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--record-video', type=Path)
    parser.add_argument('--start-seconds', type=float)
    parser.add_argument('--end-seconds', type=float)
    parser.add_argument('--labels', nargs='+', help='One plain-text label per column, including Anon.')
    parser.add_argument('--dry-run', action='store_true', help='Validate input paths and print layout without starting Kit.')
    args = parser.parse_args(argv)
    if args.start_seconds is None:
        args.start_seconds = 16. if args.preset == 'girls-segment' else 0.
    if args.end_seconds is None and args.preset == 'girls-segment':
        args.end_seconds = 21.
    if args.preset == 'girls-full' and (args.fidelity_h5 or not args.corrected_h5 or args.smooth_h5):
        parser.error('girls-full requires --corrected-h5 and exactly three columns')
    if args.preset == 'girls-segment' and not all((args.corrected_h5, args.fidelity_h5, args.smooth_h5)):
        parser.error('girls-segment requires --corrected-h5, --fidelity-h5 and --smooth-h5')
    if args.fidelity_h5 and not args.corrected_h5:
        parser.error('--fidelity-h5 requires --corrected-h5')
    if args.smooth_h5 and not args.fidelity_h5:
        parser.error('--smooth-h5 requires --fidelity-h5 and --corrected-h5')
    if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
        parser.error('Invalid start time')
    if args.end_seconds is not None and (not math.isfinite(args.end_seconds) or args.end_seconds <= args.start_seconds):
        parser.error('End time must exceed start time')
    if not math.isfinite(args.separation) or args.separation <= 0 or min(args.width, args.height) <= 0:
        parser.error('Invalid layout dimensions')
    if not math.isfinite(args.actor_z_offset):
        parser.error('Invalid actor Z offset')
    if args.record_video:
        args.record_video = args.record_video.resolve()
        if args.record_video.exists() or args.record_video.with_suffix('.partial.mp4').exists():
            parser.error('Recording output already exists')
        if args.width % 2 or args.height % 2:
            parser.error('Video dimensions must be even')
    for key in ('g1_h5', 'corrected_h5', 'fidelity_h5', 'smooth_h5', 'pmx_cache', 'audio', 'g1_usd'):
        value = getattr(args, key)
        if value is not None:
            value = value.resolve()
            if not value.is_file():
                parser.error(f'{key} not found: {value}')
            setattr(args, key, value)
    if args.labels and (len(args.labels) != len(actor_layout(args)) or
                        any(any(c in title for c in "'\\:%,[]\n\r") for title in args.labels)):
        parser.error('Supply one plain label per column without filter punctuation')
    return args


def actor_layout(args):
    roles = [('g1_h5', '/World/G1', 'G1 IK - Before')]
    if args.corrected_h5:
        roles.append(('corrected_h5', '/World/G1Corrected', 'G1 IK - Collision corrected'))
    if args.fidelity_h5:
        roles[0] = ('g1_h5', '/World/G1', 'G1 IK - Original')
        roles[1] = ('corrected_h5', '/World/G1Corrected', 'G1 - Previous correction')
        roles.append(('fidelity_h5', '/World/G1Fidelity', 'G1 - Motion preserved'))
    if args.smooth_h5:
        roles[1] = ('corrected_h5', '/World/G1Corrected', 'G1 - Old correction')
        roles.append(('smooth_h5', '/World/G1Smoothed', 'G1 - Smoothed 17-20s' if args.preset == 'girls-segment' else 'G1 - Smoothed'))
    if args.preset == 'girls-full':
        roles[0] = ('g1_h5', '/World/G1', 'G1 - Motion preserved')
        roles[1] = ('corrected_h5', '/World/G1Corrected', 'G1 - Full smooth')
    roles.append(('pmx_cache', '/World/Character', 'Anon PMX'))
    return [dict(option=key, root=root, label=args.labels[i] if args.labels else label,
                 offset_x=(i-(len(roles)-1)/2)*args.separation) for i, (key, root, label) in enumerate(roles)]


def frame_range(args, fps, count):
    start = int(round(args.start_seconds*fps))
    end = count if args.end_seconds is None else min(count, int(round(args.end_seconds*fps)))
    if not 0 <= start < end <= count:
        raise ValueError('Invalid comparison time range')
    return start, end
