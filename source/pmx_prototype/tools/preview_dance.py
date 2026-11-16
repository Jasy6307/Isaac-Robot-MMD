"""Select a character and an approved dance for live PhysX preview."""
import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_layout import PROJECT_ROOT, config_path, resolve_source
from preview_mutsumi import dance_catalog, playback_arguments


def model_dances(model):
    if model == 'mutsumi':
        return dance_catalog()
    return json.loads(resolve_source('configs/playback/anon_dances.json').read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['mutsumi', 'anon'], default='mutsumi')
    parser.add_argument('--dance', help='Dance name; defaults to Deep Blue Town for Mutsumi, Girls for Anon')
    parser.add_argument('--list-dances', action='store_true')
    parser.add_argument('--resolution', choices=['1k', '2k'])
    parser.add_argument('--render-fps', type=int, choices=[30, 60])
    parser.add_argument('--interpolate-60',action='store_true',help='Smooth 30Hz simulated poses for 60Hz display')
    parser.add_argument('--music-sync-tolerance', type=float, help='Seconds of audio drift allowed before seeking')
    parser.add_argument('--physics-threads', type=int, help='CPU PhysX worker threads (default 16)')
    parser.add_argument('--physics-detail', choices=['full','reduced','sparse'], default='full', help='Experimental reduced skirt-edge collision proxies')
    parser.add_argument('--async-render',action='store_true',help='Experimental asynchronous rendering')
    parser.add_argument('--capture-frames',type=int,nargs='*',help=argparse.SUPPRESS)
    parser.add_argument('--startup-settle-seconds', type=float)
    parser.add_argument('--no-secondary-physics', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--headless', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--steps', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.interpolate_60 and (args.render_fps != 60 or args.no_secondary_physics):
        parser.error('--interpolate-60 requires --render-fps 60 and secondary physics')
    if args.physics_detail != 'full' and args.model != 'anon':
        parser.error('reduced physics is only configured for Anon')
    if args.physics_detail != 'full' and args.no_secondary_physics:
        parser.error('reduced physics requires secondary physics to be enabled')
    catalog = model_dances(args.model)
    if args.list_dances:
        for name, item in catalog.items():
            print(f"{name}: {item['motion']}")
        return
    requested = args.dance or ('deepbluetown' if args.model == 'mutsumi' else 'girls')
    matches = [(name, item) for name, item in catalog.items()
               if requested.casefold() in [name.casefold(), *(a.casefold() for a in item.get('aliases', []))]]
    if not matches:
        parser.error(f"Dance {requested!r} is not available for {args.model}; choose from: {', '.join(catalog)}")
    name, item = matches[0]
    preset = json.loads(resolve_source('mutsumi_deepbluetown_playback.json').read_text(encoding='utf-8'))
    preset.update({key: value for key, value in item.items() if key != 'aliases'})
    preset['id'] = f'{args.model}_{name}'
    if args.model == 'anon':
        preset['pmx'] = 'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx'
        preset.pop('rig_data', None)
    resolution = args.resolution or preset['default_resolution']
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    output = args.output or PROJECT_ROOT / 'output/dance_playback' / preset['id'] / resolution / run_id
    values = playback_arguments(preset, resolution, output, args.render_fps or preset['default_render_fps'],
                                args.startup_settle_seconds, args.no_secondary_physics)
    if args.physics_threads is not None:
        if args.physics_threads < 1:
            parser.error('--physics-threads must be positive')
        values += ['--physics-threads', str(args.physics_threads)]
    if args.physics_detail != 'full':
        values += ['--skirt-edge-stride','2' if args.physics_detail == 'reduced' else '3']
    if args.async_render:
        values += ['--async-render']
    if args.interpolate_60:
        values += ['--interpolate-60']
    if args.capture_frames:
        values += ['--capture-frames',*(str(frame) for frame in args.capture_frames)]
    if args.music_sync_tolerance is not None:
        values += ['--music-sync-tolerance', str(args.music_sync_tolerance)]
    elif args.model == 'anon':
        values += ['--music-sync-tolerance', '.25']
    if args.headless:
        values += ['--headless']
    if args.steps is not None:
        values += ['--steps', str(args.steps)]
    entry = resolve_source('physx_dance.py')
    if args.dry_run:
        print(json.dumps(dict(model=args.model, preset=preset['id'], entry=str(entry), arguments=values), ensure_ascii=False, indent=2))
        return
    if args.model == 'anon':
        # The larger native cloth benefits from a larger worker task pool.
        tuned=args.physics_detail == 'sparse' and args.interpolate_60
        os.environ.setdefault('DANCE_WORKER_TASK_THREADS', '8' if tuned else '16')
        if tuned:os.environ.setdefault('DANCE_FRONTEND_TASK_THREADS', '4')
    sys.argv = [str(entry), *values]
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
