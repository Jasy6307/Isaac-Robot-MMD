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
def dance_catalog():
    return json.loads(resolve_source('configs/playback/mutsumi_dances.json').read_text(encoding='utf-8'))

def select_dance(preset,dance,catalog=None):
    catalog=catalog or dance_catalog()
    key=dance.casefold()
    for name,item in catalog.items():
        if key in [name.casefold(),*(alias.casefold() for alias in item.get('aliases',[]))]:
            return dict(preset,**{k:v for k,v in item.items() if k!='aliases'},id='mutsumi_'+name)
    raise ValueError(f'Unknown dance {dance!r}; choose from: '+', '.join(catalog))

def playback_arguments(preset,resolution,output,render_fps,settle_seconds=None,no_physics=False):
    def asset(key):
        value=Path(preset[key])
        path=value if value.is_absolute() else resolve_source(str(value)) if key=='physics_profile' else PROJECT_ROOT/value
        if not path.is_file():raise FileNotFoundError(f'Missing playback {key}: {path}')
        return path.resolve()
    profile=asset('physics_profile');settings=json.loads(profile.read_text(encoding='utf-8'))
    if settings['physics_hz']!=preset['physics_hz'] or settings['profile_id']!=preset['expected_profile_id']:
        raise ValueError('Default physics no longer matches the approved playback preset; review the preset before upgrading')
    width,height=preset['resolutions'][resolution]
    values=['--usd',str(asset('usd')),'--pmx',str(asset('pmx'))]
    if preset.get('rig_data'):values+=['--rig-data',str(asset('rig_data'))]
    values+=['--music',str(asset('music')),'--secondary-profile',str(profile),'--look',preset['look'],
        '--quality',preset['quality'],'--no-obstacles','--fit-camera-mesh','--render-fps',str(render_fps),
        '--width',str(width),'--height',str(height),'--output',str(Path(output).resolve())]
    if no_physics:
        if settle_seconds not in (None,0):raise ValueError('Physics-free playback cannot use physics prewarming')
        values+=['--no-secondary-physics']
    else:
        values+=['--secondary-worker']
        if settle_seconds is not None:values+=['--startup-settle-seconds',str(settle_seconds)]
    return values


def model_dances(model):
    if model == 'mutsumi':
        return dance_catalog()
    return json.loads(resolve_source('configs/playback/anon_dances.json').read_text(encoding='utf-8'))


def main(argv=None, *, preset_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=config_path)
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
    args = parser.parse_args(argv)
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
    preset = json.loads((args.config or resolve_source('mutsumi_deepbluetown_playback.json')).read_text(encoding='utf-8'))
    if not preset_only or args.dance:
        preset.update({key: value for key, value in item.items() if key != 'aliases'})
        preset['id'] = f'{args.model}_{name}'
    if args.model == 'anon':
        preset['pmx'] = 'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx'
        preset.pop('rig_data', None)
    resolution = args.resolution or preset['default_resolution']
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    output = args.output or PROJECT_ROOT / ('output/mutsumi_playback' if preset_only else 'output/dance_playback') / preset['id'] / resolution / run_id
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
        plan = dict(preset=preset['id'], entry=str(entry), arguments=values)
        if not preset_only: plan['model'] = args.model
        print(json.dumps(plan, ensure_ascii=False, indent=2))
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
