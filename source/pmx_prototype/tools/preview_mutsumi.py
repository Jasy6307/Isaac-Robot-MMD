"""Approved Mutsumi dance playback preset with 1K/2K resolution selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source_layout import PROJECT_ROOT,resolve_source,config_path
import argparse,json,runpy,datetime,os


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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=config_path,default=resolve_source('mutsumi_deepbluetown_playback.json'))
    parser.add_argument('--dance',help='Select an approved dance; default Deep Blue Town')
    parser.add_argument('--list-dances',action='store_true',help='List approved dances without starting Isaac')
    parser.add_argument('--resolution',choices=['1k','2k'],help='Default: approved preset resolution (2k)')
    parser.add_argument('--render-fps',type=int,choices=[30,60])
    parser.add_argument('--startup-settle-seconds',type=float,help='Override the runtime startup prewarm; default 2s, 0 disables')
    parser.add_argument('--no-secondary-physics',action='store_true',help='Play source animation without dynamic hair/skirt simulation')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--dry-run',action='store_true',help='Validate assets and print arguments without starting Isaac')
    parser.add_argument('--headless',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--steps',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.list_dances:
        for name,item in dance_catalog().items():print(f"{name}: {item['motion']}")
        return
    preset=json.loads(args.config.read_text(encoding='utf-8'))
    if args.dance:
        try:preset=select_dance(preset,args.dance)
        except ValueError as error:parser.error(str(error))
    resolution=args.resolution or preset['default_resolution']
    run_id=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}'
    output=args.output or PROJECT_ROOT/'output/mutsumi_playback'/preset['id']/resolution/run_id
    values=playback_arguments(preset,resolution,output,args.render_fps or preset['default_render_fps'],args.startup_settle_seconds,args.no_secondary_physics)
    if args.headless:values+=['--headless']
    if args.steps is not None:values+=['--steps',str(args.steps)]
    entry=resolve_source('physx_dance.py')
    if args.dry_run:print(json.dumps(dict(preset=preset['id'],entry=str(entry),arguments=values),ensure_ascii=False,indent=2));return
    sys.argv=[str(entry),*values];runpy.run_path(str(entry),run_name='__main__')


if __name__=='__main__':main()
