"""Inspect one Anon by default, or control one actor in a multi-character scene."""

from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source
import argparse
from datetime import datetime
from pathlib import Path
import os
import runpy
import sys

ROOT=PROJECT_ROOT
from anon_policy_catalog import DEFAULT_INSPECTOR_CHECKPOINT, PRESETS
from secondary_profiles import SKIRT_EDGE_STRIDES,SECONDARY_MODES
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--characters',type=int,choices=range(1,9),default=1,
                   help='Characters to show: 1 preserves the single inspector; 2-8 use the crowd scene (default: 1)')
    p.add_argument('--policy',choices=PRESETS,help='Select a named checkpoint instead of the inspector default')
    p.add_argument('--checkpoint',type=Path,help='Inspect another compatible checkpoint')
    p.add_argument('--output',type=Path)
    p.add_argument('--body-device',choices=['cpu','cuda:0'],default='cpu',help='Body physics and policy inference device (default: CPU)')
    p.add_argument('--physics-threads',type=int,help='Secondary PhysX threads per actor (default: 16 single, 4 multi-character)')
    p.add_argument('--physics-detail',choices=SKIRT_EDGE_STRIDES,
                   help='VMD preview skirt-edge proxy density (default: sparse)')
    p.add_argument('--frontend-task-threads',type=int,default=4,help='Kit task threads for the viewer (default: 4)')
    p.add_argument('--render-fps',type=int,choices=[30,60],
                   help='Display rate; live preview defaults to 60 with interpolation, recording remains 30')
    p.add_argument('--interpolate-60',action='store_true',help='Use 60 Hz display interpolation like the dance viewer')
    p.add_argument('--inline-secondary',action='store_true',help='Use the previous in-process secondary physics path')
    p.add_argument('--no-secondary-physics',action='store_true',help='Preview the policy without secondary physics')
    p.add_argument('--secondary-mode',choices=SECONDARY_MODES,help='Hair/skirt solve budget (default: light; all budgets retain nodes, collisions and 120 Hz)')
    p.add_argument('--legacy-secondary-physics',action='store_true',help='Use the historical secondary physics profile')
    p.add_argument('--record-video',type=Path,help='Record the preview at 30 FPS')
    crowd=p.add_argument_group('Multi-character scene (--characters 2-8)')
    crowd.add_argument('--controlled-character',type=int,default=1,help='Initially controlled actor, numbered from 1')
    crowd.add_argument('--random-only',action='store_true',help='Give all actors random commands')
    crowd.add_argument('--mixed-secondary',action='store_true',help='Use separate controlled/background budgets (default: light/light3); overrides --secondary-mode')
    crowd.add_argument('--controlled-secondary-mode',choices=SECONDARY_MODES,help='Controlled actor budget with --mixed-secondary (default: light)')
    crowd.add_argument('--inter-character-collisions',action='store_true',help='Enable actor body collisions (default: off)')
    crowd.add_argument('--seed',type=int,help='Background command seed (default: 42)')
    crowd.add_argument('--max-speed',type=float,help='Background planar speed limit (default: 0.55 m/s)')
    crowd.add_argument('--max-yaw',type=float,help='Background yaw limit (default: 0.4 rad/s)')
    crowd.add_argument('--arena-radius',type=float,help='Background soft return radius (default: 7 m)')
    crowd.add_argument('--serial-playback',action='store_true',help='Diagnostic: disable multi-character worker prefetch')
    a,extra=p.parse_known_args()
    a.secondary_mode=a.secondary_mode or ('full' if a.no_secondary_physics or a.legacy_secondary_physics else 'light')
    if a.controlled_secondary_mode is not None and not a.mixed_secondary:
        p.error('--controlled-secondary-mode requires --mixed-secondary')
    if a.secondary_mode!='full' and (a.no_secondary_physics or a.legacy_secondary_physics):
        p.error(f'--secondary-mode {a.secondary_mode} cannot be combined with disabled or legacy secondary physics')
    if not 1<=a.controlled_character<=a.characters:p.error('--controlled-character must be between 1 and --characters')
    if a.mixed_secondary and (a.no_secondary_physics or a.random_only):
        p.error('--mixed-secondary requires secondary physics and a manually controlled actor')
    if a.characters==1 and (a.random_only or a.mixed_secondary or a.inter_character_collisions or a.serial_playback or any(
            value is not None for value in (a.seed,a.max_speed,a.max_yaw,a.arena_radius))):
        p.error('Multi-character options require --characters 2-8')
    checkpoint=(a.checkpoint or PRESETS.get(a.policy, DEFAULT_INSPECTOR_CHECKPOINT)).resolve()
    if not checkpoint.is_file():p.error(f'Missing checkpoint: {checkpoint}')
    if a.characters>1:
        if a.inline_secondary or a.legacy_secondary_physics:
            p.error('Multi-character inspection uses separate secondary workers; inline and legacy modes are single-character options')
        if a.interpolate_60 and a.render_fps==30:p.error('--interpolate-60 requires a 60 FPS display')
        if a.record_video and (a.render_fps==60 or a.interpolate_60):
            p.error('60 FPS interpolation is for live preview; recording remains 30 FPS')
        from anon_crowd_viewer import run
        arguments=['--characters',str(a.characters),'--checkpoint',str(checkpoint),
                  '--controlled-character',str(a.controlled_character),'--body-device',a.body_device,
                  '--physics-threads',str(a.physics_threads if a.physics_threads is not None else 4),
                  '--frontend-task-threads',str(a.frontend_task_threads),
                  '--physics-detail',a.physics_detail or 'sparse',*extra]
        arguments+=['--secondary-mode',a.secondary_mode]
        if a.render_fps is not None or a.interpolate_60:
            arguments+=['--render-fps',str(60 if a.interpolate_60 else a.render_fps)]
        for name in ('output','record_video','seed','max_speed','max_yaw','arena_radius','controlled_secondary_mode'):
            value=getattr(a,name)
            if value is not None:arguments+=['--'+name.replace('_','-'),str(value)]
        for name in ('no_secondary_physics','random_only','mixed_secondary','inter_character_collisions','serial_playback'):
            if getattr(a,name):arguments.append('--'+name.replace('_','-'))
        run(arguments)
        return
    output=a.output or ROOT/'output/anon_locomotion/inspector'/f'{a.policy or "walk"}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    device=a.body_device
    use_secondary_worker=not (a.inline_secondary or a.no_secondary_physics)
    physics_detail=a.physics_detail or ('full' if a.no_secondary_physics or a.legacy_secondary_physics else 'sparse')
    render_fps=a.render_fps or (60 if a.interpolate_60 or (use_secondary_worker and not a.record_video) else 30)
    sys.argv=[str(SOURCE_DIR/'play_anon_policy.py'),'--checkpoint',str(checkpoint),'--inspector','--output',str(output),
              '--body-device',device,'--physics-threads',str(a.physics_threads if a.physics_threads is not None else 16),'--physics-detail',physics_detail,
              '--frontend-task-threads',str(a.frontend_task_threads),'--render-fps',str(render_fps),*extra]
    if use_secondary_worker:sys.argv.append('--secondary-worker')
    if a.no_secondary_physics:sys.argv.append('--no-secondary-physics')
    if a.legacy_secondary_physics:sys.argv.append('--legacy-secondary-physics')
    if a.record_video:sys.argv+=['--record-video',str(a.record_video)]
    if a.interpolate_60:sys.argv.append('--interpolate-60')
    sys.argv+=['--secondary-mode',a.secondary_mode]
    sys.path.insert(0,str(SOURCE_DIR))
    runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':main()
