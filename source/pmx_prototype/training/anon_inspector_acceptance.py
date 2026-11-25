"""Build the interactive inspector's acceptance and demonstration inputs."""

from itertools import chain
import random
from typing import NamedTuple


class PlaybackStep(NamedTuple):
    command: list
    reset: bool
    status: str
    push_character: bool = False
    manual_reset_only: bool = False


DEMO_PUSH_SECONDS = 5.0
DEMO_PUSH_KEY_HOLD_SECONDS = .5

INSPECTOR_STATIONARY_SECONDS = 3.0
# The segment number distinguishes the two forward runs and the stops.
INSPECTOR_SEGMENT_SECONDS = {
    (2, 'walk'): 3.0,
    (3, 'ramp_up_from_walk'): 10.0,
    (4, 'run_forward'): 4.0,
    (5, 'ramp_down'): 10.0,
    (6, 'stop'): 1.0,
    (9, 'run_backward'): 6.0,
    (11, 'run_left'): 6.0,
    (13, 'run_right'): 6.0,
    (15, 'run_diagonal_45'): 5.0,
    (17, 'run_diagonal_135'): 5.0,
}
DEMO_SEGMENTS = (
    ('stand', 2),
    ('walk', 4),
    ('ramp_up_from_walk', 8),
    ('run_forward', 4),
    ('ramp_down', 8),
    ('stop', 2),
    ('run_left', 6),
    ('stop', 2),
    ('run_right', 6),
    ('stop', 2),
    ('run_backward', 6),
    ('stop', 3),
    ('direction_sweep', 20),
    ('stop', 3),
    ('walk_turn_left', 6),
    ('stop', 3),
    ('walk_turn_right', 6),
    ('stop', 3),
    ('pivot_left', 4),
    ('pivot_right', 4),
    ('stand_end', 3),
)
DEMO_TURN_COMMANDS = {
    'walk_turn_left': (.50, 0., .30),
    'walk_turn_right': (.50, 0., -.30),
}


def acceptance_requests(config):
    return _playback_requests(config, demo=False)


def demo_requests(config):
    return _playback_requests(config, demo=True)


def _playback_requests(config, *, demo):
    # Load the evaluator on first use so it does not add to viewer startup time.
    from diagnostics.evaluate_parallel_walk_run import PARALLEL_ENVS, plan
    from evaluate_unified_walk_run import full_suite_settings, suite
    from locomotion_acceptance import CURRENT_PROTOCOL, STATIONARY_SEGMENTS

    dt = config.get('decimation', 4) / config.get('physics_hz', 120)
    settings = full_suite_settings(config, CURRENT_PROTOCOL)
    full_cases = suite(**settings)
    if demo:
        commands = {name: command for name, _, command in full_cases}
        commands.update(DEMO_TURN_COMMANDS)
        cases = [(name, seconds, commands[name]) for name, seconds in DEMO_SEGMENTS]
    else:
        cases = []
        for index, (name, seconds, command) in enumerate(full_cases, 1):
            if name in STATIONARY_SEGMENTS:
                seconds = INSPECTOR_STATIONARY_SECONDS
            cases.append((name, INSPECTOR_SEGMENT_SECONDS.get((index, name), seconds), command))
    serial = demo or config.get('unified_eval_parallel_envs', 1) != PARALLEL_ENVS
    groups = plan(cases, dt, diagnostic=serial)
    count = len(cases) + int(demo)
    frames = _frames(groups, dt, count)
    return chain(frames, _push_reaction_frames(config, dt, count)) if demo else frames


def _push_reaction_frames(config, dt, case_count, *, rng=None):
    from anon_inspector_controls import InspectorCommands

    # A local RNG keeps this interaction independent of policy/training randomness.
    rng = random.Random() if rng is None else rng
    controls = InspectorCommands(config)
    controls.set_speed(min(.5, controls.max_speed))
    controls.yaw_speed = min(.35, controls.max_yaw)
    singles = list('WASDQE')
    rng.shuffle(singles)
    combinations = ('WA', 'WD', 'SA', 'SD', 'WQ', 'WE', 'SQ', 'SE')
    steps = round(DEMO_PUSH_SECONDS / dt)
    hold_steps = max(1, round(DEMO_PUSH_KEY_HOLD_SECONDS / dt))
    for frame in range(steps):
        if frame % hold_steps == 0:
            slot = frame // hold_steps
            keys = singles[slot] if slot < len(singles) else rng.choice(combinations)
            controls.stop()
            for key in keys:
                controls.event(key, 'KEY_PRESS')
        status = f'{case_count:02d}/{case_count} Push Character   {(frame + 1) * dt:.1f}/{steps * dt:.1f}s'
        yield PlaybackStep(controls.command(), False, status,
                           push_character=frame == 0, manual_reset_only=True)


def _frames(groups, dt, case_count):
    for group in groups:
        # The first displayed request resets the character and starts the action.
        offset = group.warmup_steps
        for index, (name, seconds, _) in enumerate(group.segments, group.first_segment + 1):
            title = f'{index:02d}/{case_count} {name.replace("_", " ").capitalize()}'
            steps = round(seconds / dt)
            for frame in range(steps):
                status = f'{title}   {(frame + 1) * dt:.1f}/{steps * dt:.1f}s'
                request_index = offset + frame
                yield PlaybackStep(group.requested[request_index].tolist(), request_index == group.warmup_steps, status)
            offset += steps
