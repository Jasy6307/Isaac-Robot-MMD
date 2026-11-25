"""Check inspector launch defaults and independent display/solve measurements."""

from collections import deque
from contextlib import redirect_stderr
from itertools import groupby
from io import StringIO
import math
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import source_layout
from anon_inspector_acceptance import acceptance_requests, demo_requests
from anon_inspector_controls import InspectorCommands, keys_for_command
from anon_inspector_ui import PolicyInspector


FULL_ACCEPTANCE_CONFIG = dict(
    unified_walk_run=True, max_speed=1.2, physics_hz=120, decimation=4,
    unified_idle_head_acceptance=True, unified_eval_backward_walk=True,
    unified_eval_parallel_envs=4, unified_eval_backward_diagonal_run_speed=.85,
    unified_eval_stand_hold_seconds=12, unified_eval_stop_timing='compact_walk_stop',
)


class SliderModel:
    def __init__(self, value, changed):
        self.value = value
        self.changed = changed

    def get_value_as_float(self):
        return self.value

    def set_value(self, value):
        self.value = value
        self.changed(self)


class InspectorPreviewTests(unittest.TestCase):
    def launch_args(self, *options, default_checkpoint=False):
        entry = SOURCE / 'inspect_anon_policy.py'
        run_wrapper = runpy.run_path
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'policy.pt'
            checkpoint.touch()
            arguments = []
            selected = [] if default_checkpoint else ['--checkpoint', str(checkpoint)]
            with (
                patch.object(sys, 'argv', [str(entry), *selected, *options]),
                patch.object(sys, 'path', sys.path.copy()),
                patch('runpy.run_path', side_effect=lambda *a, **k: arguments.extend(sys.argv)) as player,
                patch('anon_crowd_viewer.run',side_effect=lambda argv: arguments.extend(
                    [str(SOURCE/'runtime/anon_crowd_viewer.py'),*argv])) as crowd,
            ):
                run_wrapper(str(entry), run_name='__main__')
                self.assertEqual(player.call_count+crowd.call_count,1)
            return arguments

    def test_without_arguments_loads_the_inspector_default(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'scratch_010000.pt'
            checkpoint.touch()
            with patch('anon_policy_catalog.DEFAULT_INSPECTOR_CHECKPOINT', checkpoint):
                arguments = self.launch_args(default_checkpoint=True)
            self.assertEqual(Path(arguments[arguments.index('--checkpoint') + 1]), checkpoint.resolve())

    def test_single_character_keeps_original_player_and_defaults(self):
        for options in ((),('--characters','1')):
            with self.subTest(options=options):
                arguments=self.launch_args(*options)
                self.assertEqual(Path(arguments[0]).name,'play_anon_policy.py')
                self.assertIn('--inspector',arguments)
                self.assertIn('--secondary-worker',arguments)
                for name,value in (('--body-device','cpu'),
                                   ('--physics-threads','16'),('--render-fps','60'),('--physics-detail','sparse'),('--secondary-mode','light')):
                    self.assertEqual(arguments[arguments.index(name)+1],value)
                self.assertNotIn('--characters',arguments)

    def test_four_characters_launch_crowd_with_existing_defaults(self):
        arguments=self.launch_args('--characters','4')
        self.assertEqual(Path(arguments[0]),SOURCE/'runtime/anon_crowd_viewer.py')
        for name,value in (('--characters','4'),('--controlled-character','1'),('--body-device','cpu'),
                           ('--physics-threads','4'),('--frontend-task-threads','4'),('--physics-detail','sparse'),('--secondary-mode','light')):
            self.assertEqual(arguments[arguments.index(name)+1],value)
        for option in ('--no-secondary-physics','--random-only','--inter-character-collisions','--output'):
            self.assertNotIn(option,arguments)

    def test_explicit_secondary_modes_override_defaults_for_both_launch_paths(self):
        for count in (1,4):
            for mode in ('full','light','light2','light3'):
                options=['--characters',str(count),'--secondary-mode',mode]
                arguments=self.launch_args(*options)
                self.assertEqual(arguments[arguments.index('--secondary-mode')+1],mode)
                self.assertNotIn('--no-secondary-physics',arguments)

    def test_implicit_light_does_not_break_body_only_or_legacy_diagnostics(self):
        for options in (('--no-secondary-physics',),('--characters','4','--no-secondary-physics'),('--legacy-secondary-physics',)):
            arguments=self.launch_args(*options)
            self.assertEqual(arguments[arguments.index('--secondary-mode')+1],'full')

    def test_light_mode_rejects_incompatible_physics_options(self):
        for mode in ('light','light2','light3'):
            for option in ('--no-secondary-physics','--legacy-secondary-physics'):
                with redirect_stderr(StringIO()),self.assertRaises(SystemExit):
                    self.launch_args('--secondary-mode',mode,option)

    def test_serial_playback_is_multi_character_diagnostic(self):
        arguments=self.launch_args('--characters','2','--serial-playback','--secondary-mode','light2')
        self.assertIn('--serial-playback',arguments)
        self.assertEqual(arguments[arguments.index('--secondary-mode')+1],'light2')
        with redirect_stderr(StringIO()),self.assertRaises(SystemExit):self.launch_args('--serial-playback')

    def test_mixed_secondary_is_multi_actor_only_and_requires_manual_physics(self):
        args=self.launch_args('--characters','4','--mixed-secondary','--secondary-mode','light3')
        self.assertIn('--mixed-secondary',args)
        for options in ((),('--characters','4','--random-only'),('--characters','4','--no-secondary-physics'),
                        ('--characters','4','--inline-secondary'),('--characters','4','--legacy-secondary-physics')):
            with self.subTest(options=options),redirect_stderr(StringIO()),self.assertRaises(SystemExit):
                self.launch_args(*options,'--mixed-secondary')

    def test_controlled_secondary_budget_is_forwarded_and_requires_mixed_mode(self):
        args=self.launch_args('--characters','4','--mixed-secondary','--controlled-secondary-mode','light2')
        self.assertEqual(args[args.index('--controlled-secondary-mode')+1],'light2')
        with redirect_stderr(StringIO()),self.assertRaises(SystemExit):
            self.launch_args('--characters','4','--controlled-secondary-mode','light2')

    def test_crowd_accepts_independent_display_rate_and_interpolation_alias(self):
        for options,expected in ((('--render-fps','60'),'60'),(('--render-fps','30'),'30'),(('--interpolate-60',),'60')):
            arguments=self.launch_args('--characters','2',*options)
            self.assertEqual(arguments[arguments.index('--render-fps')+1],expected)

    def test_crowd_recording_rejects_explicit_live_60_fps(self):
        with redirect_stderr(StringIO()),self.assertRaises(SystemExit):
            self.launch_args('--characters','2','--record-video','test.mp4','--render-fps','60')

    def test_crowd_forwards_shared_and_background_options(self):
        options=['--characters','4','--controlled-character','3','--body-device','cuda:0',
                 '--physics-threads','0','--frontend-task-threads','2','--physics-detail','full',
                 '--output','test-output','--record-video','test.mp4','--seed','12',
                 '--max-speed','0.4','--max-yaw','0.2','--arena-radius','8.0',
                 '--headless','--steps','12','--video-width','1280','--video-height','720',
                 '--no-secondary-physics','--inter-character-collisions','--no-real-time']
        arguments=self.launch_args(*options)
        for index,option in enumerate(options):
            if not option.startswith('--'):continue
            self.assertIn(option,arguments)
            if index+1<len(options) and not options[index+1].startswith('--'):
                self.assertEqual(arguments[arguments.index(option)+1],options[index+1])

    def test_named_policy_uses_the_same_checkpoint_in_both_modes(self):
        with TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'named.pt';checkpoint.touch()
            with patch.dict('anon_policy_catalog.PRESETS',{'walk':checkpoint}):
                arguments=self.launch_args('--characters','4','--policy','walk',default_checkpoint=True)
            self.assertEqual(Path(arguments[arguments.index('--checkpoint')+1]),checkpoint.resolve())

    def test_invalid_counts_and_unsupported_crowd_modes_fail_before_launch(self):
        cases=[('--characters','0'),('--characters','9'),('--characters','4','--controlled-character','5'),
               ('--characters','4','--render-fps','30','--interpolate-60'),
               ('--characters','4','--inline-secondary'),('--characters','4','--legacy-secondary-physics'),
               ('--random-only',)]
        for options in cases:
            with self.subTest(options=options),redirect_stderr(StringIO()),self.assertRaises(SystemExit) as error:
                self.launch_args(*options)
            self.assertEqual(error.exception.code,2)

    def test_acceptance_input_groups_use_shorter_demo_timing_and_skip_warmups(self):
        frames = list(acceptance_requests(FULL_ACCEPTANCE_CONFIG))
        self.assertEqual(len(frames), 6420)
        resets = [index for index, frame in enumerate(frames) if frame.reset]
        self.assertEqual(resets, [0, 1260, 2850, 4620])
        self.assertTrue(all('warm-up' not in frame.status for frame in frames))
        for begin, first_phase in zip(resets, ('01/40 Stand', '09/40 Run backward',
                                               '21/40 Run diagonal 315', '29/40 Reverse turn')):
            self.assertTrue(frames[begin][2].startswith(first_phase + '   '))
        np.testing.assert_allclose(frames[resets[1]][0], [-.85, 0., 0.])

        phases = [(title, list(phase)) for title, phase in groupby(frames, key=lambda frame: frame[2].split('   ')[0])]
        expected_durations = [
            3, 3, 10, 4, 10, 1, 8, 3,
            6, 3, 6, 3, 6, 3, 5, 3, 5, 3, 7, 3,
            7, 3, 24, 3, 8, 3, 8, 3,
            8, 3, 8, 3, 6, 3, 8, 3, 6, 3, 6, 3,
        ]
        self.assertEqual([len(phase) for _, phase in phases], [seconds * 30 for seconds in expected_durations])
        self.assertEqual(phases[0][0], '01/40 Stand')
        stationary_phases = [(title, phase) for title, phase in phases if title.split(' ', 1)[1] in
                             {'Stand', 'Stop', 'Stop after walk', 'Stop after pivot left', 'Stand end'}]
        self.assertEqual(len(stationary_phases), 19)
        for title, phase in stationary_phases:
            self.assertTrue(all(frame.command == [0., 0., 0.] for frame in phase))
            seconds = 1.0 if title == '06/40 Stop' else 3.0
            self.assertTrue(phase[-1][2].endswith(f'{seconds:.1f}/{seconds:.1f}s'))
        self.assertEqual(phases[-1][0], '40/40 Stand end')
        ramp_fraction = np.arange(300) / 300
        np.testing.assert_allclose([row[0][0] for row in phases[2][1]], .3 + .9 * ramp_fraction)
        np.testing.assert_allclose([row[0][0] for row in phases[4][1]], 1.2 * (1 - ramp_fraction))

    def test_auto_demo_is_one_continuous_short_playback(self):
        frames = list(demo_requests(FULL_ACCEPTANCE_CONFIG))
        self.assertEqual(len(frames), 110 * 30)
        self.assertEqual([index for index, frame in enumerate(frames) if frame.reset], [0])
        phases = {title.split(' ', 1)[1]: list(rows)
                  for title, rows in groupby(frames, key=lambda row: row[2].split('   ')[0])}
        sweep = np.asarray([row[0] for row in phases['Direction sweep']])
        self.assertEqual(len(sweep), 20 * 30)
        np.testing.assert_allclose(np.linalg.norm(sweep[:, :2], axis=1), 1.0)
        np.testing.assert_allclose(sweep[150], [0., 1., 0.], atol=1e-7)
        np.testing.assert_allclose(sweep[450], [0., -1., 0.], atol=1e-7)
        np.testing.assert_allclose(phases['Run backward'][0][0], [-.85, 0., 0.])
        np.testing.assert_allclose(phases['Walk turn left'][0][0], [.50, 0., .30])
        np.testing.assert_allclose(phases['Walk turn right'][0][0], [.50, 0., -.30])
        np.testing.assert_allclose(phases['Pivot left'][0][0], [0., 0., .6])
        np.testing.assert_allclose(phases['Pivot right'][0][0], [0., 0., -.6])
        self.assertTrue(all(frame.command == [0., 0., 0.] for frame in phases['Stand end']))
        reaction=phases['Push Character']
        self.assertEqual(len(reaction),150)
        self.assertEqual([i for i,frame in enumerate(reaction) if frame.push_character],[0])
        self.assertTrue(all(frame.manual_reset_only and not frame.reset for frame in reaction))
        self.assertTrue(reaction[-1].status.endswith('5.0/5.0s'))
        keys=[keys_for_command(frame.command) for frame in reaction]
        self.assertEqual(set().union(*keys),set('WASDQE'))
        for begin in range(0,150,15):
            self.assertTrue(all(key==keys[begin] for key in keys[begin:begin+15]))
        self.assertTrue(all(math.hypot(*frame.command[:2])<=.5+1e-7 and abs(frame.command[2])<=.35
                            for frame in reaction))

    def inspector_with_controls(self):
        inspector = PolicyInspector.__new__(PolicyInspector)
        inspector.ui = SimpleNamespace(Pixel=lambda value: value)
        inspector.controls = InspectorCommands(FULL_ACCEPTANCE_CONFIG)
        inspector.policy_config = FULL_ACCEPTANCE_CONFIG
        inspector.playback_frames = None
        inspector.playback_name = ''
        inspector.playback_status = ''
        inspector.syncing_sliders = False
        inspector.acceptance_button = SimpleNamespace(visible=True, enabled=True)
        inspector.demo_button = SimpleNamespace(visible=True, enabled=True)
        inspector.speed_slider = SimpleNamespace(enabled=True)
        inspector.yaw_slider = SimpleNamespace(enabled=True)
        inspector.speed_model = SliderModel(inspector.controls.speed, inspector.on_speed_changed)
        inspector.yaw_model = SliderModel(inspector.controls.yaw_speed, inspector.on_yaw_changed)
        inspector.speed_value = SimpleNamespace(text='')
        inspector.yaw_value = SimpleNamespace(text='')
        inspector.key_cards = {key: SimpleNamespace() for key in 'WASDQE'}
        inspector.last_keys = None
        inspector.pending = ''
        inspector.events = []
        inspector.last_focus = True
        return inspector

    def test_each_playback_button_holds_keyboard_until_stop(self):
        for start, name, other in [('start_acceptance', 'Acceptance', 'start_demo'),
                                   ('start_demo', 'Auto Demo', 'start_acceptance')]:
            with self.subTest(playback=name):
                inspector = self.inspector_with_controls()
                inspector.controls.set_speed(1.2)
                getattr(inspector, start)()
                self.assertFalse(inspector.speed_slider.enabled)
                self.assertFalse(inspector.acceptance_button.enabled)
                self.assertFalse(inspector.demo_button.enabled)
                getattr(inspector, other)()
                self.assertEqual(inspector.playback_name, name)
                self.assertEqual(inspector.next_request(), dict(command=[0., 0., 0.], reset=True, acceptance_inputs=True))
                with patch('anon_inspector_ui.has_foreground', return_value=True):
                    inspector.on_key(SimpleNamespace(type='KEY_PRESS', input='S'))
                    self.assertFalse(inspector.controls.held)
                    inspector.on_key(SimpleNamespace(type='KEY_PRESS', input='SPACE'))
                    self.assertEqual(inspector.next_request(), dict(command=[0., 0., 0.], reset=False))
                self.assertEqual(inspector.playback_status, f'{name} stopped.')
                self.assertTrue(inspector.speed_slider.enabled)
                self.assertTrue(inspector.acceptance_button.enabled)
                self.assertTrue(inspector.demo_button.enabled)
                self.assertEqual(inspector.controls.speed, 1.2)

    def test_starting_playback_discards_pending_manual_push(self):
        inspector = self.inspector_with_controls()
        inspector.push_character()
        inspector.start_demo()
        self.assertFalse(inspector.controls.consume_push())
        self.assertTrue(inspector.next_request()['acceptance_inputs'])

    def test_only_short_playback_pushes_once_during_the_final_phase(self):
        for start in ('start_demo','start_acceptance'):
            with self.subTest(start=start):
                inspector=self.inspector_with_controls()
                getattr(inspector,start)()
                requests=[]
                while inspector.playback_frames is not None:
                    requests.append(inspector.next_request())
                    request=requests[-1]
                    if request.get('manual_reset_only'):
                        inspector.show_input_controls(request['command'])
                        highlighted={key for key,card in inspector.key_cards.items()
                                     if card.style['background_color']==inspector.KEY_DOWN}
                        self.assertEqual(highlighted,set(keys_for_command(request['command'])))
                pushes=[i for i,request in enumerate(requests) if request.get('push_character')]
                self.assertEqual(pushes,[105*30] if start=='start_demo' else [])
                if start=='start_demo':
                    self.assertTrue(all(request.get('manual_reset_only') and not request['reset']
                                        for request in requests[105*30:-1]))
                with patch('anon_inspector_ui.has_foreground',return_value=True):
                    self.assertNotIn('push_character',inspector.next_request())
                self.assertIsNone(inspector.playback_frames)

    def test_interrupting_short_playback_cancels_the_final_push(self):
        for action in ('stop','reset'):
            with self.subTest(action=action):
                inspector=self.inspector_with_controls()
                inspector.start_demo()
                getattr(inspector,action)()
                with patch('anon_inspector_ui.has_foreground',return_value=True):
                    self.assertNotIn('push_character',inspector.next_request())

    def test_playback_sliders_and_keys_follow_received_input_then_restore_manual(self):
        inspector = self.inspector_with_controls()
        inspector.speed_model.set_value(.73)
        inspector.yaw_model.set_value(.31)
        inspector.solved_times = deque()
        inspector.history = []
        inspector.reset_count = 0
        inspector.last_ui_update = 0.
        inspector.last_write = 10.
        for name in ('planar_limit_label', 'yaw_limit_label',
                     'stats_label', 'hud_values', 'hud_card', 'notice_label'):
            setattr(inspector, name, SimpleNamespace(text='', visible=False))
        inspector.hud_title=SimpleNamespace(text='')
        for name in ('request_values', 'command_values', 'actual_values'):
            setattr(inspector, name, [SimpleNamespace(text='') for _ in range(3)])

        def check_frame(frame, request, keys, expected, *, automatic=True):
            state = dict(requested_command=np.array(request), command=np.zeros(3), velocity=np.zeros(3))
            with patch('anon_inspector_ui.time.monotonic', return_value=.2 * frame):
                inspector.update(state, 0, frame, acceptance_inputs=automatic)
            self.assertIn(' '.join(inspector.playback_status.split()),inspector.hud_title.text)
            self.assertNotIn('\n',inspector.hud_title.text)
            self.assertAlmostEqual(inspector.speed_model.get_value_as_float(), expected[0])
            self.assertAlmostEqual(inspector.yaw_model.get_value_as_float(), expected[1])
            self.assertEqual(inspector.speed_value.text, f'{expected[0]:.2f} m/s')
            self.assertEqual(inspector.yaw_value.text, f'{expected[1]:.2f} rad/s')
            highlighted = {key for key, card in inspector.key_cards.items()
                           if card.style['background_color'] == inspector.KEY_DOWN}
            self.assertEqual(highlighted, set(keys))

        inspector.start_demo()
        inspector.next_request()
        check_frame(1, [0., 1.2, 0.], '', (.73, .31), automatic=False)
        diagonal = .85 / math.sqrt(2)
        check_frame(2, [diagonal, -diagonal, -.6], 'WDE', (.85, .6))
        check_frame(3, [-.85, 1e-12, .6], 'SQ', (.85, .6))
        check_frame(4, [0., 0., 0.], '', (0., 0.))
        self.assertEqual((inspector.controls.speed, inspector.controls.yaw_speed), (.73, .31))
        inspector.playback_frames = iter(())
        with patch('anon_inspector_ui.has_foreground', return_value=True):
            self.assertFalse(inspector.next_request().get('acceptance_inputs'))
        self.assertTrue(inspector.speed_slider.enabled)
        self.assertEqual(inspector.playback_status, 'Auto Demo complete.')
        check_frame(5, [diagonal, -diagonal, -.6], '', (.73, .31))
        inspector.controls.event('A', 'KEY_PRESS')
        check_frame(6, [0., .73, 0.], 'A', (.73, .31), automatic=False)

    def test_default_launch_and_explicit_override(self):
        defaults = self.launch_args()
        self.assertIn('--secondary-worker', defaults)
        for option, value in (('--body-device', 'cpu'),
                              ('--physics-detail', 'sparse'), ('--frontend-task-threads', '4'), ('--render-fps', '60')):
            self.assertEqual(defaults[defaults.index(option) + 1], value)
        explicit = self.launch_args('--body-device', 'cuda:0', '--physics-detail', 'full', '--frontend-task-threads', '8', '--render-fps', '30')
        for option, value in (('--body-device', 'cuda:0'),
                              ('--physics-detail', 'full'), ('--frontend-task-threads', '8'), ('--render-fps', '30')):
            self.assertEqual(explicit[explicit.index(option) + 1], value)

    def test_modes_without_interpolation_keep_30_hz(self):
        for options in (('--inline-secondary',), ('--no-secondary-physics',), ('--record-video', 'test.mp4')):
            with self.subTest(options=options):
                arguments = self.launch_args(*options)
                self.assertEqual(arguments[arguments.index('--render-fps') + 1], '30')
                if options[0] != '--record-video':
                    self.assertNotIn('--secondary-worker', arguments)

    def test_render_activity_does_not_mask_a_stalled_solver(self):
        inspector = PolicyInspector.__new__(PolicyInspector)
        inspector.solved_times = deque([1.2, 1.3, 1.4])
        inspector.render_times = deque([1.2, 1.3, 1.4, 2.0, 2.1])
        inspector.render_frames = 5
        inspector.last_rate_update = 0.
        inspector.render_target = 60
        inspector.solve_rate_label = SimpleNamespace(text='')
        inspector.render_rate_label = SimpleNamespace(text='')
        with patch('anon_inspector_ui.time.monotonic', return_value=2.5):
            inspector.record_render()
        self.assertEqual(inspector.solve_rate_label.text, 'Solved 0 / 30 Hz')
        self.assertEqual(inspector.render_rate_label.text, 'Render 3 / 60 FPS')
        self.assertEqual(inspector.render_frames, 6)

    def test_front_back_and_side_rotate_with_the_character(self):
        inspector = PolicyInspector.__new__(PolicyInspector)
        for view, expected_offset in ((1, (0, 3.6, .2)), (2, (-4.6, 0, .7)), (3, (4.6, 0, .7))):
            with self.subTest(view=view):
                inspector.controls = SimpleNamespace(view=view)
                np.testing.assert_allclose(inspector.camera_offset(math.pi / 2), expected_offset, atol=1e-12)

    def test_angle_keeps_its_original_world_offset_and_target(self):
        inspector = PolicyInspector.__new__(PolicyInspector)
        inspector.controls = SimpleNamespace(view=0)
        inspector.reset_camera()
        inspector.camera_target(np.zeros(3), 0., 0.)
        position = np.array([1., 2., .88])
        center, offset = inspector.camera_target(position, math.pi / 2, 1 / 60)
        np.testing.assert_array_equal(center, position)
        self.assertEqual(offset, (2.3, -4, 1.15))

    def test_tracking_damps_small_pose_changes_and_snaps_on_reset(self):
        inspector = PolicyInspector.__new__(PolicyInspector)
        inspector.controls = SimpleNamespace(view=2)
        inspector.reset_camera()
        inspector.camera_target(np.zeros(3), 0., 0.)
        position = np.array([1., 0., 0.])
        center, offset = inspector.camera_target(position, math.pi / 2, 1 / 60)
        self.assertLess(center[0], .25)
        self.assertGreater(offset[1], 4.)
        center, offset = inspector.camera_target(position, math.pi / 2, 3.)
        np.testing.assert_allclose(center, position, atol=.001)
        np.testing.assert_allclose(offset, (-4.6, 0, .7), atol=.002)
        inspector.reset_camera()
        center, offset = inspector.camera_target(np.zeros(3), 0., 3.01)
        np.testing.assert_array_equal(center, np.zeros(3))
        self.assertEqual(offset, (0., 4.6, .7))


if __name__ == '__main__':
    unittest.main()
