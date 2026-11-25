"""Role switches must not carry held keys or resets to another actor."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase,main
from unittest.mock import Mock,patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
from anon_crowd_inspector import CrowdInspector
from anon_inspector_controls import InspectorCommands


class CrowdInspectorTests(TestCase):
    def inspector(self):
        value=CrowdInspector.__new__(CrowdInspector)
        value.actor_count=4;value.selected_actor=0
        value.controls=InspectorCommands(dict(max_speed=1.2,max_yaw=.6))
        value.background_paused=False;value.playback_frames=None;value.history=['old']
        value.events=[];value.actor_buttons={i:SimpleNamespace(style={}) for i in range(4)}
        value.joint_monitor=SimpleNamespace(ingest=Mock(),refresh=Mock())
        for name in ('request_values','command_values','actual_values'):
            setattr(value,name,[SimpleNamespace(text='old') for _ in range(3)])
        value.show_input_controls=Mock()
        return value

    def test_switch_releases_input_and_discards_old_actor_pending_reset(self):
        value=self.inspector();value.controls.event('W','KEY_PRESS');value.controls.request_reset()
        value.controls.event('D','KEY_PRESS');value.select_actor(2)
        self.assertEqual(value.selected_actor,2)
        self.assertEqual(value.controls.command(),[0.,0.,0.])
        self.assertFalse(value.controls.consume_reset())
        self.assertFalse(value.background_paused)
        self.assertFalse(value.history)
        value.joint_monitor.ingest.assert_called_once()

    def test_numeric_actor_shortcut_requires_foreground(self):
        value=self.inspector();value.select_actor=Mock()
        event=SimpleNamespace(type='KEY_PRESS',input='KEY_3')
        with patch('anon_crowd_inspector.has_foreground',return_value=False):value.on_key(event)
        value.select_actor.assert_not_called()
        with patch('anon_crowd_inspector.has_foreground',return_value=True):value.on_key(event)
        value.select_actor.assert_called_once_with(2)

    def test_switch_discards_push_queued_for_previous_actor(self):
        value=self.inspector();value.controls.request_push()
        value.select_actor(2)
        self.assertFalse(value.controls.consume_push())

    def test_space_and_r_leave_background_running(self):
        value=self.inspector();value.controls.event('W','KEY_PRESS')
        with patch('anon_inspector_ui.has_foreground',return_value=True):
            value.on_key(SimpleNamespace(type='KEY_PRESS',input='SPACE'))
            self.assertEqual(value.controls.command(),[0.,0.,0.])
            value.on_key(SimpleNamespace(type='KEY_PRESS',input='R'))
        self.assertTrue(value.controls.consume_reset())
        self.assertFalse(value.background_paused)

    def test_follow_camera_uses_selected_actor(self):
        value=self.inspector();value.selected_actor=2;value.reset_camera()
        poses=np.zeros((4,1,7));poses[:,:,6]=1.;poses[2,0,:3]=[7.,8.,.9]
        center,_=value.follow_camera(poses,0,1.)
        np.testing.assert_array_equal(center,[7.,8.,.9])

    def test_camera_button_and_v_resume_following(self):
        value=self.inspector();value.follow_model=SimpleNamespace(set_value=Mock())
        value.select_camera(3)
        self.assertEqual(value.controls.view,3)
        value.follow_model.set_value.assert_called_once_with(True)
        value.follow_model.set_value.reset_mock()
        with patch('anon_inspector_ui.has_foreground',return_value=True),patch('anon_crowd_inspector.has_foreground',return_value=True):
            value.on_key(SimpleNamespace(type='KEY_PRESS',input='V'))
        self.assertEqual(value.controls.view,0)
        value.follow_model.set_value.assert_called_once_with(True)


if __name__=='__main__':main()
