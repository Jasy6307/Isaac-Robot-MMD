"""Monitoring correctness and isolation from articulation/controller buffers."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from io import BytesIO
import numpy as np
import torch

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import source_layout
from anon_joint_telemetry import JointTelemetry
from anon_joint_monitor import JointMonitorState,joint_label,joint_group,joint_rows,joint_part_rows


def environment():
    target=torch.tensor([[.2,-.3,.4]])
    target_velocity=torch.tensor([[.1,.2,.3]])
    return SimpleNamespace(names=['right_knee_pitch_joint','waist_yaw_joint','left_hip_roll_joint'],
        limits=torch.tensor([[-1.,1.],[-2.,2.],[-.5,.5]]),num_actions=3,num_dofs=3,
        q=torch.tensor([[.1,-.2,.49]]),qd=torch.tensor([[1.,2.,3.]]),
        actions=torch.tensor([[.5,-.5,.25]]),physics_time=1.,
        robot=SimpleNamespace(get_dof_position_targets=lambda:target,
                              get_dof_velocity_targets=lambda:target_velocity))


class JointMonitorTests(unittest.TestCase):
    def test_batch_snapshot_reads_only_the_selected_actor(self):
        env=environment()
        env.q=torch.tensor([[.1,.2,.3],[.4,.5,.6]])
        env.qd=env.q+1;env.actions=env.q+2
        target=env.q+3;velocity=env.q+4
        env.robot.get_dof_position_targets=lambda:target
        env.robot.get_dof_velocity_targets=lambda:velocity
        state=JointTelemetry(env).read(env,index=1)
        np.testing.assert_allclose(state['joint_positions'],[.4,.5,.6])
        np.testing.assert_allclose(state['joint_position_targets'],[3.4,3.5,3.6])
        np.testing.assert_allclose(state['joint_actions'],[2.4,2.5,2.6])
        state['joint_positions'][:]=99
        np.testing.assert_allclose(env.q.numpy(),[[.1,.2,.3],[.4,.5,.6]])

    def test_readback_uses_final_drive_targets_and_never_aliases_controller(self):
        env=environment();reader=JointTelemetry(env);state=reader.read(env)
        np.testing.assert_array_equal(state['joint_position_targets'],[np.float32(.2),np.float32(-.3),np.float32(.4)])
        original_q=env.q.clone();original_actions=env.actions.clone()
        state['joint_positions'][:]=999;state['joint_actions'][:]=999
        self.assertTrue(torch.equal(env.q,original_q));self.assertTrue(torch.equal(env.actions,original_actions))
        env.robot.get_dof_position_targets().fill_(.7)
        np.testing.assert_array_equal(reader.read(env)['joint_position_targets'],np.full(3,np.float32(.7)))

    def test_partial_policy_actions_follow_explicit_dof_order(self):
        env=environment();env.num_actions=2;env.leg_ids=torch.tensor([2,0]);env.actions=torch.tensor([[.7,-.2]])
        state=JointTelemetry(env).read(env)
        np.testing.assert_array_equal(state['joint_action_mask'],[True,False,True])
        np.testing.assert_allclose(state['joint_actions'],[-.2,0,.7])

    def test_radian_conversion_error_and_nearest_limit_status(self):
        state=JointTelemetry(environment()).read(environment())
        monitor=JointMonitorState();self.assertTrue(monitor.ingest(state))
        np.testing.assert_allclose(monitor.latest['q'],np.rad2deg([.1,-.2,.49]),atol=1e-5)
        np.testing.assert_allclose(monitor.latest['error'],np.rad2deg([.1,-.1,-.09]),atol=1e-5)
        np.testing.assert_array_equal(monitor.latest['near_limit'],[False,False,True])
        self.assertIn('1/3',monitor.summary())

    def test_history_uses_simulation_time_clears_on_reset_and_checkpoint_change(self):
        env=environment();reader=JointTelemetry(env);monitor=JointMonitorState()
        for i in range(241):
            env.physics_time=i/30
            state=reader.read(env);state['checkpoint']='a';monitor.ingest(state)
        self.assertLessEqual(len(monitor.history),151)
        self.assertLessEqual(monitor.series(0)[-1],5.)
        monitor.ingest(state);self.assertLessEqual(len(monitor.history),151)
        state['joint_state_reset']=True;monitor.ingest(state);self.assertEqual(len(monitor.history),1)
        state['joint_state_reset']=False;state['joint_sample_time']+=1;monitor.ingest(state)
        self.assertEqual(len(monitor.history),2)
        state['checkpoint']='b';monitor.ingest(state);self.assertEqual(len(monitor.history),1)

    def test_bad_or_missing_samples_do_not_show_stale_measurements(self):
        env=environment();state=JointTelemetry(env).read(env);monitor=JointMonitorState()
        monitor.ingest(state);monitor.ingest({});self.assertIsNotNone(monitor.latest)
        state['joint_positions'][0]=float('nan')
        self.assertFalse(monitor.ingest(state));self.assertIsNone(monitor.latest)
        self.assertFalse(monitor.history)
        state['joint_positions']=np.zeros(2)
        self.assertFalse(monitor.ingest(state))
        self.assertFalse(monitor.ingest(dict(joint_state_valid=False,joint_state_error='unsupported')))
        self.assertEqual(monitor.error,'unsupported')

    def test_transport_payload_needs_no_pickle_or_radian_guessing(self):
        env=environment();state=JointTelemetry(env).read(env)
        buf=BytesIO();np.savez(buf,**state);buf.seek(0)
        with np.load(buf,allow_pickle=False) as file:received={k:file[k] for k in file.files}
        monitor=JointMonitorState();self.assertTrue(monitor.ingest(received))
        self.assertEqual(monitor.names,tuple(env.names))

    def test_names_and_groups_do_not_assume_alphabetical_physx_order(self):
        self.assertEqual(joint_label('left_knee_pitch_joint'),'L knee flex')
        self.assertEqual(joint_group('right_wrist_deviation_joint'),'Right arm')
        self.assertEqual(joint_group('left_ankle_roll_joint'),'Left leg')
        self.assertEqual(joint_group('neck_yaw_joint'),'Torso / Neck')
        self.assertEqual(joint_label('unmapped_joint'),'unmapped')

    def test_motion_measures_actual_speed_not_actions_or_tracking_error(self):
        env=environment();env.qd.zero_()
        state=JointTelemetry(env).read(env);monitor=JointMonitorState();monitor.ingest(state)
        np.testing.assert_array_equal(monitor.motion()['speed'],np.zeros(3))
        np.testing.assert_array_equal(monitor.motion()['level'],np.zeros(3))
        np.testing.assert_array_equal(monitor.motion()['fill'],np.zeros(3))
        self.assertIn('Moving 0/3',monitor.motion_summary())

    def test_paired_rows_preserve_all_axes_and_missing_counterparts(self):
        names=['right_knee_pitch_joint','neck_yaw_joint','left_knee_pitch_joint',
               'right_wrist_pitch_joint','left_hip_pitch_joint','custom_joint']
        grouped=dict(joint_rows(names))
        self.assertEqual(grouped['Legs'][0],['knee flex','left_knee_pitch_joint','right_knee_pitch_joint'])
        self.assertEqual(grouped['Legs'][1],['hip pitch','left_hip_pitch_joint',None])
        self.assertEqual(grouped['Arms'][0],['wrist pitch',None,'right_wrist_pitch_joint'])
        recovered=[name for rows in grouped.values() for _,left,right in rows for name in (left,right) if name]
        self.assertCountEqual(recovered,names)

    def test_anatomical_rows_combine_rotation_axes_without_losing_sides(self):
        names=['right_hip_yaw_joint','left_hip_pitch_joint','left_hip_roll_joint',
               'right_hip_pitch_joint','waist_pitch_joint','waist_yaw_joint',
               'left_wrist_pitch_joint','left_wrist_deviation_joint','custom_joint']
        grouped=dict(joint_part_rows(names))
        self.assertEqual(grouped['Legs'],[('Hip',('left_hip_roll_joint','left_hip_pitch_joint'),
                                         ('right_hip_pitch_joint','right_hip_yaw_joint'))])
        self.assertEqual(grouped['Torso / Neck'],[('Waist',('waist_pitch_joint','waist_yaw_joint'),())])
        self.assertEqual(grouped['Arms'][0][0],'Wrist')
        recovered=[n for rows in grouped.values() for _,left,right in rows for side in (left,right) for n in side]
        self.assertCountEqual(recovered,names)

    def test_group_activity_is_fastest_axis_and_preserves_individual_values(self):
        env=environment()
        env.names=['left_shoulder_roll_joint','left_shoulder_pitch_joint','left_shoulder_yaw_joint']
        env.qd[:]=torch.tensor(np.deg2rad([[20.,-60.,10.]]),dtype=torch.float32)
        monitor=JointMonitorState();monitor.ingest(JointTelemetry(env).read(env))
        part=monitor.part_motion(tuple(env.names))
        self.assertEqual(part['dominant'],'left_shoulder_pitch_joint')
        self.assertAlmostEqual(part['speed'],60.,places=4)
        self.assertAlmostEqual(part['fill'],1./3.,places=5)
        np.testing.assert_allclose(monitor.motion()['speed'],[20.,60.,10.],atol=1e-5)
        self.assertIn('Moving 1/1 joints (3 axes)',monitor.motion_summary())

    def test_motion_windows_use_absolute_velocity_and_actual_angle_range(self):
        env=environment();reader=JointTelemetry(env);monitor=JointMonitorState()
        for time,angle,velocity in ((0.,-170.,1000.),(.5,-10.,800.),(1.,20.,60.),(1.2,-20.,-120.),(1.3,5.,60.)):
            env.physics_time=time;env.q.fill_(np.deg2rad(angle));env.qd.fill_(np.deg2rad(velocity))
            monitor.ingest(reader.read(env))
        motion=monitor.motion()
        np.testing.assert_allclose(motion['speed'],80.,atol=1e-5)
        np.testing.assert_allclose(motion['swing'],40.,atol=1e-5)
        np.testing.assert_allclose(motion['fill'],80./180.,atol=1e-6)
        np.testing.assert_array_equal(motion['level'],np.full(3,2))
        env.physics_time=2.;env.qd.fill_(np.deg2rad(360.))
        monitor.ingest(reader.read(env,reset=True))
        np.testing.assert_allclose(monitor.motion()['speed'],360.,atol=1e-4)
        np.testing.assert_array_equal(monitor.motion()['fill'],np.ones(3))
        np.testing.assert_array_equal(monitor.motion()['swing'],np.zeros(3))
        np.testing.assert_array_equal(monitor.motion()['level'],np.full(3,3))


if __name__=='__main__':unittest.main()
