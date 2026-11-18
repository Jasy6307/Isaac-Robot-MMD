"""CPU tests: sustained tracking failures must not pass just by reaching the end."""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.motion_quality import (
    MotionQualityMonitor,tracking_tail_cost,up_vector_xyzw,validation_passed)


class QualityTests(unittest.TestCase):
    def metrics(self, leg=.1, height=.02, tilt=.1, normal=1.):
        return {k:torch.tensor([v]) for k,v in zip(
            ("leg_rms_rad","height_error_m","tilt_error_rad","normal_weight"),(leg,height,tilt,normal))}

    def feed(self, monitor, values, steps, start=0., active=True):
        for n in range(steps):
            monitor.update(values,torch.tensor([active]),torch.tensor([start+n/30.]))

    def test_recovered_severe_stumble_still_fails_and_keeps_first_onset(self):
        monitor=MotionQualityMonitor(1,"cpu",1/30.)
        self.feed(monitor,self.metrics(),30)
        self.feed(monitor,self.metrics(leg=.6,height=.4,tilt=1.),30,start=1.)
        self.feed(monitor,self.metrics(),30,start=2.)
        report=monitor.summary(0)
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["first_issue_s"],1.,places=5)
        self.assertAlmostEqual(report["longest_excess_s"]["height"],1.)

    def test_single_control_spike_is_allowed(self):
        monitor=MotionQualityMonitor(1,"cpu",1/30.)
        self.feed(monitor,self.metrics(leg=.8,height=.3),1)
        self.feed(monitor,self.metrics(),30,start=1/30.)
        self.assertTrue(monitor.summary(0)["passed"])

    def test_jump_leg_deviation_is_allowed_without_forcing_contact(self):
        monitor=MotionQualityMonitor(1,"cpu",1/30.)
        self.feed(monitor,self.metrics(leg=.8,normal=0.),30)
        self.assertTrue(monitor.summary(0)["passed"])
        self.assertEqual(float(tracking_tail_cost(torch.tensor([.8]),torch.tensor([.3]),
                                                 torch.tensor([.9]),torch.tensor([0.]))),0.)

    def test_alternating_failure_types_cannot_evade_persistence_gate(self):
        monitor=MotionQualityMonitor(1,"cpu",1/30.)
        for n in range(30):
            self.feed(monitor,self.metrics(leg=.6 if n%2 else .1,height=.3 if n%2==0 else .02),1,start=n/30.)
        self.assertFalse(monitor.summary(0)["passed"])

    def test_inactive_environments_and_missing_samples_are_not_successes(self):
        monitor=MotionQualityMonitor(1,"cpu",1/30.)
        self.feed(monitor,self.metrics(),30,active=False)
        self.assertFalse(monitor.summary(0)["passed"])
        self.assertEqual(monitor.summary(0)["samples"],0)

    def test_nonfinite_sample_is_failure(self):
        monitor=MotionQualityMonitor(1,"cpu",1/30.)
        self.feed(monitor,self.metrics(leg=float("nan")),1)
        self.assertFalse(monitor.summary(0)["passed"])

    def test_tail_cost_keeps_discriminating_after_exponential_reward_saturates(self):
        legs=torch.tensor([.1,.3,.6,1.])
        cost=tracking_tail_cost(legs,torch.zeros(4),torch.zeros(4),torch.ones(4))
        self.assertEqual(float(cost[0]),0.)
        self.assertTrue(bool((cost[1:]>cost[:-1]).all()))

    def test_tilt_is_reference_relative_and_yaw_alone_does_not_change_up(self):
        # xyzw: a deep reference bend can be valid; compare up directions.
        pitch=math.radians(70)
        q=torch.tensor([[0.,math.sin(pitch/2),0.,math.cos(pitch/2)]])
        up=up_vector_xyzw(q)
        self.assertAlmostEqual(float((up*up).sum()),1.,places=5)
        yaw=up_vector_xyzw(torch.tensor([[0.,0.,1.,0.]]))
        torch.testing.assert_close(yaw,torch.tensor([[0.,0.,1.]]))

    def test_completion_alone_or_unknown_quality_does_not_pass(self):
        completed=dict(completed=1,tested=1)
        self.assertFalse(validation_passed([completed]))
        self.assertFalse(validation_passed([dict(completed,quality_passed=False)]))
        self.assertTrue(validation_passed([dict(completed,quality_passed=True)]))
        self.assertFalse(validation_passed([]))


if __name__=="__main__":unittest.main()
