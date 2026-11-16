"""Steering measurements distinguish commanded turns from natural torso sway."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
import unittest
import numpy as np
from evaluate_unified_walk_run import yaw_tracking_error,linear_tracking_error,suite

class AcceptanceMetricTests(unittest.TestCase):
    def test_mean_velocity_allows_gait_oscillation_but_rejects_tracking_bias(self):
        t=np.arange(600)/30
        error=np.stack([.5*np.sin(2*np.pi*2*t)+.12,np.zeros_like(t)],-1)
        self.assertLess(linear_tracking_error(error,True,1/30),.13)
        error[:,0]+=.25
        self.assertGreater(linear_tracking_error(error,True,1/30),.36)

    def test_gait_sway_does_not_become_commanded_turn_error(self):
        errors=np.sin(np.arange(300)*2*np.pi/30)*.7
        self.assertLess(yaw_tracking_error(errors,np.zeros(300)),1e-6)
        self.assertGreater(yaw_tracking_error(errors+.4,np.full(300,.6)),.39)

    def test_changing_turn_targets_cannot_cancel_opposite_errors(self):
        targets=np.sin(np.arange(300)*2*np.pi/300)*.6
        self.assertGreater(yaw_tracking_error(-targets,targets),.3)

    def test_full_suite_includes_abrupt_forward_run_stop(self):
        cases=suite();self.assertEqual(sum(s for _,s,_ in cases),325)
        self.assertTrue(any(cases[i][0]=='run_forward' and cases[i+1][0]=='stop' for i in range(len(cases)-1)))

if __name__=='__main__':unittest.main()
