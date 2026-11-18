"""Intentional glides/pivots, extra motion, jump protection and feedback bounds."""
import json
import sys
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from source.train_workflow.utils.motion_feedback import assess_motion,MotionFeedback,PROTOCOL
from source.train_workflow.utils.auto_jump_training import AutoJumpController
from source.train_workflow.utils.quality_selection import QualitySelection,generic_profile
from types import SimpleNamespace as NS
import torch


class MotionTests(unittest.TestCase):
    def fixture(self):
        t=np.arange(121)/30.;n=len(t)
        return dict(seconds=t,joint_error=np.zeros((n,2)),command_residual=np.zeros((n,2)),
            actual_feet=np.zeros((n,2,3)),reference_feet=np.zeros((n,2,3)),reference_yaw=np.zeros((n,2)),
            expected=np.ones((n,2)),contacts=np.ones((n,2)),joint_sides=[0,1],schedules=[],duration=4.,hz=30.)

    def test_matching_intentional_glide_is_not_a_slip_candidate(self):
        x=self.fixture();x["actual_feet"][:,:,0]=x["seconds"][:,None]*.4;x["reference_feet"]=x["actual_feet"].copy()
        r=assess_motion(**x)
        self.assertTrue(r["complete"]);self.assertEqual(r["score"],0.);self.assertFalse(r["candidates"])

    def test_pivot_radius_allowance_retains_allowed_foot_origin_motion(self):
        x=self.fixture();x["reference_yaw"][:]=x["seconds"][:,None]*2.
        x["actual_feet"][:,:,0]=x["seconds"][:,None]*.3
        self.assertEqual(assess_motion(**x)["score"],0.)

    def test_unexpected_slide_with_static_reference_is_reported(self):
        x=self.fixture();x["actual_feet"][:,:,0]=x["seconds"][:,None]*.5
        r=assess_motion(**x)
        self.assertGreater(r["score"],1.)
        self.assertIn("excess_foot_origin_motion",r["candidates"][0]["reasons"])

    def test_extra_chatter_is_detected_without_any_13_second_label(self):
        x=self.fixture();x["joint_error"][:,0]=.1*np.sin(2*np.pi*6*x["seconds"])
        r=assess_motion(**x);self.assertTrue(r["candidates"])
        x["seconds"]+=20;x["duration"]=24
        shifted=assess_motion(**x)
        self.assertTrue(all(c["start_s"]>=20 for c in shifted["candidates"]))
        self.assertAlmostEqual(r["score"],shifted["score"],places=5)

    def test_large_command_without_actual_extra_motion_is_only_diagnostic(self):
        x=self.fixture();x["command_residual"][:,0]=np.sin(2*np.pi*6*x["seconds"])
        r=assess_motion(**x)
        self.assertFalse(r["candidates"]);self.assertGreater(r["windows"][0]["command_chatter_rad"],.1)

    def test_unknown_support_swing_and_jump_are_not_forced_into_locking(self):
        for kind in ("unknown","swing","jump"):
            x=self.fixture();x["joint_error"][:]=np.sin(2*np.pi*6*x["seconds"])[:,None]
            if kind=="unknown":x["expected"][:]=0
            if kind=="swing":x["contacts"][:]=0;x["expected"][:]=0
            if kind=="jump":x["schedules"]=[((0.,1.),(1.,2.),(2.,3.),(3.,4.))]
            self.assertFalse(assess_motion(**x)["candidates"])

    def test_losing_expected_contact_cannot_game_the_chatter_score(self):
        x=self.fixture();x["contacts"][:]=0
        r=assess_motion(**x)
        self.assertGreater(r["score"],1.)
        self.assertIn("intermittent_reference_support",r["candidates"][0]["reasons"])

    def test_intermittent_reference_support_does_not_hide_a_grounded_turn(self):
        x=self.fixture();x["expected"][:]=0;x["expected"][::2]=1
        x["joint_error"][:,0]=.1*np.sin(2*np.pi*6*x["seconds"])
        r=assess_motion(**x)
        self.assertTrue(r["candidates"])
        self.assertIn("extra_joint_oscillation",r["candidates"][0]["reasons"])

    def test_corrupt_clock_nonfinite_and_missing_full_coverage_are_explicit(self):
        x=self.fixture();x["seconds"][50]+=.01
        self.assertEqual(assess_motion(**x)["error"],"motion_clock_gap")
        x=self.fixture();x["joint_error"][20,0]=np.nan
        self.assertFalse(assess_motion(**x)["complete"])
        x=self.fixture();x["duration"]=10.
        self.assertFalse(assess_motion(**x)["complete"])


def record(seed,score=1.,motion=1.,candidates=None):
    return dict(seed=seed,kind="full",completed=1,tested=1,quality_passed=True,acceptance_passed=True,
        acceptance_score=score,motion_feedback_required=True,results=[dict(motion_diagnostics=dict(
            protocol=PROTOCOL,complete=True,contact_observed=True,score=motion,candidates=candidates or []))])


class FeedbackTests(unittest.TestCase):
    def test_one_seed_does_not_boost_sampling_and_two_seeds_do(self):
        candidate=dict(start_s=12.,end_s=14.,severity=2.,reasons=["extra_joint_oscillation"])
        f=MotionFeedback(84)
        f.update([record(42,candidates=[candidate]),record(43),record(44)])
        self.assertEqual(f.priority.sum(),0.)
        f.update([record(42,candidates=[candidate]),record(43,candidates=[candidate]),record(44)])
        self.assertTrue(np.all(f.priority[12:14]>0))
        prior=f.priority.copy();f.update([record(i) for i in (42,43,44)])
        self.assertTrue(np.all(f.priority<=prior))

    def test_priority_size_and_sampling_share_are_bounded_and_resume_preserved(self):
        candidates=[dict(start_s=float(i),end_s=float(i+1),severity=5.) for i in range(100)]
        f=MotionFeedback(100);f.update([record(i,candidates=candidates) for i in (42,43,44)])
        self.assertLessEqual(np.count_nonzero(f.priority),20)
        restored=MotionFeedback(100,json.loads(json.dumps(f.state_dict())))
        self.assertTrue(np.array_equal(restored.priority,f.priority))
        env=NS(device="cpu",cfg=NS(auto_motion_feedback=True))
        controller=AutoJumpController(env,dict(duration_s=100.,events=[]),ROOT/"output")
        controller.motion_feedback=restored
        p=controller.probabilities().numpy()
        self.assertAlmostEqual(float(p.sum()),1.,places=6)
        self.assertTrue(np.all(p>=.4/100.-1.e-7))
        self.assertAlmostEqual(float(p.max()),.9/100+.1/20,places=6)
        f.update([record(i,candidates=[dict(start_s=0.,end_s=20.,severity=5.)]) for i in (42,43,44)])
        self.assertLessEqual(np.count_nonzero(f.priority),20)

    def test_bad_protocol_duplicate_seed_or_missing_seed_are_rejected(self):
        with self.assertRaises(ValueError):MotionFeedback(3,dict(protocol="other",priority=[0,0,0]))
        f=MotionFeedback(3)
        with self.assertRaises(ValueError):f.update([record(42)]*3)
        with self.assertRaises(ValueError):f.update([record(42)])
        state=f.state_dict();state["criteria"]["chatter_rad"]*=2
        with self.assertRaisesRegex(ValueError,"criteria"):MotionFeedback(3,state)

    def test_missing_diagnostics_do_not_claim_recovery_or_erase_previous_priority(self):
        f=MotionFeedback(84);c=dict(start_s=12.,end_s=14.,severity=2.)
        f.update([record(i,candidates=[c]) for i in (42,43,44)])
        old=f.priority.copy();records=[record(i) for i in (42,43,44)]
        records[1]["results"][0]["motion_diagnostics"]=dict(error="unavailable")
        f.update(records);self.assertTrue(np.array_equal(old,f.priority))
        self.assertEqual(f.latest["status"],"incomplete_diagnostics")

    def test_unobserved_support_does_not_count_as_improvement(self):
        f=MotionFeedback(84);c=dict(start_s=12.,end_s=14.,severity=2.)
        f.update([record(i,candidates=[c]) for i in (42,43,44)])
        old=f.priority.copy();records=[record(i) for i in (42,43,44)]
        for r in records:r["results"][0]["motion_diagnostics"]["windows"]=[]
        f.update(records);self.assertTrue(np.array_equal(old,f.priority))


class SelectionTests(unittest.TestCase):
    def make(self):return QualitySelection(generic_profile(dict(source_sha256="test",events=[])))
    def records(self,score,motion):return [record(i,score,motion) for i in (42,43,44)]

    def test_local_improvement_can_update_an_equivalent_global_model(self):
        s=self.make();seeds=[42,43,44]
        self.assertEqual(s.confirm(400,self.records(1.,2.),seeds,"a.pt"),(True,True))
        self.assertEqual(s.confirm(800,self.records(1.01,1.5),seeds,"b.pt"),(True,True))
        self.assertEqual(s.best["checkpoint"],"b.pt")

    def test_local_gain_cannot_buy_global_regression_or_missing_motion_evidence(self):
        s=self.make();seeds=[42,43,44];s.confirm(400,self.records(1.,2.),seeds,"a.pt")
        self.assertEqual(s.confirm(800,self.records(1.1,1.),seeds,"b.pt"),(True,False))
        records=self.records(.8,1.);records[1]["results"][0].pop("motion_diagnostics")
        self.assertEqual(s.confirm(1200,records,seeds,"c.pt"),(False,False))
        self.assertEqual(s.best["checkpoint"],"a.pt")

    def test_global_improvement_cannot_hide_large_motion_regression(self):
        s=self.make();seeds=[42,43,44];s.confirm(400,self.records(1.,1.),seeds,"a.pt")
        self.assertEqual(s.confirm(800,self.records(.8,2.),seeds,"b.pt"),(True,False))

    def test_unobserved_contact_or_audit_data_cannot_qualify_a_model(self):
        for change in (dict(contact_observed=False),dict(audit_only=True)):
            s=self.make();records=self.records(1.,1.)
            records[0]["results"][0]["motion_diagnostics"].update(change)
            self.assertEqual(s.confirm(400,records,[42,43,44],"bad.pt"),(False,False))


if __name__=="__main__":unittest.main()
