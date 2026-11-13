"""Regression checks for local failures, semantic gates and best-model export."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from source.train_workflow.utils.quality_selection import (
    TRACE_FIELDS,INDEX,generic_profile,assess_trace,calibrate_profile,QualitySelection,resolve_profile,continuous_coverage)
from source.train_workflow.utils.auto_jump_training import _train_auto_impl,train_auto,first_probe_done


def record(passed=True,seed=42,kind="full",score=.5):
    return dict(kind=kind,seed=seed,completed=1,tested=1,quality_passed=True,
                acceptance_passed=passed,acceptance_score=score if passed else None,
                results=[dict(acceptance=dict(windows=[dict(start_s=0.,end_s=1.)]))])


class GateTests(unittest.TestCase):
    def setUp(self):
        self.report=dict(source_sha256="test",duration_s=3.,control_hz=30.,events=[dict(id=0,flight_reward=True,
            phases=[[.6,.8],[.8,1.],[1.,1.2],[1.2,1.5]],evidence=dict(closing_mps=1.))])
        self.profile=generic_profile(self.report)
        self.trace=np.zeros((90,len(TRACE_FIELDS)))
        t=np.arange(90)/30.;self.trace[:,0]=t
        for key,value in dict(leg=.1,height=.02,tilt=.1,normal=1.,required=1.,contact_left=1.,contact_right=1.,feet_distance=.1).items():
            self.trace[:,INDEX[key]]=value
        event=(t>=.6)&(t<1.5)
        self.trace[event,INDEX["required"]]=0;self.trace[event,INDEX["normal"]]=0
        air=(t>=1.)&(t<1.2)
        self.trace[air,INDEX["contact_left"]]=0;self.trace[air,INDEX["contact_right"]]=0
        self.trace[air & (t>1.),INDEX["qualified"]]=1

    def test_good_jump_and_transition_contacts_pass(self):
        r=assess_trace(self.trace,self.report,self.profile,full=True)
        self.assertTrue(r["passed"],r["failures"])
        self.assertAlmostEqual(r["events"][0]["air_s"],.2)

    def test_all_dances_freeze_the_same_default_contact_tolerance(self):
        for name in ("IRIS_OUT", "ji_le_jing_tu", "another_dance"):
            profile=generic_profile(dict(self.report,source_sha256=name))
            self.assertEqual(profile["swing_limits"],dict(early_contact_s=.4,early_contact_fraction=.9))
            profile["swing_limits"]["early_contact_s"]=.1
        self.assertEqual(generic_profile(self.report)["swing_limits"]["early_contact_s"],.4)

    def test_swing_collapse_fails_even_when_overall_leg_tracking_passes(self):
        self.trace[60:80, INDEX["swing_required"]] = 1
        self.trace[60:80, INDEX["swing_error"]] = .15
        r = assess_trace(self.trace, self.report, self.profile, full=True)
        self.assertIn("swing_height_failure_s", [f["check"] for f in r["failures"]])

    def test_early_swing_contact_has_local_gate_and_transition_tolerance(self):
        self.trace[60:80, INDEX["swing_required"]] = 1
        self.trace[60:63, INDEX["swing_early_contact"]] = 1
        self.assertTrue(assess_trace(self.trace, self.report, self.profile, full=True)["passed"])
        self.trace[60:74, INDEX["swing_early_contact"]] = 1
        r = assess_trace(self.trace, self.report, self.profile, full=True)
        self.assertIn("swing_early_contact_s", [f["check"] for f in r["failures"]])

    def test_repeated_short_touchdowns_cannot_evade_swing_gate(self):
        self.trace[60:90, INDEX["swing_required"]] = 1
        self.trace[60:90, INDEX["swing_early_contact"]] = 1
        self.trace[[69,79], INDEX["swing_early_contact"]] = 0
        r = assess_trace(self.trace, self.report, self.profile, full=True)
        self.assertIn("swing_early_contact_fraction", [f["check"] for f in r["failures"]])

    def test_reference_specific_visual_contact_tolerance_keeps_other_gates(self):
        profile=copy.deepcopy(self.profile)
        strict=copy.deepcopy(self.profile)
        strict["swing_limits"]=dict(early_contact_s=.15,early_contact_fraction=.25)
        self.trace[60:90,INDEX["swing_required"]]=1
        self.trace[60:70,INDEX["swing_early_contact"]]=1
        self.trace[72:78,INDEX["swing_early_contact"]]=1
        self.assertFalse(assess_trace(self.trace,self.report,strict,full=True)["passed"])
        self.assertTrue(assess_trace(self.trace,self.report,profile,full=True)["passed"])
        self.trace[60:90,INDEX["swing_early_contact"]]=1
        self.assertFalse(assess_trace(self.trace,self.report,profile,full=True)["passed"])
        self.trace[:,INDEX["swing_early_contact"]]=0
        self.trace[60:80,INDEX["swing_error"]]=.15
        self.assertFalse(assess_trace(self.trace,self.report,profile,full=True)["passed"])
        self.trace[:,INDEX["swing_error"]]=0
        self.trace[:,INDEX["tilt"]]=.8
        self.assertFalse(assess_trace(self.trace,self.report,profile,full=True)["passed"])

    def test_jump_leg_error_is_not_hidden_by_contact_success(self):
        self.trace[18:45,INDEX["leg"]]=.8
        r=assess_trace(self.trace,self.report,self.profile,full=True)
        self.assertFalse(r["passed"])
        self.assertTrue(any(f["check"]=="jump_leg_p95" for f in r["failures"]))

    def test_tail_only_full_trace_is_assessed_from_tail_start(self):
        trace=self.trace[9:45].copy()
        report=dict(self.report, tail_only=True, tail_start_seconds=.3, episode_duration_s=1.5)
        self.assertTrue(assess_trace(trace,report,generic_profile(report),full=True)["passed"])

    def test_end_hold_requires_full_trace_and_checks_tail_quality(self):
        report = dict(self.report, end_hold_seconds=2., episode_duration_s=5.)
        profile = generic_profile(report)
        short = assess_trace(self.trace, report, profile, full=True)
        self.assertIn("incomplete_full_trace", [f["check"] for f in short["failures"]])
        tail = np.repeat(self.trace[-1:], 60, axis=0)
        tail[:,0] = np.arange(90,150)/30.
        extended = np.concatenate((self.trace,tail))
        self.assertTrue(assess_trace(extended,report,profile,full=True)["passed"])
        extended[100:,INDEX["tilt"]] = 1.
        self.assertFalse(assess_trace(extended,report,profile,full=True)["passed"])

    def test_recovered_local_stumble_cannot_hide_in_a_long_easy_motion(self):
        self.trace[60:70,INDEX["height"]]=.5
        r=assess_trace(self.trace,self.report,self.profile,full=True)
        self.assertFalse(r["passed"])
        self.assertTrue(any(f["check"]=="height_window" for f in r["failures"]))

    def test_one_air_frame_is_not_a_jump_and_calibration_cannot_approve_it(self):
        self.trace[31:36,INDEX["contact_left"]]=1
        r=assess_trace(self.trace,self.report,self.profile,full=True)
        self.assertFalse(r["passed"])
        with self.assertRaisesRegex(ValueError,"missing flight"):
            calibrate_profile(self.report,[dict(results=[dict(completed=True,quality=dict(passed=True),acceptance=r)])],[])

    def test_unexpected_air_is_checked_locally_even_when_global_fraction_is_small(self):
        trace=np.tile(self.trace[:1],(900,1));trace[:,0]=np.arange(900)/30.
        trace[600:605,INDEX["unexpected_air"]]=1
        report=dict(self.report,events=[],duration_s=30.)
        r=assess_trace(trace,report,generic_profile(report),full=True)
        self.assertFalse(r["passed"])
        self.assertLess(r["metrics"]["support_air_fraction"],.05)
        self.assertTrue(any(f["check"]=="support_air_s" for f in r["failures"]))

    def test_unknown_intent_does_not_require_support_or_flight(self):
        report=dict(self.report,events=[])
        self.trace[:,INDEX["required"]]=0;self.trace[:,INDEX["contact_left"]]=0;self.trace[:,INDEX["contact_right"]]=0
        self.assertTrue(assess_trace(self.trace,report,generic_profile(report),full=True)["passed"])

    def test_missing_event_only_skips_truncated_coverage_not_full_validation(self):
        trace=self.trace[:15]
        self.assertTrue(assess_trace(trace,self.report,self.profile,full=False)["passed"])
        self.assertFalse(assess_trace(trace,self.report,self.profile,full=True)["passed"])

    def test_missing_or_nonfinite_data_never_passes(self):
        for trace in (self.trace[::2],self.trace[:0],self.trace*np.nan):
            self.assertFalse(assess_trace(trace,self.report,self.profile,full=True)["passed"])

    def test_landing_requires_actual_support_after_flight(self):
        self.trace[36:45,INDEX["contact_left"]]=0;self.trace[36:45,INDEX["contact_right"]]=0
        r=assess_trace(self.trace,self.report,self.profile,full=True)
        self.assertTrue(any(f["check"]=="landing_without_sustained_support" for f in r["failures"]))

    def test_short_support_slip_cannot_hide_below_full_motion_p95(self):
        trace=np.tile(self.trace[:1],(900,1));trace[:,0]=np.arange(900)/30.
        trace[600:630,INDEX["support_speed"]]=2.
        report=dict(self.report,events=[],duration_s=30.)
        r=assess_trace(trace,report,generic_profile(report),full=True)
        self.assertEqual(r["metrics"]["support_speed_p95"],0.)
        self.assertTrue(any(f["check"]=="support_speed_window" for f in r["failures"]))

    def test_baseline_closing_distance_is_enforced_only_for_closing_jumps(self):
        r=assess_trace(self.trace,self.report,self.profile,full=True)
        profile=calibrate_profile(self.report,[dict(results=[dict(completed=True,quality=dict(passed=True),acceptance=r)])],[])
        self.trace[30:36,INDEX["feet_distance"]]=.4
        self.assertFalse(assess_trace(self.trace,self.report,profile,full=True)["passed"])
        self.report["events"][0]["evidence"]["closing_mps"]=-1.
        self.assertTrue(assess_trace(self.trace,self.report,profile,full=True)["passed"])

    def test_profile_is_reference_bound_and_accepts_its_good_baseline(self):
        r=assess_trace(self.trace,self.report,self.profile,full=True)
        profile=calibrate_profile(self.report,[dict(results=[dict(completed=True,quality=dict(passed=True),acceptance=r)])],[])
        self.assertTrue(assess_trace(self.trace,self.report,profile,full=True)["passed"])
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            p=Path(directory)/"profile.json";p.write_text(json.dumps(profile))
            self.assertEqual(resolve_profile(self.report,p),profile)
            with self.assertRaises(ValueError):resolve_profile(dict(self.report,source_sha256="other"),p)


class SelectionTests(unittest.TestCase):
    def make(self):return QualitySelection(generic_profile(dict(source_sha256="test",events=[])),interval=400)

    def test_only_consecutive_coverage_and_full_passes_count(self):
        s=self.make()
        self.assertFalse(s.screen(400,record(kind="coverage"),record()))
        self.assertFalse(s.screen(800,record(kind="coverage"),None));self.assertEqual(s.streak,0)
        for i,ready in ((1200,False),(1600,False),(2000,True)):
            self.assertEqual(s.screen(i,record(kind="coverage"),record()),ready)
        self.assertFalse(s.screen(2800,record(kind="coverage"),record()));self.assertEqual(s.streak,1)

    def test_confirmation_cannot_omit_seeds_or_treat_legacy_quality_as_acceptance(self):
        s=self.make();seeds=[42,43,44]
        with self.assertRaises(ValueError):s.confirm(400,[record()],seeds,"best.pt")
        records=[record(seed=i) for i in seeds];records[1].pop("acceptance_passed")
        self.assertFalse(s.confirm(400,records,seeds,"best.pt")[0]);self.assertIsNone(s.best)
        records=[record(seed=i) for i in seeds];records[1]["acceptance_score"]=float("nan")
        self.assertFalse(s.confirm(400,records,seeds,"best.pt")[0])

    def test_parallel_confirmation_cannot_override_a_failed_continuous_screen(self):
        s=self.make();seeds=[42,43,44]
        s.screen(400,record(False,kind="coverage"),record())
        self.assertFalse(s.confirm(400,[record(seed=i) for i in seeds],seeds,"best.pt")[0])
        self.assertIsNone(s.best);self.assertEqual(s.streak,0)

    def test_best_survives_later_failure_and_near_ties_prefer_earlier(self):
        s=self.make();seeds=[42,43,44]
        self.assertEqual(s.confirm(400,[record(seed=i,score=1.) for i in seeds],seeds,"a.pt"),(True,True))
        self.assertEqual(s.confirm(800,[record(seed=i,score=.99) for i in seeds],seeds,"b.pt"),(True,False))
        s.confirm(1200,[record(False,seed=i) for i in seeds],seeds,"c.pt")
        self.assertEqual(s.best["checkpoint"],"a.pt")
        restored=QualitySelection(s.profile,interval=400,state=s.state_dict())
        self.assertEqual(restored.best,s.best)
        changed=QualitySelection(s.profile,interval=200,state=s.state_dict())
        self.assertEqual(changed.best,s.best)
        self.assertEqual(changed.history[:-1],s.history)
        self.assertEqual(changed.history[-1]["event"],"validation_interval_changed")
        with self.assertRaises(ValueError):QualitySelection(s.profile,patience=4,interval=200,state=s.state_dict())

    def test_changed_interval_resets_streak_without_mutating_saved_evidence(self):
        s=self.make();seeds=[42,43,44]
        for iteration in (400,800,1200):
            s.screen(iteration,record(kind="coverage"),record())
            s.confirm(iteration,[record(seed=i) for i in seeds],seeds,"best.pt")
        state=s.state_dict();old=copy.deepcopy(state)
        changed=QualitySelection(s.profile,interval=500,state=state)
        self.assertTrue(s.manual_review_ready)
        self.assertFalse(changed.manual_review_ready)
        self.assertEqual(changed.streak,0)
        self.assertEqual(changed.best,s.best)
        self.assertEqual(state,old)
        profile=copy.deepcopy(s.profile);profile["limits"]["leg_p95"]*=2
        with self.assertRaises(ValueError):QualitySelection(profile,interval=500,state=state)

    def test_never_validated_checkpoint_can_resume_with_periodic_checks(self):
        profile=self.make().profile
        saved=QualitySelection(profile,interval=10000).state_dict()
        changed=QualitySelection(profile,interval=500,state=saved)
        self.assertIsNone(changed.best)
        self.assertIsNone(changed.last_iteration)
        self.assertEqual(changed.interval,500)

    def test_manual_review_readiness_tracks_latest_confirmation_and_resumes(self):
        s=self.make();seeds=[42,43,44]
        for iteration in (400,800,1200):
            s.screen(iteration,record(kind="coverage"),record())
            self.assertFalse(s.manual_review_ready)
            s.confirm(iteration,[record(seed=i) for i in seeds],seeds,"best.pt")
        self.assertTrue(s.manual_review_ready)
        restored=QualitySelection(s.profile,interval=400,state=s.state_dict())
        self.assertTrue(restored.manual_review_ready)
        restored.screen(1600,record(kind="coverage"),record())
        self.assertFalse(restored.manual_review_ready)
        restored.confirm(1600,[record(seed=i,passed=i!=43) for i in seeds],seeds,"new.pt")
        self.assertFalse(restored.manual_review_ready)
        self.assertEqual(restored.streak,0)
        self.assertIsNotNone(restored.best)


class WorkflowTests(unittest.TestCase):
    def run_workflow(self,directory,passes,scores,early=True,resume_checkpoint=None,interrupt_at=None,
                     iterations=8,probe_interval=2,start_fraction=0.2):
        resumed=json.loads(Path(resume_checkpoint).read_text()) if resume_checkpoint else None
        report=dict(source_sha256="fixture",duration_s=1.,control_hz=30.,events=[])
        progress=NS(set_block=lambda **k:None,status=lambda *a:None,checkpoint_saved=lambda *a:None,
            quality_checked=lambda *a:None,finish=lambda passed:setattr(progress,"passed",passed),
            begin_overhead=lambda *a,**k:None,end_overhead=lambda *a,**k:None,record_overhead=lambda *a,**k:None,
            emit=lambda *a,**k:None,exports=[],checks=[])
        class Controller:
            def __init__(self,env,report,log):self.env=env;self.report=report;self.stage="short";self.progress=0.;self.selection=None
            def configure(self,p):self.stage="short";self.env._auto_jump_strength=1.
            def update(self):pass
            def load_state_dict(self,state):self.progress=state["progress"]
            def save_report(self,i):pass
            def state_dict(self):return dict(progress=self.progress,quality_selection=self.selection.state_dict())
            def validate(self,runner,wrapped,full=False,seed=42,**kwargs):
                iteration=runner.current_learning_iteration+1
                progress.checks.append((iteration,full,seed))
                kind="full" if full else "coverage"
                passed=passes.get((iteration,kind,seed),passes.get((iteration,kind),passes.get(iteration,False)))
                return record(passed,seed,"full" if full else "coverage",scores.get(iteration,1.))
            def validate_seeds(self,runner,wrapped,seeds,**kwargs):
                return [self.validate(runner,wrapped,full=True,seed=seed,**kwargs) for seed in seeds]
        class Runner:
            current_learning_iteration=0
            def __init__(self):
                self.current_learning_iteration=resumed["iteration"] if resumed else 0
                self.alg=NS(learning_rate=1.e-4,configure_learning_rate=lambda *a,**k:{})
            def learn(self,num_learning_iterations,**kwargs):
                if interrupt_at and self.current_learning_iteration<interrupt_at<=self.current_learning_iteration+num_learning_iterations:
                    raise KeyboardInterrupt()
                self.current_learning_iteration+=num_learning_iterations-1
            def save(self,path,infos=None):
                Path(path).parent.mkdir(parents=True,exist_ok=True)
                Path(path).write_text(json.dumps(dict(iteration=self.current_learning_iteration,infos=infos)))
        env=NS(reset=lambda:None,step_dt=1/30.)
        wrapped=NS(unwrapped=env)
        cfg=NS(actions=NS(joint_pos=NS(motion_h5_path="unused",motion_window_seconds=1.)),
               events=NS(reset_robot_joints=NS(params={})))
        args=NS(resume=bool(resumed),auto_jump_repair=False,auto_jump_check_starts=None,
            auto_jump_interval=probe_interval,auto_jump_probe_start_fraction=start_fraction,
            auto_quality_patience=2,auto_quality_early_stop=early,auto_validation_record=False,
            seed=42,task="fixture",dance="fixture")
        def save_bundle(path,checkpoint_path,**kwargs):
            progress.exports.append(dict(iteration=json.loads(Path(checkpoint_path).read_text())["iteration"],
                live_params=copy.deepcopy(cfg.events.reset_robot_joints.params),export_params=kwargs["env_cfg"].events.reset_robot_joints.params))
            Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(Path(checkpoint_path).read_bytes())
        modules={"source.my_task.motion_reference":NS(get_or_create_motion_buffer=lambda *a:NS(num_steps=30,export_reference=lambda:{}))}
        with patch.dict(sys.modules,modules),patch("source.train_workflow.utils.auto_jump_training.AutoJumpController",Controller),\
                patch("source.train_workflow.utils.deployment.save_bundle",save_bundle):
            try:
                _train_auto_impl(Runner(),wrapped,cfg,NS(max_iterations=iterations),report,directory,args,
                                 resumed["infos"] if resumed else None,progress)
            except KeyboardInterrupt:
                if interrupt_at is None:raise
                progress.interrupted=True
        return report,progress

    def test_full_validation_starts_at_twenty_percent_and_stays_on_the_probe_grid(self):
        self.assertEqual(first_probe_done(10000,500),2000)
        self.assertEqual(first_probe_done(8000,400),1600)
        self.assertEqual(first_probe_done(8,2),2)
        self.assertEqual(first_probe_done(10000,500,start_fraction=0),500)
        self.assertEqual(first_probe_done(4000,500,start=6000),500)
        self.assertIsNone(first_probe_done(10000,0))
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            _,progress=self.run_workflow(directory,{6:True,8:True,10:True},{6:1.,8:1.,10:1.},
                                         iterations=10,probe_interval=2,start_fraction=.5)
            self.assertEqual(sorted({item[0] for item in progress.checks}),[6,8,10])

    def test_qualified_near_ties_continue_to_limit_even_with_legacy_early_stop_flag(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            report,progress=self.run_workflow(directory,{2:True,4:True,6:True,8:True},{2:1.,4:.99,6:.99,8:.99})
            self.assertEqual(report["last_validation"]["actual_learning_iterations"],8)
            self.assertEqual(report["last_validation"]["selected_completed_iterations"],2)
            deployed=json.loads((Path(directory)/"deploy/deployment.pt").read_text())
            last=json.loads((Path(directory)/"model_7.pt").read_text())
            self.assertEqual(deployed["iteration"],1);self.assertEqual(last["iteration"],7)
            self.assertEqual(report["last_validation"]["stop_reason"],"iteration_limit")
            for iteration in (2,4,6,8):
                for seed in (42,43,44):self.assertEqual(progress.checks.count((iteration,True,seed)),1)
            self.assertTrue(progress.passed)

    def test_later_improvements_update_best_and_export_during_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            report,progress=self.run_workflow(directory,{2:True,4:True,6:True,8:True},{2:1.,4:.8,6:.7,8:.6})
            self.assertEqual(report["last_validation"]["selected_completed_iterations"],8)
            self.assertEqual([x["iteration"] for x in progress.exports],[1,3,5,7,7])
            for exported in progress.exports[:-1]:
                self.assertEqual(exported["live_params"],{})
                self.assertEqual(exported["export_params"]["motion_start_seconds"],0.)

    def test_best_bundle_is_already_usable_when_user_interrupts_before_limit(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            _,progress=self.run_workflow(directory,{2:True,4:True},{2:1.,4:.8},interrupt_at=5)
            self.assertTrue(progress.interrupted)
            deployed=json.loads((Path(directory)/"deploy/deployment.pt").read_text())
            alias=json.loads((Path(directory)/"best/model_best.pt").read_text())
            self.assertEqual(deployed["iteration"],3);self.assertEqual(alias["iteration"],3)
            self.assertTrue((Path(directory)/"auto_validation.json").exists())
            self.assertFalse((Path(directory)/"model_7.pt").exists())

    def test_ceiling_rolls_back_to_qualified_best_when_latest_model_regresses(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            report,progress=self.run_workflow(directory,{2:True,4:True},{2:1.,4:.8},early=False)
            self.assertEqual(report["last_validation"]["actual_learning_iterations"],8)
            self.assertEqual(report["last_validation"]["selected_completed_iterations"],4)
            self.assertEqual(json.loads((Path(directory)/"deploy/deployment.pt").read_text())["iteration"],3)
            self.assertTrue(progress.passed)

    def test_no_qualified_model_cannot_export_a_success_bundle(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            report,progress=self.run_workflow(directory,{},{} )
            self.assertFalse(progress.passed)
            self.assertFalse((Path(directory)/"deploy/deployment.pt").exists())
            self.assertTrue((Path(directory)/"deploy/deployment_unqualified.pt").exists())

    def test_parallel_failure_of_another_seed_updates_status_and_retains_earlier_best(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            passes={2:True,4:True,6:True,8:True,(4,"full",43):False,(6,"full",42):False,(8,"full",44):False}
            report,progress=self.run_workflow(directory,passes,{2:1.,4:.8,6:.7,8:.6})
            self.assertEqual(report["last_validation"]["selected_completed_iterations"],2)
            selection=json.loads((Path(directory)/"quality_selection.json").read_text())
            self.assertFalse(selection["latest"]["confirmed"]);self.assertEqual(selection["streak"],0)
            for iteration in (2,4,6,8):
                for seed in (42,43,44):self.assertEqual(progress.checks.count((iteration,True,seed)),1)

    def test_reset_failure_does_not_reject_verified_continuous_runtime_windows(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            passes={2:True,4:True,(2,"coverage"):False,(4,"coverage"):False}
            report,progress=self.run_workflow(directory,passes,{2:1.,4:1.})
            self.assertTrue(progress.passed)
            self.assertEqual(report["last_validation"]["stop_reason"],"iteration_limit")

    def test_resume_keeps_frozen_profile_and_copies_best_to_the_new_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as original, tempfile.TemporaryDirectory(dir=ROOT/"output") as resumed:
            self.run_workflow(original,{2:True,4:True},{2:1.,4:.99})
            report,progress=self.run_workflow(resumed,{}, {},resume_checkpoint=Path(original)/"model_3.pt")
            self.assertTrue(progress.passed)
            self.assertEqual(report["last_validation"]["selected_completed_iterations"],2)
            self.assertTrue((Path(resumed)/"best/model_best.pt").exists())
            self.assertTrue(Path(report["last_validation"]["selected_checkpoint"]).is_relative_to(resumed))

    def test_keyboard_interrupt_is_handled_so_caller_can_close_simulator(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            logger=NS(log=lambda **kwargs:None)
            env=NS()
            runner=NS(logger=logger,current_learning_iteration=0)
            args=NS(resume=False,auto_jump_interval=400)
            with patch("source.train_workflow.utils.auto_jump_training._train_auto_impl",side_effect=KeyboardInterrupt):
                train_auto(runner,NS(unwrapped=env),None,NS(max_iterations=8000),dict(duration_s=83.),directory,args)
            self.assertFalse(hasattr(env,"_auto_training_progress"))
            state=json.loads((Path(directory)/"training_status.json").read_text(encoding="utf-8"))
            self.assertEqual(state["phase"],"stopped")

    def test_atomic_best_alias_does_not_replace_a_valid_file_with_a_partial_copy(self):
        from source.train_workflow.utils.deployment import atomic_copy_file
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            source=Path(directory)/"new.pt";target=Path(directory)/"best.pt"
            source.write_bytes(b"new");target.write_bytes(b"old")
            with patch("source.train_workflow.utils.deployment.shutil.copyfile",side_effect=OSError("interrupted copy")):
                with self.assertRaises(OSError):atomic_copy_file(source,target)
            self.assertEqual(target.read_bytes(),b"old")
            atomic_copy_file(source,target)
            self.assertEqual(target.read_bytes(),b"new")
            self.assertFalse(list(Path(directory).glob("*.tmp")))


if __name__=="__main__":unittest.main()
