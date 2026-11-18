"""Progress accounting across chunks/resume, independent of Isaac or PyTorch."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.training_progress import TrainingProgress,stage_plan


class FakeLogger:
    def __init__(self):self.calls=[];self.tot_time=0.
    def log(self,**kwargs):
        self.calls.append(kwargs)
        self.tot_time+=kwargs["collect_time"]+kwargs["learn_time"]


class ProgressTests(unittest.TestCase):
    def make(self,start=0,total=1000,initial=0.,directory=None):
        logger=FakeLogger();lines=[]
        progress=TrainingProgress(logger,start=start,total=total,initial_progress=initial,
            motion_seconds=83.333333,log_dir=directory,emit=lines.append,clock=lambda:100.)
        return logger,progress,lines

    def kwargs(self,it,start,total,loss=None):
        return dict(it=it,start_it=start,total_it=total,collect_time=1.5,learn_time=.5,
                    learning_rate=1.e-4,loss_dict=loss)

    class Clock:
        def __init__(self):self.now=0.
        def __call__(self):return self.now
        def advance(self,seconds):self.now+=seconds

    def sample(self,progress,clock,kind,seconds):
        progress.validation_started(kind,42,1,3.)
        clock.advance(seconds)
        progress.validation_finished(dict(kind=kind,seed=42,completed=1,tested=1))

    def test_stage_ranges_match_curriculum_and_resume(self):
        self.assertEqual(stage_plan(8000),[
            dict(stage="short",begin=0,end=4400),dict(stage="long",begin=4400,end=6800),dict(stage="full",begin=6800,end=8000)])
        self.assertEqual(stage_plan(10000),[
            dict(stage="short",begin=0,end=5500),dict(stage="long",begin=5500,end=8500),dict(stage="full",begin=8500,end=10000)])
        # Changing rollout length must not move course transitions in sample budget.
        self.assertEqual([(s["begin"]*60,s["end"]*60) for s in stage_plan(8000)],
                         [(s["begin"]*48,s["end"]*48) for s in stage_plan(10000)])
        self.assertEqual(stage_plan(1000,1.),[dict(stage="full",begin=0,end=1000)])
        self.assertEqual(stage_plan(2000,.5),[
            dict(stage="short",begin=0,end=200),dict(stage="long",begin=200,end=1400),dict(stage="full",begin=1400,end=2000)])
        self.assertEqual(stage_plan(3),[dict(stage="short",begin=0,end=2),dict(stage="full",begin=2,end=3)])

    def test_global_header_and_eta_do_not_restart_at_chunk_boundary(self):
        logger,p,_=self.make()
        loss={"unchanged":object()}
        original=logger.log
        with p:
            p.set_block(done=0,count=200,stage="short",strength=.1,learning_rate=.001,next_probe=500)
            for it in range(200):logger.log(**self.kwargs(it,0,200,loss))
            self.assertEqual(p.snapshot()["learning_eta_s"],1600.)
            p.set_block(done=200,count=200,stage="short",strength=.2,learning_rate=.001,next_probe=500)
            logger.log(**self.kwargs(200,200,400,loss))
            self.assertEqual(logger.calls[-1]["start_it"],0)
            self.assertEqual(logger.calls[-1]["total_it"],1000)
            self.assertEqual(p.snapshot()["learning_eta_s"],1598.)
            self.assertEqual(p.snapshot()["block_completed"],1)
            self.assertIs(logger.calls[-1]["loss_dict"],loss)
        self.assertEqual(logger.log,original)

    def test_resume_distinguishes_cumulative_and_invocation_progress(self):
        logger,p,lines=self.make(start=8000,total=1000,initial=1.)
        with p:
            p.set_block(done=0,count=200,stage="full",strength=1.,learning_rate=3.e-5,next_probe=500)
            logger.log(**self.kwargs(8000,8000,8200))
            s=p.snapshot()
            self.assertEqual((s["cumulative_completed"],s["cumulative_target"]),(8001,9000))
            self.assertEqual((s["invocation_completed"],s["invocation_total"]),(1,1000))
            self.assertEqual((s["stage_completed"],s["stage_total"]),(1,1000))
            self.assertEqual(logger.calls[-1]["total_it"],9000)
        self.assertTrue(any("本次 1/1000" in line for line in lines))

    def test_learning_completion_is_not_run_completion(self):
        logger,p,_=self.make(total=1)
        with p:
            p.set_block(done=0,count=1,stage="short",strength=.1,learning_rate=.001)
            logger.log(**self.kwargs(0,0,1))
            p.validation_started("full",42,1,83.33)
            self.assertEqual(p.snapshot()["phase"],"full_validation")
            self.assertEqual(p.snapshot()["learning_eta_s"],0.)
            p.validation_progress(batch=1,batches=1,elapsed=30.,length=83.33,pending=1)
            self.assertEqual(p.snapshot()["validation"]["episode_elapsed_s"],30.)
            p.validation_finished(dict(kind="full",seed=42,completed=0,tested=1))
            p.status("exporting","export")
            p.finish(False)
            self.assertEqual(p.snapshot()["phase"],"completed")
            self.assertFalse(p.snapshot()["validation_passed"])

    def test_atomic_status_and_exception_restore_logger(self):
        with tempfile.TemporaryDirectory() as directory:
            logger,p,_=self.make(directory=directory)
            original=logger.log
            with self.assertRaisesRegex(RuntimeError,"example"):
                with p:raise RuntimeError("example")
            self.assertEqual(logger.log,original)
            state=json.loads((Path(directory)/"training_status.json").read_text(encoding="utf-8"))
            self.assertEqual(state["phase"],"failed")
            self.assertIn("RuntimeError: example",state["error"])

    def test_status_write_failure_does_not_interrupt_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            logger,p,lines=self.make(directory=directory)
            with patch("source.train_workflow.utils.training_progress.os.replace",side_effect=PermissionError("busy")):
                with p:
                    p.set_block(done=0,count=2,stage="short",strength=.1,learning_rate=.001)
                    logger.log(**self.kwargs(0,0,2))
                    logger.log(**self.kwargs(1,0,2))
            self.assertEqual(len(logger.calls),2)
            self.assertEqual(sum("写入失败" in line for line in lines),1)

    def test_quality_is_displayed_every_iteration_with_age_and_never_stops_learning(self):
        logger,p,lines=self.make(total=1000)
        with p:
            p.set_block(done=0,count=10,stage="short",strength=.1,learning_rate=.001)
            latest=dict(iteration=1,screen_passed=True,confirmed=True,score=.5,metrics=dict(leg_p95=.15),
                        windows_completed=167,windows_tested=167,jump_events_passed=27,jump_events_tested=27)
            p.quality_checked(SimpleNamespace(streak=3,patience=3,last_iteration=1,latest=latest,manual_review_ready=True,
                best=dict(iteration=1,checkpoint="best.pt"),stop_reason=None,profile=dict(kind="accepted_baseline")),True)
            logger.log(**self.kwargs(1,0,10))
            logger.log(**self.kwargs(2,0,10))
        s=p.snapshot()
        self.assertEqual((s["invocation_completed"],s["invocation_total"]),(3,1000))
        self.assertEqual(s["phase"],"learning")
        self.assertEqual(s["quality_selection"]["iterations_since_validation"],2)
        self.assertFalse(s["quality_selection"]["early_stop_enabled"])
        self.assertTrue(s["quality_selection"]["manual_review_ready"])
        self.assertEqual(s["quality_selection"]["best"]["iteration"],1)
        self.assertEqual(sum("[质量状态]" in line for line in lines),3)
        self.assertTrue(any("距当前 2 次" in line for line in lines))

    def test_manual_interrupt_is_stopped_instead_of_failed_and_restores_logger(self):
        with tempfile.TemporaryDirectory() as directory:
            logger,p,lines=self.make(directory=directory)
            original=logger.log
            with self.assertRaises(KeyboardInterrupt):
                with p:
                    p.set_block(done=0,count=10,stage="short",strength=.1,learning_rate=.001)
                    logger.log(**self.kwargs(0,0,10))
                    raise KeyboardInterrupt()
            self.assertEqual(logger.log,original)
            s=json.loads((Path(directory)/"training_status.json").read_text(encoding="utf-8"))
            self.assertEqual(s["phase"],"stopped")
            self.assertEqual(s["stop_reason"],"manual_stop")
            self.assertIsNone(s["error"])

    def test_eta_adds_remaining_validation_video_and_export(self):
        clock=self.Clock()
        lines=[]
        progress=TrainingProgress(FakeLogger(),start=0,total=8000,initial_progress=1.,
            motion_seconds=83.33,probe_interval=400,emit=lines.append,clock=clock)
        progress.expects_validation_video=True
        progress.expects_final_export=True
        progress.phase="learning"
        progress.done=7600
        progress.learning_seconds=7600.
        progress.next_probe=8000
        self.sample(progress,clock,"coverage",30.)
        self.sample(progress,clock,"full",600.)
        clock.advance(10.)
        progress.validation_finished(dict(kind="full",seed=43,completed=1,tested=1))
        clock.advance(10.)
        progress.validation_finished(dict(kind="full",seed=44,completed=1,tested=1))
        progress.phase="learning"
        progress.next_probe=8000
        progress.record_overhead("video",200.)
        progress.record_overhead("export",1.)
        state=progress.snapshot()
        self.assertEqual(progress.overhead_seconds["full"],[600.])
        self.assertEqual(state["remaining_validation_cycles"],1)
        self.assertEqual(state["learning_eta_s"],400.)
        self.assertEqual(state["overhead_eta_s"],831.)
        self.assertEqual(state["total_eta_s"],1231.)
        self.assertEqual(state["overhead_pending"],[])
        progress.emit_eta(state)
        self.assertIn("剩余约 00:20:31",lines[-1])
        self.assertIn("学习 00:06:40",lines[-1])
        self.assertIn("验证/导出 00:13:51",lines[-1])

    def test_eta_waits_for_the_first_validation_sample(self):
        _,progress,lines=self.make(total=8000)
        with progress:
            progress.set_block(done=0,count=1,stage="full",strength=1.,learning_rate=1.e-4,next_probe=400)
            progress.logger.log(**self.kwargs(0,0,1))
        self.assertIsNone(progress.snapshot()["overhead_eta_s"])
        self.assertTrue(any("验证/导出待首次验收后计入" in line for line in lines))

    def test_final_validation_is_not_double_counted_when_a_probe_lands_on_the_budget(self):
        _,progress,_=self.make(total=800,initial=1.)
        progress.phase="learning"
        progress.next_probe=800
        progress.done=800
        self.assertEqual(progress.remaining_validation_cycles(),1)
        progress.next_probe=None
        self.assertEqual(progress.remaining_validation_cycles(),0)

    def test_final_validation_is_added_when_the_budget_is_not_a_probe(self):
        _,progress,_=self.make(total=1000,initial=1.)
        progress.phase="learning"
        progress.next_probe=900
        progress.done=900
        self.assertEqual(progress.remaining_validation_cycles(),2)

    def test_recent_validation_samples_replace_earlier_ones(self):
        clock=self.Clock()
        progress=TrainingProgress(FakeLogger(),start=0,total=400,initial_progress=1.,
            motion_seconds=83.33,probe_interval=400,emit=lambda _line:None,clock=clock)
        progress.phase="learning"
        progress.next_probe=400
        progress.done=200
        for seconds in (10.,100.,100.,130.):
            progress.record_overhead("coverage",seconds)
            progress.record_overhead("full",seconds)
        overhead,_=progress.overhead_estimate(progress.remaining_validation_cycles())
        self.assertEqual(overhead,(100.+100.+130.)/3*2)

    def test_latest_failure_is_visible_while_previous_best_remains_available(self):
        _,p,lines=self.make()
        p.quality_checked(SimpleNamespace(streak=0,patience=3,last_iteration=400,manual_review_ready=False,
            latest=dict(iteration=400,screen_passed=False,confirmed=None,score=None,
                        failures=[dict(check="missed_sustained_flight",time_s=17.)]),
            best=dict(iteration=200,checkpoint="best.pt"),stop_reason=None,profile=dict(kind="generic_engineering")))
        s=p.snapshot()["quality_selection"]
        self.assertFalse(s["manual_review_ready"])
        self.assertTrue(s["manual_review_available"])
        self.assertTrue(any("未通过" in line for line in lines))
        self.assertTrue(any("missed_sustained_flight" in line for line in lines))


if __name__=="__main__":unittest.main()
