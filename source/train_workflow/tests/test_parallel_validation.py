"""Seed isolation, first-episode bookkeeping, and one-renderer integration."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("validation_random_test",ROOT/"source/my_task/mdp/validation_random.py")
random_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(random_module)
ValidationRandom=random_module.ValidationRandom
from source.train_workflow.utils.auto_jump_training import AutoJumpController
from source.train_workflow.utils.quality_selection import TRACE_FIELDS,INDEX


class SeedTests(unittest.TestCase):
    def test_seed_stream_does_not_depend_on_order_batch_size_other_resets_or_terms(self):
        first=ValidationRandom(NS(),[42,43,44])
        other=ValidationRandom(NS(),[44,42,43])
        single=ValidationRandom(NS(),[43])
        for _ in range(5):
            a=first.uniform_like(torch.zeros(8192,7),"obs:pos")
            other.uniform_like(torch.zeros(1,19),"reset:pos",torch.tensor([0]))
            b=other.uniform_like(torch.zeros(3,7),"obs:pos")
            c=single.uniform_like(torch.zeros(1,7),"obs:pos")
            self.assertTrue(torch.equal(a[:3],b[[1,2,0]]))
            self.assertTrue(torch.equal(a[1],c[0]))
            self.assertFalse(torch.equal(a[0],a[1]))

    def test_subset_reset_uses_actual_env_id_and_is_independent(self):
        all_rng=ValidationRandom(NS(),[42,43,44]);subset=ValidationRandom(NS(),[42,43,44])
        a=all_rng.uniform_like(torch.zeros(3,4),"reset")
        b=subset.uniform_like(torch.zeros(2,4),"reset",torch.tensor([2,0]))
        self.assertTrue(torch.equal(a[[2,0]],b))

    def test_scoped_noise_patch_restores_on_error_and_training_draws_unchanged(self):
        original=lambda data,cfg:data+9
        noise=NS(func=original,n_min=-.05,n_max=.05,operation="add")
        env=NS(observation_manager=NS(_group_obs_term_names={"policy":["gravity"]},
            _group_obs_term_cfgs={"policy":[NS(noise=noise)]}))
        with self.assertRaisesRegex(RuntimeError,"fixture"):
            with ValidationRandom(env,[42,43,44]):
                result=noise.func(torch.zeros(3,3),noise)
                self.assertLessEqual(float(result.abs().max()),.05)
                raise RuntimeError("fixture")
        self.assertIs(noise.func,original);self.assertFalse(hasattr(env,"_validation_random"))
        torch.manual_seed(10);expected=torch.rand(3,2)
        torch.manual_seed(10)
        self.assertTrue(torch.equal(random_module.uniform_like(env,torch.zeros(3,2),"obs"),expected))

    def test_unsupported_noise_fails_and_restores_previous_patch(self):
        fn=lambda *a:None
        noise=NS(func=fn,n_min=-1,n_max=1,operation="add")
        env=NS(observation_manager=NS(_group_obs_term_names={"policy":["a","b"]},
            _group_obs_term_cfgs={"policy":[NS(noise=noise),NS(noise=NS(func=fn))]}))
        with self.assertRaises(ValueError):
            with ValidationRandom(env,[42]):pass
        self.assertIs(noise.func,fn)


class IntegrationTests(unittest.TestCase):
    def test_reset_error_still_restores_training_rng_noise_and_probe(self):
        manager=NS(_group_obs_term_cfgs={},_group_obs_term_names={})
        def broken_reset():
            torch.rand(12)
            raise RuntimeError("reset failed")
        env=NS(num_envs=3,device="cpu",step_dt=1/30.,reset=broken_reset,observation_manager=manager)
        runner=NS(get_inference_policy=lambda **k:None,alg=NS(train_mode=lambda:None))
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            controller=AutoJumpController(env,dict(duration_s=.1,control_hz=30.,events=[],source_sha256="test"),directory)
            rng=torch.random.get_rng_state().clone()
            with patch.dict(sys.modules,{"source.my_task.mdp.validation_random":random_module}):
                with self.assertRaisesRegex(RuntimeError,"reset failed"):
                    controller.validate_seeds(runner,None,[42,43,44])
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertIsNone(controller.probe);self.assertFalse(hasattr(env,"_validation_random"))

    def test_one_fall_does_not_end_other_seeds_or_record_auto_reset_and_rng_is_restored(self):
        self._check_first_episode_bookkeeping()

    def test_motion_feedback_collects_all_seeds_even_when_pose_recording_is_disabled(self):
        self._check_first_episode_bookkeeping(motion=True,record=False)

    def _check_first_episode_bookkeeping(self,*,motion=False,record=True):
        proxy=lambda t:NS(torch=t)
        origins=torch.tensor([[100.,0.,0.],[0.,200.,0.],[300.,0.,0.],[0.,400.,0.]])
        robot=NS(joint_names=["a"],data=NS(root_state_w=proxy(torch.zeros(4,13)),joint_pos=proxy(torch.zeros(4,1))))
        cfg=NS(decimation=4,observations=NS(policy=NS(enable_corruption=True)),auto_motion_feedback=motion)
        env=NS(num_envs=4,device="cpu",step_dt=1/30.,physics_dt=1/120.,cfg=cfg,
            episode_length_buf=torch.zeros(4,dtype=torch.long),
            termination_manager=NS(time_outs=torch.zeros(4,dtype=torch.bool),terminated=torch.zeros(4,dtype=torch.bool)),
            observation_manager=NS(_group_obs_term_cfgs={},_group_obs_term_names={}))
        class Scene(dict):
            sim=NS(get_clone_plan=lambda:NS(positions=origins))
        env.scene=Scene(robot=robot)
        def reset():
            env.episode_length_buf.zero_()
            robot.data.root_state_w.torch[:,:3]=origins+torch.tensor([0.,0.,.75])
        env.reset=reset
        policy=lambda obs:torch.zeros(4,1)
        runner=NS(current_learning_iteration=399,get_inference_policy=lambda **k:policy,alg=NS(train_mode=lambda:None))
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            controller=AutoJumpController(env,dict(duration_s=.1,control_hz=30.,events=[],source_sha256="fixture"),directory)
            controller.record_validation=record
            if motion:
                controller.motion_analyzer=NS(joints=[0],assess=lambda poses,commands,trace:dict(
                    protocol="reference-motion-v1",complete=len(poses)==4,contact_observed=True,score=.1,candidates=[],command_count=len(commands)))
            def step(action):
                p=controller.probe;env.episode_length_buf+=1
                active=p["active"].clone()
                for i in active.nonzero().flatten().tolist():
                    robot.data.root_state_w.torch[i,2]=.6 if i==0 else .7+i*.01
                    p["pose_captures"][i].append(robot.data.root_state_w.torch[i,:7],robot.data.joint_pos.torch[i])
                    if "motion_commands" in p:p["motion_commands"][i].append(torch.zeros(1))
                    trace=torch.zeros(len(TRACE_FIELDS));trace[INDEX["seconds"]]=env.episode_length_buf[i]*env.step_dt
                    trace[INDEX["leg"]]=i*.01;trace[INDEX["contact_left"]]=1;trace[INDEX["normal"]]=1
                    p["trace"].append(torch.cat((torch.tensor([float(i)]),trace))[None])
                metrics={k:torch.zeros(4) for k in ("leg_rms_rad","height_error_m","tilt_error_rad")}
                metrics["normal_weight"]=torch.ones(4)
                p["quality"].update(metrics,active,env.episode_length_buf.float()*env.step_dt)
                ended=active & ((env.episode_length_buf>=3)| (torch.arange(4)==0))
                ids=ended.nonzero().flatten()
                env.termination_manager.time_outs=env.episode_length_buf>=3
                env.termination_manager.terminated=torch.arange(4)==0
                controller.reset(ids,env.episode_length_buf[ids])
                robot.data.root_state_w.torch[ids,2]=999.
                torch.rand(9) # Global RNG consumed by the reused environment.
                return torch.zeros(4,1),None,None,None
            wrapped=NS(get_observations=lambda:torch.zeros(4,1),step=step)
            assessment=lambda trace,*a,**k:dict(passed=len(trace)==3,score=.1,failures=[])
            modules={"source.my_task.mdp.validation_random":random_module,"source.sim_compat":NS(as_torch=lambda x:x)}
            rng=torch.random.get_rng_state().clone()
            with patch.dict(sys.modules,modules),patch("source.train_workflow.utils.quality_selection.assess_trace",assessment):
                records=controller.validate_seeds(runner,wrapped,[42,43,44])
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertIsNone(controller.probe);self.assertFalse(hasattr(env,"_validation_random"))
            self.assertEqual([r["seed"] for r in records],[42,43,44])
            self.assertEqual([r["completed"] for r in records],[0,1,1])
            self.assertEqual([r["acceptance_passed"] for r in records],[False,True,True])
            for i,r in enumerate(records):
                if motion:
                    self.assertTrue(r["motion_feedback_required"])
                    self.assertEqual(r["results"][0]["motion_diagnostics"]["command_count"],1 if i==0 else 3)
                if not record:
                    self.assertNotIn("pose_recording",r["results"][0]);continue
                with np.load(r["results"][0]["pose_recording"]) as archive:
                    self.assertEqual(len(archive["poses"]),2 if i==0 else 4)
                    self.assertLess(float(archive["poses"][:,2].max()),1.)
                    self.assertEqual(json.loads(str(archive["metadata"]))["seed"],42+i)
                    self.assertAlmostEqual(float(archive["poses"][0,0]),0.)

    def test_three_results_start_only_one_renderer_and_share_video(self):
        controller=AutoJumpController(NS(device="cpu"),dict(duration_s=1.,events=[]),ROOT/"output")
        controller.render_validation=True
        records=[dict(results=[dict(pose_recording=f"seed_{s}/poses.npz")]) for s in (42,43,44)]
        with patch("source.train_workflow.utils.validation_recording.render_recordings",return_value=dict(video="all.mp4")) as render:
            controller._render_validations(records,None)
        render.assert_called_once_with([f"seed_{s}/poses.npz" for s in (42,43,44)])
        self.assertTrue(all(r["results"][0]["video_recording"]["video"]=="all.mp4" for r in records))


if __name__=="__main__":unittest.main()
