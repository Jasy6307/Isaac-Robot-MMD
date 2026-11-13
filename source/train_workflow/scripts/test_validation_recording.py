"""Recording must preserve actual states and must not perturb acceptance RNG."""
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
from source.train_workflow.utils.validation_recording import ValidationPoseCapture,save_validation_scene,render_recording,load_recordings
from source.train_workflow.utils.auto_jump_training import AutoJumpController
from source.train_workflow.utils.motion_quality import MotionQualityMonitor


class RecordingTests(unittest.TestCase):
    def test_capture_is_local_immutable_and_does_not_consume_random_numbers(self):
        root=torch.tensor([100.,200.,.75,0.,0.,0.,1.]);q=torch.tensor([.1,.2])
        origin=torch.tensor([100.,200.,0.]);rng=torch.random.get_rng_state().clone()
        capture=ValidationPoseCapture(origin=origin,joint_names=["a","b"],control_hz=30.)
        capture.append(root,q);root[2]=.8;q[0]=.3;capture.append(root,q)
        origin[:]=0;root[:]=0;q[:]=0
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            path=Path(directory)/"poses.npz"
            capture.save(path,metadata=dict(seed=42),result=dict(completed=False))
            with np.load(path) as x:
                self.assertEqual(x["poses"].shape,(2,9))
                self.assertAlmostEqual(float(x["poses"][0,2]),.75)
                self.assertAlmostEqual(float(x["poses"][1,7]),.3,places=6)
                meta=json.loads(str(x["metadata"]))
                self.assertFalse(meta["result"]["completed"])
                self.assertAlmostEqual(meta["simulation_duration_s"],1/30.)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))

    def test_scene_snapshot_does_not_change_live_environment_config(self):
        cfg=NS(scene=NS(num_envs=8192),events=NS(reset_params=dict(random_start=True)),
               actions=NS(joint_pos=NS(motion_h5_path="unused.h5")))
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            path=save_validation_scene(Path(directory)/"scene.pt",env_cfg=cfg,reference=dict(q=torch.zeros(3)),
                task="test",report=dict(source_sha256="test",control_hz=30.))
            saved=torch.load(path,weights_only=False)
            self.assertEqual(saved["env_cfg"].scene.num_envs,1)
            self.assertEqual(cfg.scene.num_envs,8192)
            self.assertNotIn("checkpoint",saved)

    def test_empty_or_nonfinite_recording_is_rejected(self):
        capture=ValidationPoseCapture(origin=torch.zeros(3),joint_names=["a"],control_hz=30.)
        with self.assertRaises(ValueError):capture.save("unused.npz",metadata={},result={})
        capture.append(torch.tensor([float("nan"),0.,0.,0.,0.,0.,1.]),torch.zeros(1))
        with self.assertRaises(ValueError):capture.save("unused.npz",metadata={},result={})

    def test_explicit_music_is_preserved_without_motion_name_lookup(self):
        cfg=NS(scene=NS(num_envs=16))
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            audio=Path(directory)/"music.wav"
            audio.write_bytes(b"test")
            path=save_validation_scene(Path(directory)/"scene.pt",env_cfg=cfg,reference={},
                task="test",report=dict(source_sha256="test",control_hz=30.),audio=str(audio))
            self.assertEqual(torch.load(path,weights_only=False)["audio"],str(audio))
            with self.assertRaises(FileNotFoundError):
                save_validation_scene(Path(directory)/"missing.pt",env_cfg=cfg,reference={},
                    task="test",report=dict(source_sha256="test",control_hz=30.),audio=str(audio.with_name("missing.wav")))
    def test_actual_observe_hook_captures_terminal_pose_before_reset_once(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            proxy=lambda t:NS(torch=t)
            data=NS(root_state_w=proxy(torch.tensor([[0.,0.,.75,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]])),
                joint_pos=proxy(torch.tensor([[.1,.2]])),body_link_lin_vel_w=proxy(torch.zeros(1,2,3)),
                body_link_pose_w=proxy(torch.zeros(1,2,7)),root_lin_vel_w=proxy(torch.zeros(1,3)))
            robot=NS(data=data,body_names=["left_ankle_roll_link","right_ankle_roll_link"])
            data.joint_pos_target=proxy(torch.tensor([[.3,.4]]))
            env=NS(device="cpu",step_dt=1/30.,num_envs=1,common_step_counter=1,scene={"robot":robot},
                cfg=NS(actions=NS(joint_pos=NS(motion_h5_path="unused",motion_window_seconds=1.))))
            report=dict(duration_s=1.,source_sha256="test",control_hz=30.,events=[])
            controller=AutoJumpController(env,report,directory)
            controller.motion_analyzer=NS(joints=[0,1])
            capture=ValidationPoseCapture(origin=torch.zeros(3),joint_names=["a","b"],control_hz=30.)
            capture.append(data.root_state_w.torch[0,:7],data.joint_pos.torch[0])
            z=lambda:torch.zeros(1)
            controller.probe=dict(active=torch.tensor([True]),sums=z(),counts=z(),quality=MotionQualityMonitor(1,"cpu",1/30.),
                trace=[],pose_capture=capture,motion_commands={0:[]},flight_requests=z(),flight_missed=z(),support_samples=z(),unexpected_air=z(),
                air_run=torch.zeros(1,dtype=torch.long),air_max=torch.zeros(1,dtype=torch.long),air_samples=z(),window_samples=z(),window_air=z())
            metrics=dict(leg_rms_rad=z()+.1,height_error_m=z()+.02,tilt_error_rad=z()+.1,normal_weight=z()+1.,height_delta_m=z())
            support=dict(required=torch.tensor([True]),expected=torch.ones(1,2,dtype=torch.bool),
                contacts=torch.ones(1,2,dtype=torch.bool),unexpected_air=torch.tensor([False]),missing_feet=torch.zeros(1,2,dtype=torch.bool))
            jump=NS(phases=lambda e:torch.zeros(1,4,dtype=torch.bool),
                flight_state=lambda e:dict(air=torch.zeros(1),stance=torch.tensor([False]),rising=torch.tensor([False])))
            modules={"source.my_task.motion_reference":NS(motion_steps=lambda e:torch.ones(1,dtype=torch.long),
                get_or_create_motion_buffer=lambda *a:NS(root_pos_delta=lambda steps:torch.zeros(1,3))),
                "source.my_task.mdp":NS(jump=jump),"source.my_task.mdp.auto_jump":NS(
                    tracking_metrics=lambda *a:metrics,support_metrics=lambda e:support,
                    swing_metrics=lambda e:dict(required=torch.zeros(1,dtype=torch.bool),
                        error_m=torch.zeros(1),early_contact=torch.zeros(1,dtype=torch.bool)))}
            data.root_state_w.torch[0,2]=.6
            with patch.dict(sys.modules,modules):
                controller.observe();controller.observe()
                data.root_state_w.torch[0,2]=.75
                env.common_step_counter=2;controller.probe["active"][0]=False;controller.observe()
            self.assertEqual(len(capture.frames),2)
            self.assertEqual(len(controller.probe["motion_commands"][0]),1)
            self.assertTrue(torch.allclose(controller.probe["motion_commands"][0][0],torch.tensor([.3,.4])))
            self.assertAlmostEqual(float(capture.frames[-1][2]),.6,places=6)

    def test_encoder_missing_does_not_raise_or_change_acceptance(self):
        with patch("source.train_workflow.utils.validation_recording.find_ffmpeg",return_value=None):
            result=render_recording(ROOT/"output/unused.npz",emit=lambda s:None)
        self.assertIsNone(result["video"]);self.assertIn("ffmpeg",result["error"])

    def test_synchronized_replay_accepts_early_end_but_rejects_mismatched_clock_reference_or_seed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            def save(seed,frames,**changes):
                capture=ValidationPoseCapture(origin=torch.zeros(3),joint_names=["a"],control_hz=30.)
                for _ in range(frames):capture.append(torch.tensor([0.,0.,.7,0.,0.,0.,1.]),torch.zeros(1))
                metadata=dict(seed=seed,source_sha256="ref",completed_iterations=400,validation_protocol="test")
                metadata.update(changes)
                return capture.save(Path(directory)/f"{seed}_{frames}.npz",metadata=metadata,result=dict(completed=False))
            a=save(42,2);b=save(43,4)
            self.assertEqual([len(p) for p,m in load_recordings([a,b])],[2,4])
            with self.assertRaisesRegex(ValueError,"Duplicate"):load_recordings([a,a])
            # Checkpoint comparison is explicit; ordinary seed validation stays strict.
            self.assertEqual(len(load_recordings([a,a],checkpoint_comparison=True)),2)
            for field,value in (("source_sha256","other"),("completed_iterations",800),("validation_protocol","other")):
                bad=save(44,3,**{field:value})
                with self.assertRaisesRegex(ValueError,field):load_recordings([a,bad])


if __name__=="__main__":unittest.main()
