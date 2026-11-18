"""Pipeline orchestration checks without launching Isaac Sim."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("vmd_pipeline", ROOT / "source/train_workflow/g1_vmd_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def test_versioning_checks_whole_output_set(self):
        test_root = ROOT / "output/vmd-pipeline-tests"
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as temp:
            root = Path(temp)
            first = pipeline.select_output_paths(root, root, "IRIS_OUT_collision_corrected", True)
            self.assertEqual(first["h5"].name, "IRIS_OUT_collision_corrected.h5")
            first["video"].write_bytes(b"existing")
            second = pipeline.select_output_paths(root, root, "IRIS_OUT_collision_corrected", True)
            self.assertEqual(second["h5"].name, "IRIS_OUT_collision_corrected_v2.h5")
            second["raw"].write_bytes(b"partial")
            third = pipeline.select_output_paths(root, root, "IRIS_OUT_collision_corrected", True)
            self.assertEqual(third["h5"].name, "IRIS_OUT_collision_corrected_v3.h5")

    def test_ffmpeg_from_other_local_python(self):
        test_root = ROOT / "output/vmd-pipeline-tests"
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as temp:
            binary = Path(temp) / "Python/Python313/site-packages/imageio_ffmpeg/binaries/ffmpeg-test.exe"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"test")
            with patch.dict("os.environ", {"APPDATA": temp}), patch.dict("sys.modules", {"imageio_ffmpeg": None}), patch.object(pipeline.shutil, "which", return_value=None):
                self.assertEqual(pipeline.find_ffmpeg(), str(binary))

    def exercise(self, corrected, fail_repair=False, upper=False, lower=False, modern=False, fail_joints=False, fail_arm=False, fail_hand=False, fail_mesh=False, local=False, fail_continuity=False):
        test_root = ROOT / "output/vmd-pipeline-tests"
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as temp:
            base = Path(temp)
            vmd, audio = base / "original.vmd", base / "original.wav"
            vmd.write_bytes(b"original")
            audio.write_bytes(b"music")
            steps = []

            def run(label, command):
                command = [str(v) for v in command]
                steps.append(command)
                if '--fail-on-intersection' in command:
                    import hashlib
                    folder=Path(command[command.index('--output')+1]);folder.mkdir(parents=True,exist_ok=True)
                    motion=Path(command[command.index('--h5')+1])
                    (folder/'mesh_scan.json').write_text(json.dumps({'passed':not fail_mesh,
                        'h5_sha256':hashlib.sha256(motion.read_bytes()).hexdigest(),'sample_count':4*(630-1)+1,'sample_step_frames':.25}))
                    return
                if "--output" in command:
                    if fail_repair and "--input" in command:
                        raise RuntimeError("repair rejected")
                    Path(command[command.index("--output") + 1]).write_bytes(b"generated")
                if "--auto_record_h5" in command:
                    Path(command[command.index("--motion_csv") + 1]).with_suffix(".h5").write_bytes(b"raw")
                if "--record_avi_path" in command:
                    self.assertIn("--exit_after_playback", command)
                    self.assertNotIn("--motion_wav", command)  # music is muxed, not recorded from speakers
                    Path(command[command.index("--record_avi_path") + 1]).write_bytes(b"avi")
                if "-shortest" in command:
                    self.assertTrue(any("apad" in str(value) for value in command))
                    self.assertIn("volume=0.25,apad", command)
                    Path(command[-1]).write_bytes(b"mp4")

            class H5:
                attrs = {"fps": 30., "without_collision_validated": True,'arm_motion_validated':False if local else not fail_arm,
                         'collision_hand_geometry_mode':'neutral-hand-box-v0' if fail_hand else 'neutral-hand-box-and-articulated-finger-capsules-v1'}
                if local:
                    attrs.update(collision_repair_mode='local',collision_continuity_validated=not fail_continuity)
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def __getitem__(self, key): return range(630)

            class Capture:
                def __init__(self, path): pass
                def isOpened(self): return True
                def get(self, prop): return 630 if prop == 1 else 30.
                def release(self): pass

            modules = {"h5py": types.SimpleNamespace(File=lambda path: H5()),
                "cv2": types.SimpleNamespace(VideoCapture=Capture, CAP_PROP_FRAME_COUNT=1, CAP_PROP_FPS=2)}
            argv = ["--vmd", str(vmd), "--output_dir", str(base / "dance"),
                "--video_dir", str(base / "video"), "--name", "fresh"]
            if not (upper or lower or modern):
                argv += ['--retarget_mode','legacy']
            if corrected:
                argv.append("--collision_correction")
            if upper:
                source = base / 'source.usdc'
                source.write_bytes(b'source skeleton')
                argv += ['--upper_body_source_animation', str(source)]
            if lower:
                source = base / 'source.usdc'
                source.write_bytes(b'source skeleton')
                argv += ['--lower_body_source_animation', str(source)]
            with patch.dict("sys.modules", modules), patch.object(pipeline, "find_ffmpeg", return_value="ffmpeg"), patch.object(pipeline, "validate_final_joint_limits", return_value={'passed':not fail_joints,'joint_count':51}), patch.object(pipeline, "run_step", side_effect=run):
                if fail_continuity:
                    with self.assertRaisesRegex(RuntimeError,'local continuity was not validated'):
                        pipeline.main(argv)
                    self.assertFalse(any('--record_avi_path' in c for c in steps))
                    return
                if fail_mesh:
                    with self.assertRaisesRegex(RuntimeError,'full-motion hand mesh validation failed'):
                        pipeline.main(argv)
                    self.assertFalse(any('--record_avi_path' in c for c in steps))
                    return
                if fail_hand:
                    with self.assertRaisesRegex(RuntimeError,'articulated hand geometry was not validated'):
                        pipeline.main(argv)
                    self.assertFalse(any('--record_avi_path' in c for c in steps))
                    return
                if fail_arm:
                    with self.assertRaisesRegex(RuntimeError,'arm motion was not validated'):
                        pipeline.main(argv)
                    self.assertFalse(any('--record_avi_path' in c for c in steps))
                    return
                if fail_joints:
                    with self.assertRaisesRegex(RuntimeError,'joint constraints failed'):
                        pipeline.main(argv)
                    self.assertFalse(any('--record_avi_path' in c for c in steps))
                    self.assertFalse((base/'video/fresh.mp4').exists())
                    return
                if fail_repair:
                    with self.assertRaisesRegex(RuntimeError, "repair rejected"):
                        pipeline.main(argv)
                    self.assertFalse(any("--record_avi_path" in c for c in steps))
                    return
                pipeline.main(argv)
                self.assertTrue((base / "video/fresh.mp4").is_file())
                replay = next(c for c in steps if "--record_avi_path" in c)
                capture_command = next(c for c in steps if '--auto_record_h5' in c)
                self.assertEqual(capture_command[capture_command.index('--retarget_mode')+1],
                                 'new' if upper or lower or modern else 'legacy')
                if modern:
                    self.assertIn('--source_vmd',capture_command)
                    self.assertEqual(capture_command[capture_command.index('--source_leg_mode')+1],'fk')
                    self.assertFalse((base/'dance/fresh_raw.h5').exists())
                if upper:
                    upper_step = next(c for c in steps if '--upper_body_source_animation' in c)
                    self.assertIn('--auto_record_h5', upper_step)
                    self.assertFalse(any('--source-animation' in c for c in steps))
                    self.assertFalse((base / 'dance/fresh_upper.h5').exists())
                    if not corrected:
                        self.assertFalse((base / 'dance/fresh_raw.h5').exists())
                    if corrected:
                        repair = next(c for c in steps if '--input' in c)
                        self.assertEqual(Path(repair[repair.index('--input')+1]).name, 'fresh_raw.h5')
                expected = "fresh.h5"
                if lower:
                    capture = next(c for c in steps if '--auto_record_h5' in c)
                    self.assertIn('--lower_body_source_animation',capture)
                    self.assertFalse((base/'dance/fresh_upper.h5').exists())
                    if not corrected:self.assertFalse((base/'dance/fresh_raw.h5').exists())
                self.assertEqual(Path(replay[replay.index("--motion_h5") + 1]).name, expected)
                self.assertEqual(vmd.read_bytes(), b"original")
                pipeline.main(argv)
                self.assertTrue((base / "video/fresh_v2.mp4").is_file())
                self.assertTrue((base / "dance/fresh_v2.h5").is_file())

    def test_plain(self): self.exercise(False)
    def test_corrected(self): self.exercise(True)
    def test_local_mode_uses_its_own_certificate_without_claiming_cartesian_gesture_fit(self): self.exercise(True,local=True)
    def test_local_continuity_failure_stops_recording(self): self.exercise(True,local=True,fail_continuity=True)
    def test_failed_repair_stops_video(self): self.exercise(True, fail_repair=True)
    def test_failed_arm_motion_stops_video(self): self.exercise(True, fail_arm=True)
    def test_old_hand_geometry_cannot_be_published_as_new_validated_motion(self):self.exercise(True, fail_hand=True)
    def test_full_motion_mesh_failure_stops_video(self):self.exercise(True, fail_mesh=True)
    def test_upper_body_plain(self): self.exercise(False, upper=True)
    def test_upper_body_before_collision(self): self.exercise(True, upper=True)
    def test_lower_body_only(self): self.exercise(False, lower=True)
    def test_both_limbs_before_collision(self): self.exercise(True, upper=True, lower=True)
    def test_default_is_direct_limbs_without_extra_flags(self): self.exercise(False, modern=True)
    def test_joint_validation_failure_stops_recording(self): self.exercise(False,modern=True,fail_joints=True)


if __name__ == "__main__":
    unittest.main()
