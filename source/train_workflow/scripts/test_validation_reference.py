"""CPU checks for dance defaults, timeline alignment and PMX end-frame hold."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from source.train_workflow.utils.validation_reference import resolve_reference_video, compose_reference_video, _probe_video
from source.train_workflow.utils.validation_recording import find_ffmpeg


class ReferenceDefaultsTests(unittest.TestCase):
    def test_scene_freezes_reference_selection_without_mutating_callers(self):
        from types import SimpleNamespace as NS
        import torch
        from source.train_workflow.utils.validation_recording import save_validation_scene
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            p=Path(directory)
            cfg=NS(scene=NS(num_envs=8192),actions=NS(joint_pos=NS(motion_h5_path="unused.h5")))
            reference=dict(video="selected.mp4",start_seconds=1.5)
            saved=save_validation_scene(p/"scene.pt",env_cfg=cfg,reference={},task="test",
                report=dict(source_sha256="test",control_hz=30.),reference_video=reference)
            reference["start_seconds"]=9.
            scene=torch.load(saved,weights_only=False)
            self.assertEqual(scene["reference_video"]["start_seconds"],1.5)
            self.assertEqual(cfg.scene.num_envs,8192)

    def test_exact_dance_overrides_source_provenance_and_cli_overrides_list(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            p = Path(directory)
            motion = p/"IRIS_OUT_corrected.h5"
            motion.with_suffix(".motion.json").write_text(json.dumps({"source_dance": "IRIS_OUT"}))
            for name in ("base.mp4", "exact.mp4", "override.mp4"):
                (p/name).write_bytes(b"placeholder")
            config = p/"list.json"
            entries = {"IRIS_OUT": {"video": str(p/"base.mp4"), "start_seconds": 2.},
                       "IRIS_OUT_corrected": {"video": str(p/"exact.mp4"), "start_seconds": 3.}}
            config.write_text(json.dumps({"version": 1, "dances": entries}))
            result = resolve_reference_video(motion, config_path=config)
            self.assertEqual(result["video"], str((p/"exact.mp4").resolve()))
            self.assertEqual(result["start_seconds"], 3.)
            del entries["IRIS_OUT_corrected"]
            config.write_text(json.dumps({"version": 1, "dances": entries}))
            self.assertEqual(resolve_reference_video(motion, config_path=config)["start_seconds"], 2.)
            result = resolve_reference_video(motion, video=p/"override.mp4", start_seconds=.5, config_path=config)
            self.assertEqual(result["video"], str((p/"override.mp4").resolve()))
            self.assertEqual(result["start_seconds"], .5)
            self.assertIsNone(resolve_reference_video(motion, disabled=True))
            with self.assertRaises(FileNotFoundError):
                resolve_reference_video(motion, video=p/"missing.mp4")
            with self.assertRaises(ValueError):
                resolve_reference_video(motion, video=p/"base.mp4", start_seconds=float("nan"))

    def test_unconfigured_dance_keeps_ordinary_video(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            p = Path(directory)
            config = p/"list.json"
            config.write_text(json.dumps({"version": 1, "dances": {"unknown": {"video": None}}}))
            self.assertIsNone(resolve_reference_video(p/"unknown.h5", config_path=config))


class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = find_ffmpeg()
        if not cls.encoder:
            raise unittest.SkipTest("ffmpeg unavailable")

    def run_ffmpeg(self, *args):
        return subprocess.run([self.encoder, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *map(str,args)],
                              check=True, capture_output=True, timeout=30).stdout

    def test_real_composition_aligns_offset_freezes_end_preserves_audio_and_input(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"output") as directory:
            p = Path(directory)
            reference, validation = p/"参考 PMX.mp4", p/"validation.mp4"
            self.run_ffmpeg("-f", "lavfi", "-i", "color=red:s=160x240:r=10:d=0.2",
                "-f", "lavfi", "-i", "color=green:s=160x240:r=10:d=0.2",
                "-f", "lavfi", "-i", "color=blue:s=160x240:r=10:d=0.2",
                "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", reference)
            self.run_ffmpeg("-f", "lavfi", "-i", "color=yellow:s=480x270:r=10:d=1.2",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", validation)
            original = validation.read_bytes()
            clock = dict(fps=10., video_duration_s=1.2, source_start_s=.2, playback_rate=.5, source_frame_stride=1)
            validation.with_suffix(".json").write_text(json.dumps(clock))
            result = compose_reference_video(validation, dict(video=str(reference), start_seconds=0.),
                                             ffmpeg=self.encoder, emit=lambda _: None)
            output = result["video"]
            info = _probe_video(self.encoder, output)
            self.assertAlmostEqual(info["duration_s"], 1.2, places=1)
            self.assertEqual((info["width"],info["height"]), (720,270))
            # At half speed, source .2 (green) starts the clip; source .4 (blue)
            # appears later, then remains visible through the validation end hold.
            for seconds, channel in ((.05, 1), (.65, 2), (1.1, 2)):
                png = self.run_ffmpeg("-ss", seconds, "-i", output, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1")
                image = Image.open(io.BytesIO(png)).convert("RGB")
                pixel = image.getpixel((120,150))
                self.assertGreater(pixel[channel], max(v for i,v in enumerate(pixel) if i!=channel)+50)
                right = image.getpixel((480,150))
                self.assertGreater(min(right[:2]), 150)
            audio = self.run_ffmpeg("-i", output, "-vn", "-f", "s16le", "-ac", "1", "-ar", "8000", "pipe:1")
            samples = np.frombuffer(audio, dtype="<i2").astype(float)
            peak = np.argmax(abs(np.fft.rfft(samples)))*8000/len(samples)
            self.assertAlmostEqual(peak, 440., delta=5.)
            self.assertEqual(validation.read_bytes(), original)
            # Tail-only clips starting after the PMX ends still show its last frame.
            clock["source_start_s"] = 2.
            validation.with_suffix(".json").write_text(json.dumps(clock))
            tail = compose_reference_video(validation, dict(video=str(reference), start_seconds=0.),
                ffmpeg=self.encoder, output=p/"tail.mp4", emit=lambda _: None)
            self.assertLess(tail["reference_seek_s"], .6)
            high = compose_reference_video(validation, dict(video=str(reference), start_seconds=0.),
                ffmpeg=self.encoder, output=p/"high.mp4", emit=lambda _: None, output_height=540)
            high_info = _probe_video(self.encoder, high['video'])
            self.assertEqual((high_info['width'],high_info['height']), (1440,540))
            self.assertAlmostEqual(high_info['duration_s'],1.2,places=1)
            self.assertEqual(validation.read_bytes(),original)


if __name__ == "__main__":
    unittest.main()
