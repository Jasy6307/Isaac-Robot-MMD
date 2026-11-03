"""Provenance audio resolution and preview synchronization; no Isaac startup."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from source.train_workflow.utils.motion.audio import resolve_motion_audio, ffmpeg_audio_args


class AudioTests(unittest.TestCase):
    def test_versioned_custom_name_uses_provenance(self):
        parent = ROOT / "output/motion-audio-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root = Path(temp)
            audio = root / "IRIS_OUT.wav"
            audio.write_bytes(b"music")
            motion = root / "custom_collision_corrected_v3.h5"
            motion.write_bytes(b"motion")
            motion.with_suffix(".motion.json").write_text(json.dumps(dict(audio=audio.name)), encoding="utf-8")
            self.assertEqual(resolve_motion_audio(motion, recordings_dir=root), str(audio))
            motion.with_suffix(".motion.json").unlink()
            manifest = root / "custom.pipeline.json"
            manifest.write_text(json.dumps(dict(h5=str(motion), audio=str(audio))), encoding="utf-8")
            self.assertEqual(resolve_motion_audio(motion, recordings_dir=root), str(audio))
            unrelated = root / "another_v2.h5"
            self.assertIsNone(resolve_motion_audio(unrelated, recordings_dir=root))
            # Missing recorded music must not resolve to a different dance.
            audio.unlink()
            self.assertIsNone(resolve_motion_audio(motion, recordings_dir=root))

    def test_original_dance_and_z_edited(self):
        parent = ROOT / "output/motion-audio-tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root = Path(temp)
            audio = root / "original.wav"
            audio.write_bytes(b"music")
            for stem in ("original", "original_z_editted"):
                self.assertEqual(resolve_motion_audio(root / f"{stem}.h5", recordings_dir=root), str(audio))

    def test_offset_and_slow_motion(self):
        args = ffmpeg_audio_args("music.wav", start_seconds=5., playback_rate=.25)
        self.assertEqual(args[:4], ["-ss", "5.0", "-i", "music.wav"])
        self.assertEqual(args[-1], "atempo=0.5,atempo=0.5,apad")
        self.assertEqual(ffmpeg_audio_args("music.wav")[-1], "apad")


if __name__ == "__main__":
    unittest.main()
