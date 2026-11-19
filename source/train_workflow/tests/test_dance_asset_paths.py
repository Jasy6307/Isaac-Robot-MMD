"""Folder migration compatibility checks without starting Isaac Sim."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from source import paths
from source.train_workflow.utils.motion.resolve import resolve_dance_h5_by_name, resolve_training_motion
from source.train_workflow.utils.motion.audio import resolve_motion_audio
from source.train_workflow.utils.motion import audio as audio_module
from source.train_workflow.utils.validation_reference import resolve_reference_video
from source.train_workflow.utils.motion import sync
from source.train_workflow import g1_vmd_pipeline as pipeline


class DanceAssetPathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "output")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "media/dance"
        self.folder = self.root / "dance_one"
        self.folder.mkdir(parents=True)
        patcher = patch.object(paths, "DANCE_DIR", str(self.root))
        patcher.start()
        self.addCleanup(patcher.stop)

    def asset(self, filename, content=b"test"):
        path = self.folder / filename
        path.write_bytes(content)
        return path

    def test_legacy_paths_and_new_paths_resolve_to_same_file(self):
        for filename in ("dance_one.vmd", "dance_one_short.wav", "dance_one_v7.state.npz"):
            target = self.asset(filename)
            self.assertEqual(paths.resolve_dance_asset(self.root / filename), target)
            self.assertEqual(paths.resolve_dance_asset(target), target)
            self.assertEqual(paths.resolve_dance_asset(str(self.root / filename).replace("/", "\\")), target)

    def test_existing_flat_assets_and_unrelated_paths_win(self):
        self.asset("dance_one.h5")
        flat = self.root / "dance_one.h5"
        flat.touch()
        self.assertEqual(paths.resolve_dance_asset(flat), flat)
        unrelated = self.base / "elsewhere/dance_one.h5"
        self.assertEqual(paths.resolve_dance_asset(unrelated), unrelated)
        typo = self.root / "wrong/dance_one.h5"
        self.assertEqual(paths.resolve_dance_asset(typo), typo)

    def test_duplicate_filename_requires_explicit_folder(self):
        target = self.asset("custom.h5")
        other = self.root / "other"
        other.mkdir()
        (other / target.name).touch()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            paths.resolve_dance_asset(self.root / target.name)
        self.assertEqual(paths.resolve_dance_asset(target), target)

    def test_new_outputs_use_longest_matching_dance(self):
        short = self.root / "dance_one_short"
        short.mkdir()
        self.assertEqual(paths.resolve_dance_asset(self.root / "dance_one_short_raw.h5"), short / "dance_one_short_raw.h5")
        self.assertEqual(paths.resolve_dance_asset(self.root / "dance_one_collision_v99.h5"), self.folder / "dance_one_collision_v99.h5")

    def test_h5_name_lookup_preserves_z_edited_priority_and_variants(self):
        base = self.asset("dance_one.h5")
        edited = self.asset("dance_one_z_editted.h5")
        result, canonical = resolve_dance_h5_by_name("dance_one", dance_dir=str(self.root), compile_from_csv=False)
        self.assertEqual(Path(result), edited)
        self.assertEqual(canonical, "dance_one")
        edited.unlink()
        result, _ = resolve_dance_h5_by_name("dance_one", dance_dir=str(self.root), compile_from_csv=False)
        self.assertEqual(Path(result), base)
        variant = self.asset("dance_one_collision_v7.h5")
        result, _ = resolve_dance_h5_by_name(variant.stem, dance_dir=str(self.root), compile_from_csv=False)
        self.assertEqual(Path(result), variant)

    def test_csv_compilation_stays_in_dance_folder(self):
        csv = self.asset("dance_one_z_editted.csv")
        def compile_csv(source, output):
            self.assertEqual(Path(source), csv)
            Path(output).touch()
        with patch.object(sync, "compile_dance_csv_to_h5", side_effect=compile_csv):
            result, _ = resolve_dance_h5_by_name("dance_one", dance_dir=str(self.root))
        self.assertEqual(Path(result), csv.with_suffix(".h5"))

    def test_explicit_training_reference_takes_priority_over_dance_name(self):
        self.asset("dance_one_z_editted.h5")
        prepared = self.base / "arm_stabilized.h5"
        prepared.touch()
        with patch("source.train_workflow.utils.motion.resolve.default_dance_dir", return_value=str(self.root)):
            self.assertEqual(resolve_training_motion("dance_one", str(prepared)), (str(prepared), "dance_one"))
            self.assertEqual(resolve_training_motion(None, str(prepared)), (str(prepared), "arm_stabilized"))
            with self.assertRaises(FileNotFoundError):
                resolve_training_motion("dance_one", str(self.base / "missing.h5"))

    def test_recursive_sync_keeps_siblings_and_registers_nested_paths(self):
        self.asset("dance_one.vmd")
        self.asset("dance_one.csv")
        self.asset("dance_one.h5")
        self.asset("dance_one.wav")
        self.asset("dance_one_hand.vmd")
        config = self.base / "dances.yaml"
        registered = sync.sync_dance_assets_from_vmd(dance_dir=str(self.root), dances_config_path=str(config), media_dir=str(self.root.parent))
        self.assertEqual(registered, ["dance_one"])
        self.assertIn("dance/dance_one/dance_one.csv", config.read_text())
        self.assertIn("dance/dance_one/dance_one.wav", config.read_text())
        self.assertEqual(list(self.root.glob("*.h5")), [])
        self.assertEqual(len(list(paths.iter_dance_assets("*.vmd"))), 2)

    def test_audio_sidecars_and_old_pipeline_manifests_survive_move(self):
        motion = self.asset("dance_one_custom.h5")
        audio = self.asset("dance_one.wav")
        legacy = self.root / motion.name
        record = dict(h5=str(legacy), audio=str(self.root / audio.name))
        sidecar = motion.with_suffix(".motion.json")
        sidecar.write_text(json.dumps(record))
        self.assertEqual(resolve_motion_audio(legacy, recordings_dir=self.base), str(audio))
        sidecar.unlink()
        (self.base / "old.pipeline.json").write_text(json.dumps(record))
        self.assertEqual(resolve_motion_audio(motion, recordings_dir=self.base), str(audio))

    def test_reference_video_uses_old_motion_provenance_and_video_path(self):
        motion = self.asset("dance_one_custom.h5")
        motion.with_suffix(".motion.json").write_text(json.dumps(dict(vmd=str(self.root / "dance_one.vmd"))))
        video = self.asset("dance_one_pmx.mp4")
        config = self.base / "videos.json"
        config.write_text(json.dumps(dict(version=1, dances={"dance_one": dict(video=str(self.root / video.name))})))
        reference = resolve_reference_video(str(self.root / motion.name), config_path=config)
        self.assertEqual(reference["video"], str(video))

    def test_old_corrected_h5_without_sidecar_uses_own_folder_music(self):
        motion = self.asset("dance_one_collision_corrected_v7.h5")
        audio = self.asset("dance_one.ogg")
        with patch.object(audio_module, "DANCE_DIR", str(self.root)):
            self.assertEqual(resolve_motion_audio(motion, recordings_dir=self.base), str(audio))

    def test_pipeline_default_and_old_root_output_stay_with_dance(self):
        vmd = self.asset("dance_one_short.vmd")
        with patch.object(pipeline, "DANCE_DIR", str(self.root)):
            self.assertEqual(pipeline.dance_output_directory(vmd), self.folder)
            self.assertEqual(pipeline.dance_output_directory(vmd, self.root), self.folder)
            self.assertEqual(pipeline.dance_output_directory(vmd, self.base), self.base)
            self.assertEqual(pipeline.dance_output_directory(self.base / "new.vmd"), self.root / "new")


if __name__ == "__main__":
    unittest.main()
