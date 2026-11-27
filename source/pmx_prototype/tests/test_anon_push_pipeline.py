"""Test hard acceptance and isolated retries without launching Isaac Lab."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.push_pipeline import campaign as pipeline
from training.push_pipeline.campaign import Campaign, guard


class PushPipelineTests(unittest.TestCase):
    def test_all_hard_gates_are_required(self):
        medium = dict(total=896, recovered=833)
        clean = dict(passed=True, segments=38, falls=0)
        self.assertTrue(guard(medium, clean))
        for result in (dict(medium, recovered=832), dict(medium, total=895)):
            self.assertFalse(guard(result, clean))
        for result in (dict(clean, passed=False), dict(clean, segments=37), dict(clean, falls=1)):
            self.assertFalse(guard(medium, result))

    def test_resume_reuses_completed_stages_and_keeps_failed_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signature = {'num_envs': 2048}
            campaign = Campaign(root, signature, False)
            complete = campaign.stage('complete', lambda work: (work/'kept').touch())

            def fail(work):
                (work/'partial').touch()
                raise RuntimeError('interrupted')

            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                campaign.stage('interrupted', fail)
            resumed = Campaign(root, signature, True)
            self.assertEqual(resumed.stage('complete', lambda work: self.fail('ran twice')), complete)
            second = resumed.stage('interrupted', lambda work: (work/'done').touch())
            self.assertEqual(second.name, 'attempt_02')
            self.assertTrue((second.parent/'attempt_01/partial').is_file())
            with self.assertRaises(ValueError):
                Campaign(root, {'num_envs': 96}, True)

    def test_resume_migrates_only_the_exact_source_cleanup(self):
        migration = pipeline.read(pipeline.WORKERS/'resume_source_migration.json')
        common = {'num_envs': 8192, 'course': 'mixed4', 'reference_hash': 'unchanged'}
        before = common | {str((pipeline.SOURCE / path).resolve()): value
                           for path, value in migration['before'].items()}
        after = common | {str((pipeline.SOURCE / path).resolve()): value
                          for path, value in migration['after'].items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = Campaign(root, before, False)
            complete = original.stage('complete', lambda work: (work/'kept').touch())
            changed_source = {str(Path(pipeline.__file__).resolve()): 'changed'}
            for signature in (after | {'num_envs': 2048}, after | {'reference_hash': 'changed'},
                              after | changed_source):
                with self.assertRaises(ValueError):
                    Campaign(root, signature, True)
            self.assertEqual(pipeline.read(original.journal)['inputs_sha256'], before)
            resumed = Campaign(root, after, True)
            self.assertEqual(resumed.stage('complete', lambda work: self.fail('ran twice')), complete)
            self.assertEqual(pipeline.read(resumed.journal)['inputs_sha256'], after)


if __name__ == '__main__':
    unittest.main()
