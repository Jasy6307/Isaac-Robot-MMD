"""Read-only regression against explicitly listed accepted local motion assets."""
from pathlib import Path
import json
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from source.train_workflow.utils.grounding.pipeline import audit_motion
from source.train_workflow.utils.grounding.motion import sha256


class AcceptedGroundingTests(unittest.TestCase):
    def test_six_accepted_clips_preserve_hard_gates(self):
        torch.set_num_threads(4)
        entries = json.loads((ROOT/'tests/fixtures/grounding/accepted_runs.json').read_text(encoding='utf-8'))
        for entry in entries:
            with self.subTest(clip=entry['name']):
                for key in ('reference', 'candidate'):
                    self.assertEqual(sha256(ROOT/entry[key]), entry[key+'_sha256'])
                report = audit_motion(ROOT/entry['reference'], ROOT/entry['candidate'],
                    ROOT/'assets/g1_29dof_o6_hand_V3.usd', geometry=entry['geometry'],
                    targets=ROOT/entry['prepared']/'targets.npz')
                self.assertTrue(report['passed'], report)
                self.assertEqual(report['frames'], entry['frames'])
                self.assertEqual(report['source_contact']['source_flight_frames'], entry['source_flight_frames'])
                self.assertAlmostEqual(report['minimum_proxy_clearance_m'], entry['minimum_clearance'], delta=1e-6)


if __name__ == '__main__':
    unittest.main()
