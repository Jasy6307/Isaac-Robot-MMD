"""Verify that only targeted side pushes use their waveform's magnitude band."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_layout
from anon_push_bands import DirectionalPushBands
from anon_push_waveforms import PROFILES


class PushBandsTests(unittest.TestCase):
    def test_direction_waveform_and_extreme_are_independent(self):
        specs = {
            name: {waveform: dict(challenge_m_s=(center+offset, center+offset+.02),
                                  extreme_m_s=(center+offset+.04, center+offset+.08), extreme_probability=.5)
                   for offset, waveform in zip((0., .015, .030, .045), PROFILES)}
            for name, center in (('left', .42), ('right', .53))}
        bands = DirectionalPushBands(specs, 'cpu')
        values, extreme = bands.choose(torch.tensor([.5, .5, .25, .7]), torch.tensor([True, False, False, True]),
            torch.tensor([True, True, False, False]), torch.tensor([True, False, True, True]),
            torch.tensor([0, 2, 0, 3]), torch.tensor([.5, .5, .5, .5]), torch.tensor([.9, .2, .1, .99]))
        torch.testing.assert_close(values, torch.tensor([.43, .62, .25, .7]))
        self.assertEqual(extreme.tolist(), [False, True, False, True])
        counts = bands.summary()
        self.assertEqual(counts['left']['impulse']['count'], 1)
        self.assertEqual(counts['right']['ramp']['at_least_0.6'], 1)
        self.assertEqual(counts['left']['double']['count'], 0)


if __name__ == '__main__':
    unittest.main()
