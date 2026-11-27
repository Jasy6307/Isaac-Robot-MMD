"""Side-specific impulse distributions for standing recovery training."""

import math

import torch

from anon_push_waveforms import PROFILES


SIDES = ('left', 'right')
LEVEL_THRESHOLDS = (0., .50, .525, .55, .60, .65, .70)


class DirectionalPushBands:
    def __init__(self, settings, device):
        if set(settings) != set(SIDES) or any(set(settings[side]) != set(PROFILES) for side in SIDES):
            raise ValueError('Provide both standing directions and every force profile')
        rows = []
        for side in SIDES:
            for waveform in PROFILES:
                band = settings[side][waveform]
                low, high = map(float, band['challenge_m_s'])
                lower, upper = map(float, band['extreme_m_s'])
                probability = float(band['extreme_probability'])
                if not (0. < low <= high <= lower <= upper < math.inf and 0. <= probability <= 1.):
                    raise ValueError(f'Invalid standing push band for {side}/{waveform}')
                rows.append((low, high, lower, upper, probability))
        self.ranges = torch.tensor(rows, device=device)
        self.thresholds = torch.tensor(LEVEL_THRESHOLDS, device=device)
        self.counts = torch.zeros((len(rows), len(LEVEL_THRESHOLDS)), device=device)

    def choose(self, original, original_extreme, targeted, leftward, waveform, strength, mixture):
        index = (~leftward).long()*len(PROFILES) + waveform
        low, high, lower, upper, probability = self.ranges[index].unbind(-1)
        hard = mixture < probability
        start = torch.where(hard, lower, low)
        end = torch.where(hard, upper, high)
        impulse = torch.where(targeted, start+(end-start)*strength, original)
        observed = targeted[:, None] & (impulse[:, None] >= self.thresholds)
        self.counts.index_add_(0, index, observed.to(self.counts.dtype))
        return impulse, torch.where(targeted, hard, original_extreme)

    def summary(self):
        counts = self.counts.tolist()
        keys = ('count', *(f'at_least_{value:g}' for value in LEVEL_THRESHOLDS[1:]))
        return {side: {waveform: dict(zip(keys, map(int, counts[index*len(PROFILES)+column])))
                       for column, waveform in enumerate(PROFILES)} for index, side in enumerate(SIDES)}
