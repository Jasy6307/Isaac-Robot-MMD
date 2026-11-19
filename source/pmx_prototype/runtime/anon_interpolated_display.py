"""Present adjacent solved poses without making the 60 Hz clock drive physics."""

import math
import time

import numpy as np

from dance_pose_interpolation import PoseInterpolator


class InterpolatedDisplay:
    def __init__(self, draw, clock=time.perf_counter, sleep=time.sleep):
        self.draw = draw
        self.clock = clock
        self.sleep = sleep
        self.period = 1 / 60
        self.deadline = clock()
        self.poses = None
        self.times = []
        self.source_frames = []

    def push(self, frame, p, q, center, heading, restart=False):
        pose = dict(frame=frame, p=p, q=q, center=center, heading=heading, restart=restart)
        now = self.clock()
        if self.poses is None:
            self.poses = PoseInterpolator(pose, now)
            self.deadline = now
        else:
            self.poses.push(pose, now)

    def present_if_due(self):
        now = self.clock()
        if self.poses is None or now < self.deadline:
            return
        # Advance past missed render ticks without running catch-up renders.
        self.deadline += (int((now - self.deadline) / self.period) + 1) * self.period
        pose = self.poses.sample(now)
        previous, current = self.poses.previous, self.poses.current
        span = current['frame'] - previous['frame']
        alpha = (pose['frame'] - previous['frame']) / span if span else 1.
        center = previous['center'] * (1. - alpha) + current['center'] * alpha
        turn = math.remainder(current['heading'] - previous['heading'], math.tau)
        heading = previous['heading'] + alpha * turn
        self.draw(pose['p'], pose['q'], center, heading, pose['frame'])
        self.times.append(self.clock())
        self.source_frames.append(pose['frame'])

    def wait_until(self, deadline):
        if self.poses is None:
            self.sleep(max(0., deadline - self.clock()))
            return
        while self.clock() < deadline:
            self.present_if_due()
            remaining = min(deadline, self.deadline) - self.clock()
            if remaining > 0:
                self.sleep(remaining)

    def intervals_ms(self, warmup_policy_frames):
        intervals = np.diff(self.times) * 1000
        start = np.searchsorted(self.source_frames, warmup_policy_frames)
        return intervals[min(start, max(0, len(intervals) - 1)):]

    def finish(self):
        """A finite preview should end on its last solved pose."""
        if self.poses is None:
            return
        self.wait_until(self.poses.started + self.poses.duration)
        self.sleep(max(0., self.deadline - self.clock()))
        self.present_if_due()
