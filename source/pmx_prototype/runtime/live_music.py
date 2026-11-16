"""Seekable PCM accompaniment for the live dance, independent of render FPS.

SDL_mixer's WAV music seeking varies by backend. Play decoded PCM on our own
channel instead, slicing at a sample boundary when the user enables or resets.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from pathlib import Path
import os
import time
import wave

import sys
sys.path.insert(0, str(PROJECT_ROOT))
from source.paths import resolve_dance_asset


class LiveMusic:
    def __init__(self, path, volume=.2, drift_tolerance=.15):
        os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
        import pygame
        self.pygame = pygame
        self.path = resolve_dance_asset(path)
        with wave.open(str(self.path), 'rb') as wav:
            if wav.getcomptype() != 'NONE' or wav.getsampwidth() != 2:
                raise ValueError('Accompaniment must be an uncompressed 16-bit PCM WAV')
            self.rate, self.channels, self.samples = wav.getframerate(), wav.getnchannels(), wav.getnframes()
            self.pcm = wav.readframes(self.samples)
        self.duration = self.samples/self.rate
        self.owns_mixer = pygame.mixer.get_init() is None
        if self.owns_mixer:
            pygame.mixer.init(frequency=self.rate, size=-16, channels=self.channels, buffer=512, allowedchanges=0)
        if pygame.mixer.get_init() != (self.rate, -16, self.channels):
            raise RuntimeError('Audio mixer format does not match this WAV')
        channel_id = pygame.mixer.get_num_channels()
        pygame.mixer.set_num_channels(channel_id+1)
        self.channel = pygame.mixer.Channel(channel_id)
        self.volume = max(0., min(1., volume))
        self.drift_tolerance = float(drift_tolerance)
        if self.drift_tolerance <= 0:
            raise ValueError('drift_tolerance must be positive')
        self.channel.set_volume(self.volume)
        self.whole_sound = pygame.mixer.Sound(buffer=self.pcm)
        self.current_sound = None
        self.enabled = False
        self.starts = self.resyncs = 0
        self.origin_motion = self.origin_wall = 0.
        self.max_drift = 0.
        self.events = []

    def _start(self, position, reason):
        self.channel.stop()
        position = max(0., float(position))
        sample = min(self.samples, int(position*self.rate))
        if sample == self.samples:
            self.current_sound = None
            return
        offset = sample*self.channels*2
        # Avoid copying the entire remaining WAV into an intermediate bytes
        # object each time a seek is required; SDL still receives the same PCM.
        self.current_sound = self.whole_sound if offset == 0 else self.pygame.mixer.Sound(buffer=memoryview(self.pcm)[offset:])
        self.channel.play(self.current_sound)
        self.origin_motion = sample/self.rate
        self.origin_wall = time.perf_counter()
        self.starts += 1
        self.events.append(dict(reason=reason, motion_seconds=self.origin_motion))

    def set_enabled(self, enabled, position):
        enabled = bool(enabled)
        if enabled == self.enabled:
            return
        self.enabled = enabled
        if enabled:
            self._start(position, 'enabled')
        else:
            self.channel.stop()
            self.current_sound = None
            self.events.append(dict(reason='disabled', motion_seconds=float(position)))

    def update(self, position, restart=False):
        if not self.enabled:
            return
        if restart:
            self._start(position, 'dance_reset_or_loop')
            return
        if position >= self.duration:
            self.channel.stop()
            return
        elapsed = time.perf_counter()-self.origin_wall
        drift = self.origin_motion+elapsed-position
        self.max_drift = max(self.max_drift, abs(drift))
        if (abs(drift) > self.drift_tolerance and elapsed > .5) or not self.channel.get_busy():
            self.resyncs += 1
            self._start(position, 'drift_resync')

    def status(self):
        return dict(file=str(self.path), enabled=self.enabled, playing=bool(self.channel.get_busy()),
                    backend='pygame PCM channel', sample_rate=self.rate, duration_seconds=self.duration,
                    starts=self.starts, resyncs=self.resyncs, max_estimated_drift_seconds=self.max_drift,
                    last_start_motion_seconds=self.origin_motion, volume=self.volume,
                    drift_tolerance_seconds=self.drift_tolerance, events=self.events)

    def close(self):
        self.channel.stop()
        self.enabled = False
        self.current_sound = self.whole_sound = None
        if self.owns_mixer:
            self.pygame.mixer.quit()
