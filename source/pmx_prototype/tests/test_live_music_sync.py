"""Drift tolerance controls seeking without changing WAV playback speed."""
import sys
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
from live_music import LiveMusic

class Channel:
    def get_busy(self):return True
    def stop(self):pass

class MusicSyncTest(TestCase):
    def test_seek_uses_exact_original_pcm(self):
        music=LiveMusic.__new__(LiveMusic)
        music.pcm=bytes(range(80));music.rate=10;music.channels=2;music.samples=20
        music.pygame=SimpleNamespace(mixer=SimpleNamespace(Sound=lambda *,buffer:bytes(buffer)))
        music.channel=SimpleNamespace(stop=lambda:None,play=lambda sound:None)
        music.whole_sound=music.pcm;music.starts=0;music.events=[]
        music._start(.3,'test')
        self.assertEqual(music.current_sound,music.pcm[12:])
        self.assertEqual(music.origin_motion,.3)

    def test_tolerance_and_restart(self):
        music=LiveMusic.__new__(LiveMusic)
        music.enabled=True;music.duration=100.;music.origin_wall=0.;music.origin_motion=0.
        music.max_drift=0.;music.resyncs=0;music.channel=Channel();music.drift_tolerance=.25
        events=[]
        music._start=lambda position,reason: events.append((position,reason))
        with patch('live_music.time.perf_counter',return_value=1.):
            music.update(.8)
            self.assertFalse(events)
            music.update(.7)
            self.assertEqual(events[-1],(.7,'drift_resync'))
            self.assertEqual(music.resyncs,1)
            music.update(0.,restart=True)
            self.assertEqual(events[-1],(0.,'dance_reset_or_loop'))

if __name__=='__main__':main()
