"""Silent real-device test: enable, disable, sample-accurate seek and reset."""

# Allow both the canonical tool path and the historical root entry point.
import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from pathlib import Path
from live_music import LiveMusic

root = PROJECT_ROOT
music = LiveMusic(root/'media/dance/ji_le_jing_tu/ji_le_jing_tu.wav', volume=0)
try:
    music.set_enabled(True, 0)
    assert music.channel.get_busy()
    music.set_enabled(False, 1)
    assert not music.channel.get_busy()
    music.set_enabled(True, 12.5)
    assert music.channel.get_busy()
    assert abs(music.origin_motion-12.5) < 1/music.rate
    music.update(0, restart=True)
    assert music.channel.get_busy() and music.origin_motion == 0
    print('LIVE_MUSIC_CHECK_PASS ' + str(music.status()), flush=True)
finally:
    music.close()
