"""Import compatibility for recorded Anon push experiments."""

import sys

if __spec__ is None:
    raise SystemExit('The standalone basic push course has been retired.')

from training.push_pipeline import campaign

sys.modules[__name__] = campaign
