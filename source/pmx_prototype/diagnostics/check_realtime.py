"""Check live velocity continuity, batch parity, reset and live parameter response."""

# Allow both the canonical tool path and the historical root entry point.
import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from pathlib import Path
import numpy as np
from realtime_xpbd import RealtimeXPBD, InterpolatedXPBD, collider_poses
from secondary_motion import simulate

root = PROJECT_ROOT
s = RealtimeXPBD(root/'output/pmx_prototype/ji_le_jing_tu.usdc',
                 root/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx')
targets, ca, cb = [], [], []
for frame in range(64):
    p, q, wp, wq = s.body_pose(frame)
    targets.append(wp[s.nodes])
    a, b = collider_poses(wp, wq, s.collider_bones, s.collider_offsets, s.collider_axes)
    ca.append(a)
    cb.append(b)
system = s.system
batch, _, velocity = simulate(np.array(targets), np.array(ca), np.array(cb), system['radii'],
    system['invmass'], system['edge_array'], system['lengths'], system['compliance'],
    system['groups'], system['thickness'], 4, 12)
live, reference_p, reference_q = [], [], []
for frame in range(64):
    p, q = s.advance(frame)
    live.append(s.position.copy())
    reference_p.append(p.copy())
    reference_q.append(q.copy())
error = float(np.max(np.abs(np.array(live)-batch)))
assert error < 1e-8, error
assert np.max(np.abs(s.velocity-velocity)) < 1e-8
normal = s.position.copy()
s.reset(0)
for frame in range(64):
    s.advance(frame)
assert np.max(np.abs(s.position-normal)) < 1e-8
s.reset(0)
for frame in range(64):
    if frame == 32:
        s.gravity = 0.0
    s.advance(frame)
response = float(np.max(np.abs(s.position-normal)))
assert response > 1e-4, response
s.gravity = 9.81
stream = InterpolatedXPBD(s)
interpolated_motion = 0.0
for tick in range(127):
    p, q = stream.sample(tick/2)
    assert np.isfinite(p).all() and np.isfinite(q).all()
    assert np.max(np.abs(np.linalg.norm(q, axis=1)-1)) < 1e-5
    if tick % 2 == 0:
        assert np.max(np.abs(p-reference_p[tick//2])) < 1e-8
        assert np.max(np.abs(q-reference_q[tick//2])) < 1e-8
    else:
        interpolated_motion = max(interpolated_motion, float(np.max(np.abs(p-reference_p[tick//2]))))
assert stream.updates == 64
assert np.max(np.abs(s.position-batch[-1])) < 1e-8
assert interpolated_motion > 1e-5
print(f'REALTIME_CHECK_PASS batch_live_max_error_m={error:.3g} live_gravity_response_m={response:.6f} '
      f'render_samples=127 physics_batches={stream.updates} interpolated_motion_m={interpolated_motion:.6f}')
