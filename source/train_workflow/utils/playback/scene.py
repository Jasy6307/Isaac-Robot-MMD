"""Replay presentation settings, independent of robot motion and physics."""


def frame_full_body_replay(sim, root_position, *, width: int, height: int,
                           root_track=None) -> None:
    """Set one fixed view encompassing the complete root track plus body margin.

    Call once at recording start; camera position stays constant during motion.
    Raw/corrected recordings with the same root track use identical framing.
    """
    import numpy as np

    if hasattr(root_position, "detach"):
        root_position = root_position.detach().cpu().numpy()
    root = np.asarray(root_position, dtype=float).reshape(-1)
    if root.size < 3 or not np.isfinite(root[:3]).all():
        raise ValueError("Cannot frame replay without a finite root position")
    track = root[:3][None, :] if root_track is None else np.asarray(root_track, dtype=float)
    if track.ndim != 2 or track.shape[1] != 3 or not len(track) or not np.isfinite(track).all():
        raise ValueError("Cannot frame replay without a finite root track")
    lower, upper = track.min(axis=0), track.max(axis=0)
    center = (lower + upper) / 2
    # 4.2 m provides full-body clearance at the closest point in the dance.
    # Increase distance for sideways/vertical travel or narrower video formats.
    aspect = max(1, width) / max(1, height)
    distance = 4.2 * max(1.0, (upper[0]-lower[0]+2.4)/2.4,
                         (upper[2]-lower[2]+2.4)/2.4) * max(1.0, (16/9)/aspect)
    target = (float(center[0]), float(center[1]), float(center[2]) + 0.1)
    eye = (target[0], float(upper[1]) + distance, target[2] + 0.15)
    sim.set_camera_view(
        eye=eye, target=target
    )
    print(f"[INFO] Recording camera fixed: eye={eye}, target={target}")
