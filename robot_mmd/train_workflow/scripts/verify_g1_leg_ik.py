# -*- coding: utf-8 -*-
"""Compare FK-only vs full 6-DOF IK red->ankle errors on CSV motion."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CSV = _REPO / "robot_mmd" / "media" / "dance" / "gokurakujyodo.csv"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from robot_mmd.train_workflow.g1_joint_axis_map_raw import (
    MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT,
    MMD_ROOT_QUAT_RPY_SCALE_DEFAULT,
)
from robot_mmd.train_workflow.retarget_unitreeG1 import euler_xyz_rad_waist_extrinsic
from robot_mmd.train_workflow.utils.csv_motion_loader import (
    FootIkConfig,
    FootIkState,
    build_joint_positions_from_frame,
    get_bone_frame_lists,
    get_frame_indices,
    interpolate_bone,
    load_csv_motion,
    update_foot_ik_mmd_viz_world,
)
from robot_mmd.train_workflow.utils.g1_leg_kinematics import g1_leg_fk_pos
from robot_mmd.train_workflow.utils.mmd_fk import default_foot_ik_viz_config
from robot_mmd.train_workflow.utils.trans_util import (
    mmd_root_offset_quat_to_world,
    quat_from_waist_extrinsic_xyz,
    quat_mul,
    quat_normalize,
    remap_root_csv_euler_xyz,
    root_local_to_isaac_world,
)


def _csv_root_quat(frame: int, frames, bfl) -> list[float] | None:
    for require_dynamic in (True, False):
        for bone in ("下半身", "グルーブ", "センター親", "腰", "センター"):
            kfs = bfl.get(bone) or []
            if require_dynamic and len(kfs) <= 1:
                continue
            d = interpolate_bone(frame, bone, frames, kfs)
            if d and d.get("quat_wxyz"):
                return quat_normalize([float(v) for v in d["quat_wxyz"]])
    return None


def _root_pose(frame: int, frames, bfl, ox, oy, oz, rq0, groove: float):
    c, g = bfl.get("センター") or [], bfl.get("グルーブ") or []
    order = ("センター", "グルーブ") if len(c) > len(g) else ("グルーブ", "センター")
    rm = None
    for bone in order:
        d = interpolate_bone(frame, bone, frames, bfl.get(bone))
        if d and "pos" in d:
            rm = d
            break
    if rm is None:
        gx, gy, gz = 0.0, 0.0, 0.0
    else:
        gx, gy, gz = rm["pos"]
    rp = (ox - float(gx) * groove, oy - float(gy) * groove, oz + float(gz) * groove)
    rq = list(rq0)
    csv_q = _csv_root_quat(frame, frames, bfl)
    if csv_q:
        qw = mmd_root_offset_quat_to_world(csv_q)
        qx, qy, qz, qw_w = qw[1], qw[2], qw[3], qw[0]
        rr, rp2, ry = euler_xyz_rad_waist_extrinsic((qx, qy, qz, qw_w))
        orr, opr, oyr = remap_root_csv_euler_xyz(
            rr, rp2, ry, MMD_ROOT_QUAT_RPY_AXIS_IDX_DEFAULT, MMD_ROOT_QUAT_RPY_SCALE_DEFAULT
        )
        rq = quat_normalize(quat_mul(quat_from_waist_extrinsic_xyz(orr, opr, oyr), rq0))
    return rp, rq


def _leg_joint_names() -> list[str]:
    return [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]


def _ankle_world_from_q(
    joint_names: list[str],
    q: list[float],
    side: str,
    root_pos,
    root_quat,
) -> tuple[float, float, float]:
    keys = (
        f"{side}_hip_pitch_joint",
        f"{side}_hip_roll_joint",
        f"{side}_hip_yaw_joint",
        f"{side}_knee_joint",
        f"{side}_ankle_pitch_joint",
        f"{side}_ankle_roll_joint",
    )
    jidx = {n: i for i, n in enumerate(joint_names)}
    q6 = tuple(float(q[jidx[k]]) for k in keys)
    local = g1_leg_fk_pos(q6, side=side)
    return root_local_to_isaac_world(local, root_pos, root_quat)


def _eval_frame(
    frame: int,
    *,
    frames,
    bfl,
    bones,
    joint_names,
    default_joint_pos,
    ox,
    oy,
    oz,
    rq0,
    groove: float,
    use_ik: bool,
) -> tuple[float, float, float | None]:
    fd = {b: interpolate_bone(frame, b, frames, bfl.get(b)) for b in bones}
    fd = {k: v for k, v in fd.items() if v is not None}
    rp, rq = _root_pose(frame, frames, bfl, ox, oy, oz, rq0, groove)
    st = FootIkState()
    viz = default_foot_ik_viz_config()
    update_foot_ik_mmd_viz_world(
        st, fd, groove, foot_ik_viz_cfg=viz, target_root_pos=rp, target_root_quat_wxyz=rq, frames=frames
    )
    cfg = FootIkConfig(enable=bool(use_ik))

    q = build_joint_positions_from_frame(
        fd,
        joint_names,
        default_joint_pos,
        foot_ik_cfg=cfg,
        foot_ik_state=st,
        foot_ik_frame_idx=frame,
        foot_ik_root_pos_world=rp,
        foot_ik_root_quat_wxyz=rq,
        foot_ik_viz_cfg=viz,
    )
    red_l = st.last_left_foot_mmd_viz_world
    red_r = st.last_right_foot_mmd_viz_world
    if red_l is None or red_r is None:
        return float("nan"), float("nan"), st.last_left_ik_residual_m
    ankle_l = _ankle_world_from_q(joint_names, q.tolist(), "left", rp, rq)
    ankle_r = _ankle_world_from_q(joint_names, q.tolist(), "right", rp, rq)
    err_l = math.sqrt(sum((red_l[i] - ankle_l[i]) ** 2 for i in range(3)))
    err_r = math.sqrt(sum((red_r[i] - ankle_r[i]) ** 2 for i in range(3)))
    return err_l, err_r, st.last_left_ik_residual_m


def main() -> None:
    p = argparse.ArgumentParser(description="Verify G1 leg IK on CSV motion")
    p.add_argument(
        "--csv",
        type=str,
        default=str(_DEFAULT_CSV),
    )
    p.add_argument("--frame-step", type=int, default=120)
    p.add_argument("--ox", type=float, default=0.0463)
    p.add_argument("--oy", type=float, default=-0.0004)
    p.add_argument("--oz", type=float, default=0.7441)
    args = p.parse_args()

    frames = load_csv_motion(args.csv)
    fl = get_frame_indices(frames)
    bones = set()
    for f in frames.values():
        bones.update(f.keys())
    bfl = get_bone_frame_lists(frames, fl, bones)
    joint_names = _leg_joint_names()
    default_joint_pos = __import__("numpy").zeros(len(joint_names))
    rq0 = [0.0066, 0.0002, -0.0116, 0.9999]
    groove = 0.1
    max_frame = fl[-1] if fl else 0

    stats: dict[str, list[float]] = {"fk_l": [], "full_l": []}
    for frame in range(0, max_frame + 1, max(1, int(args.frame_step))):
        fk_l, fk_r, _ = _eval_frame(
            frame,
            frames=frames,
            bfl=bfl,
            bones=bones,
            joint_names=joint_names,
            default_joint_pos=default_joint_pos,
            ox=args.ox,
            oy=args.oy,
            oz=args.oz,
            rq0=rq0,
            groove=groove,
            use_ik=False,
        )
        fl_err, fr_err, res = _eval_frame(
            frame,
            frames=frames,
            bfl=bfl,
            bones=bones,
            joint_names=joint_names,
            default_joint_pos=default_joint_pos,
            ox=args.ox,
            oy=args.oy,
            oz=args.oz,
            rq0=rq0,
            groove=groove,
            use_ik=True,
        )
        if math.isfinite(fk_l):
            stats["fk_l"].append(fk_l)
        if math.isfinite(fl_err):
            stats["full_l"].append(fl_err)
        print(f"f={frame:5d} fkL={fk_l:.4f} fullL={fl_err:.4f} ikRes={res}")

    def _summary(name: str, vals: list[float]) -> str:
        if not vals:
            return f"{name}: (no data)"
        vals_sorted = sorted(vals)
        p95 = vals_sorted[int(0.95 * (len(vals_sorted) - 1))]
        return f"{name}: mean={sum(vals)/len(vals):.4f} p95={p95:.4f} max={max(vals):.4f}"

    print("\n=== Summary (left red->ankle, meters) ===")
    print(_summary("FK", stats["fk_l"]))
    print(_summary("full", stats["full_l"]))

    tpose = g1_leg_fk_pos((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), side="left")
    print(f"\nT-pose left ankle (pelvis frame): {tpose}")


if __name__ == "__main__":
    main()
