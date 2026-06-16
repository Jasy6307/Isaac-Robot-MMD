"""G1 29-DOF leg IK geometry constants from ``g1_29dof_mode_15_brainco_hand.urdf``.

Hip pivot is the left/right ``*_hip_pitch_joint`` origin in pelvis frame.
Thigh/shin lengths match the URDF hip_pitch->knee and knee->ankle_roll chains.
"""

from __future__ import annotations

# pelvis -> left_hip_pitch_joint origin
G1_FOOT_IK_HIP_OFFSET_Y_M = 0.064452
G1_FOOT_IK_HIP_OFFSET_Z_M = -0.1027

# |hip_pitch -> knee_joint| along hip_roll/yaw/knee chain (T-pose vector sum)
G1_FOOT_IK_THIGH_LENGTH_M = 0.340506

# |knee_joint -> ankle_roll_joint| along ankle_pitch + ankle_roll chain
G1_FOOT_IK_SHIN_LENGTH_M = 0.317568

# Viz-only: ignore tiny exceedances from near-full leg extension / numeric noise.
FOOT_IK_REACH_CLAMP_VIZ_MARGIN_M = 0.015


def g1_foot_ik_max_reach_m(*, max_reach_ratio: float = 1.0) -> float:
    return (G1_FOOT_IK_THIGH_LENGTH_M + G1_FOOT_IK_SHIN_LENGTH_M) * float(max_reach_ratio)
