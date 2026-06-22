"""Custom action terms for G1 dance tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from source.my_task.mdp.joint_groups import resolve_joint_ids
from source.my_task.mdp.root_reference import write_root_reference_from_motion
from source.my_task.motion_reference import get_or_create_motion_buffer, motion_steps

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ReferenceFrozenJointPositionAction(JointPositionAction):
    """Apply policy joint targets, then overwrite frozen joints with H5 reference."""

    cfg: "ReferenceFrozenJointPositionActionCfg"

    def __init__(self, cfg: "ReferenceFrozenJointPositionActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        asset: Articulation = self._asset
        if self.cfg.frozen_joint_name_expr:
            self._frozen_joint_ids = resolve_joint_ids(asset, list(self.cfg.frozen_joint_name_expr))
            # Indices within this action term's joint slice (all joints when joint_names=[".*"]).
            if isinstance(self._joint_ids, slice):
                self._frozen_action_cols = self._frozen_joint_ids
            else:
                joint_id_to_col = {int(j): i for i, j in enumerate(self._joint_ids)}
                self._frozen_action_cols = torch.tensor(
                    [joint_id_to_col[int(j)] for j in self._frozen_joint_ids.tolist()],
                    device=self.device,
                    dtype=torch.long,
                )
        else:
            self._frozen_joint_ids = torch.empty(0, device=self.device, dtype=torch.long)
            self._frozen_action_cols = torch.empty(0, device=self.device, dtype=torch.long)
        if self.cfg.reference_only_joint_name_expr:
            self._reference_only_joint_ids = resolve_joint_ids(
                asset, list(self.cfg.reference_only_joint_name_expr)
            )
        else:
            self._reference_only_joint_ids = torch.empty(0, device=self.device, dtype=torch.long)
        if self._reference_only_joint_ids.numel() > 0:
            if isinstance(self._joint_ids, slice):
                action_joint_ids = torch.arange(asset.num_joints, device=self.device, dtype=torch.long)
            else:
                action_joint_ids = torch.as_tensor(self._joint_ids, device=self.device, dtype=torch.long)
            overlap = set(action_joint_ids.tolist()) & set(self._reference_only_joint_ids.tolist())
            if overlap:
                raise ValueError(
                    "reference_only_joint_name_expr overlaps policy action joints: "
                    f"{sorted(overlap)}"
                )
        if self.cfg.track_root_reference and not getattr(self, "_root_track_logged", False):
            print(
                "[INFO] track_root_reference=True: root follows H5 each control step "
                "(required for O6 USD without floating-base DOFs)."
            )
            self._root_track_logged = True

    def _apply_root_reference(self) -> None:
        if not self.cfg.track_root_reference or not self.cfg.motion_h5_path:
            return
        write_root_reference_from_motion(
            self._env,
            self._asset,
            h5_path=self.cfg.motion_h5_path,
            window_seconds=float(self.cfg.motion_window_seconds),
            asset_name=self.cfg.asset_name,
        )

    def _get_reference_joint_targets_all(self) -> torch.Tensor:
        """Reference absolute joint targets for all runtime joints."""
        buf = get_or_create_motion_buffer(
            self._env,
            self.cfg.motion_h5_path,
            float(self.cfg.motion_window_seconds),
            asset_name=self.cfg.asset_name,
        )
        return buf.q_ref_abs(motion_steps(self._env))

    def _get_reference_joint_targets(self) -> torch.Tensor:
        """Reference absolute joint targets aligned with this action term's joint slice."""
        q_ref_abs_all = self._get_reference_joint_targets_all()
        if isinstance(self._joint_ids, slice):
            return q_ref_abs_all
        return q_ref_abs_all[:, self._joint_ids]

    def _scaled_raw_action(self, cols: torch.Tensor) -> torch.Tensor:
        """Return scale * raw_action for selected action columns."""
        if isinstance(self._scale, torch.Tensor):
            return self._raw_actions[:, cols] * self._scale[:, cols]
        return self._raw_actions[:, cols] * float(self._scale)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        if self._frozen_action_cols.numel() == 0:
            return
        self._raw_actions[:, self._frozen_action_cols] = 0.0
        if isinstance(self._offset, torch.Tensor):
            self._processed_actions[:, self._frozen_action_cols] = self._offset[
                :, self._frozen_action_cols
            ]
        else:
            self._processed_actions[:, self._frozen_action_cols] = float(self._offset)

    def apply_actions(self) -> None:
        self._apply_root_reference()
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        if self._frozen_joint_ids.numel() == 0:
            return
        q_ref = self._get_reference_joint_targets()
        self._asset.set_joint_position_target(
            q_ref[:, self._frozen_action_cols],
            joint_ids=self._frozen_joint_ids,
        )
        if self._reference_only_joint_ids.numel() == 0:
            return
        q_ref_all = self._get_reference_joint_targets_all()
        self._asset.set_joint_position_target(
            q_ref_all[:, self._reference_only_joint_ids],
            joint_ids=self._reference_only_joint_ids,
        )


@configclass
class ReferenceFrozenJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action with selected joints overwritten by motion reference each step."""

    class_type: type[ActionTerm] = ReferenceFrozenJointPositionAction

    frozen_joint_name_expr: list[str] = []
    """Joint name expressions forced to track ``q_ref_abs`` (policy output ignored)."""

    reference_only_joint_name_expr: list[str] = []
    """Joints driven open-loop from H5 but excluded from policy action/obs dims."""

    motion_h5_path: str = ""
    motion_window_seconds: float = 10.0
    track_root_reference: bool = False
    """When True, write H5 root pose each control step (O6 / fixed-base assets)."""


class ReferenceResidualJointPositionAction(ReferenceFrozenJointPositionAction):
    """Use reference targets as baseline and learn policy residuals around them.

    Command equation for residual-controlled joints:
    ``q_cmd = q_ref + residual_alpha * scale * raw_action``.
    """

    cfg: "ReferenceResidualJointPositionActionCfg"

    def __init__(self, cfg: "ReferenceResidualJointPositionActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        asset: Articulation = self._asset
        if self.cfg.residual_joint_name_expr:
            self._residual_joint_ids = resolve_joint_ids(asset, list(self.cfg.residual_joint_name_expr))
            if isinstance(self._joint_ids, slice):
                self._residual_action_cols = self._residual_joint_ids
            else:
                joint_id_to_col = {int(j): i for i, j in enumerate(self._joint_ids)}
                self._residual_action_cols = torch.tensor(
                    [joint_id_to_col[int(j)] for j in self._residual_joint_ids.tolist()],
                    device=self.device,
                    dtype=torch.long,
                )
        else:
            # Empty list means "all action-controlled joints".
            action_dim = self.action_dim
            self._residual_action_cols = torch.arange(
                action_dim,
                device=self.device,
                dtype=torch.long,
            )
            if isinstance(self._joint_ids, slice):
                self._residual_joint_ids = self._residual_action_cols
            else:
                self._residual_joint_ids = self._joint_ids[self._residual_action_cols]

    def _runtime_residual_alpha(self) -> float:
        alpha = getattr(self._env, "_g1_residual_alpha", self.cfg.residual_alpha)
        return max(0.0, float(alpha))

    def process_actions(self, actions: torch.Tensor) -> None:
        # Base class stores raw actions and prepares internal buffers.
        JointPositionAction.process_actions(self, actions)
        q_ref = self._get_reference_joint_targets()

        # Keep original behavior for non-residual joints unless explicitly enabled.
        if self.cfg.use_reference_residual and self._residual_action_cols.numel() > 0:
            alpha = self._runtime_residual_alpha()
            delta = self._scaled_raw_action(self._residual_action_cols)
            self._processed_actions[:, self._residual_action_cols] = (
                q_ref[:, self._residual_action_cols] + alpha * delta
            )

        # Frozen joints are still fully open-loop from reference.
        if self._frozen_action_cols.numel() > 0:
            self._raw_actions[:, self._frozen_action_cols] = 0.0
            self._processed_actions[:, self._frozen_action_cols] = q_ref[:, self._frozen_action_cols]


@configclass
class ReferenceResidualJointPositionActionCfg(ReferenceFrozenJointPositionActionCfg):
    """Reference-frozen joint action with optional residual control around reference."""

    class_type: type[ActionTerm] = ReferenceResidualJointPositionAction

    use_reference_residual: bool = True
    """If True, selected joints use ``q_ref + residual_alpha * scale * raw_action``."""

    residual_joint_name_expr: list[str] = []
    """Joint name expressions controlled by residual policy around ``q_ref``.

    Empty means all action-controlled joints.
    """

    residual_alpha: float = 1.0
    """Residual gain multiplier. Can be overridden at runtime via ``_g1_residual_alpha``."""
