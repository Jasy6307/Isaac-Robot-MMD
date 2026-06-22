"""Terrain importer with optional ground-plane Z offset."""

from __future__ import annotations

import omni.log

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass


class LoweredGroundTerrainImporter(TerrainImporter):
    """Spawns the default Isaac grid plane at ``cfg.ground_z_offset`` on Z."""

    cfg: "LoweredGroundTerrainImporterCfg"

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)) -> None:
        prim_path = self.cfg.prim_path + f"/{name}"
        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. Existing terrains: {', '.join(self.terrain_names)}."
            )
        self.terrain_prim_paths.append(prim_path)

        color = (0.0, 0.0, 0.0)
        if self.cfg.visual_material is not None:
            material = self.cfg.visual_material.to_dict()
            if "diffuse_color" in material:
                color = material["diffuse_color"]
            else:
                omni.log.warn(
                    "Visual material specified for ground plane but no diffuse color found."
                    " Using default color: (0.0, 0.0, 0.0)"
                )

        ground_plane_cfg = sim_utils.GroundPlaneCfg(
            physics_material=self.cfg.physics_material, size=size, color=color
        )
        z_off = float(self.cfg.ground_z_offset)
        translation = (0.0, 0.0, z_off) if z_off != 0.0 else None
        ground_plane_cfg.func(prim_path, ground_plane_cfg, translation=translation)


@configclass
class LoweredGroundTerrainImporterCfg(TerrainImporterCfg):
    """``TerrainImporterCfg`` with a vertical offset for ``terrain_type='plane'``."""

    class_type: type = LoweredGroundTerrainImporter

    ground_z_offset: float = 0.0
    """World-frame Z translation applied when spawning the ground plane (meters)."""
