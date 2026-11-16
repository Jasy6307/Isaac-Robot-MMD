"""Synchronize a baked Anon PMX USD animation into a live Isaac stage."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class LiveAnonScene:
    """Display the PMX PhysX cache alongside a live G1 policy rollout."""

    def __init__(
        self,
        stage,
        *,
        pmx_usd: str | Path,
        offset_x: float = 1.9,
        environment_scale: float = 0.75,
        bangs_shadow_level: float = 0.65,
        cel_brightness_scale: float = 0.88,
        face_shaping: bool = True,
    ):
        import omni.ui as ui
        import omni.usd
        import numpy as np
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdSkel, Vt
        from source.pmx_prototype.runtime.appearance_defaults import DEFAULT_LOOK
        from source.pmx_prototype.runtime.character_appearance import (
            apply_bright_character_look,
            apply_cel_look,
        )
        from source.pmx_prototype.runtime.bangs_shadow import BangsShadow

        self.np = np
        self.Vt = Vt
        self.stage = stage
        self.offset_x = float(offset_x)
        self.current_frame = 0.0
        cache_path = Path(pmx_usd).resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(f"Baked Anon USD cache does not exist: {cache_path}")
        if stage.GetPrimAtPath("/World/Character"):
            raise RuntimeError("Live Anon comparison expects /World/Character to be unused")

        self.cache_stage = Usd.Stage.Open(str(cache_path))
        if not self.cache_stage:
            raise RuntimeError(f"Could not open baked Anon USD cache: {cache_path}")
        self.cache_animation = UsdSkel.Animation.Get(self.cache_stage, "/Character/Animation")
        if not self.cache_animation:
            raise RuntimeError("Baked Anon USD does not expose /Character/Animation")
        self.cache_translations = self.cache_animation.GetTranslationsAttr()
        self.cache_rotations = self.cache_animation.GetRotationsAttr()
        samples = sorted(set(self.cache_translations.GetTimeSamples()) & set(self.cache_rotations.GetTimeSamples()))
        if len(samples) < 2:
            raise RuntimeError("Baked Anon USD has fewer than two synchronized animation samples")
        self.start_frame = float(samples[0])
        self.end_frame = float(samples[-1])
        self.fps = float(self.cache_stage.GetTimeCodesPerSecond() or 30.0)

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        character = UsdSkel.Root.Define(stage, "/World/Character")
        character.GetPrim().GetReferences().AddReference(str(cache_path))
        UsdGeom.Xformable(character).AddTranslateOp().Set(Gf.Vec3d(self.offset_x, 0.0, 0.0))
        self.animation = UsdSkel.Animation.Get(stage, "/World/Character/Animation")
        if not self.animation:
            raise RuntimeError("Referenced Anon USD does not expose /World/Character/Animation")
        self.animation.CreateJointsAttr(self.cache_animation.GetJointsAttr().Get())
        self.animation.CreateScalesAttr(self.cache_animation.GetScalesAttr().Get())

        center = (self.offset_x, 0.0, 0.9)
        apply_bright_character_look(stage, center, character_root="/World/Character")
        apply_cel_look(
            stage,
            brightness_scale=cel_brightness_scale,
            face_shaping=face_shaping,
            character_root="/World/Character",
        )
        dome = UsdLux.DomeLight.Get(stage, "/World/Light")
        if dome and dome.GetIntensityAttr():
            dome.GetIntensityAttr().Set(550.0 * float(environment_scale))
        try:
            self.bangs_shadow = BangsShadow(
                stage,
                shadow_level=bangs_shadow_level,
                character_root="/World/Character",
            )
        except ValueError as error:
            self.bangs_shadow = None
            print(f"[Appearance] Bangs occlusion unavailable: {error}", flush=True)

        camera = UsdGeom.Camera.Define(stage, "/World/LiveCompareCamera")
        camera.CreateFocalLengthAttr(50)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955 * 9.0 / 16.0)
        eye = Gf.Vec3d(self.offset_x / 2, -13.0, 2.05)
        target = Gf.Vec3d(self.offset_x / 2, 0.0, 0.85)
        UsdGeom.Xformable(camera).AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
        )
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = "/World/LiveCompareCamera"

        self.ui_window = ui.Window("G1 + Anon Live Compare", width=390, height=118)
        with self.ui_window.frame:
            with ui.VStack(spacing=4):
                ui.Label("G1 trained policy + Anon PMX / one live Isaac stage")
                self.time_label = ui.Label("Dance: 0.0 s | PMX PhysX cache synced")
                ui.Label("Use G1 Policy Eval Play / Stop or press P")
        self.advance(self.start_frame)
        print(
            f"[INFO] Live Anon comparison ready: {self.end_frame - self.start_frame + 1:.0f} cached frames, "
            f"{self.fps:.1f} fps, look={DEFAULT_LOOK}, cache={cache_path}",
            flush=True,
        )

    def advance(self, target_frame: float) -> None:
        from pxr import Usd

        target = min(self.end_frame, max(self.start_frame, float(target_frame)))
        time_code = Usd.TimeCode(target)
        translations = self.cache_translations.Get(time_code)
        rotations = self.cache_rotations.Get(time_code)
        if translations is None or rotations is None:
            raise RuntimeError(f"Baked Anon USD has no pose at frame {target:.3f}")
        translations = self.np.asarray(translations, dtype=self.np.float32)
        self.animation.GetTranslationsAttr().Set(self.Vt.Vec3fArray.FromNumpy(translations))
        self.animation.GetRotationsAttr().Set(self.Vt.QuatfArray.FromNumpy(self.np.asarray(rotations)))
        self.current_frame = target
        if self.bangs_shadow is not None:
            self.bangs_shadow.update(target)
        self.time_label.text = f"Dance: {(target - self.start_frame) / self.fps:.1f} s | PMX PhysX cache synced"

    def reset(self) -> None:
        self.advance(self.start_frame)

    def close(self) -> None:
        self.cache_stage = None
