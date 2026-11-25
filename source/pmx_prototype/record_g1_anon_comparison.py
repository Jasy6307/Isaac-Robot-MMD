"""Record G1 and Anon together in one Isaac Sim stage.

Anon is animated with the existing native PhysX secondary-motion path. G1 is
rendered from either an updated retarget H5 or captured trained-policy states
in the same stage and camera, using the existing robot USD as a kinematic visual. Both motions share one frame
clock, light rig, floor, viewport, and output video.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import hashlib

PMX_DIR = Path(__file__).resolve().parent
ROOT = PMX_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PMX_DIR) not in sys.path:
    sys.path.insert(0, str(PMX_DIR))

from source.pmx_prototype.appearance_defaults import (
    DEFAULT_LOOK,
    LOOK_CHOICES,
    resolve_appearance_args,
)
from source.pmx_prototype.secondary_profiles import (
    DEFAULT_SECONDARY_PROFILE,
    DEFAULT_SECONDARY_QUALITY,
)
from source_layout import config_path


def find_ffmpeg(explicit: str | None) -> str:
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, FileNotFoundError):
        pass
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates = sorted((Path(appdata) / "Python").glob(
            "Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe"))
        if candidates:
            return str(candidates[-1])
    raise RuntimeError("FFmpeg is required; install imageio-ffmpeg or pass --ffmpeg PATH")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-h5", type=Path, required=True)
    parser.add_argument("--g1-usd", type=Path, default=ROOT / "assets/g1_29dof_o6_hand_V3.usd")
    parser.add_argument("--pmx-usd", type=Path, required=True,
                        help="PMX body animation baked from the same source dance")
    parser.add_argument("--pmx", type=Path,
                        default=ROOT / "media/千早爱音_by_小麦弥望/Anon Chihaya.pmx")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--video-width", type=int, default=1600)
    parser.add_argument("--video-height", type=int, default=900)
    parser.add_argument("--separation", type=float, default=1.9,
                        help="Distance between the two character origins in meters")
    parser.add_argument("--secondary-profile", type=config_path,
                        default=DEFAULT_SECONDARY_PROFILE)
    parser.add_argument("--quality", choices=["hair", "cloth"], default=DEFAULT_SECONDARY_QUALITY)
    parser.add_argument("--environment-scale", type=float, default=None)
    parser.add_argument("--bangs-shadow-level", type=float, default=None)
    parser.add_argument("--cel-brightness-scale", type=float, default=None)
    parser.add_argument("--face-shaping", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--look", choices=LOOK_CHOICES, default=DEFAULT_LOOK)
    return parser


parser = build_parser()
args = resolve_appearance_args(parser.parse_args())
for name in ("g1_h5", "g1_usd", "pmx_usd", "pmx", "audio"):
    value = getattr(args, name)
    if value is not None:
        value = value.resolve()
        if not value.is_file():
            parser.error(f"{name} not found: {value}")
        setattr(args, name, value)
if not (0 <= args.environment_scale <= 2 and
        0 <= args.bangs_shadow_level <= 1 and
        0 < args.cel_brightness_scale <= 2):
    parser.error("Invalid appearance scale or shadow level")
if min(args.video_width, args.video_height) < 2 or args.video_width % 2 or args.video_height % 2:
    parser.error("Video dimensions must be positive even integers")
if not (args.separation > 0):
    parser.error("--separation must be positive")
if args.audio is None:
    args.audio = args.pmx_usd.with_suffix(".wav")
    if not args.audio.is_file():
        args.audio = None
if args.output.exists():
    parser.error(f"Output already exists; choose a new path: {args.output}")
args.output = args.output.resolve()
args.output.parent.mkdir(parents=True, exist_ok=True)
ffmpeg = find_ffmpeg(args.ffmpeg)

import h5py
from isaacsim import SimulationApp

W, H = args.video_width, args.video_height
app = SimulationApp({
    "headless": True,
    "fast_shutdown": True,
    "width": W,
    "height": H,
    "renderer": "RaytracedLighting",
    "anti_aliasing": 3,
    "extra_args": ["--/app/asyncRendering=false"],
})

import numpy as np
from PIL import Image
import torch
import omni.usd
import omni.kit.app
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdSkel, Vt
from scipy.spatial.transform import Rotation
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport
from source.train_workflow.utils.motion.self_collision import G1CollisionModel, Joint, _quat_matrix
from source.train_workflow.utils.media.viewport_avi_recorder import _capture_viewport_rgb
from physx_secondary import BodyMotion, PhysxSecondary


physics = encoder = log = None
exit_code = 1
temporary = args.output.with_name(args.output.stem + ".partial.mp4")
try:
    with h5py.File(args.g1_h5, "r") as stream:
        names = list(stream["joint_names"].asstr()[:])
        joint_positions = (stream["joint_pos_delta"][:] +
                           stream.attrs["record_baseline_joint_pos"])
        root_positions = (stream["root_pos_delta"][:] +
                          stream.attrs["record_root_anchor_pos"])
        root_delta = stream["root_quat_delta_wxyz"][:]
        root_anchor = stream.attrs["record_root_anchor_quat_wxyz"]
        fps = float(stream.attrs["fps"])
        retarget_mode = str(stream.attrs.get("retarget_mode", "unknown"))
        upper_method = str(stream.attrs.get("upper_body_retarget_method", ""))
        lower_method = str(stream.attrs.get("lower_body_retarget_method", ""))
        policy_checkpoint = str(stream.attrs.get("policy_checkpoint", ""))
    count = len(joint_positions)
    if count < 2 or abs(fps - 30.0) > 1e-6:
        raise ValueError(f"Expected nonempty 30 FPS G1 H5, got {count} frames at {fps} FPS")
    if retarget_mode not in {"new", "policy"}:
        raise ValueError(f"G1 H5 must be from the updated retarget or trained-policy chain: {retarget_mode}")

    ffmpeg_check = subprocess.run([ffmpeg, "-version"], check=True,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    del ffmpeg_check

    # Build full robot FK from the articulated USD joint tree. The visual robot
    # is made kinematic and has its joint prims removed in this playback stage;
    # its exact H5 poses then share the PMX stage without fighting PhysX.
    g1_model = G1CollisionModel(args.g1_usd, names, "cpu")
    robot_source = Usd.Stage.Open(str(args.g1_usd))
    pending = []
    for prim in robot_source.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        parent_paths = joint.GetBody0Rel().GetTargets()
        child_paths = joint.GetBody1Rel().GetTargets()
        if not parent_paths or not child_paths:
            continue
        pending.append(Joint(
            prim.GetName(), parent_paths[0].name, child_paths[0].name,
            np.array(joint.GetLocalPos0Attr().Get()),
            _quat_matrix(joint.GetLocalRot0Attr().Get()),
            np.array(joint.GetLocalPos1Attr().Get()),
            _quat_matrix(joint.GetLocalRot1Attr().Get()),
            prim.GetAttribute("physics:axis").Get(), -np.inf, np.inf,
        ))
    g1_model.joints = []
    seen = {"pelvis"}
    while pending:
        ready = [joint for joint in pending if joint.parent in seen]
        if not ready:
            raise RuntimeError("Disconnected G1 USD joint tree")
        for joint in ready:
            g1_model.joints.append(joint)
            seen.add(joint.child)
            pending.remove(joint)
    g1_model.constants = {
        joint.name: tuple(torch.tensor(value, dtype=torch.float32)
                          for value in (joint.p0, joint.r0, joint.p1, joint.r1))
        for joint in g1_model.joints
    }
    rotations, positions = g1_model.fk(torch.tensor(joint_positions, dtype=torch.float32))
    rotations = {name: value.numpy() for name, value in rotations.items()}
    positions = {name: value.numpy() for name, value in positions.items()}
    root_rotation = (
        Rotation.from_quat(root_delta[:, [1, 2, 3, 0]]) *
        Rotation.from_quat(root_anchor[[1, 2, 3, 0]])
    ).as_matrix()
    align = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    g1_offset = np.array([-args.separation / 2, 0., 0.])
    anon_offset = np.array([args.separation / 2, 0., 0.])

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    character = UsdSkel.Root.Define(stage, "/World/Character")
    character.GetPrim().GetReferences().AddReference(str(args.pmx_usd))
    motion = BodyMotion(args.pmx_usd, args.pmx, secondary_profile=args.secondary_profile)
    root_indices = np.flatnonzero(motion.parents < 0)
    if len(root_indices) == 0:
        raise RuntimeError("PMX skeleton has no root joints")
    original_pose = motion.pose

    def shifted_anon_pose(frame):
        p, q, wp, wq = original_pose(frame)
        p[root_indices] += anon_offset
        wp += anon_offset
        return p, q, wp, wq

    motion.pose = shifted_anon_pose
    physics = PhysxSecondary(stage, motion, obstacles=False, quality=args.quality)

    g1 = stage.DefinePrim("/World/G1", "Xform")
    g1.GetReferences().AddReference(str(args.g1_usd))
    joint_paths = []
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix("/World/G1"):
            continue
        if prim.IsA(UsdPhysics.Joint):
            joint_paths.append(prim.GetPath())
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body = UsdPhysics.RigidBodyAPI(prim)
            body.CreateKinematicEnabledAttr(True)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    for path in joint_paths:
        joint_prim = stage.GetPrimAtPath(path)
        if joint_prim and joint_prim.IsValid():
            joint_prim.SetActive(False)

    operations = {}
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix("/World/G1") and prim.GetName() in rotations:
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            operations[prim.GetName()] = xform.AddTransformOp()
    if set(operations) != set(rotations):
        missing = sorted(set(rotations) - set(operations))
        raise RuntimeError(f"G1 visual USD is missing FK links: {missing}")

    def publish_g1(frame):
        for name, operation in operations.items():
            matrix = np.eye(4)
            matrix[:3, :3] = align @ root_rotation[frame] @ rotations[name][frame]
            matrix[:3, 3] = (
                align @ (root_rotation[frame] @ positions[name][frame] + root_positions[frame])
                + g1_offset
            )
            operation.Set(Gf.Matrix4d(matrix.T.tolist()))

    animation = UsdSkel.Animation.Get(stage, "/World/Character/Animation")
    animation.CreateJointsAttr(motion.joints)
    animation.CreateScalesAttr([Gf.Vec3h(1)] * len(motion.joints))
    cache = UsdSkel.Cache()
    cache.Populate(UsdSkel.Root.Get(motion.source, "/Character"), Usd.PrimDefaultPredicate)
    query = cache.GetSkelQuery(motion.skeleton)
    skin = cache.GetSkinningQuery(motion.skeleton.GetPrim().GetStage().GetPrimAtPath("/Character/Body"))
    rest = UsdGeom.Mesh.Get(motion.source, "/Character/Body").GetPointsAttr().Get()
    lo = np.full(3, np.inf)
    hi = -lo.copy()
    for frame in range(motion.end_frame + 1):
        points = Vt.Vec3fArray(rest)
        if not skin.ComputeSkinnedPoints(query.ComputeSkinningTransforms(frame), points, frame):
            raise RuntimeError(f"PMX skinning failed at frame {frame}")
        xyz = np.asarray(points)
        lo = np.minimum(lo, xyz.min(0))
        hi = np.maximum(hi, xyz.max(0))
    anon_center = (lo + hi) / 2 + anon_offset
    anon_half = (hi - lo) / 2 + np.array([.25, .25, .25])
    half_x = args.separation / 2 + max(float(anon_half[0]), .36)
    half_y = max(float(anon_half[1]), .36)
    half_z = max(float(anon_half[2]), .85)
    camera_target = [0., 0., float(anon_center[2])]
    distance = max(
        half_x / (20.955 / 100),
        half_z / (20.955 / 100 * H / W),
    ) + half_y

    dome = UsdLux.DomeLight.Define(stage, "/World/Light")
    dome.CreateIntensityAttr(800)
    key = UsdLux.DistantLight.Define(stage, "/World/Key")
    key.CreateIntensityAttr(1800)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(25, -25, -35))
    floor = UsdGeom.Mesh.Define(stage, "/World/VisibleFloor")
    floor.CreatePointsAttr([(-12, -12, -.025), (12, -12, -.025),
                            (12, 12, -.025), (-12, 12, -.025)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateSubdivisionSchemeAttr("none")
    floor.CreateDisplayColorAttr([(.25, .28, .33)])

    def publish_anon(p, q):
        animation.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(q.astype(np.float32)))
        animation.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))

    p, q, _, _ = motion.pose(0)
    publish_anon(p, q)
    from character_appearance import DefaultCharacterLook, apply_bright_character_look
    bangs_shadow = None
    if args.look == DEFAULT_LOOK:
        appearance = DefaultCharacterLook(
            stage, anon_center, environment_scale=args.environment_scale, brightness_scale=args.cel_brightness_scale,
            face_shaping=args.face_shaping, bangs_shadow_level=args.bangs_shadow_level)
        bangs_shadow = appearance.bangs
    else:
        if args.look == "soft-pbr":
            apply_bright_character_look(stage, anon_center)
        dome.GetIntensityAttr().Set(float(dome.GetIntensityAttr().Get()) * args.environment_scale)

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(50)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(20.955 * H / W)
    eye = [0., -float(distance), camera_target[2] + .65]
    set_camera_view(eye=eye, target=camera_target, camera_prim_path="/World/Camera")
    viewport = get_active_viewport()
    viewport.camera_path = "/World/Camera"
    viewport.set_texture_resolution((W, H))

    publish_g1(0)
    for _ in range(20):
        app.update()
    physics.start(ctx.get_stage_id())
    for _ in range(60):
        app.update()

    log = args.output.with_suffix(".encoder.log").open("wb")
    g1_label = "G1 / Trained Policy" if retarget_mode == "policy" else "G1 / Updated Retarget"
    labels = (
        "drawbox=x=0:y=0:w=iw:h=52:color=black@0.55:t=fill,"
        "drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':"
        f"text='{g1_label}':fontcolor=white:fontsize=28:x=32:y=10,"
        "drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':"
        f"text='Anon PMX / Live PhysX':fontcolor=white:fontsize=28:x={W // 2 + 32}:y=10"
    )
    command = [ffmpeg, "-y", "-hide_banner", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", "30", "-i", "pipe:0"]
    if args.audio:
        command += ["-i", str(args.audio), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:a", "aac", "-b:a", "192k", "-af", "volume=0.25,apad"]
    command += ["-vf", labels, "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-pix_fmt", "yuv420p", "-t", str(count / 30), "-movflags", "+faststart",
                str(temporary)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log)
    trajectory_hash = hashlib.sha256()
    print(f"SHARED_SCENE_START frames={count} fps=30 size={W}x{H} retarget={retarget_mode} "
          f"upper={upper_method} lower={lower_method}", flush=True)
    for frame in range(count):
        if frame:
            substeps = physics.physics_hz // 30
            for substep in range(1, substeps + 1):
                p, q = physics.step(frame - 1 + substep / substeps)
        publish_anon(p, q)
        publish_g1(frame)
        if bangs_shadow is not None:
            bangs_shadow.update(frame)
        steps = physics.steps
        for _ in range(3):
            app.update()
        rgb = None
        for attempt in range(4):
            rgb = _capture_viewport_rgb(pre_render=None)
            if rgb is not None:
                break
            for _ in range(30):
                app.update()
        if rgb is None or rgb.shape != (H, W, 3):
            raise RuntimeError(f"Invalid viewport capture at frame {frame}: "
                               f"{None if rgb is None else rgb.shape}")
        if physics.steps != steps:
            raise RuntimeError("Viewport capture advanced the physics clock")
        trajectory_hash.update(np.ascontiguousarray(rgb).tobytes())
        if frame in {0, 60, 150, 300, count - 1}:
            Image.fromarray(rgb).save(args.output.with_name(
                f"{args.output.stem}_{frame:04d}.jpg"), quality=94)
        encoder.stdin.write(np.ascontiguousarray(rgb).tobytes())
        if frame % 60 == 0:
            print(f"SHARED_SCENE_FRAME {frame}/{count} physx_steps={physics.steps}", flush=True)
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("FFmpeg encoder failed")
    encoder = None
    log.close()
    log = None
    if temporary.stat().st_size <= 0:
        raise RuntimeError("Encoded comparison video is empty")
    os.replace(temporary, args.output)
    report = {
        "video": str(args.output),
        "g1_h5": str(args.g1_h5),
        "g1_usd": str(args.g1_usd),
        "g1_retarget_mode": retarget_mode,
        "g1_upper_body_method": upper_method,
        "g1_lower_body_method": lower_method,
        "pmx_usd": str(args.pmx_usd),
        "pmx": str(args.pmx),
        "audio": str(args.audio) if args.audio else None,
        "frames": count,
        "fps": 30,
        "duration_seconds": count / 30,
        "width": W,
        "height": H,
        "same_stage": True,
        "same_camera_floor_lighting_and_viewport": True,
        "g1_playback": (
            "kinematic visual replay driven by actual trained-policy simulation states in the shared stage"
            if retarget_mode == "policy" else
            "kinematic visual driven from updated retarget H5 FK in the shared stage"
        ),
        "g1_policy_checkpoint": policy_checkpoint or None,
        "anon_secondary_motion": "native omni.physx live hair and skirt",
        "anon_offset_x_m": float(anon_offset[0]),
        "g1_offset_x_m": float(g1_offset[0]),
        "g1_fk_validation_error_m": float(g1_model.fk_validation_error),
        "look": args.look,
        "environment_scale": args.environment_scale,
        "bangs_shadow_level": args.bangs_shadow_level,
        "cel_brightness_scale": args.cel_brightness_scale,
        "face_shaping": args.face_shaping,
        "bangs_shadow_enabled": bangs_shadow is not None,
        "secondary_profile": str(args.secondary_profile) if args.secondary_profile else None,
        "quality": args.quality,
        "trajectory_sha256": trajectory_hash.hexdigest(),
        "camera": {"eye": eye, "target": camera_target},
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SHARED_SCENE_PASS {args.output}", flush=True)
    exit_code = 0
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    for resource, cleanup in (
        (encoder, lambda: (encoder.kill(), encoder.wait())),
        (physics, lambda: physics.close()),
        (log, lambda: log.close()),
    ):
        if resource is not None:
            try:
                cleanup()
            except BaseException:
                import traceback
                traceback.print_exc()
                exit_code = 1
    app.close(skip_cleanup=True, exit_code=exit_code)
