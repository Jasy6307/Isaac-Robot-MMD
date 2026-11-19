"""Bake Anon on the source clock, then record all three actors in one Kit stage."""
import json
from pathlib import Path
import re

import numpy as np

from source.train_workflow.g1_vmd_pipeline import find_ffmpeg
from .motion import sha256

PMX_SOURCE = Path('source/pmx_prototype')
FPS = 30
LABELS = ('G1 - Collision baseline', 'G1 - Arm stabilized', 'Anon PMX')


def motion_layer(source, output, frames):
    from pxr import Usd, UsdSkel

    stage = Usd.Stage.CreateNew(str(output))
    # Relative sublayers keep the whole run folder portable.
    from os.path import relpath
    stage.GetRootLayer().subLayerPaths = [Path(relpath(source, output.parent)).as_posix()]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/Character'))
    stage.SetTimeCodesPerSecond(FPS)
    stage.SetFramesPerSecond(FPS)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(frames-1)
    return stage, UsdSkel.Animation.Get(stage, '/Character/Animation')


def prepend_neutral_pose(source, output, count):
    """2s neutral physics settle (in worker), 4s entry, then 2s first-pose hold."""
    from pxr import Usd, UsdSkel, Vt
    from scipy.spatial.transform import Rotation

    original = Usd.Stage.Open(str(source))
    anim = UsdSkel.Animation.Get(original, '/Character/Animation')
    skeleton = UsdSkel.Skeleton.Get(original, '/Character/Skeleton')
    paths = list(skeleton.GetJointsAttr().Get())
    rest = skeleton.GetRestTransformsAttr().Get()
    joints = anim.GetJointsAttr().Get()
    neutral = [rest[paths.index(joint)] for joint in joints]
    p0 = np.array([matrix.ExtractTranslation() for matrix in neutral])
    q0 = Rotation.from_quat([[*q.GetImaginary(), q.GetReal()]
                            for q in (matrix.ExtractRotationQuat() for matrix in neutral)])
    p1 = np.asarray(anim.GetTranslationsAttr().Get(0))
    q1 = Rotation.from_quat(np.asarray(anim.GetRotationsAttr().Get(0)))
    rotation_delta = (q0.inv()*q1).as_rotvec()
    ramp, hold = 4*FPS, 2*FPS
    offset = ramp+hold
    stage, target = motion_layer(source, output, count+offset)
    target.GetJointsAttr().Set(joints)
    for frame in range(offset):
        t = min(frame/(ramp-1), 1.)
        amount = t**3*(10-15*t+6*t*t)
        position = p0+(p1-p0)*amount
        quaternion = (q0*Rotation.from_rotvec(rotation_delta*amount)).as_quat()
        target.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(position.astype(np.float32)), frame)
        target.GetRotationsAttr().Set(Vt.QuatfArray.FromNumpy(quaternion.astype(np.float32)), frame)
    for frame in range(count):
        target.GetTranslationsAttr().Set(anim.GetTranslationsAttr().Get(frame), frame+offset)
        target.GetRotationsAttr().Set(anim.GetRotationsAttr().Get(frame), frame+offset)
    stage.GetRootLayer().Save()
    return offset


def trim_physics(source, baked, cache, count, offset, schema):
    from pxr import Usd, UsdSkel

    stage, target = motion_layer(source, cache, count)
    original = Usd.Stage.Open(str(source))
    baked_stage = Usd.Stage.Open(str(baked))
    raw = UsdSkel.Animation.Get(baked_stage, '/Character/Animation')
    ordered_joints = list(raw.GetJointsAttr().Get())
    dynamic = np.asarray(schema['dynamic_mask'], dtype=bool)
    if len(dynamic) != len(ordered_joints) or not dynamic.any() or dynamic.all():
        raise ValueError('PhysX and source joint schemas do not match')
    # Source animation only authors VMD-driven bones. The worker expands it to
    # the full skeleton, including rest pose joints and the secondary bones.
    caches, queries = [], []
    for scene in (original, baked_stage):
        skeleton = UsdSkel.Skeleton.Get(scene, '/Character/Skeleton')
        skel_cache = UsdSkel.Cache()
        skel_cache.Populate(UsdSkel.Root.Get(scene, '/Character'), Usd.PrimDefaultPredicate)
        if list(skeleton.GetJointsAttr().Get()) != ordered_joints:
            raise ValueError('PhysX bone ordering differs from the source skeleton')
        caches.append(skel_cache)
        queries.append(skel_cache.GetSkelQuery(skeleton))
    expected = list(range(count+offset))
    if any(attr.GetTimeSamples() != expected for attr in (raw.GetTranslationsAttr(), raw.GetRotationsAttr())):
        raise ValueError('PhysX cache has an incomplete clock')
    target.GetJointsAttr().Set(ordered_joints)
    target.GetScalesAttr().Set(raw.GetScalesAttr().Get())
    max_position, max_rotation = 0., 0.
    for frame in range(count):
        p = raw.GetTranslationsAttr().Get(frame+offset)
        q = raw.GetRotationsAttr().Get(frame+offset)
        if not np.isfinite(np.asarray(p)).all() or not np.isfinite(np.asarray(q)).all():
            raise ValueError(f'Nonfinite PMX cache at frame {frame}')
        body, baked_body = (np.asarray(query.ComputeJointSkelTransforms(time))[~dynamic]
                            for query, time in zip(queries, (frame, frame+offset)))
        difference = body-baked_body
        if not np.isfinite(difference).all():
            raise ValueError(f'Invalid composed PMX body transforms at frame {frame}')
        max_position = max(max_position, float(np.max(np.linalg.norm(difference[:, 3, :3], axis=1))))
        max_rotation = max(max_rotation, float(np.max(np.abs(difference[:, :3, :3]))))
        target.GetTranslationsAttr().Set(p, frame)
        target.GetRotationsAttr().Set(q, frame)
    if max_position > 1e-5 or max_rotation > 1e-5:
        raise ValueError(f'PMX body failed source clock verification: {max_position:g} m, {max_rotation:g} rotation matrix error')
    stage.GetDefaultPrim().SetCustomDataByKey('secondaryMotionSolver', 'PhysX with neutral pre-roll and source clock trim')
    stage.GetRootLayer().Save()
    return dict(frames=count, trimmed_frames=offset, max_body_position_error_m=max_position,
                max_body_rotation_matrix_error=max_rotation)


def bake_anon(run, source, pmx, count):
    folder = run.directory/'pmx'
    folder.mkdir()
    preroll = folder/'preroll.usdc'
    offset = prepend_neutral_pose(source, preroll, count)
    baked, cache = folder/'physx_full.usdc', folder/'anon_physx.usdc'
    worker = folder/'worker'
    run.python('pmx_physx', PMX_SOURCE/'runtime/dance_secondary_worker.py', '--usd', preroll,
               '--pmx', pmx, '--secondary-profile', PMX_SOURCE/'configs/physics/secondary_default.json',
               '--quality', 'cloth', '--physics-threads', 12, '--skirt-edge-stride', 3,
               '--startup-settle-seconds', 2, '--render-fps', FPS, '--steps', count+offset,
               '--output', worker, '--bake-usd', baked, '--runtime-audit', timeout=180+(count+offset)*2)
    report = json.loads((worker/'worker_report.json').read_text(encoding='utf-8'))
    if report['status'] != 'complete' or report['frames'] != count+offset or report['loops'] != 0:
        raise ValueError('PMX physics bailed out or wrapped the clip')
    schema = json.loads((worker/'motion_schema.json').read_text(encoding='utf-8'))
    run.report('pmx_clock', trim_physics(source, baked, cache, count, offset, schema))
    return cache


def record_comparison(run, left, middle, cache, audio, count, width, height):
    video = run.directory/'recording'/'comparison.mp4'
    video.parent.mkdir()
    audio_arg = ('--audio', audio) if audio is not None else ()
    run.python('isaac_record', PMX_SOURCE/'play_g1_anon_comparison.py', '--g1-h5', left,
               '--corrected-h5', middle, '--pmx-cache', cache, *audio_arg, '--headless',
               '--width', width, '--height', height, '--actor-z-offset', -.015, '--record-video', video,
               '--labels', *LABELS, timeout=180+count*2)
    report = json.loads(video.with_suffix('.json').read_text(encoding='utf-8'))
    if (not report['same_stage'] or report['frames'] != count or report['fps'] != FPS or
            tuple(report['comparison_order']) != LABELS):
        raise ValueError('Recorder metadata does not match the requested scene and clock')
    for name, path in (('g1', left), ('corrected_h5', middle), ('pmx', cache)):
        if report[name+'_sha256'] != sha256(path):
            raise ValueError(f'Recorder input changed: {path}')
    sound = ('-map', '0:a:0') if audio else ()
    log = run.command('decode_video', find_ffmpeg(), '-nostdin', '-xerror', '-stats_period', '1',
                      '-progress', 'pipe:1', '-nostats', '-i', video, '-map', '0:v:0', *sound,
                      '-f', 'null', '-', timeout=120+count/10)
    decoded = log.read_text(encoding='utf-8', errors='replace')
    frames = re.findall(r'^frame=(\d+)', decoded, re.MULTILINE)
    times = re.findall(r'^out_time_us=(\d+)', decoded, re.MULTILINE)
    if (not frames or int(frames[-1]) != count or not times or
            abs(int(times[-1])/1e6-count/FPS) > 1/FPS or 'progress=end' not in decoded or
            not re.search(rf'Stream #0:0.*Video: h264.*\b{width}x{height}\b.*\b30 fps\b', decoded) or
            (audio and not re.search(r'Stream #0:\d+.*Audio: aac', decoded))):
        raise ValueError('Decoded recording failed frame, duration, size or audio verification')
    return video
