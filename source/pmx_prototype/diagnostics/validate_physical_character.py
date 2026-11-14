"""Build and exercise Anon's physical prerequisite (no RL policy).

Cases: fixed-base joint sweep, free-base PD hold, unpowered gravity drop.
Each invocation writes a USD, full manifest, metrics and optional screenshot.
"""

# Allow both the canonical tool path and the historical root entry point.
import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = PROJECT_ROOT
sys.path.insert(0, str(SOURCE_DIR))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', choices=['joints', 'stand', 'drop'], default='stand')
parser.add_argument('--usd', type=Path, default=ROOT/'output/pmx_prototype/character.usdc')
parser.add_argument('--output', type=Path, default=ROOT/'output/anon_physical')
parser.add_argument('--seconds', type=float, default=8.)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--capture', action='store_true')
parser.add_argument('--interactive', action='store_true', help='Keep running after the bounded test')
args = parser.parse_args()
if args.seconds < 2 or not args.usd.is_file():
    parser.error('Need an existing character USD and at least two seconds')
from isaacsim import SimulationApp
app = SimulationApp({'headless': args.headless, 'width': 1000, 'height': 1000,
                     'renderer': 'RaytracedLighting', 'fast_shutdown': True,
                     'extra_args': ['--/app/asyncRendering=false']})
import numpy as np
from scipy.spatial.transform import Rotation
import omni.usd
import omni.physx
import omni.physics.tensors as tensors
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdLux, UsdSkel, PhysxSchema, Vt
from physical_character import build, SkinBridge


def run():
    out = args.output/args.case
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(dict(case=args.case, status='running', passed=False)), encoding='utf-8')
    context = omni.usd.get_context(); context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, 'Z'); UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, '/World'); stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    scene = UsdPhysics.Scene.Define(stage, '/World/Physics')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); scene.CreateGravityMagnitudeAttr(9.81)
    pxscene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    pxscene.CreateEnableGPUDynamicsAttr(False); pxscene.CreateBroadphaseTypeAttr('MBP')
    pxscene.CreateTimeStepsPerSecondAttr(240)
    floor = UsdGeom.Cube.Define(stage, '/World/Ground')
    floor.CreateSizeAttr(1.)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.05)); floor.AddScaleOp().Set(Gf.Vec3f(20, 20, .1))
    floor.CreateDisplayColorAttr([(.18, .2, .24)])
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    manifest = build(stage, args.usd, fixed=args.case == 'joints',
                     lift=.35 if args.case == 'joints' else (.25 if args.case == 'drop' else .003))
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    light = UsdLux.DomeLight.Define(stage, '/World/Light'); light.CreateIntensityAttr(1000.)
    key = UsdLux.DistantLight.Define(stage, '/World/Key'); key.CreateIntensityAttr(1800.)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(25, -25, -35))
    camera = UsdGeom.Camera.Define(stage, '/World/Camera')
    camera.CreateFocalLengthAttr(50.)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(2.4, -4.5, 2.),
                                                      Gf.Vec3d(0, 0, .85), Gf.Vec3d(0, 0, 1)).GetInverse())
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    viewport = get_active_viewport()
    if viewport:
        viewport.camera_path = '/World/Camera'
        viewport.set_texture_resolution((1000, 1000))
    # Raw manual PhysX stepping makes elapsed simulated time independent of rendering.
    interface = omni.physx.get_physx_simulation_interface()
    interface.attach_stage(context.get_stage_id())
    dt = 1/240
    interface.simulate(dt, 0.); interface.fetch_results()
    view = tensors.create_simulation_view('numpy', context.get_stage_id(), backend='physx')
    robot = view.create_articulation_view('/World/Anon')
    if robot.count != 1:
        raise RuntimeError(f'Expected one articulation, got {robot.count}')
    meta = robot.shared_metatype
    names = list(meta.dof_names); link_names = list(meta.link_names)
    if len(names) != len(manifest['joints']) or meta.fixed_base != (args.case == 'joints'):
        raise RuntimeError(f'Unexpected articulation topology: {names}, fixed={meta.fixed_base}')
    bridge = SkinBridge(stage, manifest, link_names)
    ids = np.array([0], dtype=np.uint32)
    targets = np.zeros((1, len(names)), dtype=np.float32)
    limits = robot.get_dof_limits().copy()[0]
    initial = robot.get_link_transforms().copy()[0]
    expected_gains = np.array([next(j['stiffness'] for j in manifest['joints'] if j['name'] == name) for name in names])
    expected_damping = np.array([next(j['damping'] for j in manifest['joints'] if j['name'] == name) for name in names])
    gains_ok = bool(np.allclose(robot.get_dof_stiffnesses()[0], expected_gains, rtol=1e-5)
                    and np.allclose(robot.get_dof_dampings()[0], expected_damping, rtol=1e-5)
                    and np.allclose(robot.get_dof_max_velocities()[0], 12., rtol=1e-5))
    unpowered = False
    if args.case == 'drop':
        robot.set_dof_stiffnesses(targets, ids)
        robot.set_dof_dampings(targets, ids)
        unpowered = bool(np.allclose(robot.get_dof_stiffnesses(), 0.)
                         and np.allclose(robot.get_dof_dampings(), 0.))
    pelvis_index = link_names.index('pelvis')
    contact = view.create_rigid_contact_view('/World/Anon/Bodies/*_ankle_roll', ['/World/Ground'])
    start_z = float(initial[pelvis_index, 2])
    traces, commanded, measured, contact_forces, base_poses, link_poses = [], [], [], [], [], []
    # Ignore physics proxy geometry for character screenshots, without disabling collisions.
    for prim in stage.Traverse():
        if prim.GetName() == 'Collider':
            UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
    bridge.update(initial)
    stage.GetRootLayer().Export(str(out/'scene.usda'))
    if args.case == 'stand':
        # Reference the articulation subtree only: no floor, lights, or world support.
        asset_path = args.output/'anon_physical.usda'
        asset = Usd.Stage.CreateNew(str(asset_path.resolve()))
        asset_root = asset.DefinePrim('/Anon', 'Xform')
        asset_root.GetReferences().AddReference('./stand/scene.usda', '/World/Anon')
        asset.SetDefaultPrim(asset_root)
        UsdGeom.SetStageUpAxis(asset, 'Z'); UsdGeom.SetStageMetersPerUnit(asset, 1.)
        asset.GetRootLayer().Save()
    for _ in range(30): app.update()
    if viewport:
        viewport.camera_path = '/World/Camera'
    n = int(args.seconds/dt)
    print(f'ANON_PHYSICAL_READY case={args.case} dofs={len(names)} links={len(link_names)} fixed={meta.fixed_base}', flush=True)
    screenshot = None
    for step in range(n):
        t = step*dt
        if args.case == 'joints':
            # Every axis gets its own bounded 0.6 s excitation, avoiding simultaneous extreme poses.
            slot = min(int(t/.6), len(names)-1)
            targets.fill(0.)
            amplitude = min(.18, float(limits[slot, 1])*.5)
            targets[0, slot] = amplitude*np.sin(np.pi*((t % .6)/.6))**2
        robot.set_dof_position_targets(targets, ids)
        interface.simulate(dt, (step+1)*dt); interface.fetch_results()
        q = robot.get_dof_positions().copy()[0]
        poses = robot.get_link_transforms().copy()[0]
        contact_forces.append(contact.get_net_contact_forces(dt).copy())
        base_poses.append(poses[pelvis_index].copy())
        link_poses.append(poses.copy())
        if not np.isfinite(q).all() or not np.isfinite(poses).all():
            raise RuntimeError(f'Non-finite physics state at {t}')
        tilt = float(np.degrees(np.arccos(np.clip(Rotation.from_quat(poses[pelvis_index, 3:]).apply([0, 0, 1])[2], -1, 1))))
        traces.append([t, float(poses[pelvis_index, 2]), tilt, float(np.max(np.maximum(limits[:, 0]-q, q-limits[:, 1])))])
        commanded.append(targets[0].copy()); measured.append(q)
        if step % 8 == 0:
            bridge.update(poses)
            app.update()
        if args.capture and step == min(n-1, 240 if args.case == 'drop' else n-8):
            bridge.update(poses)
            for _ in range(5): app.update()
            screenshot = str((out/'physical.png').resolve())
            capture_viewport_to_file(viewport, screenshot)
            for _ in range(15): app.update()
    traces = np.array(traces)
    measured, commanded = np.array(measured), np.array(commanded)
    final = robot.get_link_transforms().copy()[0]
    response = np.max(np.abs(measured), axis=0)
    # Check actual skin bone world poses against their owners (full hierarchy roundtrip).
    wp, wq = bridge.update(final)
    local = UsdSkel.Animation.Get(stage, '/World/Anon/LivePose').GetTranslationsAttr().Get()
    quat = UsdSkel.Animation.Get(stage, '/World/Anon/LivePose').GetRotationsAttr().Get()
    reconstructed_p, reconstructed_q = [], []
    for i, parent in enumerate(manifest['parents']):
        r = Rotation.from_quat([*quat[i].GetImaginary(), quat[i].GetReal()])
        if parent >= 0:
            reconstructed_p.append(reconstructed_p[parent]+reconstructed_q[parent].apply(local[i]))
            reconstructed_q.append(reconstructed_q[parent]*r)
        else:
            reconstructed_p.append(np.array(local[i])); reconstructed_q.append(r)
    skin_error = float(np.max(np.linalg.norm(np.array(reconstructed_p)-wp, axis=1)))
    cache = UsdSkel.Cache()
    cache.Populate(UsdSkel.Root.Get(stage, '/World/Anon/Visual'), Usd.PrimDefaultPredicate)
    skel_query = cache.GetSkelQuery(UsdSkel.Skeleton.Get(stage, '/World/Anon/Visual/Skeleton'))
    mesh = UsdGeom.Mesh.Get(stage, '/World/Anon/Visual/Body')
    deformed = Vt.Vec3fArray(mesh.GetPointsAttr().Get())
    skin_ok = bool(cache.GetSkinningQuery(mesh.GetPrim()).ComputeSkinnedPoints(
        skel_query.ComputeSkinningTransforms(), deformed))
    skin_ok = skin_ok and bool(np.isfinite(np.asarray(deformed)).all())
    checks = dict(finite=True, correct_base=bool(meta.fixed_base) == (args.case == 'joints'),
                  joint_limits=bool(traces[:, 3].max() < .04), skin_roundtrip=skin_error < 1e-5,
                  cpu_skinning=skin_ok, drive_units=gains_ok)
    asset_path = args.output/'anon_physical.usda'
    if asset_path.is_file():
        exported = Usd.Stage.Open(str(asset_path.resolve()))
        root = exported.GetDefaultPrim()
        export_ok = bool(root and root.HasAPI(UsdPhysics.ArticulationRootAPI))
        rigid_count, dof_count = 0, 0
        for prim in exported.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_count += 1
                mass = UsdPhysics.MassAPI(prim)
                export_ok = export_ok and mass.GetMassAttr().Get() > 0 and min(mass.GetDiagonalInertiaAttr().Get()) > 0
            if prim.IsA(UsdPhysics.RevoluteJoint): dof_count += 1
            if prim.IsA(UsdPhysics.Joint):
                for rel in (UsdPhysics.Joint(prim).GetBody0Rel(), UsdPhysics.Joint(prim).GetBody1Rel()):
                    targets_in_asset = rel.GetTargets()
                    export_ok = export_ok and bool(targets_in_asset) and all(exported.GetPrimAtPath(p) for p in targets_in_asset)
        checks['exported_asset'] = bool(export_ok and rigid_count == 27 and dof_count == 23)
    mean_foot_forces = np.array(contact_forces)[-240:, :, 2].mean(axis=0)
    if args.case == 'joints':
        active_response = []
        for k in range(len(names)):
            start, end = int(k*.6/dt), min(len(measured), int((k+1)*.6/dt))
            active_response.append(float(measured[start:end, k].max()-measured[start, k]) if end > start else 0.)
        response = np.array(active_response)
        checks['every_joint_responds'] = bool(np.all(response > .04) and args.seconds >= len(names)*.6)
    elif args.case == 'stand':
        checks['unsupported_stand'] = bool(traces[:, 1].min() > start_z-.12 and traces[:, 2].max() < 20)
        checks['both_feet_support_weight'] = bool(np.all(mean_foot_forces > manifest['total_mass_kg']*9.81*.1)
                                                  and abs(mean_foot_forces.sum()/(manifest['total_mass_kg']*9.81)-1.) < .15)
    else:
        checks['gravity_response'] = bool(start_z-traces[:120, 1].min() > .10)
        checks['all_drives_disabled'] = unpowered
    report = dict(case=args.case, status='complete', passed=all(checks.values()), checks=checks, seconds=n*dt,
                  dofs=len(names), links=len(link_names), fixed_base=bool(meta.fixed_base),
                  total_mass_kg=manifest['total_mass_kg'], start_pelvis_z=start_z,
                  min_pelvis_z=float(traces[:, 1].min()), final_pelvis_z=float(traces[-1, 1]),
                  max_tilt_deg=float(traces[:, 2].max()), max_limit_violation_rad=float(max(0, traces[:, 3].max())),
                  skin_roundtrip_error_m=skin_error, joint_response_rad=dict(zip(names, response.tolist())),
                  mean_foot_vertical_forces_n=mean_foot_forces.tolist(),
                  screenshot=screenshot, policy=None,
                  limitations=['No balance policy: stand uses zero-target joint PD only.',
                               'Hair and clothing follow rigid bone owners; secondary physics disabled.',
                               'Mass and actuator parameters require tuning before locomotion training.'])
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(out/'trace.npz', state=traces, positions=measured, targets=commanded, dof_names=names,
                        contact_forces=np.array(contact_forces), base_poses=np.array(base_poses),
                        link_poses=np.array(link_poses), link_names=link_names)
    print('ANON_PHYSICAL_RESULT '+json.dumps(report), flush=True)
    while args.interactive and app.is_running():
        interface.simulate(dt, n*dt); interface.fetch_results(); n += 1
        if n % 8 == 0:
            bridge.update(robot.get_link_transforms().copy()[0]); app.update()
            time.sleep(1/30)
    interface.detach_stage()
    return 0 if report['passed'] else 1


result = 2
try:
    result = run()
except Exception as exc:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    failure_dir = args.output/args.case
    failure_dir.mkdir(parents=True, exist_ok=True)
    (failure_dir/'report.json').write_text(json.dumps(dict(case=args.case, status='error', passed=False,
                                                         error=str(exc)), indent=2), encoding='utf-8')
    result = 2
finally:
    app.close(exit_code=result)
sys.exit(result)
