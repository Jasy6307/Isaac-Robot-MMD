"""Shared asset-writing steps for the 25- and 31-DOF reference bodies.

Call after SimulationApp has started. Version-specific schemas, drives and
fixed-parent mappings are deliberately retained for existing checkpoints.
"""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory


PLAGENHOEF_MASS_PROFILE = 'plagenhoef_female_v1'
_LIMB_MASSES_KG = dict(shoulder_yaw=1.35, forearm_twist=.76, hand=.19,
                      hip_pitch=5.83, knee_pitch=2.71, ankle_roll=.61)
_MAIN_MASSES_KG = dict(head=4.01, waist_pitch=14.64, pelvis=8.05)
_MAIN_MASSES_KG.update({f'{side}_{name}': mass for side in ('left', 'right')
                       for name, mass in _LIMB_MASSES_KG.items()})


def resolve_body_asset(root, config):
    full_body = config.get('full_body_reference', False)
    wrists = config.get('articulated_wrists', False)
    mass_profile = config.get('body_mass_profile')
    if mass_profile is not None and not (full_body and wrists):
        raise ValueError('body_mass_profile requires full_body_reference and articulated_wrists (31 DOF)')
    if full_body:
        return ensure_reference_asset(root, version=4 if wrists else 3, mass_profile=mass_profile)
    return Path(config.get('asset', root / 'output/anon_physical/anon_physical.usda'))


def _verify_mass_asset(folder):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    calibration = manifest['mass_calibration']
    if (calibration['profile'] != PLAGENHOEF_MASS_PROFILE
            or any(manifest['links'][name]['mass'] != mass
                   for name, mass in _MAIN_MASSES_KG.items())):
        raise ValueError(f'Wrong mass profile in {folder}')
    for name in ('anon_physical.usda', 'rig.usda'):
        if sha256((folder / name).read_bytes()).hexdigest() != calibration['asset_sha256'][name]:
            raise ValueError(f'Calibrated body asset has changed: {folder / name}')
    return folder / 'anon_physical.usda'


def _ensure_plagenhoef_asset(root):
    from pxr import Gf, Sdf

    output = (root / 'output').resolve()
    folder = output / f'anon_physical_v4_{PLAGENHOEF_MASS_PROFILE}'
    if folder.exists():
        return _verify_mass_asset(folder)

    baseline = ensure_reference_asset(root, version=4).parent
    inputs = {name: (baseline / name).read_bytes()
              for name in ('anon_physical.usda', 'rig.usda', 'manifest.json')}
    manifest = json.loads(inputs['manifest.json'])
    links = manifest['links']
    if len(links) != 35 or len(manifest['joints']) != 31:
        raise ValueError('Plagenhoef mass profile requires the 35-body, 31-DOF reference asset')

    # Clone the authored layer so the collider geometry, COM, axes and drives
    # agree with the baseline. For fixed geometry each principal inertia is
    # linear in that body's mass.
    rig = Sdf.Layer.CreateAnonymous('rig.usda')
    if not rig.ImportFromString(inputs['rig.usda'].decode('utf-8')):
        raise RuntimeError(f'Could not load the baseline body layer: {baseline}')
    for name, mass in _MAIN_MASSES_KG.items():
        body = links[name]
        old_mass = body['mass']
        mass_attr = rig.GetAttributeAtPath(body['path'] + '.physics:mass')
        inertia_attr = rig.GetAttributeAtPath(body['path'] + '.physics:diagonalInertia')
        if not math.isclose(mass_attr.default, old_mass, rel_tol=1e-6):
            raise ValueError(f'Mass differs between rig and manifest for {name}')
        mass_attr.default = mass
        inertia_attr.default = Gf.Vec3f(*(value * mass / old_mass for value in inertia_attr.default))
        body['mass'] = mass

    total = math.fsum(body['mass'] for body in links.values())
    if not math.isclose(total, 50.64, abs_tol=1e-8):
        raise ValueError(f'Expected 50.64 kg after mass calibration, got {total:g} kg')
    manifest['total_mass_kg'] = total
    with TemporaryDirectory(prefix='.anon-mass-', dir=output) as temporary:
        staged = Path(temporary) / folder.name
        staged.mkdir()
        (staged / 'anon_physical.usda').write_bytes(inputs['anon_physical.usda'])
        if not rig.Export(str(staged / 'rig.usda')):
            raise RuntimeError('Could not write the calibrated body layer')
        manifest['mass_calibration'] = dict(
            profile=PLAGENHOEF_MASS_PROFILE,
            baseline_sha256={name: sha256(data).hexdigest() for name, data in inputs.items()},
            asset_sha256={name: sha256((staged / name).read_bytes()).hexdigest()
                          for name in ('anon_physical.usda', 'rig.usda')},
            inertia_rule='For fixed geometry and COM, scale each principal inertia by new_mass/old_mass.',
        )
        (staged / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        if not staged.resolve().is_relative_to(output) or not folder.resolve().is_relative_to(output):
            raise ValueError('Calibrated body assets must stay inside the output directory')
        try:
            staged.rename(folder)
        except FileExistsError:
            # Another process can build the same profile at the same time.
            pass
    return _verify_mass_asset(folder)


def ensure_reference_asset(root, version, mass_profile=None):
    if version not in (3, 4):
        raise ValueError(f'Unsupported reference body version: {version}')
    if mass_profile is not None:
        if version != 4 or mass_profile != PLAGENHOEF_MASS_PROFILE:
            raise ValueError(f'Unsupported mass profile for reference body v{version}: {mass_profile!r}')
        return _ensure_plagenhoef_asset(root)

    wrists = version == 4
    out = root / f'output/anon_physical_v{version}'
    out.mkdir(parents=True, exist_ok=True)
    asset = out / 'anon_physical.usda'
    if asset.exists():
        return asset

    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from physical_character import build, SkinBridge

    stage = Usd.Stage.CreateNew(str(out / 'rig.usda'))
    options = dict(articulated_neck=True)
    if wrists:
        options['articulated_wrists'] = True
    manifest = build(stage, root / 'output/pmx_prototype/character.usdc', **options)
    manifest['schema'] = f'anon_physical_v{version}_{31 if wrists else 25}dof'
    if wrists:
        manifest['fixed_parents'] = {
            'head': 'neck_pitch',
            **{side + '_hand': side + '_wrist_deviation' for side in ('left', 'right')},
        }
    for joint in manifest['joints']:
        name = joint['name']
        if name.startswith('waist'):
            gains = (180., 12.)
        elif name.startswith('neck'):
            gains = (40., 4.)
        elif 'shoulder' in name:
            gains = (90., 8.)
        elif not wrists and 'elbow' in name:
            gains = (60., 6.)
        else:
            continue
        joint['stiffness'], joint['damping'] = gains
        drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath('/World/Anon/Joints/' + name), 'angular'
        )
        drive.GetStiffnessAttr().Set(gains[0] * np.pi / 180)
        drive.GetDampingAttr().Set(gains[1] * np.pi / 180)

    links = list(manifest['links'])
    transforms = np.zeros((len(links), 7))
    transforms[:, 6] = 1.
    transforms[:, :3] = [
        np.array(manifest['links'][name]['origin']) + [0, 0, manifest['lift']]
        for name in links
    ]
    SkinBridge(stage, manifest, links).update(transforms)
    for prim in stage.Traverse():
        if prim.GetName() == 'Collider':
            UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
    stage.GetRootLayer().Save()

    wrapper = Usd.Stage.CreateNew(str(asset))
    character = wrapper.DefinePrim('/Anon', 'Xform')
    character.GetReferences().AddReference('./rig.usda', '/World/Anon')
    wrapper.SetDefaultPrim(character)
    UsdGeom.SetStageUpAxis(wrapper, 'Z')
    UsdGeom.SetStageMetersPerUnit(wrapper, 1.)
    wrapper.GetRootLayer().Save()
    if wrists:
        assert len(manifest['joints']) == 31
    (out / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return asset
