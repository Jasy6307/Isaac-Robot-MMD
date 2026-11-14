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

import json


def ensure_reference_asset(root, version):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from physical_character import build, SkinBridge

    if version not in (3, 4):
        raise ValueError(f'Unsupported reference body version: {version}')
    wrists = version == 4
    out = root / f'output/anon_physical_v{version}'
    out.mkdir(parents=True, exist_ok=True)
    asset = out / 'anon_physical.usda'
    if asset.exists():
        return asset

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
