"""Create the separate 25-DOF reference-guided body. Called after Kit starts."""
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from physical_character import build, SkinBridge

ROOT = Path(__file__).resolve().parents[2]


def ensure_asset():
    out = ROOT/'output/anon_physical_v3'
    out.mkdir(parents=True, exist_ok=True)
    asset = out/'anon_physical.usda'
    if asset.exists(): return asset
    stage = Usd.Stage.CreateNew(str(out/'rig.usda'))
    m = build(stage, ROOT/'output/pmx_prototype/character.usdc', articulated_neck=True)
    m['schema'] = 'anon_physical_v3_25dof'
    for j in m['joints']:
        if j['name'].startswith('waist'): gains = (180., 12.)
        elif j['name'].startswith('neck'): gains = (40., 4.)
        elif 'shoulder' in j['name']: gains = (90., 8.)
        elif 'elbow' in j['name']: gains = (60., 6.)
        else: continue
        j['stiffness'], j['damping'] = gains
        drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath('/World/Anon/Joints/'+j['name']), 'angular')
        drive.GetStiffnessAttr().Set(gains[0]*np.pi/180)
        drive.GetDampingAttr().Set(gains[1]*np.pi/180)
    links = list(m['links'])
    t = np.zeros((len(links), 7)); t[:, 6] = 1.
    t[:, :3] = [np.array(m['links'][n]['origin'])+[0, 0, m['lift']] for n in links]
    SkinBridge(stage, m, links).update(t)
    for prim in stage.Traverse():
        if prim.GetName() == 'Collider': UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
    stage.GetRootLayer().Save()
    wrapper = Usd.Stage.CreateNew(str(asset))
    root = wrapper.DefinePrim('/Anon', 'Xform')
    root.GetReferences().AddReference('./rig.usda', '/World/Anon')
    wrapper.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(wrapper, 'Z'); UsdGeom.SetStageMetersPerUnit(wrapper, 1.)
    wrapper.GetRootLayer().Save()
    (out/'manifest.json').write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding='utf-8')
    return asset
