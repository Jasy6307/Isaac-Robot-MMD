"""Separate 31-DOF model with forearm twist and two-axis wrists; keep v3 intact."""
import json
from pathlib import Path
import numpy as np
if __name__=='__main__':
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True})
from pxr import Usd,UsdGeom,UsdPhysics
from physical_character import build,SkinBridge
ROOT=Path(__file__).resolve().parents[2]

def ensure_asset():
    out=ROOT/'output/anon_physical_v4';out.mkdir(exist_ok=True,parents=True)
    asset=out/'anon_physical.usda'
    if asset.exists():return asset
    stage=Usd.Stage.CreateNew(str(out/'rig.usda'))
    m=build(stage,ROOT/'output/pmx_prototype/character.usdc',articulated_neck=True,articulated_wrists=True)
    m['schema']='anon_physical_v4_31dof'
    m['fixed_parents']={'head':'neck_pitch',**{s+'_hand':s+'_wrist_deviation' for s in ['left','right']}}
    for j in m['joints']:
        gains=(180.,12.) if j['name'].startswith('waist') else (40.,4.) if j['name'].startswith('neck') else (90.,8.) if 'shoulder' in j['name'] else None
        if gains:
            j['stiffness'],j['damping']=gains
            drive=UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath('/World/Anon/Joints/'+j['name']),'angular')
            drive.GetStiffnessAttr().Set(gains[0]*np.pi/180);drive.GetDampingAttr().Set(gains[1]*np.pi/180)
    links=list(m['links']);t=np.zeros((len(links),7));t[:,6]=1
    t[:,:3]=[np.array(m['links'][n]['origin'])+[0,0,m['lift']] for n in links]
    SkinBridge(stage,m,links).update(t)
    for prim in stage.Traverse():
        if prim.GetName()=='Collider':UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
    stage.GetRootLayer().Save()
    wrapper=Usd.Stage.CreateNew(str(asset));root=wrapper.DefinePrim('/Anon','Xform')
    root.GetReferences().AddReference('./rig.usda','/World/Anon');wrapper.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(wrapper,'Z');UsdGeom.SetStageMetersPerUnit(wrapper,1.);wrapper.GetRootLayer().Save()
    assert len(m['joints'])==31
    (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    return asset

if __name__=='__main__':
    print(ensure_asset());app.close()
