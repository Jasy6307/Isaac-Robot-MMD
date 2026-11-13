"""Local render-only semantic mapping for converted Mutsumi split meshes."""
import numpy as np
from pxr import UsdGeom,UsdShade,UsdSkel,Vt,Sdf


def adapt_mutsumi_materials(stage,character_root):
    body=UsdGeom.Mesh.Get(stage,character_root+'/Body')
    parts=[stage.GetPrimAtPath(character_root+f'/Part_{i:02d}') for i in (0,2,3,10)]
    if not body or not all(parts) or body.ComputeVisibility()!='invisible':return None
    skeleton=UsdSkel.Skeleton.Get(stage,character_root+'/Skeleton')
    names=list(skeleton.GetJointNamesAttr().Get())
    front=[i for i,n in enumerate(names) if n.lower().startswith('front hair')]
    if not front or '頭' not in names:return None
    bind=UsdSkel.BindingAPI(body);ji=np.asarray(bind.GetJointIndicesPrimvar().Get());jw=np.asarray(bind.GetJointWeightsPrimvar().Get())
    width=bind.GetJointIndicesPrimvar().GetElementSize();ji=ji.reshape(-1,width);jw=jw.reshape(-1,width)
    head_weight=np.sum(jw*np.isin(ji,[names.index('頭')]),axis=1)
    front_weight=np.sum(jw*np.isin(ji,front),axis=1)
    triangles=np.asarray(body.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    body_subsets={}
    for subset in UsdGeom.Subset.GetAllGeomSubsets(body):
        mat=UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
        if mat:body_subsets[str(mat.GetPath())]=np.asarray(subset.GetIndicesAttr().Get(),dtype=int)
    def subset(mesh,name,indices,material,family):
        sub=UsdGeom.Subset.Define(stage,mesh.GetPath().AppendChild(name));sub.CreateElementTypeAttr('face');sub.CreateFamilyNameAttr(family);sub.CreateIndicesAttr(Vt.IntArray(indices.tolist()))
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(material)
    count=0;receivers=[]
    for i in (0,3):
        mesh=UsdGeom.Mesh.Get(stage,character_root+f'/Part_{i:02d}')
        material=UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()[0]
        source=str(material.GetPath());material.GetPrim().SetDisplayName(f'mutsumi_skin_body_{i:02d}')
        face=UsdShade.Material.Define(stage,character_root+f'/Looks/MutsumiFace_{i:02d}')
        face.GetPrim().GetReferences().AddInternalReference(source);face.GetPrim().SetDisplayName(f'mutsumi_face_skin_{i:02d}')
        faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3);ids=np.flatnonzero(np.mean(head_weight[faces],axis=1)>.5)
        if not len(ids):continue
        subset(mesh,'ToonFace',ids,face,'materialBind')
        original=body_subsets[source];face_ids=original[np.mean(head_weight[triangles[original]],axis=1)>.5]
        subset(body,f'ToonFace_{i:02d}',face_ids,face,'appearanceSemantics');count+=len(face_ids);receivers.append(str(mesh.GetPath()))
    eye=UsdShade.MaterialBindingAPI(parts[1]).ComputeBoundMaterial()[0];eye.GetPrim().SetDisplayName('mutsumi_eye')
    for i in (1,):
        material=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(character_root+f'/Part_{i:02d}')).ComputeBoundMaterial()[0]
        material.GetPrim().SetDisplayName(f'mutsumi_skin_body_{i:02d}')
    hair=UsdShade.MaterialBindingAPI(parts[3]).ComputeBoundMaterial()[0];original=body_subsets[str(hair.GetPath())]
    bang_ids=original[np.mean(front_weight[triangles[original]],axis=1)>.25]
    marker=UsdShade.Material.Define(stage,character_root+'/Looks/MutsumiBangsSemantic');marker.GetPrim().GetReferences().AddInternalReference(str(hair.GetPath()));marker.GetPrim().SetDisplayName('mutsumi_front_bangs')
    subset(body,'ToonBangs',bang_ids,marker,'appearanceSemantics')
    stage.GetPrimAtPath(character_root).CreateAttribute('appearance:shadowReceivers',Sdf.ValueTypeNames.StringArray).Set(receivers)
    return dict(adapter='mutsumi_split_mesh',face_triangles=count,bangs_triangles=len(bang_ids),shadow_receivers=receivers,eyes_preserved=True)
