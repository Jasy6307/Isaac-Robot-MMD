"""Shared character looks and approved cel look 08; preserve original ground."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from pxr import Gf, Sdf, UsdShade, UsdGeom, UsdLux


def apply_bright_character_look(stage, center=(0., 0., 0.),character_root="/World/Character"):
    """Apply local character overrides and soft lights to a Z-up preview stage."""
    character = stage.GetPrimAtPath(character_root)
    if not character:
        raise ValueError('Expected /World/Character in preview stage')
    from pxr import Usd
    for prim in Usd.PrimRange(character):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != 'UsdPreviewSurface':
            continue
        label = prim.GetParent().GetDisplayName().lower()
        eyes = any(word in label for word in ('eye', '瞳', '虹彩'))
        hair = any(word in label for word in ('hair', '髪', '发'))
        reflection = .035 if eyes else (.018 if hair else .008)
        shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(reflection))
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.38 if eyes else (.68 if hair else .9))
    dome = UsdLux.DomeLight.Define(stage, '/World/Light')
    dome.CreateIntensityAttr(550)
    dome.CreateColorAttr(Gf.Vec3f(.96, .98, 1.))
    for path, intensity, color, angle, rotation in (
        ('/World/Key', 1350, (1., .94, .89), 25., (25, -25, -35)),
        ('/World/Fill', 450, (.94, .97, 1.), 25., (55, 25, 140)),
    ):
        light = UsdLux.DistantLight.Define(stage, path)
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(*color))
        light.CreateAngleAttr(angle)
        xf = UsdGeom.Xformable(light)
        op = next((op for op in xf.GetOrderedXformOps()
                   if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ), None)
        (op or xf.AddRotateXYZOp()).Set(Gf.Vec3f(*rotation))
    face = UsdLux.SphereLight.Define(stage, '/World/FaceSoftbox')
    face.CreateRadiusAttr(1.4)
    face.CreateIntensityAttr(650)
    face.CreateColorAttr(Gf.Vec3f(1., .97, .94))
    face.CreateNormalizeAttr(True)
    xf = UsdGeom.Xformable(face)
    op = next((op for op in xf.GetOrderedXformOps()
               if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    (op or xf.AddTranslateOp()).Set(Gf.Vec3d(float(center[0]), float(center[1])-2.5, 1.7))
    return face


def add_character_outline(stage, width=.002,character_root="/World/Character",mesh_name='Body'):
    """Skinned inverted hull. It follows the existing skeleton, without physics."""
    import numpy as np
    import carb
    from pxr import Vt
    source=character_root+'/'+mesh_name
    destination=character_root+'/ToonOutline'+('' if mesh_name=='Body' else '_'+mesh_name)
    original_mesh=UsdGeom.Mesh.Get(stage,source)
    if mesh_name=='Body' and original_mesh.ComputeVisibility()=='invisible':
        for prim in list(stage.GetPrimAtPath(character_root).GetChildren()):
            if prim.IsA(UsdGeom.Mesh) and prim.GetName().startswith('Part_') and UsdGeom.Imageable(prim).ComputeVisibility()!='invisible':
                add_character_outline(stage,width,character_root,prim.GetName())
        return
    kept=[]
    hair_vertices=[]
    hair_opacity=None
    bindings=[];assigned=set();all_faces=np.asarray(original_mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    for subset in UsdGeom.Subset.GetAllGeomSubsets(original_mesh):
        mat=UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
        indices=np.asarray(subset.GetIndicesAttr().Get(),dtype=int);bindings.append((mat,indices));assigned.update(indices.tolist())
    remaining=np.asarray(sorted(set(range(len(all_faces)))-assigned),dtype=int)
    if len(remaining):bindings.append((UsdShade.MaterialBindingAPI(original_mesh.GetPrim()).ComputeBoundMaterial()[0],remaining))
    for mat,indices in bindings:
        name=mat.GetPrim().GetDisplayName().lower() if mat else ''
        # Small layered facial parts and fingers should retain texture lines.
        if any(word in name for word in ('face','skin','eye','瞳','目','眼','眉','睫','指甲','腮','裙纱')):
            continue
        kept.extend(indices.tolist())
        if any(word in name for word in ('hair','发','髪')):
            surface=UsdShade.Shader(stage.GetPrimAtPath(mat.GetPath().AppendChild('Surface')))
            if surface:hair_opacity=surface.GetInput('opacity').GetConnectedSource()
            all_faces=np.asarray(original_mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            face_ids=indices
            triangles=all_faces[face_ids]
            edges=triangles[:,[0,1,1,2,2,0]].reshape(-1,2)
            graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),
                             shape=(len(original_mesh.GetPointsAttr().Get()),)*2).tocsr()
            _,labels=connected_components(graph,directed=False)
            positions=np.asarray(original_mesh.GetPointsAttr().Get())
            reject=set()
            for component in np.unique(labels[triangles[:,0]]):
                selected=labels[triangles[:,0]]==component
                vertices=np.unique(triangles[selected])
                spans=np.sort(np.ptp(positions[vertices],axis=0))
                # Tiny detached strands keep their texture silhouette, no hull.
                if selected.sum()<24 or spans[-2]<.008:
                    reject.update(face_ids[selected].tolist())
            kept=[index for index in kept if index not in reject]
            hair_vertices.extend(triangles.flatten())
    if not kept:return
    stage.DefinePrim(destination,'Mesh').GetReferences().AddInternalReference(source)
    mesh=UsdGeom.Mesh.Get(stage,destination)
    for child in list(mesh.GetPrim().GetChildren()):
        child.SetActive(False)
    points=np.asarray(mesh.GetPointsAttr().Get())
    normals=np.asarray(mesh.GetNormalsAttr().Get())
    normals=normals.copy()
    extrusion=np.full((len(points),1),width)
    if hair_vertices:
        ids=np.unique(hair_vertices)
        # Average coincident seam normals and use a much thinner hair shell.
        _,inverse=np.unique(np.round(points[ids],5),axis=0,return_inverse=True)
        sums=np.zeros((inverse.max()+1,3));np.add.at(sums,inverse,normals[ids])
        smooth=sums[inverse]
        smooth/=np.maximum(np.linalg.norm(smooth,axis=1,keepdims=True),1e-8)
        normals[ids]=smooth
        extrusion[ids]=.00065
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy((points+extrusion*normals).astype(np.float32)))
    faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    if kept:
        faces=faces[np.asarray(sorted(kept),dtype=int)]
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]*len(faces)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces[:,::-1].flatten().tolist()))
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy((-normals).astype(np.float32)))
    mesh.CreateDoubleSidedAttr(False)
    mesh.GetPrim().CreateAttribute('singleSided',Sdf.ValueTypeNames.Bool).Set(True)
    # Flattened texture paths on this copy are irrelevant: all faces use ink.
    ink_path='/World/Looks/ToonInk'+('_'+mesh_name if mesh_name!='Body' and hair_opacity else '')
    material=UsdShade.Material.Define(stage,ink_path)
    shader=UsdShade.Shader.Define(stage,ink_path+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.006,.004,.009))
    shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.006,.004,.009))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1.)
    shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput('specularColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.))
    if mesh_name!='Body' and hair_opacity:
        shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).ConnectToSource(*hair_opacity)
        shader.CreateInput('opacityThreshold',Sdf.ValueTypeNames.Float).Set(.1)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    carb.settings.get_settings().set_bool('/rtx/hydra/faceCulling/enabled',True)


def apply_cel_look(stage,brightness_scale=1.,face_shaping=False,character_root="/World/Character"):
    """Native MDL two-band shader; leave eyes, original ground and lights intact."""
    from pathlib import Path
    from pxr import Usd
    mdl=str(resolve_source('mmd_cel.mdl').resolve())
    for prim in list(Usd.PrimRange(stage.GetPrimAtPath(character_root))):
        if not prim.IsA(UsdShade.Material):continue
        material=UsdShade.Material(prim)
        surface=UsdShade.Shader(stage.GetPrimAtPath(prim.GetPath().AppendChild('Surface')))
        if not surface or surface.GetIdAttr().Get()!='UsdPreviewSurface':continue
        name=prim.GetDisplayName().lower()
        if any(word in name for word in ('eye','瞳','高光','虹彩')):continue
        source=surface.GetInput('diffuseColor').GetConnectedSource()
        cel=UsdShade.Shader.Define(stage,prim.GetPath().AppendChild('CelSurface'))
        cel.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        cel.SetSourceAsset(Sdf.AssetPath(mdl),'mdl')
        cel.SetSourceAssetSubIdentifier('mmd_cel','mdl')
        cel.CreateOutput('out',Sdf.ValueTypeNames.Token)
        if source:
            tex=UsdShade.Shader(source[0].GetPrim())
            asset=tex.GetInput('file').Get()
            if asset:
                inp=cel.CreateInput('albedo',Sdf.ValueTypeNames.Asset)
                inp.Set(Sdf.AssetPath(asset.resolvedPath or asset.path))
                inp.GetAttr().SetColorSpace('sRGB')
        is_face=any(word in name for word in ('face','顔','脸'))
        cel.CreateInput('brightness',Sdf.ValueTypeNames.Float).Set(.58*brightness_scale)
        cel.CreateInput('threshold',Sdf.ValueTypeNames.Float).Set((.45 if face_shaping else -.1) if is_face else .25)
        cel.CreateInput('shadow_level',Sdf.ValueTypeNames.Float).Set((.75 if face_shaping else .82) if is_face else .64)
        cel.CreateInput('face_shaping',Sdf.ValueTypeNames.Bool).Set(is_face and face_shaping)
        cel.CreateInput('shadow_tint',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.,.96,.93) if is_face and face_shaping else Gf.Vec3f(1.))
        cel.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(float(surface.GetInput('opacity').Get() or 1.))
        material.CreateSurfaceOutput('mdl').ConnectToSource(cel.ConnectableAPI(),'out')
    add_character_outline(stage,character_root=character_root)


class DefaultCharacterLook:
    """Apply approved look 08 without touching floor bindings or geometry."""
    def __init__(self,stage,center=(0.,0.,0.),character_root='/World/Character'):
        from source.pmx_prototype.appearance_defaults import ENVIRONMENT_SCALE,CEL_BRIGHTNESS_SCALE,FACE_SHAPING,BANGS_SHADOW_LEVEL
        from source.pmx_prototype.character_semantics import adapt_mutsumi_materials
        self.semantic_mapping=adapt_mutsumi_materials(stage,character_root)
        if self.semantic_mapping:print('[Appearance] '+str(self.semantic_mapping),flush=True)
        self.face_light=apply_bright_character_look(stage,center,character_root=character_root)
        UsdLux.DomeLight.Get(stage,'/World/Light').GetIntensityAttr().Set(550*ENVIRONMENT_SCALE)
        apply_cel_look(stage,brightness_scale=CEL_BRIGHTNESS_SCALE,face_shaping=FACE_SHAPING,character_root=character_root)
        from source.pmx_prototype.bangs_shadow import BangsShadow
        self.bangs=None
        try:
            self.bangs=BangsShadow(stage,shadow_level=BANGS_SHADOW_LEVEL,character_root=character_root)
        except ValueError as error:
            print(f'[Appearance] Look 08 active; bangs occlusion unavailable: {error}',flush=True)

    def update(self,frame,time_code=None):
        if self.bangs is not None:
            self.bangs.update(frame,time_code=time_code)
