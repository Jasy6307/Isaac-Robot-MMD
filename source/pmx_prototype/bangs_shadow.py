"""Live directional bangs occlusion, evaluated from the actual skinned mesh.

Only the face receives this artist-controlled hard shadow. No floor/light edits.
"""
import numpy as np
import sys
try:
    import coverage
except ImportError:
    coverage=None
if coverage is not None and not hasattr(getattr(coverage,'types',None),'Tracer'):
    sys.modules['coverage']=None
try:
    from numba import njit
finally:
    if coverage is not None:sys.modules['coverage']=coverage
from pxr import Usd, UsdGeom, UsdSkel, UsdShade, Sdf, Vt


@njit(cache=True)
def cast_shadow(points, triangles, direction):
    result=np.zeros(len(points),dtype=np.float32)
    for i in range(len(points)):
        origin=points[i]+direction*.0003
        for j in range(len(triangles)):
            a,b,c=triangles[j]
            x=b[0]-a[0];y=b[1]-a[1];z=b[2]-a[2]
            X=c[0]-a[0];Y=c[1]-a[1];Z=c[2]-a[2]
            hx=direction[1]*Z-direction[2]*Y
            hy=direction[2]*X-direction[0]*Z
            hz=direction[0]*Y-direction[1]*X
            det=x*hx+y*hy+z*hz
            if abs(det)<1e-10:continue
            inverse=1./det
            sx=origin[0]-a[0];sy=origin[1]-a[1];sz=origin[2]-a[2]
            u=inverse*(sx*hx+sy*hy+sz*hz)
            if u<0. or u>1.:continue
            qx=sy*z-sz*y;qy=sz*x-sx*z;qz=sx*y-sy*x
            v=inverse*(direction[0]*qx+direction[1]*qy+direction[2]*qz)
            if v<0. or u+v>1.:continue
            distance=inverse*(X*qx+Y*qy+Z*qz)
            if .0001<distance<.18:
                result[i]=1.;break
    return result


class BangsShadow:
    def __init__(self,stage,shadow_level=.72,character_root="/World/Character"):
        self.mesh=UsdGeom.Mesh.Get(stage,character_root+'/Body')
        faces=np.asarray(self.mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        face_ids=[];bangs=[]
        for subset in UsdGeom.Subset.GetAllGeomSubsets(self.mesh):
            mat=UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
            name=mat.GetPrim().GetDisplayName().lower() if mat else ''
            if any(word in name for word in ('face','顔','脸')):
                face_ids.extend(subset.GetIndicesAttr().Get())
                cel=UsdShade.Shader(stage.GetPrimAtPath(mat.GetPath().AppendChild('CelSurface')))
                cel.CreateInput('receive_bangs',Sdf.ValueTypeNames.Bool).Set(True)
                cel.CreateInput('bangs_shadow_level',Sdf.ValueTypeNames.Float).Set(float(shadow_level))
            if any(word in name for word in ('前发','前髪','bang','fringe')):
                bangs.extend(subset.GetIndicesAttr().Get())
        if not face_ids or not bangs:
            raise ValueError('Face and bangs material subsets are required for bangs shadows')
        self.face_vertices=np.unique(faces[np.asarray(face_ids)].flatten())
        self.bangs=faces[np.asarray(bangs)]
        self.rest=self.mesh.GetPointsAttr().Get()
        self.cache=UsdSkel.Cache()
        self.cache.Populate(UsdSkel.Root.Get(stage,character_root),Usd.PrimDefaultPredicate)
        self.query=self.cache.GetSkelQuery(UsdSkel.Skeleton.Get(stage,character_root+'/Skeleton'))
        self.skin=self.cache.GetSkinningQuery(self.mesh.GetPrim())
        self.mask=UsdGeom.PrimvarsAPI(self.mesh).CreatePrimvar('bang_shadow',Sdf.ValueTypeNames.FloatArray,'vertex')
        receivers=stage.GetPrimAtPath(character_root).GetAttribute('appearance:shadowReceivers').Get() or []
        self.receiver_masks=[UsdGeom.PrimvarsAPI(UsdGeom.Mesh.Get(stage,p)).CreatePrimvar('bang_shadow',Sdf.ValueTypeNames.FloatArray,'vertex') for p in receivers]
        self.direction=np.array([-.35,-.65,.68],dtype=np.float64)
        self.direction/=np.linalg.norm(self.direction)
        self.last_frame=None;self.updates=0
        # Compile before starting music or timing the render loop.
        cast_shadow(np.zeros((1,3)),np.zeros((1,3,3)),self.direction)

    def update(self,frame,time_code=None):
        tick=int(frame)
        if tick==self.last_frame:return
        self.last_frame=tick
        points=Vt.Vec3fArray(self.rest)
        transforms=self.query.ComputeSkinningTransforms(Usd.TimeCode.Default() if time_code is None else time_code)
        if not self.skin.ComputeSkinnedPoints(transforms,points,Usd.TimeCode.Default() if time_code is None else time_code):
            raise RuntimeError('Bangs shadow skinning failed')
        xyz=np.asarray(points,dtype=np.float64)
        mask=cast_shadow(xyz[self.face_vertices],xyz[self.bangs],self.direction)
        values=np.zeros(len(xyz),dtype=np.float32)
        values[self.face_vertices]=mask
        self.mask.Set(Vt.FloatArray(values.tolist()))
        for receiver in self.receiver_masks:receiver.Set(Vt.FloatArray(values.tolist()))
        self.updates+=1
        if self.updates%60==1:
            print(f'BANGS_SHADOW frame={frame:.1f} occluded={int(mask.sum())}/{len(mask)}',flush=True)
