"""Add a skinned PhysX skirt lattice; retain the approved body animation."""

from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source
from pathlib import Path
import argparse
import json
import numpy as np
from pxr import Usd, UsdSkel, UsdGeom, Gf, Vt
from convert_pmx import read_pmx

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser()
p.add_argument('--usd',type=Path,default=ROOT/'output/mutsumi_pmx/IRIS_OUT_clearance.usdc')
p.add_argument('--pmx',type=Path,default=ROOT/'media/mutsmi/mutsmi v1.pmx')
p.add_argument('--profile', choices=['legacy','smooth'], default='smooth')
p.add_argument('--skirt-length-scale', type=float, default=1.0,
               help='Vertical skirt length ratio about the fixed waist; 0 < ratio <= 1')
p.add_argument('--skirt-max-stretch', type=float, default=1.02,
               help='Hard native D6 length cap ratio for skirt edges and waist tethers')
p.add_argument('--output',type=Path,default=ROOT/'output/mutsumi_physx')
a=p.parse_args()
if not np.isfinite(a.skirt_length_scale) or not 0 < a.skirt_length_scale <= 1:
 p.error('skirt-length-scale must be finite and in (0, 1]')
if not np.isfinite(a.skirt_max_stretch) or not 1 <= a.skirt_max_stretch <= 1.1:
 p.error('skirt-max-stretch must be finite and in [1, 1.1]')
a.output.mkdir(parents=True,exist_ok=True)
model=read_pmx(a.pmx); source=Usd.Stage.Open(str(a.usd.resolve()))
dest=a.output/'character.usdc'
stage=Usd.Stage.CreateNew(str(dest));stage.GetRootLayer().subLayerPaths=[str(a.usd.resolve()).replace('\\','/')]
stage.SetDefaultPrim(stage.GetPrimAtPath('/Character'))
stage.SetTimeCodesPerSecond(30);stage.SetStartTimeCode(0);stage.SetEndTimeCode(source.GetEndTimeCode())
sk=UsdSkel.Skeleton.Get(stage,'/Character/Skeleton')
joints=list(sk.GetJointsAttr().Get());names=list(sk.GetJointNamesAttr().Get())
rest=list(sk.GetRestTransformsAttr().Get());bind=list(sk.GetBindTransformsAttr().Get())
mesh=UsdGeom.Mesh.Get(stage,'/Character/Part_09')
points=np.asarray(mesh.GetPointsAttr().Get()).copy();faces=np.array(mesh.GetFaceVertexIndicesAttr().Get());ids=np.unique(faces)
raw=np.asarray([v[0] for v in model['vertices']]);scale=np.ptp(points[:,2])/np.ptp(raw[:,1]);base=raw[:,1].min()
cloth=points[ids]; top,bottom=cloth[:,2].max(),cloth[:,2].min()
original_length=float(top-bottom)
if a.skirt_length_scale != 1:
 points[ids,2]=top+(points[ids,2]-top)*a.skirt_length_scale
 mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
 normals=np.asarray(mesh.GetNormalsAttr().Get()).copy()
 normals[ids,2]/=a.skirt_length_scale
 normals[ids]/=np.maximum(np.linalg.norm(normals[ids],axis=1)[:,None],1e-12)
 mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
 cloth=points[ids];bottom=cloth[:,2].min()
skirt_geometry=dict(length_scale=a.skirt_length_scale,original_length_m=original_length,
                   shortened_length_m=float(top-bottom),hem_raise_m=original_length-float(top-bottom))
center=np.array([0.,np.median(cloth[:,1])])
angles=np.arctan2(cloth[:,1]-center[1],cloth[:,0]-center[0])%(2*np.pi)
radii=np.linalg.norm(cloth[:,:2]-center,axis=1)
rows,cols=9,21; hip=names.index('下半身'); bones=[];bodies=[];grid=np.empty((rows,cols),int)
radius_grid=np.empty((rows,cols))
for row in range(rows):
 for col in range(cols):
  angle=col*2*np.pi/cols;z=top-(top-bottom)*row/(rows-1)
  da=np.abs((angles-angle+np.pi)%(2*np.pi)-np.pi)
  score=da**2+((cloth[:,2]-z)/(top-bottom)*3)**2
  radius_grid[row,col]=np.median(radii[np.argsort(score)[:12]])
for col in range(cols):
 for row in range(rows):
  angle=col*2*np.pi/cols;z=top-(top-bottom)*row/(rows-1)
  da=np.abs((angles-angle+np.pi)%(2*np.pi)-np.pi)
  score=da**2+((cloth[:,2]-z)/(top-bottom)*3)**2
  radius=float(radius_grid[row,col])
  position=np.array([center[0]+radius*np.cos(angle),center[1]+radius*np.sin(angle),z])
  parent=hip if row==0 else grid[row-1,col]
  index=len(joints); grid[row,col]=index
  name=f'qun_{row}_{col}';path=joints[parent]+'/'+name
  parent_position=np.array(bind[parent].ExtractTranslation())
  joints.append(path);names.append(name)
  rest.append(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*(position-parent_position))))
  bind.append(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*position)))
  pmx_position=(position/scale+np.array([0,0,base]))[[0,2,1]]
  pmx_parent=127 if row==0 else len(model['bones'])+len(bones)-1
  bones.append(dict(name=name,position=pmx_position.tolist(),parent=pmx_parent,append=None,flags=0,ik=None,layer=0))
  bodies.append(dict(name=name,bone=len(model['bones'])+len(bones)-1,group=8,mask=65535,shape=0,size=[.012/scale,0,0],position=pmx_position.tolist(),rotation=[0,0,0],mode=1))
sk.GetJointsAttr().Set(joints);sk.GetJointNamesAttr().Set(names)
sk.GetRestTransformsAttr().Set(rest);sk.GetBindTransformsAttr().Set(bind)
# Cubic B-spline displacement weights eliminate bilinear cell-boundary creases.
binding=UsdSkel.BindingAPI(mesh);ji=np.asarray(binding.GetJointIndicesPrimvar().Get()).copy().reshape(-1,4)
jw=np.asarray(binding.GetJointWeightsPrimvar().Get()).copy().reshape(-1,4)
if a.profile=='smooth':
 old_ji,old_jw=ji,jw
 ji=np.zeros((len(points),16),dtype=np.int32);jw=np.zeros((len(points),16),dtype=np.float32)
 ji[:,:4]=old_ji;jw[:,:4]=old_jw
 binding.GetJointIndicesPrimvar().SetElementSize(16)
 binding.GetJointWeightsPrimvar().SetElementSize(16)
def cubic(t):
 return np.array([(1-t)**3,3*t**3-6*t*t+4,-3*t**3+3*t*t+3*t+1,t**3])/6
for vertex,angle in zip(ids,angles):
 v=np.clip((top-points[vertex,2])/(top-bottom)*(rows-1),0,rows-1-1e-6)
 u=angle/(2*np.pi)*cols;r=int(v);c=int(u)%cols;fv=v-r;fu=u-int(u)
 if a.profile=='smooth':
  weights=np.outer(cubic(fv),cubic(fu)).flatten()
  ji[vertex]=[grid[np.clip(r+dr,0,rows-1),(c+dc)%cols] for dr in (-1,0,1,2) for dc in (-1,0,1,2)]
  jw[vertex]=weights
  continue
 ji[vertex]=[grid[r,c],grid[r,(c+1)%cols],grid[r+1,c],grid[r+1,(c+1)%cols]]
 jw[vertex]=[(1-fv)*(1-fu),(1-fv)*fu,fv*(1-fu),fv*fu]
binding.GetJointIndicesPrimvar().Set(Vt.IntArray.FromNumpy(ji.flatten().astype(np.int32)))
binding.GetJointWeightsPrimvar().Set(Vt.FloatArray.FromNumpy(jw.flatten().astype(np.float32)))
stage.GetRootLayer().Save()
(a.output/'rig.json').write_text(json.dumps(dict(bones=bones,bodies=bodies,rows=rows,columns=cols,skirt_geometry=skirt_geometry,skirt_max_stretch=a.skirt_max_stretch,profile=a.profile,skirt_fabric_relaxation=a.profile=='smooth',locked_hair_bones=[i for i,b in enumerate(model['bones']) if b['name'].startswith('front hair')] if a.profile=='smooth' else [],cloth_translation_only=a.profile=='smooth',credits='很ying的哈呵'),ensure_ascii=False,indent=2),encoding='utf-8')
print('SKIRT_GEOMETRY',json.dumps(skirt_geometry))
print('MUTSUMI_PHYSX_RIG_READY',len(bones),'skirt nodes',len(ids),'skirt vertices')
