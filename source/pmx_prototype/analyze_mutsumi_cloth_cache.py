"""Matched-time cloth diagnostics against common leg capsule proxies, not triangle contact proof."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdSkel,UsdGeom,Vt,Gf
from convert_pmx import read_pmx
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=7011);p.add_argument('--stride',type=int,default=30);a=p.parse_args()
s=Usd.Stage.Open(str(a.cache.resolve()));sk=UsdSkel.Skeleton.Get(s,'/Character/Skeleton');names=list(sk.GetJointNamesAttr().Get());bind=sk.GetBindTransformsAttr().Get();rest=np.array([x.ExtractTranslation() for x in bind]);by={n:i for i,n in enumerate(names)}
c=UsdSkel.Cache();c.Populate(UsdSkel.Root.Get(s,'/Character'),Usd.PrimDefaultPredicate);q=c.GetSkelQuery(sk);mesh=UsdGeom.Mesh.Get(s,'/Character/Part_09');skin=c.GetSkinningQuery(mesh.GetPrim());points=mesh.GetPointsAttr().Get();used=np.unique(mesh.GetFaceVertexIndicesAttr().Get())
m=read_pmx(Path('media/mutsmi/mutsmi v1.pmx'),physics=True);raw=np.array([v[0] for v in m['vertices']]);scale=np.ptp(np.array(points)[:,2])/np.ptp(raw[:,1]);zbase=raw[:,1].min();proxies=[]
for b in m['rigid_bodies']:
 if b['bone']<0:continue
 name=m['bones'][b['bone']]['name']
 if name not in ['左足','右足','左ひざ','右ひざ']:continue
 i=by[name];center=(np.array(b['position'])[[0,2,1]]-[0,0,zbase])*scale;axis=Rotation.from_euler('xyz',b['rotation']).apply([0,1,0])[[0,2,1]]*b['size'][1]*scale/2
 proxies.append((i,center-rest[i]-axis,center-rest[i]+axis,b['size'][0]*scale))
rows=1+max(int(n.split('_')[1]) for n in names if n.startswith('qun_'));grid=np.array([[by[f'qun_{r}_{col}'] for col in range(21)] for r in range(rows)])
records=[]
for f in range(0,min(a.frames,int(s.GetEndTimeCode())+1),a.stride):
 mats=q.ComputeJointSkelTransforms(f);pos=np.array([x.ExtractTranslation() for x in mats]);v=pos[grid]
 u,w=v[1:-1]-v[:-2],v[2:]-v[1:-1];angles=np.degrees(np.arccos(np.clip(np.sum(u*w,axis=-1)/np.maximum(np.linalg.norm(u,axis=-1)*np.linalg.norm(w,axis=-1),1e-10),-1,1)))
 pp=Vt.Vec3fArray(points);assert skin.ComputeSkinnedPoints(q.ComputeSkinningTransforms(f),pp,f);vertices=np.array(pp)[used]
 depth=np.full(len(vertices),-np.inf)
 for i,aa,bb,radius in proxies:
  pa=np.array(mats[i].Transform(Gf.Vec3d(*aa)));pb=np.array(mats[i].Transform(Gf.Vec3d(*bb)));axis=pb-pa;t=np.clip((vertices-pa)@axis/max(np.dot(axis,axis),1e-12),0,1);dist=np.linalg.norm(vertices-(pa+t[:,None]*axis),axis=-1);depth=np.maximum(depth,radius-dist)
 records.append(dict(frame=f,fold_fraction=float(np.mean(angles>90)),leg_proxy_vertex_fraction_over_2mm=float(np.mean(depth>.002)),leg_proxy_vertex_penetration_p95_mm=float(np.percentile(np.maximum(depth,0),95)*1000),leg_proxy_vertex_penetration_max_mm=float(max(0,depth.max())*1000)))
r=dict(cache=str(a.cache.resolve()),sample_stride_frames=a.stride,sample_count=len(records),common_leg_proxy_radius_scale=1.,metrics={k:float(np.mean([v[k] for v in records])) for k in records[0] if k!='frame'},samples=records,limitation='Skinned vertices versus common PMX leg capsule proxies; not mesh-mesh ground truth')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2),encoding='utf-8');print('CLOTH_CACHE_AUDIT',a.cache,r['metrics'],flush=True)
