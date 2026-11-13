"""Full-motion visible triangle mesh intersections; no vertex subsampling."""
from pathlib import Path
import sys,json,argparse,hashlib,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,h5py,torch,trimesh,fcl
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from pxr import Usd,UsdGeom,UsdPhysics
from source.train_workflow.utils.motion.self_collision import G1CollisionModel,Joint,_quat_matrix

parser=argparse.ArgumentParser();parser.add_argument('--h5',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stride',type=int,default=1)
parser.add_argument('--fail-on-intersection',action='store_true')
args=parser.parse_args()
from source.paths import resolve_dance_asset
if args.h5:
    args.h5 = resolve_dance_asset(args.h5)
out=args.output;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
if args.stride<1:parser.error('--stride must be positive')
with h5py.File(args.h5) as stream:
 names=list(stream['joint_names'].asstr()[:]);q=stream['joint_pos_delta'][:]+stream.attrs['record_baseline_joint_pos'];fps=float(stream.attrs['fps'])
if q.ndim!=2 or q.shape[1]!=len(names) or not len(q) or not np.isfinite(q).all() or not np.isfinite(fps) or fps<=0:
 raise ValueError('Invalid finite H5 trajectory or sample rate')
coordinates=np.arange(4*(len(q)-1)+1,dtype=float)/4
coordinates=coordinates[::args.stride]
lo=np.floor(coordinates).astype(int);hi=np.minimum(lo+1,len(q)-1);alpha=coordinates-lo
poses=(q[lo]*(1-alpha[:,None])+q[hi]*alpha[:,None]).astype(np.float32)
usd=ROOT/'assets/g1_29dof_o6_hand_V3.usd';stage=Usd.Stage.Open(str(usd));paths={};pending=[];adjacent=set();parents={}
for prim in stage.Traverse():
 if not prim.IsA(UsdPhysics.Joint):continue
 j=UsdPhysics.Joint(prim);b0=j.GetBody0Rel().GetTargets();b1=j.GetBody1Rel().GetTargets()
 if not b0 or not b1:continue
 paths[b0[0].name]=b0[0];paths[b1[0].name]=b1[0];adjacent.add(tuple(sorted((b0[0].name,b1[0].name))));parents[b1[0].name]=b0[0].name
 pending.append(Joint(prim.GetName(),b0[0].name,b1[0].name,np.array(j.GetLocalPos0Attr().Get()),_quat_matrix(j.GetLocalRot0Attr().Get()),np.array(j.GetLocalPos1Attr().Get()),_quat_matrix(j.GetLocalRot1Attr().Get()),prim.GetAttribute('physics:axis').Get(),-np.inf,np.inf))
model=G1CollisionModel(usd,names,'cpu');model.joints=[];seen={'pelvis'}
while pending:
 ready=[j for j in pending if j.parent in seen]
 if not ready:raise RuntimeError('Disconnected USD joint tree')
 for j in ready:model.joints.append(j);seen.add(j.child);pending.remove(j)
model.constants={j.name:tuple(model._tensor(v) for v in (j.p0,j.r0,j.p1,j.r1)) for j in model.joints}
with torch.no_grad():rot,pos=model.fk(model._tensor(poses))
rot={k:v.numpy() for k,v in rot.items()};pos={k:v.numpy() for k,v in pos.items()}
cache=UsdGeom.XformCache();meshes={};objects={};corners={};component_vertices={}
for body,path in paths.items():
 if body not in rot:continue
 inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T);points=[];faces=[];offset=0
 for prim in Usd.PrimRange(stage.GetPrimAtPath(path),Usd.TraverseInstanceProxies()):
  if not prim.IsA(UsdGeom.Mesh) or '/visuals/' not in str(prim.GetPath()):continue
  mesh=UsdGeom.Mesh(prim)
  if UsdGeom.Imageable(prim).ComputeVisibility()==UsdGeom.Tokens.invisible:continue
  verts=np.asarray(mesh.GetPointsAttr().Get(),dtype=float);mat=inv@np.array(cache.GetLocalToWorldTransform(prim)).T
  points.append(verts@mat[:3,:3].T+mat[:3,3]);indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get());counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());holes=set(mesh.GetHoleIndicesAttr().Get() or []);cursor=0
  for polygon_index,count in enumerate(counts):
   polygon=indices[cursor:cursor+count]+offset;cursor+=count
   if polygon_index in holes:continue
   faces.extend((polygon[0],polygon[k],polygon[k+1]) for k in range(1,count-1))
  offset+=len(verts)
 if not points or not faces:continue
 mesh=trimesh.Trimesh(vertices=np.concatenate(points),faces=np.asarray(faces),process=True);meshes[body]=mesh
 geometry=trimesh.collision.mesh_to_BVH(mesh);objects[body]=fcl.CollisionObject(geometry)
 lower,upper=mesh.bounds;corners[body]=np.array([[x,y,z] for x in (lower[0],upper[0]) for y in (lower[1],upper[1]) for z in (lower[2],upper[2])])
 if body.startswith(('lh_','rh_')):
  edges=mesh.edges_unique;graph=sparse.coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(mesh.vertices),len(mesh.vertices)))
  _,labels=connected_components(graph,directed=False);component_vertices[body]=np.unique(labels,return_index=True)[1]
 print('MESH',body,len(mesh.vertices),len(mesh.faces),'closed',mesh.is_watertight,flush=True)
bodies=list(meshes);body_index={n:i for i,n in enumerate(bodies)};hands={n for n in bodies if n.startswith(('lh_','rh_'))}
pairs=[];excluded=[]
for i,a in enumerate(bodies):
 for b in bodies[i+1:]:
  if a not in hands and b not in hands:continue
  pair=tuple(sorted((a,b)))
  if pair in adjacent:excluded.append(pair);continue
  pairs.append(pair)
pair_indices=np.asarray([[body_index[a],body_index[b]] for a,b in pairs]);first,second=pair_indices.T
request=fcl.CollisionRequest(num_max_contacts=8,enable_contact=True);hits={};queries=0;started=time.monotonic()
for sample,coordinate in enumerate(coordinates):
 bounds=[]
 for body in bodies:
  obj=objects[body];obj.setTransform(fcl.Transform(rot[body][sample].astype(float),pos[body][sample].astype(float)))
  world=corners[body]@rot[body][sample].T+pos[body][sample];bounds.append([world.min(0),world.max(0)])
 bounds=np.asarray(bounds)
 possible=np.where((np.maximum(bounds[first,0],bounds[second,0])<np.minimum(bounds[first,1],bounds[second,1])-1e-8).all(-1))[0]
 for pair_index in possible:
  a,b=pairs[pair_index];result=fcl.CollisionResult();queries+=1
  intersect=fcl.collide(objects[a],objects[b],request,result)>0
  contained=False;depth=0.
  if not intersect and 'head_link' in (a,b):
   hand=b if a=='head_link' else a
   if hand in hands:
    points=meshes[hand].vertices[component_vertices[hand]]
    world=points@rot[hand][sample].T+pos[hand][sample]
    local=(world-pos['head_link'][sample])@rot['head_link'][sample]
    lower,upper=meshes['head_link'].bounds;inside=((local>=lower)&(local<=upper)).all(-1)
    if inside.any() and meshes['head_link'].contains(local[inside]).any():contained=True
  if not intersect and not contained:continue
  key=a+' / '+b;entry=hits.setdefault(key,{'pair':[a,b],'sample_coordinates':[],'surface_intersection_samples':0,'contained_component_samples':0,'max_fcl_contact_depth_m':0.,'examples':[]})
  entry['sample_coordinates'].append(float(coordinate));entry['surface_intersection_samples']+=int(intersect);entry['contained_component_samples']+=int(contained)
  contacts=result.contacts
  if contacts:depth=max(float(c.penetration_depth) for c in contacts)
  entry['max_fcl_contact_depth_m']=max(entry['max_fcl_contact_depth_m'],depth)
  if len(entry['examples'])<3:
   entry['examples'].append({'coordinate':float(coordinate),'seconds':coordinate/fps,'contacts':[{'triangle_a':int(c.b1),'triangle_b':int(c.b2),'position':np.asarray(c.pos).tolist(),'depth':float(c.penetration_depth)} for c in contacts[:3]],'contained_component':contained})
 if sample%60==0 or sample==len(coordinates)-1:
  print('SCAN',sample+1,'/',len(coordinates),'time_s',round(coordinate/fps,3),'pairs_with_intersections',len(hits),'narrow_queries',queries,'elapsed',round(time.monotonic()-started,1),flush=True)
  (out/'progress.json').write_text(json.dumps({'completed_samples':sample+1,'total_samples':len(coordinates),'pairs_with_intersections':len(hits),'elapsed_seconds':time.monotonic()-started}))
report={'passed':not bool(hits),'h5':str(args.h5.resolve()),'h5_sha256':hashlib.sha256(args.h5.read_bytes()).hexdigest(),'usd_sha256':hashlib.sha256(usd.read_bytes()).hexdigest(),
 'method':'visible-triangle-BVH-FCL-v1','fcl_version':getattr(fcl,'__version__','unknown'),'frames':len(q),'fps':fps,'sample_count':len(coordinates),
 'sample_step_frames':args.stride/4,'scan_coordinates':coordinates.tolist(),'hand_links':sorted(hands),'tested_pairs':[list(p) for p in pairs],
 'excluded_direct_joint_pairs':[list(p) for p in excluded],'meshes':{n:{'vertices':len(m.vertices),'triangles':len(m.faces),'watertight':m.is_watertight} for n,m in meshes.items()},
 'narrow_phase_queries':queries,'intersections':list(hits.values()),'elapsed_seconds':time.monotonic()-started,
 'note':'FCL triangle surface intersection is reported without silently discarding CAD contact; mesh penetration depth needs separate interpretation. Directly joint-connected parts alone are excluded.'}
(out/'mesh_scan.json').write_text(json.dumps(report,indent=2));print('MESH_SCAN_DONE',len(hits),'pairs',report['elapsed_seconds'],flush=True)
if hits and args.fail_on_intersection:raise SystemExit(2)
