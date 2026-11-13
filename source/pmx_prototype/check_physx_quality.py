"""CPU invariants for bend forces and smooth skin frames, without starting Kit."""
import ast
from pathlib import Path
import numpy as np
from realtime_xpbd import njit, reconstruct, swing, multiply, rotate, fk

tree = ast.parse(Path(__file__).with_name('physx_secondary.py').read_text(encoding='utf-8'))
namespace = dict(np=np, reconstruct=reconstruct, swing=swing, multiply=multiply, rotate=rotate)
for name in ('curvature_forces', 'reconstruct_smooth', 'capsule_approach_damping'):
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'quality_kernel_check', 'exec'), namespace)
    namespace[name] = njit(cache=False)(namespace[name])
force = namespace['curvature_forces']
target = np.array([[0.,0.,0.], [0.,0.,.1], [0.,0.,.2]])
triplets = np.array([[0,1,2]])
k, d, dt = np.array([4.]), np.array([.006]), 1/120
assert np.max(np.abs(force(target,target,target,target,triplets,k,d,dt))) == 0
bent = target.copy(); bent[1,0] = .02
f = force(bent,target,bent,target,triplets,k,d,dt)
assert f[1,0] < 0 and np.max(np.abs(f.sum(axis=0))) < 1e-12
translation = np.array([3.,-2.,1.])
translated = force(bent+translation,target+translation,bent+translation,target+translation,triplets,k,d,dt)
assert np.max(np.abs(f-translated)) < 1e-12
friction = force(target,target,bent,target,triplets,np.array([0.]),d,dt)
assert np.sum(friction*(target-bent)/dt) < 0

wp = np.array([[0.,0.,.1*i] for i in range(5)])
q = np.tile([0.,0.,0.,1.],(5,1))
parents = np.array([-1,0,1,2,3])
lp = wp.copy(); lp[1:] -= wp[:-1]
dynamic = np.array([False,True,True,True,True])
native = wp.copy(); native[2,0] = .025; native[3,0] = .01
outp,outq,worldp,worldq = namespace['reconstruct_smooth'](
    lp,q,wp,q,parents,dynamic,native,np.array([1,2,3,4,3]),
    np.array([0,1,1,2,3]),np.array([0,2,3,4,4]),
    np.array([0,1,1,1,2]),np.array([0,3,4,4,4]))
assert np.max(np.abs(worldp-native)) < 1e-12
assert np.max(np.abs(fk(outp,outq,parents)[0]-native)) < 1e-10
assert np.max(np.abs(np.linalg.norm(outq,axis=1)-1)) < 1e-10
# A skirt patch twisted 90 degrees around its unchanged vertical tangent:
# the second material direction must recover roll, without moving any node.
patch = np.array([[.1*(c-1),0.,.1*r] for r in range(3) for c in range(3)])
turned = patch.copy(); turned[:,1] = patch[:,0]; turned[:,0] = 0
parents = np.full(9,-1,dtype=np.int64)
identity = np.tile([0.,0.,0.,1.],(9,1))
up = np.array([max(r-1,0)*3+c for r in range(3) for c in range(3)])
down = np.array([min(r+1,2)*3+c for r in range(3) for c in range(3)])
left = np.array([r*3+max(c-1,0) for r in range(3) for c in range(3)])
right = np.array([r*3+min(c+1,2) for r in range(3) for c in range(3)])
outp,outq,worldp,worldq = namespace['reconstruct_smooth'](
    patch,identity,patch,identity,parents,np.ones(9,dtype=np.bool_),turned,down,
    up,down,up,down,left,right)
assert np.max(np.abs(worldp-turned)) < 1e-12
assert np.max(np.abs(fk(outp,outq,parents)[0]-turned)) < 1e-10
for quat in worldq:
    assert np.linalg.norm(rotate(quat,np.array([1.,0.,0.]))-np.array([0.,1.,0.])) < 1e-9
cushion=namespace['capsule_approach_damping']
ca=np.array([[0.,0.,-.1]]);cb=np.array([[0.,0.,.1]])
points=np.array([[.105,0.,0.],[.15,0.,0.]])
velocity=np.array([[-1.,0.,0.],[-1.,0.,0.]])
pr=np.full(2,.02);br=np.array([.08])
args=(pr,ca,cb,ca,cb,br,1/480,.02,.5,.24)
f=cushion(points,velocity,*args)
assert f[0,0]>0 and np.max(np.abs(f[1]))==0
assert np.sum(f*velocity)<0
assert np.max(np.abs(cushion(points,-velocity,*args)))==0
assert np.max(np.abs(cushion(points,np.zeros_like(velocity),*args)))==0
shift=np.array([3.,-2.,1.])
moved=cushion(points+shift,velocity,pr,ca+shift,cb+shift,ca+shift,cb+shift,br,1/480,.02,.5,.24)
assert np.max(np.abs(moved-f))<1e-10
boundary=np.array([[.12,0.,0.],[.12-1e-6,0.,0.]])
assert np.max(np.abs(cushion(boundary,velocity,*args)))<1e-8
support=cushion(points,np.zeros_like(velocity),*args,2.)
assert support[0,0]>0 and np.max(np.abs(support[1]))==0
assert np.max(np.abs(cushion(boundary,np.zeros_like(velocity),*args,2.)))<1e-8
print('PHYSX_QUALITY_CHECK_PASS restoring_force balanced_force dissipative_damping unchanged_native_positions cloth_material_frame contact_cushion_dissipative_continuous')
