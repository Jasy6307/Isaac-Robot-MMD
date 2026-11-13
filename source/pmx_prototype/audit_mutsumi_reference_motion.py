"""Compare pre/post clearance teacher skeletons, in physical world metres."""
from pathlib import Path
import argparse,json,unicodedata
import numpy as np
from pxr import Usd,UsdSkel,UsdGeom
p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--current',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
names=['左足IK','右足IK','左足首','右足首','上半身','下半身','頭','左腕','右腕']
def read(path):
 s=Usd.Stage.Open(str(path));sk=UsdSkel.Skeleton.Get(s,'/Character/Skeleton')
 joints=[unicodedata.normalize('NFKC',str(x)) for x in sk.GetJointNamesAttr().Get()];ids=[joints.index(x) for x in names]
 cache=UsdSkel.Cache();cache.Populate(UsdSkel.Root.Get(s,'/Character'),Usd.PrimDefaultPredicate);query=cache.GetSkelQuery(sk)
 xf=UsdGeom.XformCache().GetLocalToWorldTransform(sk.GetPrim());values=[]
 for frame in range(int(s.GetEndTimeCode())+1):
  mats=query.ComputeJointSkelTransforms(frame)
  values.append([tuple(xf.Transform(mats[i].ExtractTranslation())) for i in ids])
 return np.asarray(values)
old,new=read(a.previous),read(a.current)
assert old.shape==new.shape,(old.shape,new.shape)
assert np.isfinite(new).all(),'Nonfinite skeleton position'
d=np.linalg.norm(old-new,axis=-1)
body=json.loads(a.current.with_suffix('.json').read_text(encoding='utf-8'))
assert body['leg_clearance'] is None,'Extra clearance still active'
assert np.max(d[:,4:])<1e-6,'Nonleg joint changed'
r=dict(frames=len(new),extra_leg_clearance=False,max_world_change_m={name:float(d[:,i].max()) for i,name in enumerate(names)},native_ik_errors=body.get('ik'),upper_body_unchanged=True,finite=True,previous=str(a.previous.resolve()),current=str(a.current.resolve()))
a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print('MOTION_AUDIT_PASS',a.current.parent.name,len(new),float(d[:,:2].max()),flush=True)
