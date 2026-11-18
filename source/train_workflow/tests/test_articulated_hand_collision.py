"""Visual mesh coverage and articulated-finger collision regression."""
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from pxr import Usd,UsdGeom,UsdPhysics
from source.train_workflow.utils.motion.self_collision import G1CollisionModel
from source.train_workflow.utils.motion.sync import get_default_compile_joint_names


class ArticulatedHandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names=get_default_compile_joint_names();cls.usd=ROOT/'assets/g1_29dof_o6_hand_V3.usd'
        cls.model=G1CollisionModel(cls.usd,cls.names,'cpu');cls.stage=Usd.Stage.Open(str(cls.usd));cls.paths={}
        cls.cache=UsdGeom.XformCache()
        for prim in cls.stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                for path in UsdPhysics.Joint(prim).GetBody1Rel().GetTargets():cls.paths[path.name]=path

    @classmethod
    def vertices(cls,body):
        prim=cls.stage.GetPrimAtPath(cls.paths[body]);inverse=np.linalg.inv(np.array(cls.cache.GetLocalToWorldTransform(prim)).T)
        points=[]
        for child in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
            if not child.IsA(UsdGeom.Mesh) or '/visuals/' not in str(child.GetPath()):continue
            transform=inverse@np.array(cls.cache.GetLocalToWorldTransform(child)).T
            p=np.asarray(UsdGeom.Mesh(child).GetPointsAttr().Get(),dtype=float)
            points.append(p@transform[:3,:3].T+transform[:3,3])
        return np.concatenate(points)

    def test_all_finger_links_have_articulated_head_collision_pairs(self):
        for parts in self.model.hand_parts.values():
            for name,_ in parts:
                if 'hand_base_link' not in name:self.assertIn(f'{name} / head_link',self.model.pair_labels)

    def test_all_leg_housings_are_covered_as_hand_obstacles(self):
        shapes={s.name:s for s in self.model.shapes}
        for side in ('left','right'):
            for part in ('hip_pitch','hip_roll','ankle_pitch'):
                name=f'{side}_{part}_link';self.assertIn(name,shapes)
                self.assertIn(f'lh_hand_base_link / {name}',self.model.pair_labels)
                self.assertIn(f'rh_thumb_distal / {name}',self.model.pair_labels)
                shape=shapes[name];points=self.vertices(name);axis=shape.b-shape.a
                t=np.clip((points-shape.a)@axis/max(float(axis@axis),1e-12),0,1)
                self.assertLessEqual(float((np.linalg.norm(points-shape.a-t[:,None]*axis,axis=-1)-shape.radius).max()),1e-7)

    def test_finger_capsules_cover_every_visual_vertex(self):
        shapes={s.name:s for s in self.model.shapes}
        for parts in self.model.hand_parts.values():
            for name,_ in parts:
                if 'hand_base_link' in name:continue
                shape=shapes[name];points=self.vertices(name);axis=shape.b-shape.a
                t=np.clip((points-shape.a)@axis/max(float(axis@axis),1e-12),0,1)
                signed=np.linalg.norm(points-shape.a-t[:,None]*axis,axis=-1)-shape.radius
                self.assertLessEqual(float(signed.max()),1e-7,name)

    def test_bent_thumb_leaves_neutral_box_but_remains_in_articulated_capsule(self):
        q=torch.zeros((1,len(self.names)))
        for joint,value in [('lh_thumb_cmc_yaw',.4),('lh_thumb_cmc_pitch',.04),('lh_thumb_ip',.09)]:q[:,self.names.index(joint)]=value
        r,p=self.model.fk(q);body='lh_thumb_distal';points=self.vertices(body)
        world=points@r[body][0].numpy().T+p[body][0].numpy()
        local=(world-p['lh_hand_base_link'][0].numpy())@r['lh_hand_base_link'][0].numpy()
        shape=next(s for s in self.model.shapes if s.name=='lh_hand_base_link')
        self.assertGreater(float((abs(local-shape.a)-shape.b).max()),.01)

    def test_collision_follows_finger_angles_and_has_finite_gradients(self):
        q=torch.zeros((2,len(self.names)));q[1,self.names.index('lh_thumb_cmc_yaw')]=.4;q.requires_grad_(True)
        distances=self.model.distances(*self.model.fk(q));column=self.model.pair_labels.index('lh_thumb_distal / head_link')
        self.assertGreater(float(abs(distances[0,column]-distances[1,column]).detach()),1e-4)
        distances[:,column].sum().backward();self.assertTrue(torch.isfinite(q.grad).all())


if __name__=='__main__':unittest.main()
