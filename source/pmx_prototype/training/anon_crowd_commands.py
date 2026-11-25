"""Seeded per-character commands for the separate crowd exploration viewer."""
import math
import numpy as np


class RandomCrowdCommands:
    def __init__(self,count,seed=42,*,max_speed=.55,max_yaw=.4,radius=7.):
        if count<1 or max_speed<=0 or max_yaw<0 or radius<=2:
            raise ValueError('Invalid crowd count, speed, turn limit or arena radius')
        self.count=count;self.seed=seed;self.max_speed=max_speed;self.max_yaw=max_yaw;self.radius=radius
        self.reset()

    def reset(self):
        self.generators=[np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(self.count)]
        self.until=np.zeros(self.count);self.commands=np.zeros((self.count,3),np.float32)

    def sample(self,time,positions,headings,*,paused=False):
        positions=np.asarray(positions);headings=np.asarray(headings)
        if positions.shape!=(self.count,2) or headings.shape!=(self.count,):
            raise ValueError('Crowd command positions/headings have the wrong shape')
        if paused:return np.zeros_like(self.commands)
        for i,rng in enumerate(self.generators):
            if time<1.:continue  # Start from a settled stand, as in the inspector.
            if time>=self.until[i]:
                angle=rng.uniform(-math.pi,math.pi);speed=rng.uniform(min(.15,self.max_speed*.5),self.max_speed)
                self.commands[i]=[speed*math.cos(angle),speed*math.sin(angle),rng.uniform(-self.max_yaw,self.max_yaw)]
                if rng.random()<.15:self.commands[i]=0.
                self.until[i]=time+rng.uniform(2.,5.)
            distance=np.linalg.norm(positions[i])
            if distance>self.radius-1.5:
                # Request an ordinary policy translation toward the arena center.
                # Never teleport or write a root pose to keep a character inside.
                world=-positions[i]/max(distance,1e-6)*min(.35,self.max_speed)
                c,s=np.cos(headings[i]),np.sin(headings[i])
                self.commands[i]=[s*world[0]-c*world[1],c*world[0]+s*world[1],0.]
        return self.commands.copy()


class BodyPoseMapper:
    """Convert actual body-link world transforms to PMX local bone poses."""
    def __init__(self,manifest,links):
        self.owners=np.array([links.index(n) for n in manifest['bone_owners']])
        self.rest=np.asarray(manifest['rest_positions'])
        self.origins=np.array([manifest['links'][n]['origin'] for n in manifest['bone_owners']])
        self.parents=np.asarray(manifest['parents']);self.children=np.flatnonzero(self.parents>=0)

    def __call__(self,transforms):
        from scipy.spatial.transform import Rotation
        t=transforms[self.owners];rotations=Rotation.from_quat(t[:,3:])
        wp=t[:,:3]+rotations.apply(self.rest-self.origins);wq=t[:,3:].copy()
        p,q=wp.copy(),wq.copy();parent=self.parents[self.children]
        inverse=Rotation.from_quat(wq[parent]).inv()
        p[self.children]=inverse.apply(wp[self.children]-wp[parent])
        q[self.children]=(inverse*Rotation.from_quat(wq[self.children])).as_quat()
        return p,q,wp,wq
