"""Contact cleanup of captured foot paths; retain mocap swing shape and timing."""
import numpy as np


def plant_feet(positions,rotations,corners,contacts,mean_velocity,dt):
    count=len(positions);total=3*count
    source=np.tile(positions,(3,1,1))+np.arange(total)[:,None,None]*dt*mean_velocity
    orientation=np.tile(rotations,(3,1,1,1));contact=np.tile(contacts,(3,1))
    corrected=cleanup_world_feet(source,orientation,corners,contact,dt)
    return positions+(corrected-source)[count:2*count]


def cleanup_world_feet(source,orientation,corners,contact,dt):
    total=len(source);correction=np.zeros_like(source)
    for foot in range(2):
        edges=np.diff(np.r_[False,contact[:,foot],False].astype(int))
        starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        for first,last in zip(starts,ends):
            desired=source[first,foot].copy()
            for k in range(first,last):
                if k>first:
                    old=(orientation[k-1,foot]@corners[foot].T).T
                    near=old[:,2]<old[:,2].min()+.003
                    point=corners[foot,near].mean(0)
                    desired[:2]-=((orientation[k,foot]-orientation[k-1,foot])@point)[:2]
                desired[2]=.002-(orientation[k,foot]@corners[foot].T).T[:,2].min()
                correction[k,foot]=desired-source[k,foot]
        for last,next_first in zip(ends[:-1],starts[1:]):
            a=last-1;b=next_first
            if b<=a+1:continue
            d0=correction[a,foot];d1=correction[b,foot]
            m0=(correction[a,foot]-correction[max(0,a-1),foot])/dt
            m1=(correction[min(total-1,b+1),foot]-correction[b,foot])/dt
            duration=(b-a)*dt
            for k in range(a+1,b):
                s=(k-a)/(b-a)
                correction[k,foot]=(2*s**3-3*s**2+1)*d0+(s**3-2*s**2+s)*duration*m0+(-2*s**3+3*s**2)*d1+(s**3-s**2)*duration*m1
    return source+correction
