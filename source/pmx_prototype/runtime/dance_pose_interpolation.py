"""Interpolate adjacent 30 Hz solved poses for a 60 Hz viewer."""
import numpy as np


def interpolate_pose(a, b, alpha):
    alpha=max(0.,min(1.,float(alpha)))
    if alpha == 0.: return a['p'],a['q']
    if alpha == 1.: return b['p'],b['q']
    p=(1.-alpha)*a['p']+alpha*b['p']
    q0=np.asarray(a['q']);q1=np.asarray(b['q'])
    dot=np.sum(q0*q1,axis=-1,keepdims=True)
    q1=np.where(dot<0.,-q1,q1);dot=np.clip(np.abs(dot),0.,1.)
    theta=np.arccos(dot);denom=np.sin(theta)
    safe=np.where(denom>1e-8,denom,1.)
    w0=np.where(denom>1e-8,np.sin((1.-alpha)*theta)/safe,1.-alpha)
    w1=np.where(denom>1e-8,np.sin(alpha*theta)/safe,alpha)
    q=w0*q0+w1*q1
    q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
    return p,q


class PoseInterpolator:
    def __init__(self,state,now,source_fps=30):
        self.previous=self.current=state
        self.started=now
        self.duration=1./source_fps
    def push(self,state,now):
        if state.get('restart') or state['frame']<self.current['frame']:
            self.previous=self.current=state
        else:
            self.previous=self.current
            self.current=state
        self.started=now
    def sample(self,now):
        alpha=max(0.,min(1.,(now-self.started)/self.duration))
        p,q=interpolate_pose(self.previous,self.current,alpha)
        state=dict(self.current,p=p,q=q,frame=self.previous['frame']+(self.current['frame']-self.previous['frame'])*alpha)
        state['restart']=False
        return state
