"""One render clock, independent pose interpolation and reset for each actor."""
import math
import time
import numpy as np
from dance_pose_interpolation import PoseInterpolator


class CrowdInterpolatedDisplay:
    def __init__(self,draw,clock=time.perf_counter,sleep=time.sleep):
        self.draw=draw;self.clock=clock;self.sleep=sleep
        self.period=1/60;self.deadline=clock();self.poses=[]
        self.times=[];self.source_frames=[];self.interpolated_frames=0

    def push(self,frame,poses,centers,headings,restarts):
        count=len(poses)
        if not count or any(len(values)!=count for values in (centers,headings,restarts)):
            raise ValueError('Crowd display actor counts differ')
        if self.poses and len(self.poses)!=count:raise ValueError('Crowd display actor count changed')
        now=self.clock();first=not self.poses
        for i,(p,q) in enumerate(poses):
            pose=dict(frame=frame,p=p,q=q,center=np.array(centers[i],dtype=float,copy=True),
                heading=float(headings[i]),restart=bool(restarts[i]))
            if first:self.poses.append(PoseInterpolator(pose,now))
            else:self.poses[i].push(pose,now)
        if first:self.deadline=now

    def present_if_due(self):
        now=self.clock()
        if not self.poses or now<self.deadline:return False
        self.deadline+=(int((now-self.deadline)/self.period)+1)*self.period
        samples=[];interpolated=False
        for poses in self.poses:
            pose=poses.sample(now);previous,current=poses.previous,poses.current
            span=current['frame']-previous['frame']
            alpha=(pose['frame']-previous['frame'])/span if span else 1.
            pose['center']=previous['center']*(1-alpha)+current['center']*alpha
            pose['heading']=previous['heading']+math.remainder(current['heading']-previous['heading'],math.tau)*alpha
            interpolated|=span>0 and 0.<alpha<1.
            samples.append(pose)
        if self.draw(samples) is False:return False
        self.times.append(self.clock());self.source_frames.append(min(pose['frame'] for pose in samples))
        self.interpolated_frames+=int(interpolated)
        return True

    def wait_until(self,deadline):
        if not self.poses:
            self.sleep(max(0.,deadline-self.clock()));return
        while self.clock()<deadline:
            self.present_if_due()
            remaining=min(deadline,self.deadline)-self.clock()
            if remaining>0:self.sleep(remaining)

    def finish(self):
        if not self.poses:return
        self.wait_until(max(poses.started+poses.duration for poses in self.poses))
        self.sleep(max(0.,self.deadline-self.clock()));self.present_if_due()
