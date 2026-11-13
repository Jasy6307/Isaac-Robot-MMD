"""Deterministic fixed and continuously changing velocity commands."""
import numpy as np


def segments(demo=False,boundary=False,pivots=False):
    if boundary:return [(f'boundary_{i}_{sign}',4) for i in range(8) for sign in ['pos','neg']]+[('stop',4)]
    if demo:return [('stand',3),('diagonal_45',5),('diagonal_135',5),('diagonal_225',5),('diagonal_315',5),('speed_ramp',16),('direction_sweep',16),('combined_turn',20)]+([('pivot_pos',6),('pivot_neg',6)] if pivots else [])+[('stop',4)]
    return [(f'heading_{i}',4) for i in range(16)]+[(f'speed_{v}',5) for v in [.05,.1,.2,.3]]+ [('speed_ramp',16),('direction_sweep',16),('combined_turn',20),('stop',4)]


def command(label,t,seconds):
    if label in ['stop','stand']:return [0.,0.,0.]
    if label.startswith('pivot_'):return [0.,0.,.6 if label=='pivot_pos' else -.6]
    if label.startswith('boundary_'):
        _,i,sign=label.split('_');angle=int(i)*np.pi/4
        return [.4*np.cos(angle),.4*np.sin(angle),.6 if sign=='pos' else -.6]
    if label.startswith('heading_'):
        angle=int(label.split('_')[1])*np.pi/8;return [.4*np.cos(angle),.4*np.sin(angle),0.]
    if label.startswith('diagonal_'):
        angle=int(label.split('_')[1])*np.pi/180;return [.4*np.cos(angle),.4*np.sin(angle),0.]
    if label.startswith('speed_') and label!='speed_ramp':return [float(label.split('_')[1]),0.,0.]
    if label=='speed_ramp':return [.4*np.sin(np.pi*t/seconds)**2,0.,0.]
    if label=='direction_sweep':
        a=2*np.pi*t/seconds;return [.28*np.cos(a),.28*np.sin(a),0.]
    a=2*np.pi*t/seconds
    return [.4*np.cos(a),.4*np.sin(a),.6*np.sin(a)]


def demo_command(t,pivots=False):
    for label,seconds in segments(True,pivots=pivots):
        if t<seconds:return command(label,t,seconds)
        t-=seconds
    return [0.,0.,0.]
