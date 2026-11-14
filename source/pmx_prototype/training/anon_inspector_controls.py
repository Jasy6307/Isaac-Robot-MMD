"""Keyboard command semantics, independent of Isaac's UI and simulation."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import math

def keyboard_event_names(event):
    kind=getattr(event.type,'name',str(event.type)).split('.')[-1]
    if kind not in {'KEY_PRESS','KEY_REPEAT','KEY_RELEASE'}:return None
    return getattr(event.input,'name',str(event.input)).split('.')[-1],kind


class InspectorCommands:
    MOVEMENT={'W','A','S','D','Q','E'}
    PLUS={'EQUAL','NUMPAD_ADD','NUMPAD_EQUAL'}
    MINUS={'MINUS','NUMPAD_SUBTRACT'}

    def __init__(self,config):
        self.fixed_forward=False;self.speed=.2;self.yaw_speed=.35
        self.held=set();self.pulses=set();self.enabled=True;self.reset_pending=False;self.view=0
        self.switching=False;self.walk_speed=.2;self.walk_yaw=.35
        self.configure(config)

    def configure(self,config):
        if not self.fixed_forward:self.walk_speed=self.speed;self.walk_yaw=self.yaw_speed
        self.fixed_forward=bool(config.get('amp_forward_only'))
        self.max_speed=float(config.get('max_speed',.4))
        self.max_yaw=float(config.get('max_yaw',.6))
        self.speed=float(config.get('amp_forward_speed',.4)) if self.fixed_forward else min(self.walk_speed,self.max_speed)
        self.yaw_speed=min(self.walk_yaw,self.max_yaw)
        self.stop();self.reset_pending=False
        self.notice='Hold movement keys; release to stop.'

    def stop(self):
        self.held.clear()
        self.pulses.clear()

    def set_speed(self,value):
        if not self.fixed_forward:self.speed=round(max(0.,min(self.max_speed,float(value))),3)

    def event(self,key,kind):
        tracked=key in self.MOVEMENT|self.PLUS|self.MINUS|{'SPACE','R','V'}
        if not tracked:return False
        if kind=='KEY_RELEASE':self.held.discard(key);return True
        if self.switching or not self.enabled:return True
        if key in self.MOVEMENT:
            if kind=='KEY_PRESS':
                if self.fixed_forward and key!='W':
                    self.stop()
                    self.notice='This policy supports forward/stop only. Switch to WALK for WASD and Q/E.'
                else:self.held.add(key);self.pulses.add(key);self.notice='Keyboard control active.'
        elif kind in {'KEY_PRESS','KEY_REPEAT'} and key in self.PLUS|self.MINUS:
            if self.fixed_forward:self.notice=f'Fixed trained speed: {self.speed:.2f} m/s. Switch to WALK for variable speed.'
            else:self.set_speed(self.speed+(.05 if key in self.PLUS else -.05));self.notice=f'Speed set to {self.speed:.2f} m/s.'
        elif kind=='KEY_PRESS':
            if key=='SPACE':self.stop();self.notice='Stop requested. Release and press a movement key to resume.'
            elif key=='R':self.stop();self.reset_pending=True;self.notice='Reset requested.'
            elif key=='V':self.view=(self.view+1)%4
        return True

    def command(self,consume=True):
        if self.switching or not self.enabled:return [0.,0.,0.]
        keys=self.held|self.pulses
        if consume:self.pulses.clear()
        if self.fixed_forward:return [self.speed if 'W' in keys else 0.,0.,0.]
        forward=float('W' in keys)-float('S' in keys)
        left=float('A' in keys)-float('D' in keys)
        scale=self.speed/max(1.,math.hypot(forward,left))
        yaw=(float('Q' in keys)-float('E' in keys))*self.yaw_speed
        return [forward*scale,left*scale,yaw]

    def consume_reset(self):
        value=self.reset_pending;self.reset_pending=False;return value
