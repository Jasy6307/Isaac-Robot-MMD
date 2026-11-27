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


def keys_for_command(command):
    """Map forward, left and yaw requests to the corresponding key cards."""
    keys = set()
    for value, (positive, negative) in zip(command, (('W', 'S'), ('A', 'D'), ('Q', 'E'))):
        if value > 1e-4:
            keys.add(positive)
        elif value < -1e-4:
            keys.add(negative)
    return frozenset(keys)


class InspectorCommands:
    MOVEMENT={'W','A','S','D','Q','E'}
    PLUS={'EQUAL','NUMPAD_ADD','NUMPAD_EQUAL'}
    MINUS={'MINUS','NUMPAD_SUBTRACT'}

    def __init__(self,config):
        self.fixed_forward=False;self.speed=.2;self.yaw_speed=.35
        self.held=set();self.pulses=set();self.enabled=True;self.reset_pending=False;self.push_pending=False;self.view=0
        self.walk_speed=.2;self.walk_yaw=.35
        self.configure(config)

    def configure(self,config):
        if not self.fixed_forward:self.walk_speed=self.speed;self.walk_yaw=self.yaw_speed
        self.fixed_forward=bool(config.get('amp_forward_only'))
        self.max_speed=float(config.get('max_speed',.4))
        self.max_yaw=float(config.get('max_yaw',.6))
        self.speed=float(config.get('amp_forward_speed',.4)) if self.fixed_forward else min(self.walk_speed,self.max_speed)
        self.yaw_speed=min(self.walk_yaw,self.max_yaw)
        self.stop();self.reset_pending=False;self.push_pending=False
        self.notice=''

    def stop(self):
        self.held.clear()
        self.pulses.clear()

    def set_speed(self,value):
        if not self.fixed_forward:self.speed=round(max(0.,min(self.max_speed,float(value))),3)

    def event(self,key,kind):
        tracked=key in self.MOVEMENT|self.PLUS|self.MINUS|{'SPACE','R','V'}
        if not tracked:return False
        if kind=='KEY_RELEASE':self.held.discard(key);return True
        if not self.enabled:return True
        if key in self.MOVEMENT:
            if kind=='KEY_PRESS':
                if self.fixed_forward and key!='W':
                    self.stop()
                    self.notice='Forward only [W].'
                else:self.held.add(key);self.pulses.add(key);self.notice=''
        elif kind in {'KEY_PRESS','KEY_REPEAT'} and key in self.PLUS|self.MINUS:
            if self.fixed_forward:self.notice=f'Fixed speed: {self.speed:.2f} m/s.'
            else:self.set_speed(self.speed+(.05 if key in self.PLUS else -.05));self.notice=''
        elif kind=='KEY_PRESS':
            if key=='SPACE':self.stop();self.notice='Stop requested.'
            elif key=='R':self.request_reset()
            elif key=='V':self.view=(self.view+1)%4
        return True

    def command(self,consume=True):
        if not self.enabled:return [0.,0.,0.]
        keys=self.held|self.pulses
        if consume:self.pulses.clear()
        if self.fixed_forward:return [self.speed if 'W' in keys else 0.,0.,0.]
        forward=float('W' in keys)-float('S' in keys)
        left=float('A' in keys)-float('D' in keys)
        scale=self.speed/max(1.,math.hypot(forward,left))
        yaw=(float('Q' in keys)-float('E' in keys))*self.yaw_speed
        return [forward*scale,left*scale,yaw]

    def request_reset(self):
        self.stop()
        self.push_pending=False
        self.reset_pending=True
        self.notice='Reset requested.'

    def consume_reset(self):
        value=self.reset_pending;self.reset_pending=False;return value

    def request_push(self, settings=None):
        self.reset_pending=False
        self.push_pending=True if settings is None else settings
        if settings is None:
            self.notice='Random balance push requested. Press R to reset after falling.'
        else:
            angle=settings['angle_degrees']
            direction='random direction' if angle is None else f'{angle:+.0f}°'
            self.notice=f'Push {settings["delta_speed"]:.3f} m/s, {direction}. Press R to reset.'

    def consume_push(self):
        value=self.push_pending;self.push_pending=False;return value
