"""Live Isaac UI, keyboard subscription and inspection telemetry."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import ctypes
import json
import os
import time
from anon_inspector_controls import InspectorCommands,keyboard_event_names


def has_foreground():
    if os.name!='nt':return True
    pid=ctypes.c_ulong()
    ctypes.windll.user32.GetForegroundWindow.restype=ctypes.c_void_p
    ctypes.windll.user32.GetWindowThreadProcessId.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    ctypes.windll.user32.GetWindowThreadProcessId(ctypes.windll.user32.GetForegroundWindow(),ctypes.byref(pid))
    return pid.value==os.getpid()


class PolicyInspector:
    CAMERA_OFFSETS=[[2.3,-4,1.15],[3.6,0,.2],[0,4.6,.7],[0,-4.6,.7]]
    def __init__(self,config,checkpoint,output,switch_policy,catalog,active):
        import carb.input
        import omni.appwindow
        import omni.ui as ui
        self.ui=ui;self.controls=InspectorCommands(config);self.output=output;self.last_write=0.
        self.events=[];self.history=[];self.last_focus=True;self.reset_count=0
        self.catalog=catalog;self.active=active;self.pending='';self.buttons={}
        self.input=carb.input.acquire_input_interface()
        self.keyboard=omni.appwindow.get_default_app_window().get_keyboard()
        self.subscription=self.input.subscribe_to_keyboard_events(self.keyboard,self.on_key)
        for name in ['Stage','Render Settings','Property','Content','Console']:
            window=ui.Workspace.get_window(name)
            if window:window.visible=False
        c=self.controls
        self.panel=ui.Window('Anon Policy Inspector',width=460,height=700)
        self.panel.position_x=16;self.panel.position_y=80
        with self.panel.frame:
            with ui.VStack(spacing=4):
                ui.Label('ANON | POLICY INSPECTOR',height=25,style={'font_size':21})
                self.model_label=ui.Label('',height=22)
                self.scope_label=ui.Label('',height=25,style={'color':0xff70dfff})
                ui.Label('WASD: move   Q/E: turn   +/-: speed',height=20)
                ui.Label('Release: stop   Space: stop   R: reset   V: view',height=20)
                ui.Label('Movement speed (m/s), +/- step = 0.05',height=18)
                self.speed_model=ui.SimpleFloatModel(c.speed)
                self.speed_slider=ui.FloatSlider(model=self.speed_model,min=0.,max=c.max_speed,height=23)
                self.speed_model.add_value_changed_fn(lambda m:c.set_speed(m.get_value_as_float()))
                ui.Label('Turning speed (rad/s)',height=18)
                self.yaw_model=ui.SimpleFloatModel(c.yaw_speed)
                self.yaw_slider=ui.FloatSlider(model=self.yaw_model,min=0.,max=c.max_yaw,height=23)
                self.yaw_model.add_value_changed_fn(lambda m:setattr(c,'yaw_speed',m.get_value_as_float()))
                with ui.HStack(height=27):
                    self.enable_model=ui.SimpleBoolModel(True)
                    ui.CheckBox(model=self.enable_model,width=24)
                    ui.Label('Enable keyboard (click viewport before moving)')
                    def enabled(m):c.enabled=m.get_value_as_bool();c.stop()
                    self.enable_model.add_value_changed_fn(enabled)
                with ui.HStack(height=30,spacing=5):
                    ui.Button('STOP [Space]',clicked_fn=self.stop)
                    ui.Button('Reset [R]',clicked_fn=lambda:c.event('R','KEY_PRESS'))
                with ui.HStack(height=28,spacing=4):
                    for i,name in enumerate(['3/4','Side','Rear','Front']):
                        ui.Button(name,clicked_fn=lambda index=i:setattr(c,'view',index))
                self.command_label=ui.Label('',height=22)
                self.actual_label=ui.Label('',height=22)
                self.stats_label=ui.Label('',height=22)
                self.notice_label=ui.Label('',word_wrap=True,height=45)
                ui.Label('Look 08 | hair/skirt PhysX ON | learned clearance',height=23)
                if len(catalog)>1:
                    ui.Label('Switch model here (same window; stops before switching):',height=20)
                    for key,item in catalog.items():
                        self.buttons[key]=ui.Button(item['label'],height=28,clicked_fn=lambda target=key:self.switch(switch_policy,target))
                else:
                    ui.Label('One policy | speed continuously selects walk / run',height=23)
                ui.Spacer(height=2)
        self.apply_policy(config,active)

    def apply_policy(self,config,active):
        self.controls.configure(config);self.controls.switching=False;self.active=active;self.pending=''
        c=self.controls
        self.model_label.text=self.catalog[active]['label']
        scope='WALK + RUN' if config.get('unified_walk_run') else 'WALK'
        self.scope_label.text=f'RUN: W / stop only, fixed {c.speed:.2f} m/s' if c.fixed_forward else f'{scope}: WASD 0 - {c.max_speed:.2f} m/s | Q/E +/-{c.max_yaw:.2f}'
        self.speed_slider.max=c.max_speed;self.yaw_slider.max=c.max_yaw
        self.speed_model.set_value(c.speed);self.yaw_model.set_value(c.yaw_speed)
        self.refresh_buttons()

    def refresh_buttons(self):
        for key,button in self.buttons.items():button.enabled=not self.pending and key!=self.active and self.catalog[key]['available']
        self.speed_slider.enabled=self.yaw_slider.enabled=not self.pending and not self.controls.fixed_forward

    def switch(self,callback,preset):
        if self.pending or preset==self.active:return
        self.stop();self.controls.reset_pending=False;self.controls.switching=True;self.pending=preset
        self.controls.notice='Stopping, then switching to '+self.catalog[preset]['label']
        self.refresh_buttons();callback(preset)
    def stop(self):self.controls.stop();self.controls.notice='Stop requested.'

    def on_key(self,event):
        names=keyboard_event_names(event)
        if names is None:return False
        key,kind=names
        if kind!='KEY_RELEASE' and not has_foreground():return False
        handled=self.controls.event(key,kind)
        if handled:
            self.events.append(dict(key=key,kind=kind));self.events=self.events[-24:]
        return handled

    def command(self):
        focus=has_foreground()
        if not focus:
            self.controls.stop()
            if self.last_focus:self.controls.notice='Window lost focus: stopped. Click viewport and press movement keys again.'
        self.last_focus=focus
        return self.controls.command()

    def camera_offset(self):return self.CAMERA_OFFSETS[self.controls.view]

    def update(self,state,falls,frames,wall):
        c=self.controls;command=state['command'].tolist();actual=state['velocity'].tolist()
        if not self.history or command!=self.history[-1]['command']:
            self.history.append(dict(frame=frames,command=command));self.history=self.history[-32:]
        if bool(state.get('reset',False)) and str(state.get('reset_reason',''))!='policy_switch':self.reset_count+=1
        if str(state.get('switch_error','')):
            self.pending='';c.switching=False;c.stop();c.notice=str(state['switch_error']);self.refresh_buttons()
        if not c.fixed_forward and abs(self.speed_model.get_value_as_float()-c.speed)>.0001:self.speed_model.set_value(c.speed)
        self.command_label.text=f'Command: fwd {command[0]:+.2f} / left {command[1]:+.2f} / yaw {command[2]:+.2f}'
        self.actual_label.text=f'Actual:    fwd {actual[0]:+.2f} / left {actual[1]:+.2f} / yaw {actual[2]:+.2f}'
        self.stats_label.text=f'{frames/max(wall,.001):.1f} FPS | {frames/30/max(wall,.001):.2f}x real time | falls {falls} | resets {self.reset_count}'
        self.notice_label.text=c.notice
        if time.monotonic()-self.last_write>.25:
            data=dict(requested_command=c.command(consume=False),applied_command=command,actual_velocity=actual,
                      speed=c.speed,held=sorted(c.held),focused=self.last_focus,falls=falls,resets=self.reset_count,
                      simulated_seconds=frames/30,fixed_forward=c.fixed_forward,notice=c.notice,events=self.events,command_history=self.history)
            data.update(policy_id=self.active,model=self.catalog[self.active]['label'],switch_pending=self.pending,
                        worker_pid=int(state['worker_pid']),physics_stage_id=int(state['physics_stage_id']),viewer_pid=os.getpid())
            temporary=self.output/'inspection_state.tmp.json';temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
            os.replace(temporary,self.output/'inspection_state.json');self.last_write=time.monotonic()

    def close(self):
        self.controls.stop()
        self.input.unsubscribe_to_keyboard_events(self.keyboard,self.subscription)
