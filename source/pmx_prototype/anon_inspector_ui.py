"""Live Isaac UI, keyboard subscription and inspection telemetry."""
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
    def __init__(self,config,checkpoint,output,restart):
        import carb.input
        import omni.appwindow
        import omni.ui as ui
        self.ui=ui;self.controls=InspectorCommands(config);self.output=output;self.last_write=0.
        self.events=[];self.history=[];self.last_focus=True;self.reset_count=0
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
                ui.Label(checkpoint.parent.name,height=22)
                scope=f'RUN: forward {c.speed:.2f} m/s + stop' if c.fixed_forward else f'WALK: any direction, 0 - {c.max_speed:.2f} m/s'
                ui.Label(scope,height=25,style={'color':0xff70dfff})
                ui.Label('WASD: move   Q/E: turn   +/-: speed',height=20)
                ui.Label('Release: stop   Space: stop   R: reset   V: view',height=20)
                if c.fixed_forward:ui.Label('RUN supports W / stop only. A/S/D/Q/E and speed changes are unavailable.',word_wrap=True,height=40)
                else:
                    ui.Label('Movement speed (m/s), +/- step = 0.05',height=18)
                    self.speed_model=ui.SimpleFloatModel(c.speed)
                    ui.FloatSlider(model=self.speed_model,min=0.,max=c.max_speed,height=23)
                    self.speed_model.add_value_changed_fn(lambda m:c.set_speed(m.get_value_as_float()))
                    ui.Label('Turning speed (rad/s)',height=18)
                    self.yaw_model=ui.SimpleFloatModel(c.yaw_speed)
                    ui.FloatSlider(model=self.yaw_model,min=0.,max=c.max_yaw,height=23)
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
                ui.Label('Switch trained policy (restarts preview):',height=20)
                with ui.HStack(height=30,spacing=5):
                    ui.Button('Forward RUN 1.2',clicked_fn=lambda:self.switch(restart,'run'))
                    ui.Button('Omni WALK (31 DOF)',clicked_fn=lambda:self.switch(restart,'walk'))
                ui.Spacer(height=2)

    def switch(self,restart,preset):self.stop();restart(preset)
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
        if bool(state.get('reset',False)):self.reset_count+=1
        if not c.fixed_forward and abs(self.speed_model.get_value_as_float()-c.speed)>.0001:self.speed_model.set_value(c.speed)
        self.command_label.text=f'Command: fwd {command[0]:+.2f} / left {command[1]:+.2f} / yaw {command[2]:+.2f}'
        self.actual_label.text=f'Actual:    fwd {actual[0]:+.2f} / left {actual[1]:+.2f} / yaw {actual[2]:+.2f}'
        self.stats_label.text=f'Sim {frames/30:.1f}s | falls {falls} | resets {self.reset_count} | {frames/30/max(wall,.001):.2f}x real time'
        self.notice_label.text=c.notice
        if time.monotonic()-self.last_write>.25:
            data=dict(requested_command=c.command(consume=False),applied_command=command,actual_velocity=actual,
                      speed=c.speed,held=sorted(c.held),focused=self.last_focus,falls=falls,resets=self.reset_count,
                      simulated_seconds=frames/30,fixed_forward=c.fixed_forward,notice=c.notice,events=self.events,command_history=self.history)
            temporary=self.output/'inspection_state.tmp.json';temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
            os.replace(temporary,self.output/'inspection_state.json');self.last_write=time.monotonic()

    def close(self):
        self.controls.stop()
        self.input.unsubscribe_to_keyboard_events(self.keyboard,self.subscription)
