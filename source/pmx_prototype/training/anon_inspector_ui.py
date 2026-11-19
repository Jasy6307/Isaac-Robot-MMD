"""Live Isaac UI, keyboard subscription and inspection telemetry."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import asyncio
import ctypes
import json
import math
import os
import time
from collections import deque
from pathlib import Path
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
    FOLLOW_POSITION_TIME=.16
    FOLLOW_HEADING_TIME=.28
    TEXT=0xfff0eeeb
    MUTED=0xffb4aaa2
    ACCENT=0xffe7c786
    CARD=0xff34302a
    KEY_DOWN=0xff796144
    BUTTON_FONT=17
    KEY_RADIUS=6
    FOCUS_NOTICE='Click the viewport to focus.'

    def __init__(self,config,output,catalog,active,render_target):
        import carb.input
        import omni.appwindow
        import omni.ui as ui
        self.ui=ui;self.controls=InspectorCommands(config);self.output=output;self.last_write=0.;self.last_ui_update=0.
        self.solved_times=deque();self.render_times=deque();self.render_frames=0;self.last_rate_update=0.
        self.render_target=render_target
        self.events=[];self.history=[];self.last_focus=True;self.reset_count=0
        self.catalog=catalog;self.active=active;self.pending='';self.key_cards={};self.last_keys=None
        self.reset_camera()
        self.font_style={}
        self.input=carb.input.acquire_input_interface()
        self.keyboard=omni.appwindow.get_default_app_window().get_keyboard()
        self.subscription=self.input.subscribe_to_keyboard_events(self.keyboard,self.on_key)
        for name in ['Render Settings','Content','Console']:
            window=ui.Workspace.get_window(name)
            if window:window.visible=False
        c=self.controls
        self.panel=ui.Window('Policy Inspector',width=380,height=640)
        self.panel.frame.style={
            'Button':{'background_color':0xff433a32,'color':self.TEXT,'border_radius':6,'font_size':self.BUTTON_FONT,**self.font_style},
            'Button:hovered':{'background_color':0xff665447},
            'Button:pressed':{'background_color':0xff796144},
            'FloatSlider':{'color':self.ACCENT,'background_color':0xff49423a,'border_radius':3},
        }
        with self.panel.frame:
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                                   vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED) as self.scroll:
                with ui.HStack():
                    ui.Spacer(width=12)
                    with ui.VStack(spacing=3,width=ui.Fraction(1)):
                        ui.Spacer(height=6)
                        self.model_label=self.label('',height=36,size=16,color=self.ACCENT,word_wrap=True)
                        self.separator()
                        with ui.HStack(height=22,spacing=3):
                            ui.Spacer(width=66)
                            for title in ['Fwd m/s','Side m/s','Yaw rad/s']:
                                self.label(title,size=14,color=self.MUTED,width=ui.Fraction(1),alignment=ui.Alignment.CENTER)
                        self.request_values=self.value_row('Request')
                        self.command_values=self.value_row('Applied')
                        self.actual_values=self.value_row('Actual')
                        with ui.HStack(height=24,spacing=6):
                            self.solve_rate_label=self.label('Solved 0 / 30 Hz',size=15,width=ui.Fraction(1))
                            self.render_rate_label=self.label(f'Render 0 / {render_target} FPS',size=15,width=ui.Fraction(1))
                        with ui.HStack(height=24):
                            self.motion_label=self.label('',size=15,color=self.ACCENT,width=ui.Fraction(1))
                            self.stats_label=self.label('',size=14,color=self.MUTED,alignment=ui.Alignment.RIGHT_CENTER)
                        with ui.HStack(height=42,spacing=6):
                            self.stop_button=ui.Button('STOP  [Space]',clicked_fn=self.stop,width=ui.Fraction(1),
                                                       style={'background_color':0xff594836,'color':self.TEXT,'border_radius':6,
                                                              'font_size':self.BUTTON_FONT,**self.font_style})
                            self.reset_button=ui.Button('RESET  [R]',clicked_fn=c.request_reset,width=ui.Fraction(1))
                        self.separator()
                        with ui.HStack(height=24):
                            self.label('Speed [-/+]',width=ui.Fraction(1))
                            self.speed_value=self.label('',color=self.ACCENT,alignment=ui.Alignment.RIGHT_CENTER)
                        self.speed_model=ui.SimpleFloatModel(c.speed)
                        self.speed_slider=ui.FloatSlider(model=self.speed_model,min=0.,max=c.max_speed,height=26)
                        self.speed_model.add_value_changed_fn(lambda m:c.set_speed(m.get_value_as_float()))
                        with ui.HStack(height=24,spacing=7):
                            self.label('Yaw [Q/E]')
                            self.yaw_limit_label=self.label('',size=14,color=self.MUTED,width=ui.Fraction(1),alignment=ui.Alignment.CENTER)
                            self.yaw_value=self.label('',color=self.ACCENT,alignment=ui.Alignment.RIGHT_CENTER)
                        self.yaw_model=ui.SimpleFloatModel(c.yaw_speed)
                        self.yaw_slider=ui.FloatSlider(model=self.yaw_model,min=0.,max=c.max_yaw,height=26)
                        self.yaw_model.add_value_changed_fn(lambda m:setattr(c,'yaw_speed',m.get_value_as_float()))
                        self.separator()
                        with ui.HStack(height=28):
                            self.enable_model=ui.SimpleBoolModel(True)
                            ui.CheckBox(model=self.enable_model,width=26,height=26)
                            self.label('Keyboard',width=ui.Fraction(1))
                            def enabled(m):c.enabled=m.get_value_as_bool();c.stop()
                            self.enable_model.add_value_changed_fn(enabled)
                        for row in [[('Q','Turn L'),('W','Forward'),('E','Turn R')],[('A','Left'),('S','Back'),('D','Right')]]:
                            with ui.HStack(height=46,spacing=5):
                                for key,description in row:
                                    with ui.ZStack(width=ui.Fraction(1)):
                                        self.key_cards[key]=ui.Rectangle(style={'background_color':self.CARD,'border_radius':self.KEY_RADIUS})
                                        with ui.VStack():
                                            self.label(key,height=26,size=20,alignment=ui.Alignment.CENTER)
                                            self.label(description,height=18,size=14,alignment=ui.Alignment.CENTER)
                        self.separator()
                        self.label('Camera [V]',height=24)
                        with ui.HStack(height=36,spacing=4):
                            for i,name in enumerate(['Angle','Side','Back','Front']):
                                ui.Button(name,clicked_fn=lambda index=i:setattr(c,'view',index),width=ui.Fraction(1))
                        self.notice_label=self.label('',word_wrap=True,height=40,size=14,color=self.ACCENT,visible=False)
                        ui.Spacer(height=8)
                    ui.Spacer(width=12)
        self.apply_policy(config,active)
        self.dock_task=asyncio.ensure_future(self.dock_with_property())

    def label(self,text,height=0,size=15,color=None,**kwargs):
        style={'font_size':size,'color':self.TEXT if color is None else color,**self.font_style}
        return self.ui.Label(text,height=height,style=style,**kwargs)

    def separator(self):
        self.ui.Rectangle(height=1,style={'background_color':0xff514940})

    def value_row(self,title):
        with self.ui.HStack(height=25,spacing=3):
            self.label(title,width=66,size=14,color=self.MUTED)
            return [self.label('',size=17,width=self.ui.Fraction(1),alignment=self.ui.Alignment.CENTER) for _ in range(3)]

    async def dock_with_property(self):
        """Use the Property tab group; restore the usual right-hand layout if absent."""
        import omni.kit.app
        ui=self.ui
        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()
            prop=ui.Workspace.get_window('Property')
            if prop is None:continue
            prop.visible=True
            stage=ui.Workspace.get_window('Stage')
            if stage is not None:stage.visible=True
            if not prop.docked:
                viewport=ui.Workspace.get_window('Viewport')
                if viewport is None or stage is None:continue
                if not stage.docked:
                    stage.dock_in(viewport,ui.DockPosition.RIGHT,.3)
                    await omni.kit.app.get_app().next_update_async()
                prop.dock_in(stage,ui.DockPosition.BOTTOM,.5)
                await omni.kit.app.get_app().next_update_async()
            self.panel.deferred_dock_in('Property',ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
            return
        viewport=ui.Workspace.get_window('Viewport')
        if viewport is not None:self.panel.dock_in(viewport,ui.DockPosition.RIGHT,.3)

    def apply_policy(self,config,active):
        self.controls.configure(config);self.controls.switching=False;self.active=active;self.pending=''
        c=self.controls
        self.model_label.text=self.catalog[active]['label']
        self.speed_slider.max=c.max_speed;self.yaw_slider.max=c.max_yaw
        self.speed_model.set_value(c.speed);self.yaw_model.set_value(c.yaw_speed)
        self.speed_value.text=f'{c.speed:.2f} m/s';self.yaw_value.text=f'{c.yaw_speed:.2f} rad/s'
        self.yaw_limit_label.text=f'Limit ±{c.max_yaw:.2f}'
        self.refresh_sliders()

    def refresh_sliders(self):
        self.speed_slider.enabled=self.yaw_slider.enabled=not self.pending and not self.controls.fixed_forward

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
            if self.last_focus:self.controls.notice=self.FOCUS_NOTICE
        elif not self.last_focus and self.controls.notice==self.FOCUS_NOTICE:
            self.controls.notice=''
        self.last_focus=focus
        return self.controls.command()

    def camera_offset(self,heading):
        x,y,z=self.CAMERA_OFFSETS[self.controls.view]
        if self.controls.view==0:return x,y,z
        cos_yaw,sin_yaw=math.cos(heading),math.sin(heading)
        return cos_yaw*x-sin_yaw*y,sin_yaw*x+cos_yaw*y,z

    def reset_camera(self):
        self.follow_time=None

    def camera_target(self,center,heading,now):
        # Run on render time so damping feels the same at 30 and 60 FPS.
        if self.follow_time is None:
            self.follow_center=center.copy()
            self.follow_heading=heading
        else:
            elapsed=max(0.,now-self.follow_time)
            position_weight=-math.expm1(-elapsed/self.FOLLOW_POSITION_TIME)
            heading_weight=-math.expm1(-elapsed/self.FOLLOW_HEADING_TIME)
            self.follow_center+=position_weight*(center-self.follow_center)
            self.follow_heading+=heading_weight*math.remainder(heading-self.follow_heading,math.tau)
        self.follow_time=now
        if self.controls.view==0:return center,self.camera_offset(0.)
        return self.follow_center,self.camera_offset(self.follow_heading)

    def motion_status(self,command):
        c=self.controls
        if self.pending:return 'Switching'
        if not c.enabled:return 'Keyboard off'
        if not self.last_focus:return 'Not focused'
        speed=(command[0]**2+command[1]**2)**.5
        if speed<.01:return 'Turning' if abs(command[2])>.01 else 'Stopped'
        if c.fixed_forward:return 'Forward'
        return 'Walking' if speed<.55 else ('Walk / run' if speed<.9 else 'Running')

    def recent_rates(self,now):
        cutoff=now-1.
        for samples in (self.solved_times,self.render_times):
            while samples and samples[0]<=cutoff:samples.popleft()
        return len(self.solved_times),len(self.render_times)

    def record_render(self):
        now=time.monotonic()
        self.render_times.append(now);self.render_frames+=1
        if now-self.last_rate_update>=.2:
            solved,rendered=self.recent_rates(now)
            self.solve_rate_label.text=f'Solved {solved} / 30 Hz'
            self.render_rate_label.text=f'Render {rendered} / {self.render_target} FPS'
            self.last_rate_update=now

    def update(self,state,falls,frames):
        now=time.monotonic()
        self.solved_times.append(now)
        c=self.controls;command=state['command'].tolist();actual=state['velocity'].tolist()
        requested=state['requested_command'].tolist() if 'requested_command' in state else c.command(consume=False)
        yaw_limit=float(state.get('yaw_limit',c.max_yaw))
        if not self.history or command!=self.history[-1]['command']:
            self.history.append(dict(frame=frames,command=command));self.history=self.history[-32:]
        if bool(state.get('reset',False)) and str(state.get('reset_reason',''))!='policy_switch':self.reset_count+=1
        pending=str(state.get('switch_pending',''))
        if pending!=self.pending:
            self.pending=pending;c.switching=bool(pending);self.refresh_sliders()
        if str(state.get('switch_error','')):
            self.pending='';c.switching=False;c.stop();c.notice=str(state['switch_error']);self.refresh_sliders()
        if not c.fixed_forward and abs(self.speed_model.get_value_as_float()-c.speed)>.0001:self.speed_model.set_value(c.speed)
        if now-self.last_ui_update>=.1:
            for label,value in zip(self.request_values,requested):label.text=f'{value:+.2f}'
            for label,value in zip(self.command_values,command):label.text=f'{value:+.2f}'
            for label,value in zip(self.actual_values,actual):label.text=f'{value:+.2f}'
            self.yaw_limit_label.text=f'Limit ±{yaw_limit:.2f}'
            self.motion_label.text='Fallen [R]' if bool(state.get('fallen',False)) else self.motion_status(command)
            self.speed_value.text=f'{c.speed:.2f} m/s';self.yaw_value.text=f'{c.yaw_speed:.2f} rad/s'
            self.stats_label.text=f'Falls {falls}  |  Resets {self.reset_count}'
            if self.notice_label.text!=c.notice:
                self.notice_label.text=c.notice
                self.notice_label.visible=bool(c.notice)
            keys=frozenset(c.held)
            if keys!=self.last_keys:
                for key,card in self.key_cards.items():
                    card.style={'background_color':self.KEY_DOWN if key in keys else self.CARD,'border_radius':self.KEY_RADIUS}
                self.last_keys=keys
            self.last_ui_update=now
        if now-self.last_write>.25:
            data=dict(requested_command=requested,applied_command=command,yaw_limit=yaw_limit,actual_velocity=actual,
                      speed=c.speed,held=sorted(c.held),focused=self.last_focus,falls=falls,resets=self.reset_count,
                      simulated_seconds=frames/30,fixed_forward=c.fixed_forward,fallen=bool(state.get('fallen',False)),
                      notice=c.notice,events=self.events,command_history=self.history)
            data.update(policy_id=self.active,model=self.catalog[self.active]['label'],switch_pending=self.pending,
                        worker_pid=int(state['worker_pid']),physics_stage_id=int(state['physics_stage_id']),viewer_pid=os.getpid())
            solved,rendered=self.recent_rates(now)
            data.update(display_fps=rendered,solved_pose_fps=solved,render_frames=self.render_frames,solved_frames=frames)
            temporary=self.output/'inspection_state.tmp.json';temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
            os.replace(temporary,self.output/'inspection_state.json');self.last_write=now

    def close(self):
        self.controls.stop()
        if not self.dock_task.done():self.dock_task.cancel()
        self.input.unsubscribe_to_keyboard_events(self.keyboard,self.subscription)
        self.panel.destroy()
