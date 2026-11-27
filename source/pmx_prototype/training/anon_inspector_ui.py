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
from anon_character_push import MANUAL_PUSH_MAX, MANUAL_PUSH_MIN
from anon_inspector_acceptance import acceptance_requests, demo_requests
from anon_inspector_controls import InspectorCommands,keyboard_event_names,keys_for_command


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
    HUD_WIDTH=420

    def __init__(self,config,output,catalog,active,render_target,*,allow_acceptance=True,joint_monitor=True,header_builder=None):
        import carb.input
        import omni.appwindow
        import omni.ui as ui
        self.ui=ui;self.controls=InspectorCommands(config);self.output=output;self.last_write=0.;self.last_ui_update=0.
        self.solved_times=deque();self.render_times=deque();self.render_frames=0;self.last_rate_update=0.
        self.render_target=render_target
        self.events=[];self.history=[];self.last_focus=True;self.reset_count=0
        self.catalog=catalog;self.active=active;self.key_cards={};self.last_keys=None
        self.keyboard_hud=None;self.hud_values=None;self.hud_title=None
        self.hud_card=None
        self.joint_monitor=None
        self.allow_acceptance = allow_acceptance
        self.policy_config = config
        self.playback_frames = None
        self.playback_name = ''
        self.playback_status = ''
        self.syncing_sliders = False
        self.reset_camera()
        self.font_style={}
        self.input=carb.input.acquire_input_interface()
        self.keyboard=omni.appwindow.get_default_app_window().get_keyboard()
        self.subscription=self.input.subscribe_to_keyboard_events(self.keyboard,self.on_key)
        for name in ['Render Settings','Console']:
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
                        if header_builder is not None:header_builder()
                        self.label('Motion status',height=28,size=19,color=self.ACCENT)
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
                        self.stats_label=self.label('',height=24,size=14,color=self.MUTED,
                                                    alignment=ui.Alignment.RIGHT_CENTER)
                        self.notice_label=self.label('',word_wrap=True,height=40,size=14,color=self.ACCENT,visible=False)
                        if joint_monitor:
                            from anon_joint_monitor_ui import JointMonitorWidget
                            self.joint_monitor=JointMonitorWidget(invalidate=self.panel.frame.invalidate_raster)
                        ui.Spacer(height=8)
                    ui.Spacer(width=12)
        self.build_controls_panel()
        self.build_keyboard_hud()
        self.acceptance_button.visible = self.demo_button.visible = allow_acceptance and bool(config.get('unified_walk_run'))
        self.speed_slider.max=c.max_speed;self.yaw_slider.max=c.max_yaw
        self.show_input_controls()
        self.planar_limit_label.text=''
        self.yaw_limit_label.text=f'Limit ±{c.max_yaw:.2f}'
        self.refresh_sliders()
        self.dock_task=asyncio.ensure_future(self.dock_panels())

    def build_controls_panel(self):
        """Keep playback, speed and camera controls together in the bottom tab group."""
        ui=self.ui;c=self.controls
        self.controls_panel=ui.Window('Policy Controls',width=960,height=260)
        self.controls_panel.frame.style=self.panel.frame.style
        with self.controls_panel.frame:
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                                   vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
                with ui.VStack(spacing=6):
                    ui.Spacer(height=6)
                    with ui.HStack(spacing=16):
                        ui.Spacer(width=0)
                        with ui.VStack(width=400,spacing=6):
                            self.label('Checkpoint playback',height=22,size=15,color=self.ACCENT)
                            with ui.HStack(height=38,spacing=6):
                                self.acceptance_button=ui.Button('Auto Run Eval',clicked_fn=self.start_acceptance,width=ui.Fraction(1))
                                self.demo_button=ui.Button('Auto Run Eval (Short)',clicked_fn=self.start_demo,width=ui.Fraction(1))
                            with ui.HStack(height=38,spacing=6):
                                self.stop_button=ui.Button('STOP  [Space]',clicked_fn=self.stop,width=ui.Fraction(1),
                                    style={'background_color':0xff594836,'color':self.TEXT,'border_radius':6,
                                           'font_size':self.BUTTON_FONT,**self.font_style})
                                self.reset_button=ui.Button('RESET  [R]',clicked_fn=self.reset,width=ui.Fraction(1))
                            with ui.HStack(height=24,spacing=6):
                                self.label('Push (m/s)',width=98,size=14)
                                self.push_strength=ui.SimpleFloatModel(.525)
                                ui.FloatSlider(model=self.push_strength,min=MANUAL_PUSH_MIN,max=MANUAL_PUSH_MAX,
                                               width=ui.Fraction(1))
                            with ui.HStack(height=22,spacing=4):
                                for strength in (.20,.525,.70,.80,1.00):
                                    ui.Button(f'{strength:g}',width=ui.Fraction(1),style={'font_size':13},
                                              clicked_fn=lambda value=strength:self.push_strength.set_value(value))
                            with ui.HStack(height=24,spacing=4):
                                self.push_random=ui.SimpleBoolModel(True)
                                ui.CheckBox(model=self.push_random,width=22)
                                self.label('Random',width=62,size=13)
                                self.push_angle=ui.SimpleFloatModel(0.)
                                angle=ui.FloatSlider(model=self.push_angle,min=-180.,max=180.,width=ui.Fraction(1),
                                                     tooltip='0° forward, +90° left, -90° right, ±180° back, relative to the character')
                                angle.enabled=False
                                self.push_random.add_value_changed_fn(lambda model:setattr(angle,'enabled',not model.get_value_as_bool()))
                            self.push_button=ui.Button('Push once [P]',clicked_fn=self.push_from_controls,height=31,
                                tooltip='One horizontal impulse. Hold a movement key and press P to push mid-stride.')
                        with ui.VStack(spacing=4,width=ui.Fraction(1)):
                            with ui.HStack(height=24):
                                self.label('Speed [-/+]')
                                self.planar_limit_label=self.label('',size=14,color=self.MUTED,width=ui.Fraction(1),alignment=ui.Alignment.CENTER)
                                self.speed_value=self.label('',color=self.ACCENT,alignment=ui.Alignment.RIGHT_CENTER)
                            self.speed_model=ui.SimpleFloatModel(c.speed)
                            self.speed_slider=ui.FloatSlider(model=self.speed_model,min=0.,max=c.max_speed,height=26)
                            self.speed_model.add_value_changed_fn(self.on_speed_changed)
                            with ui.HStack(height=24,spacing=7):
                                self.label('Yaw [Q/E]')
                                self.yaw_limit_label=self.label('',size=14,color=self.MUTED,width=ui.Fraction(1),alignment=ui.Alignment.CENTER)
                                self.yaw_value=self.label('',color=self.ACCENT,alignment=ui.Alignment.RIGHT_CENTER)
                            self.yaw_model=ui.SimpleFloatModel(c.yaw_speed)
                            self.yaw_slider=ui.FloatSlider(model=self.yaw_model,min=0.,max=c.max_yaw,height=26)
                            self.yaw_model.add_value_changed_fn(self.on_yaw_changed)
                        with ui.VStack(width=220,spacing=6):
                            self.label('Camera [V]',height=22,size=15,color=self.ACCENT)
                            self.camera_buttons={}
                            for row in ((0,1),(2,3)):
                                with ui.HStack(height=38,spacing=6):
                                    for i in row:
                                        name=('Angle','Side','Back','Front')[i]
                                        self.camera_buttons[name]=ui.Button(name,clicked_fn=lambda index=i:setattr(c,'view',index),width=ui.Fraction(1))
                            with ui.HStack(height=28,spacing=4):
                                self.enable_model=ui.SimpleBoolModel(True)
                                ui.CheckBox(model=self.enable_model,width=26,height=26)
                                self.label('viewport HUD',width=ui.Fraction(1),size=14)
                                def enabled(m):c.enabled=m.get_value_as_bool();c.stop()
                                self.enable_model.add_value_changed_fn(enabled)
                        ui.Spacer(width=0)
                    ui.Spacer(height=6)

    def build_keyboard_hud(self):
        """Draw a passive keyboard guide inside the viewport, outside the docked panel."""
        from omni.kit.viewport.utility import get_active_viewport_window
        viewport_window=get_active_viewport_window()
        if viewport_window is None:
            import carb
            carb.log_warn('Policy keyboard HUD unavailable: no active viewport window')
            return
        ui=self.ui
        self.keyboard_hud=viewport_window.get_frame('anon.policy.keyboard')
        self.hud_viewport=viewport_window.viewport_api
        self.hud_layout=None
        self.keyboard_hud.clear()
        self.keyboard_hud.visible=True
        self.keyboard_hud.opaque_for_mouse_events=False
        self.key_cards={};self.last_keys=None
        with self.keyboard_hud:
            with ui.VStack(opaque_for_mouse_events=False):
                ui.Spacer()
                with ui.HStack(height=208,opaque_for_mouse_events=False):
                    self.hud_left_padding=ui.Spacer(width=18)
                    self.build_keyboard_card()
                    ui.Spacer()
                self.hud_bottom_padding=ui.Spacer(height=18)

    def build_keyboard_card(self):
        ui=self.ui
        with ui.ZStack(width=self.HUD_WIDTH,height=208,opaque_for_mouse_events=False) as self.hud_card:
            ui.Rectangle(style={'background_color':0xbc26221f,'border_radius':10},
                         opaque_for_mouse_events=False)
            with ui.VStack():
                ui.Spacer(height=12)
                with ui.HStack():
                    ui.Spacer(width=12)
                    with ui.VStack(spacing=4):
                        self.hud_title=self.label('Keyboard',height=18,size=14,color=self.ACCENT,
                            word_wrap=False,elided_text=True,opaque_for_mouse_events=False)
                        self.hud_values=self.label('',height=18,size=12,color=self.MUTED)
                        for row in ((('Q','Turn L'),('W','Forward'),('E','Turn R')),
                                    (('A','Left'),('S','Back'),('D','Right'))):
                            with ui.HStack(height=46,spacing=5):
                                for key,description in row:
                                    with ui.ZStack(width=ui.Fraction(1),opaque_for_mouse_events=False):
                                        self.key_cards[key]=ui.Rectangle(style={
                                            'background_color':self.CARD,'border_radius':self.KEY_RADIUS})
                                        with ui.VStack(spacing=0):
                                            ui.Spacer(height=3)
                                            self.label(key,height=24,size=19,alignment=ui.Alignment.CENTER)
                                            self.label(description,height=18,size=12,alignment=ui.Alignment.CENTER)
                                            ui.Spacer(height=1)
                        self.label('Space  Stop       R  Reset',height=18,size=12,color=self.MUTED)
                        self.label('-/+  Speed       V  Camera',height=18,size=12,color=self.MUTED)
                    ui.Spacer(width=12)
                ui.Spacer(height=12)

    def layout_keyboard_hud(self):
        """Keep the guide inside the rendered image when the viewport has black bars."""
        frame=getattr(self,'keyboard_hud',None)
        if frame is None:return
        width,height=frame.computed_width,frame.computed_height
        rx,ry=self.hud_viewport.resolution
        if min(width,height,rx,ry)<=0:return
        layout=(width,height,rx,ry)
        if layout==self.hud_layout:return
        scale=min(width/rx,height/ry)
        left=18+max(0.,(width-rx*scale)/2)
        bottom=18+max(0.,(height-ry*scale)/2)
        self.hud_left_padding.width=self.ui.Pixel(left)
        self.hud_bottom_padding.height=self.ui.Pixel(bottom)
        self.hud_layout=layout

    def label(self,text,height=0,size=15,color=None,**kwargs):
        style={'font_size':size,'color':self.TEXT if color is None else color,**self.font_style}
        return self.ui.Label(text,height=height,style=style,**kwargs)

    def value_row(self,title):
        with self.ui.HStack(height=25,spacing=3):
            self.label(title,width=66,size=14,color=self.MUTED)
            return [self.label('',size=17,width=self.ui.Fraction(1),alignment=self.ui.Alignment.CENTER) for _ in range(3)]

    async def dock_panels(self):
        await self.dock_with_property()
        await self.dock_controls_with_content()

    async def dock_controls_with_content(self):
        """Share Content's bottom tabs and select the new control panel."""
        import omni.kit.app
        ui=self.ui
        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()
            content=ui.Workspace.get_window('Content')
            viewport=ui.Workspace.get_window('Viewport')
            if content is None or viewport is None:continue
            content.visible=True
            # Apply the default split even when Kit restored a previous layout.
            content.dock_in(viewport,ui.DockPosition.BOTTOM,.26)
            await omni.kit.app.get_app().next_update_async()
            self.controls_panel.deferred_dock_in('Content',ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
            return
        viewport=ui.Workspace.get_window('Viewport')
        if viewport is not None:self.controls_panel.dock_in(viewport,ui.DockPosition.BOTTOM,.26)

    async def dock_with_property(self):
        """Use Property's lower-right tabs below Stage, with Inspector selected."""
        import omni.kit.app
        ui=self.ui
        for _ in range(120):
            await omni.kit.app.get_app().next_update_async()
            prop=ui.Workspace.get_window('Property')
            if prop is None:continue
            prop.visible=True
            stage=ui.Workspace.get_window('Stage')
            if stage is not None:stage.visible=True
            # Keep the lower-right group at 65% even after a saved Kit layout.
            if stage is not None:
                viewport=ui.Workspace.get_window('Viewport')
                if viewport is None or stage is None:continue
                if not stage.docked:
                    stage.dock_in(viewport,ui.DockPosition.RIGHT,.3)
                    await omni.kit.app.get_app().next_update_async()
                prop.dock_in(stage,ui.DockPosition.BOTTOM,.65)
                await omni.kit.app.get_app().next_update_async()
            self.panel.deferred_dock_in('Property',ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
            return
        viewport=ui.Workspace.get_window('Viewport')
        if viewport is not None:self.panel.dock_in(viewport,ui.DockPosition.RIGHT,.3)


    def on_speed_changed(self, model):
        if not self.syncing_sliders:
            self.controls.set_speed(model.get_value_as_float())

    def on_yaw_changed(self, model):
        if not self.syncing_sliders:
            self.controls.yaw_speed = model.get_value_as_float()

    def show_input_controls(self, requested=None):
        c = self.controls
        if requested is None:
            speed, yaw, keys = c.speed, c.yaw_speed, frozenset(c.held)
        else:
            speed = math.hypot(requested[0], requested[1])
            yaw = abs(requested[2])
            keys = keys_for_command(requested)

        self.syncing_sliders = True
        try:
            for model, value in ((self.speed_model, speed), (self.yaw_model, yaw)):
                if abs(model.get_value_as_float() - value) > 1e-4:
                    model.set_value(value)
        finally:
            self.syncing_sliders = False
        self.speed_value.text = f'{speed:.2f} m/s'
        self.yaw_value.text = f'{yaw:.2f} rad/s'
        if getattr(self,'hud_values',None) is not None:
            self.hud_values.text=f'Speed {speed:.2f} m/s    Turn {yaw:.2f} rad/s'
            self.update_hud_title()
        if keys != self.last_keys:
            for key, card in self.key_cards.items():
                card.style = {'background_color': self.KEY_DOWN if key in keys else self.CARD,
                              'border_radius': self.KEY_RADIUS}
            self.last_keys = keys

    def update_hud_title(self,prefix=''):
        if getattr(self,'hud_title',None) is None:return
        status=' '.join(self.playback_status.split())
        if self.playback_frames is not None:
            title=self.playback_name+(f' | {status}' if status else '')
        else:
            title=status or ('Keyboard' if self.controls.enabled else 'Keyboard off')
        self.hud_title.text=prefix+title

    def refresh_sliders(self):
        available = self.playback_frames is None
        self.speed_slider.enabled = self.yaw_slider.enabled = available and not self.controls.fixed_forward
        self.acceptance_button.enabled = self.demo_button.enabled = available

    def start_acceptance(self):
        self.start_playback(self.acceptance_button, 'Acceptance', acceptance_requests)

    def start_demo(self):
        self.start_playback(self.demo_button, 'Auto Demo', demo_requests)

    def start_playback(self, button, name, requests):
        if not button.enabled or not button.visible:
            return
        self.controls.stop()
        self.controls.consume_reset()
        self.controls.consume_push()
        try:
            self.playback_frames = requests(self.policy_config)
        except ValueError as error:
            self.playback_status = f'{name} unavailable. {error}'
            return
        self.playback_name = name
        self.playback_status = f'Starting {name}.'
        self.refresh_sliders()

    def finish_playback(self, reason):
        if self.playback_frames is not None:
            self.playback_frames = None
            self.playback_status = f'{self.playback_name} {reason}.'
            self.controls.stop()
            self.show_input_controls()
            self.refresh_sliders()

    def stop(self):
        self.finish_playback('stopped')
        self.controls.stop()
        self.controls.consume_push()
        self.controls.notice='Stop requested.'

    def reset(self):
        self.finish_playback('stopped')
        self.controls.request_reset()

    def push_character(self, settings=None):
        self.finish_playback('interrupted by push')
        self.controls.request_push(settings)

    def push_from_controls(self):
        strength=min(MANUAL_PUSH_MAX,max(MANUAL_PUSH_MIN,self.push_strength.get_value_as_float()))
        angle=None if self.push_random.get_value_as_bool() else round(self.push_angle.get_value_as_float())
        self.push_character(dict(delta_speed=round(strength,3),angle_degrees=angle))

    def on_key(self,event):
        names=keyboard_event_names(event)
        if names is None:return False
        key,kind=names
        if kind!='KEY_RELEASE' and not has_foreground():return False
        if key == 'P':
            if kind == 'KEY_PRESS' and self.controls.enabled:
                self.push_from_controls()
            return True
        if self.playback_frames is not None and key != 'V':
            if kind == 'KEY_PRESS' and self.controls.enabled:
                if key == 'SPACE':
                    self.stop()
                elif key == 'R':
                    self.reset()
            handled = key in self.controls.MOVEMENT | self.controls.PLUS | self.controls.MINUS | {'SPACE', 'R'}
        else:
            handled=self.controls.event(key,kind)
        if handled:
            self.events.append(dict(key=key,kind=kind));self.events=self.events[-24:]
        return handled

    def command(self):
        focus=has_foreground()
        if not focus:
            self.controls.stop()
        self.last_focus=focus
        return self.controls.command()

    def next_request(self):
        if self.playback_frames is not None:
            step = next(self.playback_frames, None)
            if step is not None:
                self.playback_status = step.status
                request = dict(command=step.command, reset=step.reset, acceptance_inputs=True)
                if step.push_character:request['push_character']=True
                if step.manual_reset_only:request['manual_reset_only']=True
                return request
            self.finish_playback('complete')
        request = dict(command=self.command(), reset=self.controls.consume_reset())
        push=self.controls.consume_push()
        if push:request['push_character']=push
        return request

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

    def recent_rates(self,now):
        cutoff=now-1.
        for samples in (self.solved_times,self.render_times):
            while samples and samples[0]<=cutoff:samples.popleft()
        return len(self.solved_times),len(self.render_times)

    def record_render(self):
        now=time.monotonic()
        self.layout_keyboard_hud()
        self.render_times.append(now);self.render_frames+=1
        if now-self.last_rate_update>=.2:
            solved,rendered=self.recent_rates(now)
            self.solve_rate_label.text=f'Solved {solved} / 30 Hz'
            self.render_rate_label.text=f'Render {rendered} / {self.render_target} FPS'
            self.last_rate_update=now

    def update(self,state,falls,frames,*,acceptance_inputs=False):
        now=time.monotonic()
        if getattr(self,'joint_monitor',None) is not None:self.joint_monitor.ingest(state)
        self.solved_times.append(now)
        c=self.controls;command=state['command'].tolist();actual=state['velocity'].tolist()
        requested=state['requested_command'].tolist() if 'requested_command' in state else c.command(consume=False)
        yaw_limit=float(state.get('yaw_limit',c.max_yaw))
        planar_limit=float(state['planar_speed_limit']) if 'planar_speed_limit' in state else None
        if not self.history or command!=self.history[-1]['command']:
            self.history.append(dict(frame=frames,command=command));self.history=self.history[-32:]
        if bool(state.get('reset',False)):self.reset_count+=1
        if now-self.last_ui_update>=.1:
            if getattr(self,'joint_monitor',None) is not None:self.joint_monitor.refresh()
            for label,value in zip(self.request_values,requested):label.text=f'{value:+.2f}'
            for label,value in zip(self.command_values,command):label.text=f'{value:+.2f}'
            for label,value in zip(self.actual_values,actual):label.text=f'{value:+.2f}'
            if acceptance_inputs and self.playback_frames is not None:
                self.planar_limit_label.text=self.playback_name
                self.show_input_controls(requested)
            else:
                self.planar_limit_label.text=f'Limit {planar_limit:.2f}' if planar_limit is not None else ''
                self.show_input_controls()
            self.yaw_limit_label.text=f'Limit ±{yaw_limit:.2f}'
            self.stats_label.text=f'Falls {falls}  |  Resets {self.reset_count}'
            if self.notice_label.text!=c.notice:
                self.notice_label.text=c.notice
                self.notice_label.visible=bool(c.notice)
            self.last_ui_update=now
        if now-self.last_write>.25:
            data=dict(requested_command=requested,applied_command=command,yaw_limit=yaw_limit,actual_velocity=actual,
                      speed=c.speed,held=sorted(c.held),focused=self.last_focus,falls=falls,resets=self.reset_count,
                      simulated_seconds=frames/30,fixed_forward=c.fixed_forward,fallen=bool(state.get('fallen',False)),
                      notice=c.notice,events=self.events,command_history=self.history)
            if planar_limit is not None:data['planar_speed_limit']=planar_limit
            data.update(policy_id=self.active,model=self.catalog[self.active]['label'],
                        environment_backend='lab-direct',
                        worker_pid=int(state['worker_pid']),physics_stage_id=int(state['physics_stage_id']),viewer_pid=os.getpid())
            solved,rendered=self.recent_rates(now)
            data.update(display_fps=rendered,solved_pose_fps=solved,render_frames=self.render_frames,solved_frames=frames)
            temporary=self.output/'inspection_state.tmp.json';temporary.write_text(json.dumps(data,indent=2),encoding='utf-8')
            os.replace(temporary,self.output/'inspection_state.json');self.last_write=now

    def close(self):
        self.controls.stop()
        if not self.dock_task.done():self.dock_task.cancel()
        self.input.unsubscribe_to_keyboard_events(self.keyboard,self.subscription)
        if self.keyboard_hud is not None:
            self.keyboard_hud.clear()
            self.keyboard_hud.visible=False
            self.keyboard_hud=None
        self.key_cards.clear()
        if self.joint_monitor is not None:self.joint_monitor.close()
        self.controls_panel.destroy()
        self.panel.destroy()
