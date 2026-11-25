"""The single-character Inspector controls one row of the crowd batch."""
import math
import numpy as np
from anon_inspector_ui import PolicyInspector,has_foreground
from anon_inspector_controls import keyboard_event_names


class CrowdInspector(PolicyInspector):
    def __init__(self,config,output,catalog,active,render_target,*,count,selected=0,actor_collisions=False):
        self.actor_count=count;self.selected_actor=selected;self.actor_collisions=actor_collisions
        self.background_paused=False;self.reset_all_pending=False;self.actor_buttons={}
        self.follow_selected=True
        super().__init__(config,output,catalog,active,render_target,header_builder=self.build_actor_header)
        for index,name in enumerate(('Angle','Side','Back','Front')):
            self.camera_buttons[name].set_clicked_fn(lambda i=index:self.select_camera(i))

    def select_camera(self,index):
        self.controls.view=index;self.follow_model.set_value(True);self.reset_camera()

    def build_actor_header(self):
        ui=self.ui
        self.label(f'Controlled actor [1-{self.actor_count}]',height=22,size=16,color=self.ACCENT)
        for start in range(0,self.actor_count,4):
            with ui.HStack(height=30,spacing=5):
                for i in range(start,min(start+4,self.actor_count)):
                    self.actor_buttons[i]=ui.Button(f'Anon {i+1:02d}',clicked_fn=lambda n=i:self.select_actor(n))
        with ui.HStack(height=28,spacing=6):
            self.follow_model=ui.SimpleBoolModel(True)
            ui.CheckBox(model=self.follow_model,width=22)
            self.label('Follow',width=52,size=14)
            self.follow_model.add_value_changed_fn(self.on_follow_changed)
            self.background_button=ui.Button('Pause background',clicked_fn=self.toggle_background,style={'font_size':14})
            self.reset_all_button=ui.Button('Reset all',width=88,clicked_fn=self.request_reset_all,style={'font_size':14})
        self.crowd_status_label=self.label('Loading crowd...',height=36,size=13,color=self.MUTED,word_wrap=True)
        self.refresh_actor_buttons()

    def refresh_actor_buttons(self):
        for index,button in self.actor_buttons.items():
            button.style={'font_size':15,'color':self.ACCENT if index==self.selected_actor else self.TEXT}

    def select_actor(self,index):
        if not 0<=index<self.actor_count or index==self.selected_actor:return
        self.finish_playback('stopped')
        self.controls.stop();self.controls.consume_reset();self.controls.consume_push()
        self.selected_actor=index;self.reset_camera();self.history.clear()
        if self.joint_monitor is not None:
            self.joint_monitor.ingest(dict(joint_state_valid=False,joint_state_error='Waiting for selected actor'))
            self.joint_monitor.refresh()
        for values in (self.request_values,self.command_values,self.actual_values):
            for label in values:label.text='—'
        self.refresh_actor_buttons();self.show_input_controls()

    def on_follow_changed(self,model):
        self.follow_selected=model.get_value_as_bool();self.reset_camera()

    def toggle_background(self):
        self.background_paused=not self.background_paused
        self.background_button.text='Resume background' if self.background_paused else 'Pause background'

    def request_reset_all(self):
        self.stop();self.controls.consume_reset();self.controls.consume_push();self.reset_all_pending=True

    def consume_reset_all(self):
        pending=self.reset_all_pending;self.reset_all_pending=False;return pending

    def on_key(self,event):
        names=keyboard_event_names(event)
        if names is not None:
            key,kind=names
            number=key.removeprefix('KEY_').removeprefix('NUMPAD_')
            if number.isdigit() and 1<=int(number)<=self.actor_count:
                if kind=='KEY_PRESS' and has_foreground():self.select_actor(int(number)-1)
                return True
        handled=super().on_key(event)
        if handled and names==('V','KEY_PRESS') and has_foreground() and self.controls.enabled:
            self.follow_model.set_value(True)
        return handled

    def show_input_controls(self,requested=None):
        super().show_input_controls(requested)
        self.update_hud_title(f'Anon {self.selected_actor+1:02d} | ')

    def update_crowd(self,state,falls,resets,frame,*,acceptance_inputs=False):
        index=self.selected_actor
        self.crowd_status_label.text=(f'Background: {"stopped" if self.background_paused else "random"}'
            f' | Actor collisions: {"on" if self.actor_collisions else "off"}')
        # Selection may change while a packet is in flight. Do not display the
        # previous actor's telemetry or attach it to the new actor's curves.
        if int(state['controlled_actor'])!=index:return
        selected=dict(state)
        selected.update(poses=state['poses'][index],command=state['commands'][index],velocity=state['velocity'][index],
            requested_command=state['requested_commands'][index],reset=bool(state['resets'][index]),
            fallen=bool(state['fallen'][index]),physics_stage_id=state['physics_stage_id'])
        selected.update(yaw_limit=float(state['yaw_limits'][index]),planar_speed_limit=float(state['planar_speed_limits'][index]))
        self.reset_count=int(resets[index])-int(selected['reset'])
        super().update(selected,int(falls[index]),frame,acceptance_inputs=acceptance_inputs)

    def follow_camera(self,poses,pelvis,now):
        from scipy.spatial.transform import Rotation
        center=np.asarray(poses[self.selected_actor,pelvis,:3],dtype=float)
        forward=Rotation.from_quat(poses[self.selected_actor,pelvis,3:]).apply([0.,-1.,0.])
        heading=math.atan2(forward[0],-forward[1]) if math.hypot(forward[0],forward[1])>1e-4 else 0.
        return self.camera_target(center,heading,now)
