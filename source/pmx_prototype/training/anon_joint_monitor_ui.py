"""A passive movement overview with optional detailed joint trends."""
from anon_joint_monitor import MOTION_LEVELS, JointMonitorState, joint_label, joint_part, joint_part_rows


class JointMonitorWidget:
    TEXT=0xfff0eeeb
    MUTED=0xffb4aaa2
    TARGET=0xff86c7e7
    ACTUAL=0xffefc774
    ERROR=0xff8c92ed
    NORMAL=0xffa9c9a2
    MOTION_COLORS=(MUTED,ACTUAL,NORMAL,TARGET)

    def __init__(self,*,invalidate=None):
        import omni.ui as ui
        self.ui=ui
        self.state=JointMonitorState()
        self.rows={};self.cells={};self.member_lookup={};self.axis_buttons={};self.axis_names=()
        self.displayed_names=();self.selected='';self.curves=False
        self.font={}
        self.frame=ui.Frame(height=428)
        self.invalidate=invalidate or self.frame.invalidate_raster
        with self.frame:
            with ui.VStack(spacing=4):
                with ui.HStack(height=30,spacing=6):
                    self.label('Joint motion',size=19,color=self.TARGET)
                    self.overview_button=ui.Button('Back',width=72,visible=False,
                        clicked_fn=lambda:self.show_view(False),style={'font_size':16})
                self.summary_label=self.label('Waiting for joint data',height=44,size=15,word_wrap=True)
                self.overview_frame=ui.Frame(height=340)
                with self.overview_frame:
                    with ui.ScrollingFrame(height=340,
                            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF) as self.list_scroll:
                        self.rows_frame=ui.Frame()
                self.curves_frame=ui.Frame(height=550,visible=False)
                with self.curves_frame:
                    with ui.VStack(spacing=3):
                        self.selected_label=self.label('Select a joint to view its history',height=40,size=15,color=self.TARGET,word_wrap=True)
                        self.axis_frame=ui.Frame(height=34,visible=False)
                        self.detail_label=self.label('',height=72,size=14,word_wrap=True)
                        self.label('Angle°   Target (gold) / Actual (blue)',height=24,size=14,color=self.MUTED)
                        with ui.ZStack(height=84):
                            ui.Rectangle(style={'background_color':0xff232323})
                            self.target_plot=ui.Plot(ui.Type.LINE,-1.,1.,0.,0.,style={'color':self.TARGET,'background_color':0})
                            self.actual_plot=ui.Plot(ui.Type.LINE,-1.,1.,0.,0.,style={'color':self.ACTUAL,'background_color':0})
                        self.scale_label=self.label('',height=24,size=13,color=self.MUTED,word_wrap=True)
                        self.label('Tracking error° (target - actual)',height=24,size=14,color=self.ERROR)
                        self.error_plot=ui.Plot(ui.Type.LINE,-1.,1.,0.,0.,height=54,
                            style={'color':self.ERROR,'background_color':0xff232323})
                        self.time_label=self.label('Last 0.0 s (up to 5 s)',height=36,size=13,color=self.MUTED,word_wrap=True)
                        self.action_label=self.label('',height=52,size=13,color=self.MUTED,word_wrap=True)
        self.show_view(False)

    def label(self,text,*,size=12,color=None,**kwargs):
        return self.ui.Label(text,style={'font_size':size,'color':self.TEXT if color is None else color},**kwargs)

    def ingest(self,state):
        self.state.ingest(state)

    def rebuild_rows(self):
        ui=self.ui
        self.rows_frame.clear();self.rows={};self.cells={};self.member_lookup={}
        self.axis_names=()
        self.displayed_names=self.state.names
        if self.selected not in self.displayed_names:
            self.selected=next((n for n in self.displayed_names if 'left_knee' in n),self.displayed_names[0])
        with self.rows_frame:
            with ui.VStack(spacing=2):
                with ui.HStack(height=22):
                    self.label('Left',size=14,color=self.MUTED,alignment=ui.Alignment.CENTER)
                    self.label('Joint',width=112,size=14,color=self.MUTED,alignment=ui.Alignment.CENTER)
                    self.label('Right',size=14,color=self.MUTED,alignment=ui.Alignment.CENTER)
                for group_index,(group,rows) in enumerate(joint_part_rows(self.displayed_names)):
                    if group_index:
                        self.ui.Rectangle(height=1,style={'background_color':0xff505050})
                    paired=group in ('Legs','Arms')
                    for label,left,right in rows:
                        with ui.HStack(height=25,spacing=6):
                            if paired:
                                self.motion_cell(left,reverse=False)
                                self.label(label,width=112,size=15,alignment=ui.Alignment.CENTER)
                                self.motion_cell(right,reverse=True)
                            else:
                                members=left or right
                                button=ui.Button(label,width=140,clicked_fn=lambda n=members:self.select_part(n),
                                                 style={'font_size':16})
                                fill,tail=self.motion_bar()
                                speed=self.label('—',width=74,size=16,alignment=ui.Alignment.RIGHT_CENTER)
                                self.register_cell(members,(button,fill,tail,speed))

    def register_cell(self,members,cell):
        self.cells[members]=cell
        for name in members:
            self.rows[name]=cell
            self.member_lookup[name]=members

    def motion_bar(self):
        ui=self.ui
        with ui.VStack():
            ui.Spacer()
            with ui.ZStack(height=14):
                ui.Rectangle(style={'background_color':0xff343434,'border_radius':4})
                with ui.HStack():
                    fill=ui.Rectangle(width=ui.Fraction(.0001),visible=False,
                        style={'background_color':self.MUTED,'border_radius':4})
                    tail=ui.Spacer(width=ui.Fraction(1.))
            ui.Spacer()
        return fill,tail

    def motion_cell(self,members,*,reverse):
        ui=self.ui
        with ui.HStack(spacing=4):
            if not members:
                self.label('—',size=16,alignment=ui.Alignment.CENTER)
                return
            if reverse:fill,tail=self.motion_bar()
            button=ui.Button('—',width=67,clicked_fn=lambda n=members:self.select_part(n),
                             style={'font_size':16,'background_color':0xff343434})
            if not reverse:fill,tail=self.motion_bar()
            self.register_cell(members,(button,fill,tail,button))

    def select(self,name):
        self.selected=name
        self.show_view(True)

    def select_part(self,members):
        value=self.state.part_motion(members)
        if value is not None:self.select(value['dominant'])

    def update_axis_buttons(self):
        members=self.member_lookup[self.selected]
        self.axis_frame.visible=len(members)>1
        if self.axis_names!=members:
            self.axis_frame.clear();self.axis_buttons={};self.axis_names=members
            with self.axis_frame:
                with self.ui.HStack(spacing=6):
                    for name in members:
                        self.axis_buttons[name]=self.ui.Button(joint_part(name)[2],
                            clicked_fn=lambda n=name:self.select(n),style={'font_size':16})
        for name,button in self.axis_buttons.items():
            button.style={'font_size':16,'color':self.TARGET if name==self.selected else self.TEXT}

    def show_view(self,curves):
        self.curves=curves
        self.frame.height=self.ui.Pixel(640 if curves else 428)
        self.overview_frame.visible=not curves
        self.curves_frame.visible=curves
        self.overview_button.visible=curves
        self.refresh()

    def refresh(self):
        self.invalidate()
        self.summary_label.text=self.state.summary() if self.curves else self.state.motion_summary()
        values=self.state.latest
        if values is None:
            for button,fill,_,speed in self.cells.values():
                fill.visible=False;speed.text='—'
                button.tooltip='Joint data unavailable';speed.tooltip='Joint data unavailable'
            self.axis_frame.visible=False
            self.selected_label.text='Joint data unavailable'
            for label in (self.detail_label,self.scale_label,self.action_label):label.text=''
            self.time_label.text='Waiting for valid samples'
            for plot in (self.target_plot,self.actual_plot,self.error_plot):plot.set_data(0.,0.)
            return
        if self.displayed_names!=self.state.names:self.rebuild_rows()
        motion=self.state.motion()
        for members,(button,fill,tail,speed) in self.cells.items():
            value=self.state.part_motion(members,motion)
            amount=value['fill'];level=value['level'];color=self.MOTION_COLORS[level]
            fill.visible=amount>0
            fill.width=self.ui.Fraction(max(amount,.0001))
            tail.width=self.ui.Fraction(max(1.-amount,.0001))
            fill.style={'background_color':color,'border_radius':4}
            measured=value['speed']
            speed.text=f'{measured:.0f}°/s' if measured<1000 else f'{measured/1000:.1f}k°/s'
            button.style={'font_size':16,'color':self.TARGET if self.curves and self.selected in members else self.TEXT}
            speed.style={'font_size':16,'color':color}
            details=[]
            for name in members:
                i=self.displayed_names.index(name)
                details.append(f'{joint_part(name)[2]}: {motion["speed"][i]:.1f}°/s | swing {motion["swing"][i]:.1f}°')
            tooltip=(f'{MOTION_LEVELS[level]} | fastest axis: {joint_part(value["dominant"])[2]}\n'
                     + '\n'.join(details)+'\nMeasured axis motion, not torque or motor load.')
            speed.tooltip=tooltip;button.tooltip=tooltip
        if not self.curves:return
        self.update_axis_buttons()
        i=self.displayed_names.index(self.selected)
        self.selected_label.text=f'{joint_label(self.selected)}  |  {self.selected}'
        self.detail_label.text=(f'{MOTION_LEVELS[int(motion["level"][i])]}  {motion["speed"][i]:.1f}°/s | 1 s swing {motion["swing"][i]:.1f}°\n'
            f'Target {values["target"][i]:+.1f}°   Actual {values["q"][i]:+.1f}°\n'
            f'Error {values["error"][i]:+.1f}°   Instant vel {values["qd"][i]:+.1f}°/s')
        target,actual,error,duration=self.state.series(i)
        low=min((*target,*actual));high=max((*target,*actual));pad=max(2.,(high-low)*.1)
        for plot,data in ((self.target_plot,target),(self.actual_plot,actual)):
            plot.scale_min=low-pad;plot.scale_max=high+pad;plot.set_data(*data)
        peak=max(1.,max(abs(x) for x in error)*1.1)
        self.error_plot.scale_min=-peak;self.error_plot.scale_max=peak;self.error_plot.set_data(*error)
        self.scale_label.text=f'Angle range: {low-pad:.1f}° to {high+pad:.1f}°'
        self.time_label.text=f'Last {duration:.1f} s (max 5 s, simulation time)   Past -> Now'
        action=f'{values["action"][i]:+.2f}' if values['action_mask'][i] else 'not controlled'
        low,high=values['limits'][i]
        self.action_label.text=(f'Policy residual: {action} (unitless, not torque)\n'
            f'Limits {low:.1f}° to {high:.1f}° | margin {values["margin"][i]:.1f}°')

    def close(self):
        self.frame.clear();self.rows.clear();self.cells.clear();self.member_lookup.clear();self.axis_buttons.clear()
        self.state.history.clear()
