"""Encode one rendered frame per control step, with explicit simulation timing."""
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def find_ffmpeg():
    executable=shutil.which('ffmpeg')
    if executable:return executable
    candidates=sorted((Path.home()/'AppData/Roaming/Python').glob('Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe'))
    if not candidates:raise FileNotFoundError('ffmpeg is required for --record-video')
    return str(candidates[-1])


class CrowdVideoRecorder:
    def __init__(self,path,size,output,*,radius=7.,fps=30):
        self.path=Path(path).resolve();self.width,self.height=size;self.fps=fps;self.frames=0
        if self.path.exists():raise FileExistsError(f'Refusing to overwrite {self.path}')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.log=(Path(output)/'video_encoder.log').open('w',encoding='utf-8')
        self.process=subprocess.Popen([find_ffmpeg(),'-n','-v','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{self.width}x{self.height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264',
            '-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(self.path)],
            stdin=subprocess.PIPE,stderr=self.log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        font=Path('C:/Windows/Fonts/arial.ttf')
        self.title_font=ImageFont.truetype(str(font),max(18,round(self.height*25/1080)))
        self.font=ImageFont.truetype(str(font),max(15,round(self.height*20/1080)))
        self.set_camera([radius*1.2,-radius*1.65,radius*1.25],[0.,0.,.6])

    def set_camera(self,eye,target):
        self.eye=np.asarray(eye,dtype=float)
        self.forward=np.asarray(target,dtype=float)-self.eye;self.forward/=np.linalg.norm(self.forward)
        self.right=np.cross(self.forward,[0.,0.,1.]);self.right/=np.linalg.norm(self.right)
        self.up=np.cross(self.right,self.forward)

    def write(self,rgb,frame,pelvis_positions,falls,*,controlled_actor=None):
        if rgb.shape!=(self.height,self.width,3):raise ValueError(f'Unexpected viewport size {rgb.shape}')
        image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
        scale=self.height/1080;pad=round(20*scale)
        draw.rectangle((0,0,self.width,round(78*scale)),fill=(19,26,35))
        mode='independent random commands' if controlled_actor is None else f'controlled Anon {controlled_actor+1:02d} + random background'
        draw.text((pad,round(8*scale)),f'{len(pelvis_positions)} Anons | {mode} | live policy inference',
                  font=self.title_font,fill=(242,245,250))
        draw.text((pad,round(43*scale)),f'Simulation time {(frame+1)/self.fps:.2f} s | Falls {int(sum(falls))}',
                  font=self.font,fill=(134,218,248))
        focal=self.width*35/36
        for i,position in enumerate(pelvis_positions):
            delta=np.asarray(position)+np.array([0.,0.,.98])-self.eye;depth=np.dot(delta,self.forward)
            if depth<=0:continue
            x=self.width/2+focal*np.dot(delta,self.right)/depth
            y=self.height/2-focal*np.dot(delta,self.up)/depth
            if not pad<x<self.width-pad or not 85*scale<y<self.height-50*scale:continue
            text=f'{i+1:02d}';box=draw.textbbox((x,y),text,font=self.font)
            draw.rounded_rectangle((box[0]-5,box[1]-3,box[2]+5,box[3]+3),radius=4,fill=(19,26,35))
            draw.text((x,y),text,font=self.font,fill=(245,235,175))
        footer=round(40*scale)
        draw.rectangle((0,self.height-footer,self.width,self.height),fill=(19,26,35))
        draw.text((pad,self.height-footer+round(8*scale)),
                  f'{self.fps} FPS simulation-time recording | Playback speed does not represent live performance',
                  font=self.font,fill=(220,226,235))
        self.process.stdin.write(np.asarray(image).tobytes());self.frames+=1
        if frame%300==0:image.save(self.path.parent/f'{self.path.stem}_frame_{frame:04d}.jpg',quality=92)

    def finish(self):
        self.process.stdin.close()
        code=self.process.wait(timeout=120);self.log.close()
        if code:raise RuntimeError(f'Video encoding failed with exit code {code}')

    def abort(self):
        if self.process.poll() is None:self.process.kill();self.process.wait(timeout=15)
        self.log.close()
