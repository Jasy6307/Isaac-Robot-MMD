"""Compose native portrait panels into an unwarped 1080p side/rear comparison."""

# Allow both the canonical tool path and the historical root entry point.
import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import argparse
import json
from pathlib import Path
import subprocess
from PIL import Image,ImageDraw,ImageFont

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--panels',nargs=3,type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--titles',nargs=3,help='Explicit panel names for the comparison stage')
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
ff=str(next((Path.home()/'AppData/Roaming/Python').glob('Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe')))
reports=[json.loads((x/'report.json').read_text()) for x in a.panels]
for report in reports[1:]:
    for key in ['camera_offsets','resolution','secondary_settings','appearance_defaults_sha256','frames_per_view']:
        assert report[key]==reports[0][key],key
    if 'support_source_sha256' in report and 'support_source_sha256' in reports[0]:
        assert report['support_source_sha256']==reports[0]['support_source_sha256'],'Appearance/physics implementation changed'
parts=[]
for view in ['side','rear']:
    dest=a.output.with_name(a.output.stem+'_'+view+'.mp4');parts.append(dest)
    cmd=[ff,'-y','-v','error']
    for folder in a.panels:cmd+=['-i',str(folder/(view+'.mp4'))]
    filters='[0:v][1:v][2:v]hstack=inputs=3[v]'
    if a.titles:
        label=a.output.with_name(a.output.stem+'_'+view+'_titles.png')
        im=Image.new('RGB',(1920,90),(17,23,33));draw=ImageDraw.Draw(im)
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
        for i,title in enumerate(a.titles):
            draw.text((i*640+16,12),title,font=font,fill='white')
            speed=reports[i].get('commanded_forward_speed',.4)
            draw.text((i*640+16,54),('侧面' if view=='side' else '背面')+f' · 同机位 / {speed:.1f} m/s',font=small,fill=(120,218,255))
        im.save(label);cmd+=['-loop','1','-i',str(label)]
        filters='[0:v][1:v][2:v]hstack=inputs=3[base];[base][3:v]overlay=0:0:shortest=1[v]'
    cmd+=['-filter_complex',filters,'-map','[v]','-frames:v',str(reports[0]['frames_per_view']),'-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p',str(dest)]
    subprocess.run(cmd,check=True)
subprocess.run([ff,'-y','-v','error','-i',str(parts[0]),'-i',str(parts[1]),'-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0[v]','-map','[v]','-c:v','libx264','-crf','16','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)],check=True)
subprocess.run([ff,'-v','error','-i',str(a.output),'-f','null','-'],check=True)
meta=dict(panels=[str(x.resolve()) for x in a.panels],titles=a.titles,resolution=[1920,1080],fps=30,frames=reports[0]['frames_per_view']*2,full_decode_passed=True,spatial_warp=False,sequence=['side','rear'])
a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(a.output)
