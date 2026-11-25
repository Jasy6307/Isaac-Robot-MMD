"""Native-height 1080p side-by-side excerpts from identical live command demos."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline-report',type=Path,required=True)
    p.add_argument('--candidate-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path)
    a=p.parse_args()
    reports=[json.loads(path.read_text(encoding='utf-8')) for path in [a.baseline_report,a.candidate_report]]
    for r in reports:
        assert r['video_resolution']==[1920,1080] and r['frames']==2160 and r['simulated_seconds']==72
        assert r['secondary_physics'] and r['runtime_compensation_removed'] and r['falls']==0
    for key in ['body_device','character_appearance','appearance_source_sha256','secondary_source_sha256',
                'quality','physics_hz','secondary_settings','anti_aliasing']:
        assert reports[0][key]==reports[1][key],f'Comparison conditions differ: {key}'
    ff=a.ffmpeg or next((Path.home()/'AppData/Roaming/Python').glob('Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe'))
    # Crop only background horizontally: each character keeps all 1080 source
    # scanlines. No enlargement of a low-resolution render is involved.
    filters=[]
    for source in [0,1]:
        filters.append(f'[{source}:v]split=3[s{source}a][s{source}b][s{source}c]')
        for suffix,start,end in [('a',31,37),('b',40,46),('c',49,55)]:
            filters.append(f'[s{source}{suffix}]trim=start={start}:end={end},setpts=PTS-STARTPTS,crop=960:1080:480:0[p{source}{suffix}]')
        filters.append(''.join(f'[p{source}{s}]' for s in 'abc')+f'concat=n=3:v=1:a=0[v{source}]')
    filters.append('[v0][v1]hstack=inputs=2,drawbox=x=0:y=0:w=iw:h=130:color=0x101820:t=fill,'
                   'drawbox=x=0:y=1022:w=iw:h=58:color=0x101820:t=fill,'
                   "drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='OLD - Unified 31 DOF':x=28:y=22:fontsize=34:fontcolor=white,"
                   "drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='NEW - Stance slip refinement':x=988:y=22:fontsize=34:fontcolor=white,"
                   "drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='Backward 1.20 m/s':x=(w-tw)/2:y=77:fontsize=30:fontcolor=0x79d8ff:enable='lt(t,6)',"
                   "drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='Left 1.20 m/s':x=(w-tw)/2:y=77:fontsize=30:fontcolor=0x79d8ff:enable='gte(t,6)*lt(t,12)',"
                   "drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='Right 1.20 m/s':x=(w-tw)/2:y=77:fontsize=30:fontcolor=0x79d8ff:enable='gte(t,12)',"
                   "drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='Same camera and appearance | Hair + skirt PhysX ON | No runtime compensation':x=(w-tw)/2:y=1038:fontsize=25:fontcolor=white[out]")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    command=[str(ff),'-v','error','-y','-i',str(a.baseline),'-i',str(a.candidate),
             '-filter_complex',';'.join(filters),'-map','[out]','-an','-c:v','libx264','-crf','16',
             '-preset','slow','-pix_fmt','yuv420p','-r','30','-movflags','+faststart',str(a.output)]
    subprocess.run(command,check=True)
    metadata=dict(resolution=[1920,1080],fps=30,seconds=18,source_resolution=[1920,1080],
                  source_ranges_seconds=[[31,37],[40,46],[49,55]],
                  composition='Two center 960x1080 crops at native height; horizontal background removed only',
                  baseline=str(a.baseline.resolve()),candidate=str(a.candidate.resolve()),
                  source_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.baseline,a.candidate]},
                  matched_conditions=True,secondary_physics=True,runtime_compensation_removed=True)
    a.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(a.output.resolve())


if __name__=='__main__':main()
