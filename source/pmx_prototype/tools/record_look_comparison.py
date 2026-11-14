"""Record every Anon IRISOUT look and build synchronized comparison videos."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

from pathlib import Path
import argparse
import subprocess
import json
import shutil
import sys
import time

ROOT=PROJECT_ROOT
sys.path.insert(0,str(ROOT))
from source.pmx_prototype.appearance_defaults import LOOK_CHOICES
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'media/recordings/IRIS_OUT_look_comparison_current')
parser.add_argument('--compose-only',action='store_true')
args=parser.parse_args()
OUT=args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
ENCODER=next((Path.home()/'AppData/Roaming/Python').glob('Python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe'))
FONT=ROOT/'output/iris-out-physx-preview/comparison_font.ttc'
if not FONT.exists():shutil.copyfile('C:/Windows/Fonts/msyh.ttc',FONT)
font=FONT.relative_to(ROOT).as_posix()
modes=[
    ('original','原版','通用材质与照明'),
    ('soft-pbr','提亮 PBR','低反射 · 脸部柔光'),
    ('mmd-toon','精调赛璐璐（默认）','环境75% · 材质88% · 脸部塑形'),
]
assert tuple(item[0] for item in modes)==LOOK_CHOICES
manifest=dict(started=time.strftime('%Y-%m-%d %H:%M:%S'),modes=[],state='recording')
if args.compose_only:
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert [item['key'] for item in manifest['modes']]==[item[0] for item in modes]
def save():
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
def ffmpeg(args,log):
    with (OUT/log).open('wb') as stream:
        subprocess.run([str(ENCODER),'-y','-hide_banner',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
def label(index):
    _,title,description=modes[index]
    return (f"drawtext=fontfile={font}:expansion=none:text='{title}':x=24:y=22:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.65,"
            f"drawtext=fontfile={font}:expansion=none:text='{description}':x=24:y=68:fontsize=23:fontcolor=white:box=1:boxcolor=black@0.65")

save()
try:
    for index,(key,title,description) in enumerate([] if args.compose_only else modes):
        path=OUT/f'{index+1:02d}_{key}.mp4'
        print(f'RECORDING {index+1}/{len(modes)} {key}',flush=True)
        command=[str(ROOT/'run_isaac61.bat'),'source/pmx_prototype/record_physx_video.py',
            '--usd','output/iris-out-audit/IRIS_OUT_pmx.usdc','--output',str(path),
            '--audio','media/dance/IRIS_OUT/IRIS_OUT.wav','--ffmpeg',str(ENCODER),
            '--video-width','1920','--video-height','1080',
            '--look',key,'--save-poses' if index==0 else '--load-poses',str(OUT/'shared_poses.npz')]
        with (OUT/f'{index+1:02d}_render.log').open('wb') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        assert report['frames']==465 and report['width']==1920 and report['height']==1080
        ffmpeg(['-i',str(path),'-f','null','-'],f'{index+1:02d}_decode.log')
        manifest['modes'].append(dict(key=key,title=title,video=str(path),report=report))
        save()
        print(f'RECORDED {index+1}/{len(modes)} {key}',flush=True)
    reports=[item['report'] for item in manifest['modes']]
    assert all(r['camera']==reports[0]['camera'] for r in reports)
    manifest['identical_trajectory_hashes']=len({r['trajectory_sha256'] for r in reports})==1
    assert manifest['identical_trajectory_hashes'],'Rendered joint trajectories differ'
    manifest['ground_appearance']='original'
    manifest['state']='composing';save()
    inputs=[]
    for item in manifest['modes']:inputs+=['-i',item['video']]
    filters=[f'[{i}:v]crop=960:1080:480:0,setsar=1,{label(i)}[v{i}]' for i in range(len(modes))]
    filters.append(''.join(f'[v{i}]' for i in range(len(modes)))+
        f'xstack=inputs={len(modes)}:layout='+ '|'.join(f'{960*i}_0' for i in range(len(modes)))+'[v]')
    overview=OUT/'IRIS_OUT_all_looks_3panel.mp4'
    ffmpeg([*inputs,'-filter_complex',';'.join(filters),'-map','[v]','-map','0:a:0',
        '-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-c:a','copy',
        '-t','15.5','-movflags','+faststart',str(overview)],'overview_encode.log')
    pair_paths=[]
    for pair,(a,b) in enumerate(((0,1),(1,2),(0,2))):
        pair_path=OUT/f'pair_{pair+1}.mp4';pair_paths.append(pair_path)
        filters=(f'[0:v]crop=960:1080:480:0,setsar=1,{label(a)}[a];'
                 f'[1:v]crop=960:1080:480:0,setsar=1,{label(b)}[b];[a][b]hstack=inputs=2[v]')
        ffmpeg(['-i',manifest['modes'][a]['video'],'-i',manifest['modes'][b]['video'],
            '-filter_complex',filters,'-map','[v]','-map','0:a:0','-c:v','libx264','-crf','18',
            '-preset','fast','-pix_fmt','yuv420p','-c:a','copy','-t','15.5',str(pair_path)],f'pair_{pair+1}_encode.log')
    concat=OUT/'pairs.txt'
    concat.write_text(''.join(f"file '{p.name}'\n" for p in pair_paths),encoding='utf-8')
    pairs=OUT/'IRIS_OUT_look_pairs_1080p.mp4'
    ffmpeg(['-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(pairs)],'pairs_concat.log')
    for path in (overview,pairs):ffmpeg(['-i',str(path),'-f','null','-'],path.stem+'_decode.log')
    ffmpeg(['-ss','2','-i',str(overview),'-frames:v','1','-update','1',str(OUT/'overview.jpg')],'overview_still.log')
    manifest.update(state='complete',overview=str(overview),pair_comparison=str(pairs))
    save();print('LOOK_COMPARISON_COMPLETE',flush=True)
except BaseException as error:
    manifest.update(state='failed',error=repr(error));save();raise
