"""Build all local dance reference movies with the approved Mutsumi PhysX rig.

Run with a regular Python containing imageio-ffmpeg. Each Isaac worker uses the
project's verified runtime. A manifest supports safe resume; defaults are only
updated after every requested movie is decoded and verified.
"""

from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source, config_path
from pathlib import Path
import argparse, hashlib, json, os, struct, subprocess, sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from source.paths import resolve_dance_asset
from source.pmx_prototype.secondary_profiles import DEFAULT_SECONDARY_PROFILE
from source.pmx_prototype.appearance_defaults import DEFAULT_LOOK,LOOK_CHOICES

p=argparse.ArgumentParser()
p.add_argument('--dances',nargs='+',default=['IRIS_OUT','ni_zui_zhong_yao','hal_ia_yeah_oh_ahhh_oh','ji_le_jing_tu','girls','deepbluetown'])
p.add_argument('--preserve-existing-dances',nargs='*',default=['IRIS_OUT'],help='Keep approved existing movies byte-for-byte, even with --force; pass an empty list to opt out')
p.add_argument('--leg-clearance-dances',nargs='*',default=[],help='Explicit opt-in only; alters foot trajectories and may distort choreography')
p.add_argument('--force',action='store_true',help='Regenerate even verified movies')
p.add_argument('--secondary-profile',type=config_path,default=DEFAULT_SECONDARY_PROFILE,help='Secondary physics profile; defaults to the approved V10 hair-contact/drape and existing fabric/waist/side-hair template at120Hz')
p.add_argument('--legacy-secondary-physics',action='store_true',help='Use the original Mutsumi rig recipe without the new default profile')
p.add_argument('--look',choices=LOOK_CHOICES,default=DEFAULT_LOOK)
p.add_argument('--mesh-camera-dances',nargs='*',default=[],help='Fit rendered mesh bounds for low-pose clips')
p.add_argument('--output',type=Path,default=ROOT/'output/mutsumi_validation_references')
a=p.parse_args()
import imageio_ffmpeg

a.output.mkdir(parents=True,exist_ok=True)
manifest_path=a.output/'manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else dict(version=1,dances={})
encoder=imageio_ffmpeg.get_ffmpeg_exe()
pmx=ROOT/'media/mutsmi/mutsmi v1.pmx'
asset=ROOT/'output/mutsumi_pmx/character.usdc'
approved=ROOT/'output/mutsumi_physx_short10_release/rig.json'
recipe=json.loads(approved.read_text(encoding='utf-8'))
def frame_count(path):
    data=path.read_bytes()
    if not data.startswith(b'Vocaloid Motion Data 0002'):
        raise ValueError(f'Unsupported VMD header: {path}')
    count=struct.unpack_from('<I',data,50)[0]
    if not count or 54+111*count>len(data): raise ValueError(f'Truncated/empty VMD: {path}')
    return max(frame for _,frame in struct.iter_unpack('<15sI92x',data[54:54+111*count]))+1
inventory={dance:frame_count(resolve_dance_asset(ROOT/f'media/dance/{dance}.vmd')) for dance in a.dances}
env=os.environ.copy();env['PYTHONUNBUFFERED']='1'

def save():
    temporary=manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,manifest_path)

def run(script,args,log):
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen([str(ROOT/'run_isaac61.bat'),str(ROOT/script),*map(str,args)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        code=process.wait()
    if code: raise RuntimeError(f'Worker failed ({code}): {log}')
    if script.endswith('physx_dance.py') and 'PHYSX_LIVE_PASS ' not in log.read_text(encoding='utf-8',errors='replace'):
        raise RuntimeError(f'Renderer did not report successful completion: {log}')

manifest['status']='running';manifest.pop('error',None);save()
try:
 for dance in a.dances:
    folder=a.output/dance;folder.mkdir(exist_ok=True)
    vmd=resolve_dance_asset(ROOT/f'media/dance/{dance}.vmd')
    audio=next((vmd.with_suffix(ext) for ext in ('.wav','.ogg','.mp3') if vmd.with_suffix(ext).is_file()),vmd.with_suffix('.wav'))
    count=inventory[dance]
    video=vmd.with_name(dance+'_mutsumi_reference.mp4')
    if dance in a.preserve_existing_dances and video.exists():
        sidecar=video.with_suffix('.json')
        preserved=json.loads(sidecar.read_text(encoding='utf-8'))
        if preserved.get('status')!='verified' or preserved.get('video_sha256')!=hashlib.sha256(video.read_bytes()).hexdigest():
            raise RuntimeError(f'Approved movie integrity check failed; refusing to regenerate protected dance: {dance}')
        manifest['dances'][dance]=preserved;save()
        print('REFERENCE_PRESERVED',dance,flush=True);continue
    clearance=dance in a.leg_clearance_dances
    mesh_camera=dance in a.mesh_camera_dances
    policy_data=dict(version=3,leg_clearance=clearance,look=a.look,ground='original',render_fps=30,resolution=[720,900])
    if mesh_camera:policy_data['camera_fit']='skinned_mesh'
    policy=json.dumps(policy_data,sort_keys=True).encode()
    profile_path=a.secondary_profile.resolve() if a.secondary_profile and not a.legacy_secondary_physics else None
    profile_bytes=profile_path.read_bytes() if profile_path else b''
    signature=hashlib.sha256(vmd.read_bytes()+approved.read_bytes()+policy+profile_bytes).hexdigest()
    old=manifest['dances'].get(dance,{})
    if (not a.force and old.get('status')=='verified' and old.get('signature')==signature and video.exists()
            and old.get('video_sha256')==hashlib.sha256(video.read_bytes()).hexdigest()):
        print('REFERENCE_RESUME',dance,flush=True);continue
    entry=dict(status='baking',signature=signature,vmd=str(vmd),video=str(video),frames=count,fps=30,duration_seconds=count/30,
               actor='若叶睦',credits='很ying的哈呵',source_first_frame=0,translation_scale=.3115215949,
               stance_reference='Anon Chihaya PMX',leg_clearance=clearance,retarget_policy_version=2,rig_recipe=str(approved),look=a.look,ground_appearance='original')
    manifest['dances'][dance]=entry;save();print('REFERENCE_START',dance,count,flush=True)
    body=folder/'body.usdc'
    run('source/pmx_prototype/load_vmd.py',['--pmx',pmx,'--vmd',vmd,'--asset',asset,'--output',body,'--translation-scale','.3115215949','--stance-reference',ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx']+(['--leg-clearance'] if clearance else []),folder/'bake.log')
    run('source/pmx_prototype/prepare_mutsumi_physx.py',['--usd',body,'--pmx',pmx,'--output',folder,'--profile','smooth','--skirt-length-scale','.9'],folder/'rig.log')
    rig_path=folder/'rig.json';rig=json.loads(rig_path.read_text(encoding='utf-8'))
    for field in ('skirt_contact_smoothing','skirt_contact_release','skirt_max_stretch','locked_hair_bones','cloth_translation_only','skirt_fabric_relaxation'):
        if field in recipe:rig[field]=recipe[field]
    if profile_path:
        profile=json.loads(profile_bytes.decode('utf-8'))
        rig.update(profile)
        if rig.get('skirt_contact_smoothing') and 'skirt_linear_damping' in profile.get('appearance',{}):
            rig['skirt_contact_smoothing']['linear_damping']=profile['appearance']['skirt_linear_damping']
        entry['appearance_profile']=str(profile_path)
        entry['appearance_profile_sha256']=hashlib.sha256(profile_bytes).hexdigest()
    rig_path.write_text(json.dumps(rig,ensure_ascii=False,indent=2),encoding='utf-8')
    entry['status']='recording';save();print('REFERENCE_RECORD',dance,flush=True)
    pending=video.with_name(video.stem+'.partial.mp4')
    run('source/pmx_prototype/physx_dance.py',['--usd',folder/'character.usdc','--pmx',pmx,'--rig-data',rig_path,
        '--music',audio,'--output',folder/'render','--headless','--benchmark','--no-music','--no-obstacles','--quality','cloth','--legacy-secondary-physics',
        '--steps',count,'--render-fps',30,'--width',720,'--height',900,'--reference-clock','--fit-camera','--look',a.look,
        '--record-video',pending,'--ffmpeg',encoder,'--bake-usd',folder/'physics.usdc','--capture-frames',0,count//2,count-1]+(['--fit-camera-mesh'] if mesh_camera else []),folder/'record.log')
    report=json.loads((folder/'render/physx_report.json').read_text(encoding='utf-8'))
    if report['status']!='complete' or not report['reference_clock'] or report['source_first_frame']!=0:
        raise RuntimeError(f'Incomplete or misaligned movie: {dance}')
    if report['look']!=a.look or report['ground_appearance']!='original':raise RuntimeError(f'Appearance verification failed: {dance}')
    if profile_path and report['physics_hz']!=profile['physics_hz']:raise RuntimeError(f'Physics clock mismatch: {dance}')
    entry.update(physics_hz=report['physics_hz'],physics_steps=report['physics_steps'],camera_fit=report.get('camera_fit','joints'))
    reader=imageio_ffmpeg.read_frames(str(pending));metadata=next(reader);decoded=sum(1 for _ in reader)
    if decoded!=count or metadata['fps']!=30 or metadata['size']!=(720,900):
        raise RuntimeError(f'Frame verification failed: {dance}: {decoded}, {metadata}')
    os.replace(pending,video)
    entry.update(status='verified',decoded_frames=decoded,resolution=[720,900],video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),physics_report=str(folder/'render/physx_report.json'))
    video.with_suffix('.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf-8')
    save();print('REFERENCE_VERIFIED',dance,decoded,flush=True)
 config=ROOT/'source/train_workflow/validation_videos.json'
 data=json.loads(config.read_text(encoding='utf-8-sig'))
 for dance in a.dances:
    entry=manifest['dances'][dance]
    data['dances'][dance]=dict(video=Path(entry['video']).relative_to(ROOT).as_posix(),start_seconds=0.,actor='若叶睦',label='Mutsumi PhysX reference')
    if dance.endswith('_short'):
        for variant in Path(entry['vmd']).parent.glob(dance+'_*.h5'):
            data['dances'][variant.stem]={**data['dances'][dance],'alias_of':dance}
 temporary=config.with_suffix('.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(temporary,config)
 manifest['status']='complete';save();print('MUTSUMI_REFERENCES_COMPLETE',len(a.dances),flush=True)
except BaseException as error:
 manifest['status']='failed';manifest['error']=repr(error);save();raise
