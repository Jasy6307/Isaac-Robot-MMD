"""Prepare a rig-calibrated source animation for default in-memory retargeting."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import uuid

from source.paths import resolve_dance_asset

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PMX = ROOT/'media/千早爱音_by_小麦弥望/Anon Chihaya.pmx'
DEFAULT_ASSET = ROOT/'output/pmx_prototype/character.usdc'


def resolve_vmd(csv_path, explicit=None):
    csv_path = resolve_dance_asset(csv_path)
    if explicit:
        candidate = resolve_dance_asset(explicit)
    elif csv_path.with_suffix('.vmd').is_file():
        candidate = csv_path.with_suffix('.vmd').resolve()
    else:
        meta = csv_path.with_suffix('.motion.json')
        value = json.loads(meta.read_text(encoding='utf-8')).get('vmd') if meta.is_file() else None
        if not value:
            raise ValueError('New retargeting needs the source VMD: use --source_vmd or a same-stem VMD/.motion.json; use --retarget_mode legacy for comparison')
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = meta.parent/candidate
    candidate = resolve_dance_asset(candidate)
    if not candidate.is_file() or candidate.suffix.lower() != '.vmd':
        raise ValueError(f'Source VMD does not exist: {candidate}')
    return candidate.resolve()


def prepare_source_animation(csv_path, *, vmd=None, leg_mode='fk', pmx=DEFAULT_PMX, asset=DEFAULT_ASSET):
    vmd = resolve_vmd(csv_path, vmd)
    pmx, asset = Path(pmx).resolve(), Path(asset).resolve()
    if leg_mode not in ('fk','ik'):
        raise ValueError('Source leg mode must be fk or ik')
    digest = hashlib.sha256()
    # Include the baker implementation so fixes invalidate old cached poses.
    baker = ROOT/'source/pmx_prototype/load_vmd.py'
    for path in (vmd,pmx,asset,baker):
        if not path.is_file():
            raise ValueError(f'Retargeting source asset is missing: {path}')
        digest.update(str(path).encode());digest.update(path.read_bytes())
    digest.update(leg_mode.encode())
    folder = ROOT/'output/retarget-source-cache';folder.mkdir(parents=True,exist_ok=True)
    output = folder/f'{digest.hexdigest()[:24]}_{leg_mode}.usdc'
    if output.is_file() and output.with_suffix('.json').is_file():
        return output
    temporary = folder/f'{output.stem}_{uuid.uuid4().hex}.usdc'
    try:
        subprocess.run([sys.executable,str(baker),'--vmd',str(vmd),'--pmx',str(pmx),
                        '--asset',str(asset),'--leg-mode',leg_mode,'--output',str(temporary)],cwd=ROOT,check=True)
        report = json.loads(temporary.with_suffix('.json').read_text(encoding='utf-8'))
        report['usd'] = str(output);report['source_cache_sha256'] = digest.hexdigest()
        temporary.replace(output)
        temporary.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.with_suffix('.json').replace(output.with_suffix('.json'))
    finally:
        temporary.unlink(missing_ok=True);temporary.with_suffix('.json').unlink(missing_ok=True)
    return output
