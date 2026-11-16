"""Cache only camera-fit geometry bounds, invalidated by USD layer contents."""
from pathlib import Path
import hashlib,json,time,uuid
import numpy as np


def cached_camera_bounds(stage,compute,directory,mode,enabled=True):
    tick=time.perf_counter();directory=Path(directory)
    def identity():
        layers={str(Path(layer.realPath).resolve()):hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
            for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
        recipe=dict(version=1,mode=mode,sample_step=15,usd_layers=layers)
        key=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
        return key,recipe
    key,recipe=identity();path=directory/(key+'.json')
    if enabled and path.is_file():
        try:
            record=json.loads(path.read_text(encoding='utf-8'))
            lo,hi=np.asarray(record['lo'],dtype=float),np.asarray(record['hi'],dtype=float)
            if record['identity']==recipe and lo.shape==hi.shape==(3,) and np.isfinite([lo,hi]).all() and np.all(hi>lo):
                return lo,hi,dict(hit=True,path=str(path),seconds=time.perf_counter()-tick)
        except (OSError,ValueError,KeyError,TypeError):pass
    lo,hi=compute()
    if enabled:
        key,recipe=identity();path=directory/(key+'.json');directory.mkdir(parents=True,exist_ok=True)
        temporary=directory/(key+'.'+uuid.uuid4().hex+'.tmp')
        temporary.write_text(json.dumps(dict(identity=recipe,lo=lo.tolist(),hi=hi.tolist())),encoding='utf-8')
        temporary.replace(path)
    return lo,hi,dict(hit=False,enabled=enabled,path=str(path),seconds=time.perf_counter()-tick)
