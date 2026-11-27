"""Package a calibration that preserves the qualified low-speed intervention threshold."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

from tensordict import TensorDict
import torch

HERE=Path(__file__).resolve().parent
from measure import sha256 as digest
sys.path.insert(0, str(HERE.parents[1]))
import source_layout
from anon_recovery_actor import RecoveryActor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'wider', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    original = json.loads((source.parent/'agent_config.json').read_text(encoding='utf-8'))
    slower = original['actor']['recovery']
    broad = json.loads((args.wider.parent/'agent_config.json').read_text(encoding='utf-8'))['actor']['recovery']
    config=copy.deepcopy(original)
    config['actor']['recovery'].update(threshold=broad['threshold'],width=broad['width'],gain=broad['gain'],
        low_speed=dict(threshold=slower['threshold'],width=slower['width'],gain=slower['gain']))
    shutil.copyfile(source,destination/'policy_final.pt')
    shutil.copyfile(source.parent/'env_config.json',destination/'env_config.json')
    (destination/'agent_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')

    actor_cfg=copy.deepcopy(config['actor'])
    del actor_cfg['class_name']
    raw=torch.zeros(4,248)
    raw[:,9:11]=torch.tensor([[0.,0.],[.04,0.],[.4,0.],[1.2,0.]])
    inp=TensorDict({'policy':raw},batch_size=[len(raw)])
    actor=RecoveryActor(inp,config['obs_groups'],'actor',31,**actor_cfg)
    state=torch.load(source,map_location='cpu',weights_only=True)
    actor.load_state_dict(state['actor_state_dict'])
    actor.eval()
    export=torch.jit.script(actor.as_jit().eval())
    torch.testing.assert_close(export(raw),actor(inp))
    torch.jit.save(export,str(destination/'policy_jit.pt'))
    output = destination/'policy_final.pt'
    audit = dict(copied_weights_from=str(source), checkpoint=str(output), sha256=digest(output))
    (destination/'training.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
