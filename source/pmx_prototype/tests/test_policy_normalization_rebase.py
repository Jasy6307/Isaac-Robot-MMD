"""Raw-input actions must survive a change in observation statistics."""

import sys as _sys
from pathlib import Path as _Path
_source_dir = _Path(__file__).resolve().parents[1]
if str(_source_dir) not in _sys.path:
    _sys.path.insert(0, str(_source_dir))
from source_layout import PROJECT_ROOT, SOURCE_DIR, resolve_source

import unittest
import torch
from policy_normalization_rebase import rebase_state

class RebaseTests(unittest.TestCase):
    def test_affine_rebase_preserves_raw_network_function(self):
        torch.manual_seed(42)
        old={'obs_normalizer._mean':torch.randn(1,248),'obs_normalizer._std':torch.rand(1,248)+.06,
             'obs_normalizer._var':torch.ones(1,248),'obs_normalizer.count':torch.tensor(100),
             'mlp.0.weight':torch.randn(256,248)*.05,'mlp.0.bias':torch.randn(256)}
        new={k:v.clone() for k,v in old.items()}
        new['obs_normalizer._mean']=torch.randn(1,248)*2
        new['obs_normalizer._std']=torch.rand(1,248)*3+.1
        result=rebase_state(old,new)
        raw=torch.randn(512,248)*3
        forward=lambda s:torch.nn.functional.elu(torch.nn.functional.linear((raw-s['obs_normalizer._mean'])/(s['obs_normalizer._std']+.01),s['mlp.0.weight'],s['mlp.0.bias']))
        self.assertTrue(torch.allclose(forward(old),forward(result),atol=1e-4,rtol=1e-5))
        self.assertTrue(torch.equal(result['obs_normalizer._mean'],new['obs_normalizer._mean']))
        self.assertTrue(torch.equal(old['obs_normalizer.count'],torch.tensor(100)))

if __name__=='__main__':unittest.main()
