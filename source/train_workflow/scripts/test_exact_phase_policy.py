"""Pure CPU tests in Isaac Python: composition equals the validated controls."""
import sys
import unittest
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.exact_phase_policy import ExactPhaseResidualPolicy
from source.train_workflow.scripts.compose_phase_checkpoint import compose


class ExactTests(unittest.TestCase):
    def test_from_scratch_gradients_are_separated_by_gate(self):
        obs=TensorDict({"policy":torch.randn(8,163)},batch_size=[8])
        model=ExactPhaseResidualPolicy(obs,{"actor":["policy"]},"actor",29,
            hidden_dims=[32,16],obs_normalization=True,freeze_base=False,freeze_normalization=False)
        for gate in (0.,1.):
            model.zero_grad();obs["policy"][:,-1]=gate
            model(obs).square().mean().backward()
            normal=sum(float(p.grad.abs().sum()) for p in model.mlp.parameters() if p.grad is not None)
            jump=sum(float(p.grad.abs().sum()) for p in model.jump_mlp.parameters() if p.grad is not None)
            self.assertEqual(jump if gate==0 else normal,0.)
            self.assertGreater(normal if gate==0 else jump,0.)

    def test_from_scratch_normalization_uses_separate_motion_groups(self):
        x=torch.randn(8,163);x[:4,-1]=0.;x[4:,-1]=1.
        obs=TensorDict({"policy":x},batch_size=[8])
        model=ExactPhaseResidualPolicy(obs,{"actor":["policy"]},"actor",29,
            hidden_dims=[32,16],obs_normalization=True,freeze_base=False,freeze_normalization=False)
        model.update_normalization(obs)
        torch.testing.assert_close(model.obs_normalizer._mean[:,:153],x[:4,:153].mean(0,keepdim=True))
        torch.testing.assert_close(model.jump_normalizer._mean,x[4:,:162].mean(0,keepdim=True))

    def test_compiled_policy_matches_both_sources_and_smooth_composition(self):
        torch.manual_seed(12)
        groups={"actor":["policy"],"critic":["policy"]}
        models=[]
        for dim in (153,162):
            obs=TensorDict({"policy":torch.randn(10,dim)},batch_size=[10])
            actor=MLPModel(obs,groups,"actor",29,hidden_dims=[32,16],obs_normalization=True)
            critic=MLPModel(obs,groups,"critic",1,hidden_dims=[32,16],obs_normalization=True)
            models.append((actor,critic))
        a,b=models
        checkpoint=compose(dict(actor_state_dict=a[0].state_dict(),critic_state_dict=a[1].state_dict()),
                           dict(actor_state_dict=b[0].state_dict(),critic_state_dict=b[1].state_dict()))
        x=torch.randn(10,163);x[:,-1]=torch.linspace(0,1,10)
        obs=TensorDict({"policy":x},batch_size=[10])
        model=ExactPhaseResidualPolicy(obs,groups,"actor",29,hidden_dims=[32,16],obs_normalization=True)
        model.load_state_dict(checkpoint["actor_state_dict"])
        normal=a[0]({"policy":x[:,:153]});jump=b[0]({"policy":x[:,:162]})
        expected=normal+x[:,-1:]*(jump-normal)
        torch.testing.assert_close(model(obs),expected,atol=1e-6,rtol=1e-5)
        before={k:v.clone() for k,v in model.obs_normalizer.state_dict().items()}
        model.update_normalization(obs)
        for k,v in before.items():self.assertTrue(torch.equal(model.obs_normalizer.state_dict()[k],v))
        self.assertFalse(any(p.requires_grad for p in model.mlp.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.jump_mlp.parameters()))


if __name__=="__main__":unittest.main()
