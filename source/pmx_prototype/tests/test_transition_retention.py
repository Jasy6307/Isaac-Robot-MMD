import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
from tensordict import TensorDict

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
from unified_walk_run_reference import sample_commands
from retention_anchor import install


class TransitionRetentionTests(unittest.TestCase):
    def test_order_and_pool_isolation(self):
        cfg=json.loads((Path(__file__).parent/'fixtures/transition_sampling.json').read_text())
        cfg['unified_transition_env_fraction']=.05
        n=1000;ids=torch.arange(n);mask=(ids%100>=90)&(ids%100<95)
        env=SimpleNamespace(num_envs=n,device='cpu',cfg=cfg,evaluation=False,iteration=7600,commands=torch.zeros(n,3),
                            continuous_ref_commands=torch.zeros(n,3),command_age=torch.zeros(n))
        sample_commands(env,ids,on_reset=True)
        for _ in range(4):
            before=env.transition_stage.clone();sample_commands(env,ids)
            self.assertTrue(torch.equal(env.transition_stage[mask],(before[mask]+1)%3))
            bank=torch.tensor([[1.1,0,.6],[1.1,0,-.6],[-.9,0,.6]])
            self.assertTrue(torch.equal(env.commands[mask],bank[env.transition_stage[mask]]))
            self.assertTrue(((env.unified_command_hold[mask]>=5)&(env.unified_command_hold[mask]<=6)).all())
        self.assertTrue((env.transition_stage[~mask]==0).all())

    def test_retention_reduces_teacher_error_without_altering_critic(self):
        class Actor(torch.nn.Module):
            def __init__(self):super().__init__();self.layer=torch.nn.Linear(3,2,bias=False)
            def forward(self,obs):return self.layer(obs['policy'])
        torch.manual_seed(1);actor=Actor();critic=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.SGD([*actor.parameters(),critic],lr=.1)
        initial=actor.layer.weight.detach().clone();observations=TensorDict({'policy':torch.ones(100,3)},batch_size=[100])
        def update():
            with torch.no_grad():actor.layer.weight.add_(.1)
            return {}
        alg=SimpleNamespace(actor=actor,optimizer=optimizer,update=update)
        env=SimpleNamespace(device='cpu',ids=torch.arange(100),cfg={'train_retention_steps':2,'train_retention_weight':2.},get_observations=lambda:observations)
        install(SimpleNamespace(alg=alg),env);logs=alg.update()
        self.assertLess(float((actor.layer.weight-initial).square().mean()),.01)
        self.assertTrue(torch.equal(critic,torch.ones(1)))
        self.assertGreater(logs['retention_action_rmse'],0)


if __name__=='__main__':unittest.main()
