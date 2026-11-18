"""Raw-frame mirrors, rollout eligibility and actual RSL symmetry gradients."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import source_layout
from lateral_symmetry import AnatomicalMirror,augmentation,eligibility,configure,install_normalization,paired_reference_std


class MirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.fixture=json.loads((Path(__file__).parent/'fixtures/lateral_symmetry_manifest.json').read_text())

    def setUp(self):
        f=self.fixture
        self.m=AnatomicalMirror([j['name'] for j in f['joints']],f['key_names'],f,f['offsets'])
        self.e=SimpleNamespace(lateral_mirror=self.m)

    def test_involution_and_vector_parity(self):
        x=torch.randn(23,248);x[:,10]+=3
        y=self.m.observations(x)
        torch.testing.assert_close(self.m.observations(y),x,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(y[:,:3],x[:,:3]*torch.tensor([-1,1,1]))
        torch.testing.assert_close(y[:,3:6],x[:,3:6]*torch.tensor([1,-1,-1]))
        torch.testing.assert_close(y[:,237:241],x[:,237:241]*torch.tensor([1,-1,-1,1]))
        torch.testing.assert_close(y[:,202:204],x[:,202:204].flip(-1))

    def test_measured_phase_offset_not_assumed_half_cycle(self):
        x=torch.zeros(1,248);x[:,10]=1.2;x[:,205]=1
        y=self.m.observations(x)
        phase=torch.atan2(y[:,204],y[:,205])/(2*torch.pi)
        self.assertAlmostEqual(float(phase),.4375,places=6) # -.5625 modulo one
        self.assertNotAlmostEqual(float(phase),.5,places=3)

    def test_amp_window_preserves_time_and_swaps_anatomy(self):
        x=torch.randn(7,4,96);y=self.m.amp(x)
        torch.testing.assert_close(self.m.amp(y),x)
        torch.testing.assert_close(y[...,0],x[...,0])
        torch.testing.assert_close(y[...,10:41],self.m.joints(x[...,10:41]))
        torch.testing.assert_close(y[...,72:].reshape(7,4,8,3),x[...,72:].reshape(7,4,8,3)[...,self.m.key_index,:]*self.m.polar)

    def test_phase_reward_scales_match_paired_joints(self):
        q=torch.randn(128,31)*torch.arange(1,32)[None,:]/10
        std=paired_reference_std(q,self.m,.06)
        torch.testing.assert_close(std,std[self.m.index])
        self.assertTrue((std>=.06).all())

    def test_amp_reward_is_invariant_with_biased_discriminator(self):
        from amp_motion_prior import AMPMotionPrior
        discriminator=torch.nn.Linear(384,1)
        prior=SimpleNamespace(mirror=self.m,feature_size=96,discriminator=discriminator,
            normalize=lambda x:(x-.31)*1.7)
        x=torch.randn(16,384);mx=self.m.amp(x.reshape(16,4,96)).flatten(1)
        a=AMPMotionPrior.score(prior,x);b=AMPMotionPrior.score(prior,mx)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        prior.mirror=None
        torch.testing.assert_close(AMPMotionPrior.score(prior,x),discriminator(prior.normalize(x)).squeeze(-1),atol=0,rtol=0)

    def test_only_settled_pure_lateral_run_is_eligible(self):
        target=torch.tensor([[0,1.2,0],[0,-1.2,0],[.2,1.2,0],[0,1.2,.2],[0,.4,0],[0,0,0],[0,1.2,0]],dtype=torch.float32)
        e=SimpleNamespace(commands=target,continuous_ref_commands=target.clone(),continuous_ref_rate=torch.zeros_like(target),device='cpu')
        e.continuous_ref_commands[-1,1]=.4
        self.assertEqual(eligibility(e).flatten().tolist(),[True,True,False,False,False,False,False])
        e.continuous_ref_rate[0,0]=1
        self.assertFalse(bool(eligibility(e)[0]))

    def test_batch_metadata_survives_augmentation_and_mean_action_call(self):
        x=torch.randn(8,248);x[:,10]=1.2;a=torch.randn(8,31)
        mask=torch.tensor([[True],[False]]*4)
        obs=TensorDict(dict(policy=x,mirror_eligible=mask),batch_size=[8])
        before=obs.clone();o,b=augmentation(self.e,obs,a)
        self.assertEqual(o.batch_size,torch.Size([16]))
        torch.testing.assert_close(o['policy'][8:][~mask.flatten()],x[~mask.flatten()])
        torch.testing.assert_close(o['policy'][8:][mask.flatten()],self.m.observations(x)[mask.flatten()])
        _,means=augmentation(self.e,actions=a)
        torch.testing.assert_close(means,b)
        torch.testing.assert_close(obs['policy'],before['policy'])

    def test_real_rsl_mirror_loss_backpropagates(self):
        from rsl_rl.extensions import Symmetry
        from rsl_rl.models import MLPModel
        obs=TensorDict(dict(policy=torch.randn(8,248),mirror_eligible=torch.ones(8,1,dtype=torch.bool)),batch_size=[8])
        actor=MLPModel(obs,{'actor':['policy']},'actor',31,hidden_dims=[16],activation='elu',obs_normalization=True)
        ext=Symmetry(self.e,augmentation,use_data_augmentation=True,use_mirror_loss=True,mirror_loss_coeff=.05)
        paired,_=augmentation(self.e,obs=obs)
        batch=SimpleNamespace(observations=paired)
        loss=ext.compute_loss(actor,batch,8);(.05*loss).backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(loss.detach()),0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None),0)

    def test_install_preserves_normalizer_until_rollout_update(self):
        from rsl_rl.models import MLPModel
        x=torch.randn(8,248);x[:,10]=1.2
        obs=TensorDict(dict(policy=x,mirror_eligible=torch.ones(8,1,dtype=torch.bool)),batch_size=[8])
        models=[MLPModel(obs,{'actor':['policy']},'actor',31,hidden_dims=[8],activation='elu',obs_normalization=True) for _ in range(2)]
        runner=SimpleNamespace(alg=SimpleNamespace(actor=models[0],critic=models[1]))
        self.e.cfg=dict(train_lateral_mirror=True,train_mirror_augmentation=True)
        before=[{k:v.clone() for k,v in m.state_dict().items()} for m in models]
        install_normalization(runner,self.e)
        for m,s in zip(models,before):
            for key,value in m.state_dict().items():torch.testing.assert_close(value,s[key],rtol=0,atol=0)
            m.train();m.update_normalization(obs)
            self.assertEqual(float(m.obs_normalizer.count),16)

    def test_disabled_configuration_is_no_op(self):
        cfg={'algorithm':{'symmetry_cfg':None}}
        configure(cfg,SimpleNamespace(cfg={}))
        self.assertIsNone(cfg['algorithm']['symmetry_cfg'])

    def test_real_ppo_rollout_storage_and_update_with_metadata(self):
        from rsl_rl.algorithms import PPO
        for augment in [False,True]:
            with self.subTest(augmentation=augment):
                obs=TensorDict(dict(policy=torch.randn(8,248),mirror_eligible=torch.tensor([[True],[False]]*4)),batch_size=[8])
                self.e.num_envs=8;self.e.num_actions=31
                self.e.cfg=dict(train_lateral_mirror=True,train_mirror_augmentation=augment)
                cfg=dict(num_steps_per_env=4,multi_gpu=None,obs_groups={'actor':['policy'],'critic':['policy']},
                    actor=dict(class_name='MLPModel',hidden_dims=[16],activation='elu',obs_normalization=True,
                               distribution_cfg=dict(class_name='GaussianDistribution',init_std=.08,std_type='log')),
                    critic=dict(class_name='MLPModel',hidden_dims=[16],activation='elu',obs_normalization=True),
                    algorithm=dict(class_name='PPO',num_learning_epochs=1,num_mini_batches=2,
                        symmetry_cfg=dict(data_augmentation_func='lateral_symmetry:augmentation',
                            use_data_augmentation=augment,use_mirror_loss=True,mirror_loss_coeff=.05)))
                alg=PPO.construct_algorithm(obs,self.e,cfg,'cpu')
                install_normalization(SimpleNamespace(alg=alg),self.e)
                self.assertEqual(alg.actor.obs_dim,248)
                with torch.no_grad():
                    for step in range(4):
                        alg.act(obs)
                        obs=TensorDict(dict(policy=torch.randn(8,248),mirror_eligible=obs['mirror_eligible'].clone()),batch_size=[8])
                        alg.process_env_step(obs,torch.rand(8),torch.zeros(8,dtype=torch.long),{})
                    alg.compute_returns(obs)
                losses=alg.update()
                self.assertIn('symmetry',losses)
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))


if __name__=='__main__':unittest.main()
