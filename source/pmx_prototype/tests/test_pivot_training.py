"""Sampling and reward acceptance checks; never creates a training runner."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import source_layout
from unified_walk_run_reference import sample_commands
from pivot_stability import pivot_stability_terms

CONFIG=SOURCE/'configs/training/unified_pivot_recovery.json'


class PivotTrainingTests(unittest.TestCase):
    def make_env(self,n=12288):
        cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
        return SimpleNamespace(num_envs=n,device='cpu',cfg=cfg,evaluation=False,iteration=7500,
            commands=torch.zeros(n,3),continuous_ref_commands=torch.zeros(n,3),command_age=torch.ones(n))

    def test_reserved_pools_survive_resampling_and_force_balanced_reversals(self):
        torch.manual_seed(42)
        for n in (8192,12288):
            env=self.make_env(n);ids=torch.arange(n);bucket=ids%100
            pure=(bucket>=70)&(bucket<80);reverse=(bucket>=80)&(bucket<90);turn=pure|reverse
            env.commands[:,0]=1.2 # Exercise the stop-overwrite branch.
            for reset in (True,False,False,False):
                before=env.commands.clone();sample_commands(env,ids,on_reset=reset)
                self.assertTrue((env.commands[turn,:2]==0).all())
                self.assertTrue((env.commands[turn,2].abs()>=.15).all())
                self.assertTrue((env.commands[turn,2].abs()<=.6).all())
                self.assertAlmostEqual(float((env.commands[pure,2]>0).float().mean()),.5,delta=.01)
                if not reset:self.assertTrue(torch.equal(env.commands[reverse,2],-before[reverse,2]))
                self.assertTrue(((env.unified_command_hold[turn]>=6)&(env.unified_command_hold[turn]<=8)).all())
            # Partial environment resets must not change any other commands.
            before=env.commands.clone();sample_commands(env,ids[reverse][::3],on_reset=True)
            untouched=torch.ones(n,dtype=torch.bool);untouched[ids[reverse][::3]]=False
            self.assertTrue(torch.equal(before[untouched],env.commands[untouched]))

    def test_boundary_sampling_and_evaluation_scope(self):
        torch.manual_seed(10);env=self.make_env();ids=torch.arange(env.num_envs);pure=(ids%100>=70)&(ids%100<80)
        samples=[]
        for _ in range(100):
            sample_commands(env,ids,on_reset=True);samples.append(env.commands[pure,2].abs())
        fraction=float((torch.cat(samples)==.6).float().mean())
        self.assertAlmostEqual(fraction,.5,delta=.01)
        env.evaluation=True;sample_commands(env,ids,on_reset=True)
        self.assertTrue((env.commands==0).all())
        env.evaluation=False;env.cfg['unified_reversal_env_fraction']=.3
        with self.assertRaises(ValueError):sample_commands(env,ids)

    def test_support_landing_and_drift_costs_are_gated_and_monotone(self):
        cfg=self.make_env(1).cfg
        commands=torch.tensor([[0.,0.,-.6]]*6)
        commands[4]=torch.tensor([.3,0.,-.6]);commands[5]=0
        ref=commands.clone()
        bottom=torch.tensor([[0.,.1],[0.,.1],[.1,.1],[0.,0.],[.1,.1],[.1,.1]])
        contact=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[1.,1.],[1.,1.],[1.,1.]])
        velocity=torch.zeros(6,6);velocity[3,1]=.3
        original=[x.clone() for x in (commands,ref,bottom,contact,velocity)]
        terms=pivot_stability_terms(commands,ref,bottom,contact,velocity,cfg)
        self.assertEqual(float(terms['reward_rate'][0]),0.) # Valid support + intended swing.
        self.assertGreater(float(terms['landing'][1]),0.) # Missing scheduled landing.
        self.assertEqual(float(terms['support'][1]),0.) # Another foot still supports.
        self.assertGreater(float(terms['support'][2]),0.) # No grounded foot.
        self.assertGreater(float(terms['drift'][3]),0.)
        self.assertTrue((terms['reward_rate'][4:]==0).all()) # Walking/standing unaffected.
        expected=-sum(cfg['pivot_'+key+'_weight']*terms[key] for key in ['support','landing','drift'])
        self.assertTrue(torch.allclose(terms['reward_rate'],expected))
        changed=dict(cfg,pivot_landing_weight=cfg['pivot_landing_weight']*2)
        doubled=pivot_stability_terms(commands,ref,bottom,contact,velocity,changed)
        self.assertLess(float(doubled['reward_rate'][1]),float(terms['reward_rate'][1]))
        rotated=velocity.clone();rotated[:,:2]=torch.stack([-velocity[:,1],velocity[:,0]],-1)
        mirrored=commands.clone();mirrored[:,2]*=-1
        other=pivot_stability_terms(mirrored,mirrored,bottom,contact,rotated,cfg)
        self.assertTrue(torch.equal(terms['reward_rate'],other['reward_rate']))
        for x,y in zip((commands,ref,bottom,contact,velocity),original):self.assertTrue(torch.equal(x,y))
        zero={**cfg,**{'pivot_'+key+'_weight':0 for key in ['support','landing','drift']}}
        self.assertTrue((pivot_stability_terms(commands,ref,bottom,contact,velocity,zero)['reward_rate']==0).all())

    def test_drift_deadzone_and_turn_entry_do_not_penalize_normal_sway(self):
        cfg=self.make_env(1).cfg
        command=torch.tensor([[0.,0.,-.6]]*3);ref=command.clone();ref[2,0]=.3
        bottom=torch.zeros(3,2);contact=torch.ones(3,2);velocity=torch.zeros(3,6)
        velocity[:,0]=torch.tensor([.08,.3,.3])
        terms=pivot_stability_terms(command,ref,bottom,contact,velocity,cfg)
        self.assertEqual(float(terms['drift'][0]),0.)
        self.assertGreater(float(terms['drift'][1]),0.)
        self.assertFalse(bool(terms['active'][2])) # Still decelerating from walking.
        self.assertTrue(torch.isfinite(terms['reward_rate']).all())


if __name__=='__main__':
    torch.set_num_threads(8)
    unittest.main()
