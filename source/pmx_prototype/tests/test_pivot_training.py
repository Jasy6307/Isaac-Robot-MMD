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
PIVOT_STOP_CONFIG=SOURCE/'configs/training/unified/pivot_stop_sampling_resume.json'
LOW_SPEED_CONFIG=Path(__file__).parent/'fixtures/low_speed_sampling.json'


class PivotTrainingTests(unittest.TestCase):
    def make_env(self,n=12288,config=CONFIG):
        cfg=json.loads(config.read_text(encoding='utf-8'))
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

    def test_pivot_stop_pool_alternates_exact_limits_and_preserves_other_pools(self):
        baseline = self.make_env(8192, LOW_SPEED_CONFIG)
        env = self.make_env(8192, LOW_SPEED_CONFIG)
        env.cfg.update(json.loads(PIVOT_STOP_CONFIG.read_text(encoding='utf-8')))
        ids = torch.arange(env.num_envs, dtype=torch.int32)
        bucket = ids.remainder(100)
        pool = (bucket >= 90) & (bucket < 96)
        signs = torch.where(bucket[pool].remainder(2) == 0, 1., -1.)
        self.assertEqual(int((signs == 1).sum()), 244)
        self.assertEqual(int((signs == -1).sum()), 244)

        for on_reset, yaw, hold in ((True, .6, 6.), (False, 0., 8.), (False, .6, 6.), (False, 0., 8.)):
            previous_plan = env.continuous_ref_commands.clone()
            torch.manual_seed(42)
            sample_commands(baseline, ids, on_reset=on_reset)
            torch.manual_seed(42)
            sample_commands(env, ids, on_reset=on_reset)
            torch.testing.assert_close(env.commands[pool, 2], yaw * signs, rtol=0., atol=0.)
            self.assertEqual(int(env.commands[pool, :2].count_nonzero()), 0)
            durations = env.unified_command_hold[pool]
            self.assertTrue(((durations >= hold) & (durations < hold + 2.)).all())
            self.assertTrue(torch.equal(env.commands[~pool], baseline.commands[~pool]))
            self.assertTrue(torch.equal(env.unified_command_hold[~pool], baseline.unified_command_hold[~pool]))
            if not on_reset:
                self.assertTrue(torch.equal(env.continuous_ref_commands, previous_plan))

        reset_ids = ids[pool][::7]
        original = env.commands.clone()
        sample_commands(env, reset_ids, on_reset=True)
        self.assertTrue((env.commands[reset_ids, 2].abs() == .6).all())
        original[reset_ids] = env.commands[reset_ids]
        self.assertTrue(torch.equal(env.commands, original))
        env.evaluation = True
        sample_commands(env, ids, on_reset=True)
        self.assertEqual(int(env.commands.count_nonzero()), 0)

    def test_pivot_stop_pool_rejects_overlapping_or_unpaired_reservations(self):
        env = self.make_env(8192, LOW_SPEED_CONFIG)
        ids = torch.arange(env.num_envs)
        for fraction in (.12, .05):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                env.cfg['unified_pivot_stop_env_fraction'] = fraction
                sample_commands(env, ids, on_reset=True)

    def test_initial_standing_is_reset_only_and_uses_the_general_pool(self):
        baseline = self.make_env(8192, LOW_SPEED_CONFIG)
        baseline.cfg.update(json.loads(PIVOT_STOP_CONFIG.read_text(encoding='utf-8')))
        env = self.make_env(8192, LOW_SPEED_CONFIG)
        env.cfg.update(baseline.cfg, unified_stand_reset_env_fraction=.01)
        ids = torch.arange(env.num_envs, dtype=torch.int32)
        stand = ids.remainder(100) == 96
        self.assertEqual(int(stand.sum()), 81)
        torch.manual_seed(42)
        sample_commands(baseline, ids, on_reset=True)
        torch.manual_seed(42)
        sample_commands(env, ids, on_reset=True)
        self.assertTrue((env.commands[stand] == 0.).all())
        self.assertTrue((env.continuous_ref_commands[stand] == 0.).all())
        self.assertTrue((env.unified_command_hold[stand] == env.cfg['unified_idle_hold_seconds']).all())
        self.assertTrue(torch.equal(env.commands[~stand], baseline.commands[~stand]))
        self.assertTrue(torch.equal(env.unified_command_hold[~stand], baseline.unified_command_hold[~stand]))
        baseline.commands.copy_(env.commands)
        baseline.unified_command_hold.copy_(env.unified_command_hold)
        torch.manual_seed(43)
        sample_commands(baseline, ids)
        torch.manual_seed(43)
        sample_commands(env, ids)
        self.assertTrue(torch.equal(env.commands, baseline.commands))
        self.assertTrue(torch.equal(env.unified_command_hold, baseline.unified_command_hold))
        self.assertTrue((env.commands[stand] != 0.).any())

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
