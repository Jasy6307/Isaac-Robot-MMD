"""Check that the residual actor preserves unsupported commands and its frozen base."""

from pathlib import Path
import math
import sys
import unittest

from tensordict import TensorDict
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_layout
from anon_lateral_actor import LateralRecoveryActor
from anon_recovery_actor import RecoveryActor


class RecoveryActorTests(unittest.TestCase):
    def test_shape_gate_and_export(self):
        raw=torch.zeros(5,248)
        raw[:,9:11]=torch.tensor([[0.,1.2],[-1.2,0.],[1.2,0.],[0.,.4],[0.,0.]])
        obs=TensorDict({'policy':raw},batch_size=[len(raw)])
        actor=RecoveryActor(obs,{'actor':['policy']},'actor',31,
                            dict(joint_indices=list(range(15)),threshold=.8,width=.1,gain=1.,freeze_gate=True),
                            hidden_dims=[16],obs_normalization=True)
        with torch.no_grad():
            actor.correction.gate[-1].weight.zero_()
            actor.correction.gate[-1].bias.fill_(12.)
            actor.correction.residual[-1].bias.fill_(.25)
            base=actor.mlp(actor.obs_normalizer(raw))
            corrected=actor(obs)
            torch.testing.assert_close(corrected[:2],base[:2],rtol=0.,atol=0.)
            torch.testing.assert_close(corrected[:,15:],base[:,15:],rtol=0.,atol=0.)
            torch.testing.assert_close(corrected[2:,:15]-base[2:,:15],torch.full((3,15),.25))
            exported=torch.jit.script(actor.as_jit().eval())
            torch.testing.assert_close(exported(raw),corrected)
        trainable=[name for name,tensor in actor.named_parameters() if tensor.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith('correction.residual.') for name in trainable))

    def test_straight_run_calibration_keeps_turning_conservative(self):
        raw = torch.zeros(3, 248)
        raw[:, 9:12] = torch.tensor([[1.1, 0., 0.], [1.1, 0., -.4], [.4, 0., 0.]])
        obs = TensorDict({'policy': raw}, batch_size=[3])
        actor = RecoveryActor(obs, {'actor': ['policy']}, 'actor', 31,
            dict(joint_indices=list(range(15)), threshold=.85, width=.1, gain=1.,
                 low_speed=dict(threshold=.98, width=.018, gain=.82),
                 straight_run=dict(threshold=.9, width=.05, gain=.6)),
            hidden_dims=[16], obs_normalization=True)
        with torch.no_grad():
            actor.correction.gate[-1].weight.zero_()
            actor.correction.gate[-1].bias.fill_(math.log(19.))
            actor.correction.residual[-1].bias.fill_(.25)
            base = actor.mlp(actor.obs_normalizer(raw))
            corrected = actor(obs)
            expected = torch.tensor([.15, 0., .25])[:, None].expand(3, 15)
            torch.testing.assert_close(corrected[:, :15] - base[:, :15], expected)
            torch.testing.assert_close(corrected[:, 15:], base[:, 15:], rtol=0., atol=0.)
            torch.testing.assert_close(torch.jit.script(actor.as_jit().eval())(raw), corrected)

    def test_standing_correction_preserves_moving_actions(self):
        raw = torch.zeros(4, 248)
        raw[:, 9:12] = torch.tensor([[0., 0., 0.], [.4, 0., 0.], [1.2, 0., 0.], [0., 0., .4]])
        obs = TensorDict({'policy': raw}, batch_size=[len(raw)])
        actor = RecoveryActor(obs, {'actor': ['policy']}, 'actor', 31,
            dict(joint_indices=list(range(15)), threshold=.8, width=.1, gain=1.,
                 freeze_gate=True, standing=True), hidden_dims=[16], obs_normalization=True)
        with torch.no_grad():
            correction = actor.correction
            correction.gate[-1].weight.zero_()
            correction.gate[-1].bias.fill_(12.)
            self.assertIsNotNone(correction.protected_residual)
            baseline = actor(obs)
            correction.residual[-1].bias.fill_(.20)
            result = actor(obs)
            torch.testing.assert_close(result[1:], baseline[1:], rtol=0., atol=0.)
            torch.testing.assert_close(result[0, :15] - baseline[0, :15], torch.full((15,), .20))
            torch.testing.assert_close(torch.jit.script(actor.as_jit().eval())(raw), result)
        self.assertTrue(all(name.startswith('correction.residual.') for name, param in actor.named_parameters()
                            if param.requires_grad))

    def test_lateral_standing_branch_preserves_other_commands_and_exports(self):
        raw = torch.zeros(7, 248)
        raw[:6, :2] = torch.tensor([[.90, 0.], [-.90, 0.], [0., -1.40],
                                    [.9, .8], [.4, 0.], [.9, 0.]])
        raw[5, 9] = .4
        obs = TensorDict({'policy': raw}, batch_size=[len(raw)])
        actor = LateralRecoveryActor(obs, {'actor': ['policy']}, 'actor', 31,
            dict(joint_indices=list(range(15)), threshold=.8, width=.1, gain=1.,
                 freeze_gate=True, standing=True, lateral_standing={}), hidden_dims=[16], obs_normalization=True)
        with torch.no_grad():
            baseline = actor(obs)
            actor.correction.residual[-1].bias.fill_(.25)
            result = actor(obs)
            torch.testing.assert_close(result[2:], baseline[2:], rtol=0., atol=0.)
            torch.testing.assert_close(result[:2, :15] - baseline[:2, :15], torch.full((2, 15), .25))
            torch.testing.assert_close(result[:, 15:], baseline[:, 15:], rtol=0., atol=0.)
            torch.testing.assert_close(torch.jit.script(actor.as_jit().eval())(raw), result)
        self.assertTrue(all(name.startswith('correction.residual.') for name, param in actor.named_parameters()
                            if param.requires_grad))

    def test_lateral_standing_guard_protects_a_still_moving_stop_reference(self):
        raw = torch.zeros(4, 248)
        raw[:, 0] = .9
        raw[1, 105] = .17
        raw[2, 206] = -.17
        raw[3, 136] = .05
        obs = TensorDict({'policy': raw}, batch_size=[len(raw)])
        guard = dict(lower_pose=[-.01] * 31, upper_pose=[.01] * 31,
                     lower_rate=[-.002] * 31, upper_rate=[.002] * 31)
        parameters = dict(joint_indices=list(range(15)), threshold=.8, width=.1, gain=1.,
                          freeze_gate=True, standing=True)
        ungated = LateralRecoveryActor(obs, {'actor': ['policy']}, 'actor', 31,
                     {**parameters, 'lateral_standing': {}}, hidden_dims=[16], obs_normalization=True)
        actor = LateralRecoveryActor(obs, {'actor': ['policy']}, 'actor', 31,
                     {**parameters, 'lateral_standing': {'idle_reference': guard}}, hidden_dims=[16], obs_normalization=True)
        actor.load_state_dict(ungated.state_dict())
        with torch.no_grad():
            baseline = actor(obs)
            actor.correction.residual[-1].bias.fill_(.25)
            guarded = actor(obs)
            torch.testing.assert_close(guarded[1:], baseline[1:], rtol=0., atol=0.)
            torch.testing.assert_close(guarded[0, :15] - baseline[0, :15], torch.full((15,), .25))
            torch.testing.assert_close(torch.jit.script(actor.as_jit().eval())(raw), guarded)
