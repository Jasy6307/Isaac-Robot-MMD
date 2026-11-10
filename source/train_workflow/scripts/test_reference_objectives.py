"""Behavior checks for unified rewards: ambiguity, motion shifts and saturation."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.reference_objectives import (
    tracking_score, contact_evidence, contact_score, exploration_strength)
from source.train_workflow.utils.auto_jump_training import AutoJumpController
from types import SimpleNamespace


class ReferenceObjectiveTests(unittest.TestCase):
    def test_large_tracking_error_still_distinguishes_progress(self):
        error = torch.tensor([0., 1., 25., 100.], requires_grad=True)
        score = tracking_score(error)
        self.assertEqual(score[0], 1.)
        self.assertTrue(bool((score[:-1] > score[1:]).all()))
        score.sum().backward()
        self.assertTrue(bool((error.grad < 0.).all()))

    def test_contacts_reward_both_correct_support_and_correct_release(self):
        contacts = torch.tensor([[1, 0], [1, 1], [0, 0]], dtype=torch.bool)
        support = torch.tensor([[1., 0.]]).expand(3, -1)
        air = torch.tensor([[0., 1.]]).expand(3, -1)
        torch.testing.assert_close(contact_score(contacts, support, air), torch.tensor([1., .5, .5]))
        self.assertEqual(contact_score(contacts, support*0, air*0).sum(), 0.)

    def test_contact_targets_do_not_require_single_foot_or_jump_labels(self):
        feet = np.zeros((120, 2, 3))
        feet[30:60, 1, 2] = .2
        feet[70:90, :, 2] = .2
        support, air = contact_evidence(feet, 30., np.ones(120, bool))
        np.testing.assert_allclose(support[10], [1., 1.])
        np.testing.assert_allclose(support[40], [1., 0.])
        np.testing.assert_allclose(air[40], [0., 1.])
        np.testing.assert_allclose(air[80], [1., 1.])
        self.assertFalse((support*air > 0).any())
        # Translating the reference or adding leading standing frames does
        # not change the intent of this isolated motion segment.
        shifted = feet + np.array([10., -20., 2.])
        s2, a2 = contact_evidence(shifted, 30., np.ones(120, bool))
        np.testing.assert_allclose(support, s2, atol=1.e-12)
        np.testing.assert_allclose(air, a2, atol=1.e-12)
        extended = np.concatenate((np.zeros((60, 2, 3)), feet))
        s3, a3 = contact_evidence(extended, 30., np.ones(180, bool))
        np.testing.assert_allclose(support[5:], s3[65:], atol=1.e-12)
        np.testing.assert_allclose(air[5:], a3[65:], atol=1.e-12)

    def test_ambiguous_and_invalid_contacts_do_not_force_a_label(self):
        feet = np.zeros((90, 2, 3)); feet[:, 1, 2] = .035
        support, air = contact_evidence(feet, 30., np.ones(90, bool))
        np.testing.assert_allclose(support[:, 1], 0., atol=1.e-12)
        np.testing.assert_allclose(air[:, 1], 0., atol=1.e-12)
        support, air = contact_evidence(feet, 30., np.zeros(90, bool))
        self.assertFalse(support.any()); self.assertFalse(air.any())

    def test_jump_aid_retires_instead_of_growing_to_dominate(self):
        self.assertEqual([exploration_strength(x) for x in [0., .25, .5, 1.]], [1., .5, 0., 0.])

    def test_old_sampler_state_cannot_silently_resume_new_reward_contract(self):
        report = dict(duration_s=3., source_sha256="test", events=[], reward_contract={"version": "new"})
        controller = AutoJumpController(SimpleNamespace(device="cpu",step_dt=1/30.), report,
            Path(__file__).resolve().parents[3]/"output/auto-jump/tests")
        state = controller.state_dict(); state.pop("reward_contract")
        with self.assertRaisesRegex(ValueError, "Reward contract changed"):
            controller.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
