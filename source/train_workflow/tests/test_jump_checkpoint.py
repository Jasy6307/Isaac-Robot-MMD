"""Verify warm-start migration preserves both actor and critic outputs."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch

spec = importlib.util.spec_from_file_location("jump_checkpoint", Path(__file__).parents[1] / "scripts/prepare_jump_checkpoint.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MigrationTests(unittest.TestCase):
    def test_output_equivalence_with_arbitrary_new_observations(self):
        torch.manual_seed(42)
        checkpoint = {"iter": 11998, "optimizer_state_dict": {"state": "old"}, "infos": None}
        for group in ("actor_state_dict", "critic_state_dict"):
            checkpoint[group] = {"mlp.0.weight": torch.randn(12, 153),
                                 "mlp.0.bias": torch.randn(12),
                                 "obs_normalizer._mean": torch.randn(1, 153),
                                 "obs_normalizer._var": torch.ones(1, 153),
                                 "obs_normalizer._std": torch.ones(1, 153) * 2}
        migrated = module.expand_checkpoint(checkpoint)
        x = torch.randn(10, 153)
        expanded = torch.cat((x, torch.randn(10, 8) * 10), dim=-1)
        for group in ("actor_state_dict", "critic_state_dict"):
            old, new = checkpoint[group], migrated[group]
            a = ((x-old["obs_normalizer._mean"])/old["obs_normalizer._std"]) @ old["mlp.0.weight"].T
            b = ((expanded-new["obs_normalizer._mean"])/new["obs_normalizer._std"]) @ new["mlp.0.weight"].T
            torch.testing.assert_close(a, b)
            self.assertEqual(old["mlp.0.weight"].shape[1], 153)
        self.assertNotIn("optimizer_state_dict", migrated)
        self.assertEqual(migrated["iter"], 0)

    def test_wrong_source_layout_rejected(self):
        with self.assertRaises(ValueError):
            module.expand_checkpoint({"actor_state_dict": {"mlp.0.weight": torch.zeros(12, 161)}})

    def test_v2_restart_changes_exploration_without_changing_mean_weights(self):
        weight = torch.randn(12, 161)
        source = {"actor_state_dict": {"mlp.0.weight": weight,
                                       "distribution.std_param": torch.ones(29) * 1.6},
                  "optimizer_state_dict": {}, "iter": 299}
        result = module.restart_specialist(source)
        torch.testing.assert_close(result["actor_state_dict"]["mlp.0.weight"], weight)
        torch.testing.assert_close(result["actor_state_dict"]["distribution.std_param"], torch.ones(29) * .45)
        self.assertNotIn("optimizer_state_dict", result)
        self.assertEqual(source["iter"], 299)


if __name__ == "__main__":
    unittest.main()
