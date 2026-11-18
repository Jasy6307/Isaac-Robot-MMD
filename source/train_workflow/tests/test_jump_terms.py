"""CPU checks for jump phase and takeoff gating, without launching Isaac."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[3]
tree = ast.parse((ROOT / "source/my_task/mdp/jump.py").read_text(encoding="utf-8"))
names = {"phases", "upward_velocity", "flight_state", "airborne_close", "phase_joint_tracking"}
names.add("sustained_flight")
names.add("full_step_ground_contacts")
nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
         or (isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "PHASES" for x in n.targets))]
ns = dict(torch=torch, motion_steps=lambda env: env.steps,
          ground_forces=lambda env: env.forces,
          joint_pos_tracking_exp=lambda env, *args: torch.ones(env.num_envs))
ns["sim_compat"] = NS(as_torch=lambda value: value)
exec(compile(ast.Module(body=nodes, type_ignores=[]), "jump_terms_cpu", "exec"), ns)


class TermTests(unittest.TestCase):
    def env(self):
        return NS(num_envs=1, device="cpu", step_dt=1/30., common_step_counter=0,
                  steps=torch.tensor([505]), forces=torch.tensor([[100., 100.]]),
                  episode_length_buf=torch.tensor([10]),
                  scene={"robot": NS(data=NS(root_lin_vel_w=NS(torch=torch.tensor([[0.,0.,.5]]))))})

    def test_grounded_leg_closing_does_not_get_flight_reward(self):
        env = self.env()
        env.steps[:] = 515
        self.assertEqual(ns["airborne_close"](env).item(), 0.)

    def test_flight_requires_prior_stance_and_sustained_air(self):
        env = self.env()
        self.assertEqual(ns["airborne_close"](env).item(), 0.)
        env.common_step_counter += 1
        env.steps[:] = 514
        env.forces.zero_()
        self.assertEqual(ns["airborne_close"](env).item(), 0.)
        # Repeated queries within one control step cannot manufacture airtime.
        self.assertEqual(ns["airborne_close"](env).item(), 0.)
        env.common_step_counter += 1
        env.steps[:] = 515
        self.assertEqual(ns["airborne_close"](env).item(), 1.)
        env.common_step_counter += 1
        env.episode_length_buf.zero_()
        self.assertEqual(ns["airborne_close"](env).item(), 0.)

    def test_joint_tracking_relaxed_only_during_push(self):
        env = self.env()
        env.steps[:] = 509
        self.assertAlmostEqual(ns["phase_joint_tracking"](env, "unused").item(), .3)
        env.steps[:] = 515
        self.assertAlmostEqual(ns["phase_joint_tracking"](env, "unused").item(), 1.)

    def test_v2_rewards_duration_not_a_single_instant(self):
        env = self.env()
        ns["sustained_flight"](env)
        values = []
        env.forces.zero_()
        for i in range(4):
            env.common_step_counter += 1
            env.steps[:] = 514 + i
            values.append(ns["sustained_flight"](env).item())
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[-1], 1.)

    def test_substep_contact_cannot_be_counted_as_continuous_flight(self):
        history = torch.zeros(1, 4, 1, 1, 3)
        history[0, 3, 0, 0, 2] = 100.
        env = NS(cfg=NS(decimation=4), scene={
            "jump_left_ground": NS(data=NS(normal_force_matrix_w_history=history)),
            "jump_right_ground": NS(data=NS(normal_force_matrix_w_history=torch.zeros_like(history)))})
        self.assertEqual(ns["full_step_ground_contacts"](env).tolist(), [[True, False]])

    def test_three_events_and_ordinary_motion(self):
        env = self.env()
        env.cfg = NS(jump_phase_schedules=(ns["PHASES"],
                    ((20.48,20.60),(20.60,20.75),(20.75,20.95),(20.95,21.15)),
                    ((47.88,48.02),(48.02,48.18),(48.18,48.38),(48.38,48.58))))
        for frame in (515,624,1447):
            env.steps[:] = frame
            self.assertTrue(ns["phases"](env)[0,2])
        env.steps[:] = 1590
        self.assertFalse(ns["phases"](env).any())


if __name__ == "__main__":
    unittest.main()
