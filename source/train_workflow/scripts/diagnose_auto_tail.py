"""Read-only policy rollouts from fixed starts to diagnose tail starvation."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
from source import sim_compat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", required=True, type=Path)
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--starts", type=float, nargs="+", default=[0.,8.,10.,12.,12.5])
parser.add_argument("--modes", type=str, nargs="+", default=["normal", "base_tail", "jump_tail", "reference"],
                    choices=["normal", "base", "jump", "base_tail", "jump_tail", "reference"],
                    help="Policy ablations; *_tail overrides the gate only after 13.8s.")
parser.add_argument("--full", action="store_true", help="Run the complete 17.5s rollout from time zero.")
parser.add_argument("--disable_fall_termination", action="store_true",
                    help="Diagnostic only: keep time_out but disable fall/orientation termination.")
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
sim_compat.add_legacy_headless_arg(parser)
args = parser.parse_args()
sim_compat.prepare_launcher_args(args, default_visualizer="none")
app = AppLauncher(args).app

import gymnasium as gym
import source.my_task
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from source.my_task.motion_reference.motion_buffer import register_packaged_reference
from source.train_workflow.utils.auto_jump_training import AutoJumpController
from source.train_workflow.utils.deployment import load_bundle


def main():
    print("TAIL_ABLATION_MODES", args.modes, flush=True)
    bundle = load_bundle(args.bundle)
    cfg = bundle["env_cfg"]
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.auto_motion_feedback = False
    cfg.seed = 42
    if args.disable_fall_termination:
        # Keep the physical dynamics and rewards unchanged; only prevent an
        # early reset so the trace can reveal where the tail diverges.
        cfg.terminations.fall_height = None
        cfg.terminations.bad_orientation = None
    agent = bundle["agent_cfg"]
    agent.device = args.device
    register_packaged_reference(cfg.actions.joint_pos.motion_h5_path,bundle["reference"])
    args.output.mkdir(parents=True,exist_ok=True)
    wrapped = RslRlVecEnvWrapper(gym.make(bundle["task"],cfg=cfg),clip_actions=agent.clip_actions)
    try:
        controller = AutoJumpController(wrapped.unwrapped,cfg.auto_jump_report,args.output)
        controller.configure(1.)
        wrapped.unwrapped._auto_jump_controller = controller
        runner = OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=args.device)
        runner.load(str(args.checkpoint.resolve()))
        records=[]
        original_get_policy = runner.get_inference_policy

        def _rewrite_gate(observation, gate, mask=None):
            """Rewrite only the final jump gate, preserving all other observations."""
            if torch.is_tensor(observation):
                result = observation.clone()
                if mask is None:
                    result[:, -1] = float(gate)
                else:
                    result[mask, -1] = float(gate)
                return result
            # This path is for non-concatenated observation groups.  The current
            # task uses a flat tensor, but keep the diagnostic explicit if that changes.
            if hasattr(observation, "clone") and hasattr(observation, "keys"):
                result = observation.clone()
                # RslRlVecEnvWrapper currently returns a TensorDict with a
                # flat ``policy`` leaf of shape [N, 163].
                for key in ("policy", "obs"):
                    try:
                        value = result[key]
                    except (KeyError, TypeError):
                        continue
                    if torch.is_tensor(value) and value.shape[-1] >= 163:
                        value = value.clone()
                        if mask is None:
                            value[..., -1] = float(gate)
                        else:
                            value[mask, -1] = float(gate)
                        result[key] = value
                        return result
                for key in result.keys():
                    value = result[key]
                    if torch.is_tensor(value) and value.shape[-1] == 1 and key in ("jump_state", "jump_gate"):
                        result[key] = value.new_full(value.shape, float(gate))
                        return result
            raise TypeError(f"Unsupported observation type for gate ablation: {type(observation)!r}")

        for mode in args.modes:
            print("TAIL_ABLATION_MODE_START", mode, flush=True)
            gate = {"normal": None, "base": 0., "jump": 1.,
                    "base_tail": 0., "jump_tail": 1., "reference": None}[mode]
            wrapped.unwrapped._g1_residual_alpha = 0.0 if mode == "reference" else 1.0
            tail_only = mode.endswith("_tail")

            def _get_policy(*policy_args, _gate=gate, **policy_kwargs):
                policy = original_get_policy(*policy_args, **policy_kwargs)
                if _gate is None:
                    return policy
                def _policy(observation):
                    mask = None
                    if tail_only:
                        from source.my_task.motion_reference import motion_steps
                        seconds = motion_steps(wrapped.unwrapped).float() * wrapped.unwrapped.step_dt
                        mask = seconds >= 13.8
                    return policy(_rewrite_gate(observation, _gate, mask))
                return _policy

            runner.get_inference_policy = _get_policy
            if args.full:
                # Match the training validator: one parallel call with the
                # three deterministic seed streams, rather than three
                # separate single-replica resets.
                batch_records = controller.validate(runner, wrapped, full=True,
                    seeds=args.seeds, trace_dir=args.output / mode)
                for seed, record in zip(args.seeds, batch_records):
                    records.append(dict(mode=mode, seed=seed, record=record))
                    print("TAIL_ABLATION", json.dumps(dict(mode=mode, seed=seed,
                        results=record["results"])), flush=True)
            else:
                for seed in args.seeds:
                    try:
                        record=controller.validate(runner,wrapped,seed=seed,
                            starts_override=args.starts,
                            trace_dir=args.output/f"{mode}_seed_{seed}")
                    except BaseException as error:
                        print("TAIL_ABLATION_ERROR", mode, seed, repr(error), flush=True)
                        raise
                    records.append(dict(mode=mode,seed=seed,record=record))
                    print("TAIL_ABLATION",json.dumps(dict(mode=mode,seed=seed,results=record["results"])),flush=True)
            (args.output/"results.json").write_text(json.dumps(dict(checkpoint=str(args.checkpoint.resolve()),
                protocol="parallel full 17.5s rollouts; no learning; gate ablations; same seed streams as training validator" if args.full else
                    "fixed-start 3s rollouts; no learning; gate ablations; reference starts clamped to duration-3s by validation",
                records=records),indent=2),encoding="utf-8")
    finally:
        wrapped.close()

try:
    main()
finally:
    app.close()
