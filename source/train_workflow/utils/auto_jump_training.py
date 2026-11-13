"""One runner, automatic curriculum, bounded sampling and independent probes.

Probe failures are local execution evidence, not ground-truth dance labels.
The sampler always retains uniform coverage, including non-jump motion.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch


def prepare_auto_config(cfg, h5_path, window_seconds, review_path=None, *, hold_seconds=2.0,
                        tail_only=False, tail_start_seconds=None):
    if not math.isfinite(hold_seconds) or hold_seconds < 0:
        raise ValueError("End hold must be finite and nonnegative")
    import h5py
    from source.train_workflow.utils.motion.auto_jump import analyze_h5, event_schedules
    with h5py.File(h5_path) as f:
        fps = float(f.attrs["fps"])
        duration = len(f["joint_pos_delta"]) / fps
    if window_seconds > duration + 1e-5:
        raise ValueError("Auto training window exceeds source duration; refusing held-tail analysis")
    report = analyze_h5(h5_path, cfg.scene.robot.spawn.usd_path,
                        frames=max(3, int(math.ceil(window_seconds * fps))))
    cfg.auto_jump_report = report
    if getattr(cfg, "reference_reward_version", None):
        report["reward_contract"] = dict(version=cfg.reference_reward_version,
            scales=cfg.reference_reward_scales,
            weights={name: term.weight for name,term in vars(cfg.rewards).items()
                     if term is not None and hasattr(term, "weight")})
    report["duration_s"] = float(window_seconds)
    report["end_hold_seconds"] = float(hold_seconds)
    report["episode_duration_s"] = float(window_seconds + hold_seconds)
    report["control_hz"] = 1. / (cfg.sim.dt * cfg.decimation)
    report["jump_gate"] = dict(
        pre_seconds=float(getattr(cfg, "jump_gate_pre_seconds", .6)),
        post_seconds=float(getattr(cfg, "jump_gate_post_seconds", .6)),
        ramp_seconds=float(getattr(cfg, "jump_gate_ramp_seconds", .3)),
    )
    report["tail_only"] = bool(tail_only)
    if tail_only:
        if review_path:
            raise ValueError("Tail-only training does not combine with a support review")
        flight_events = [e for e in report["events"] if e.get("flight_reward")]
        if not flight_events:
            raise ValueError("Tail-only training requires a high-confidence flight event")
        last_event = max(flight_events, key=lambda e: float(e["phases"][-1][1]))
        derived = max(0.0, float(last_event["phases"][0][0]) - 1.0)
        start = derived if tail_start_seconds is None else float(tail_start_seconds)
        if not math.isfinite(start) or start < 0.0 or start >= window_seconds - 1e-5:
            raise ValueError(f"Tail-only start must be in [0, {window_seconds}), got {start}")
        report["tail_start_seconds"] = start
        report["tail_event_id"] = int(last_event["id"])
        report["tail_duration_s"] = float(window_seconds + hold_seconds - start)
        print(f"[AUTO TAIL] 只训练末段：{start:.3f}s -> {window_seconds + hold_seconds:.3f}s；"
              f"前段 0–{start:.3f}s 不采样；目标事件 id={last_event['id']}")
    if review_path:
        from source.train_workflow.utils.motion.auto_jump import apply_review
        apply_review(report, json.loads(Path(review_path).read_text(encoding="utf-8")))
    cfg.jump_phase_schedules = event_schedules(report)
    cfg.episode_length_s = window_seconds + hold_seconds
    cfg.events.reset_robot_joints.params.update(segment_seconds=cfg.episode_length_s,
        random_start=False, motion_start_seconds=None)
    print(f"[AUTO JUMP] {len(report['events'])} hypotheses; "
          f"{len(cfg.jump_phase_schedules)} high-confidence reference flight requests; "
          f"gate={report['jump_gate']}; no reviewed labels loaded")
    return report


def curriculum_stage(progress):
    if not 0 <= progress <= 1:
        raise ValueError("Progress must be in [0,1]")
    return "short" if progress < .55 else "long" if progress < .85 else "full"


def first_probe_done(total, interval, *, start=0, start_fraction=0.2):
    """First in-invocation completion count that keeps the interval, 2*interval, ... schedule.

    Probes before start_fraction of the cumulative budget are omitted. A probe that
    lands exactly on that fraction still runs. The returned count may pass `total`;
    the training loop then keeps the final validation only.
    """
    if interval <= 0:
        return None
    if total < 1 or start < 0 or not 0 <= start_fraction <= 1:
        raise ValueError("Invalid probe schedule")
    basis = int(round(float(start_fraction) * 1000))
    min_cumulative = ((start + total) * basis + 999) // 1000
    min_done = max(0, min_cumulative - start)
    steps = 1 if min_done == 0 else math.ceil(min_done / interval)
    return steps * interval


def install_checkpoint_metadata(runner, controller, *, start, total, initial_progress, on_save=None):
    original_save = runner.save

    def save_with_curriculum(path, infos=None):
        # Intermediate PPO saves must be resumable even if a chunk is interrupted.
        metadata = dict(infos or {})
        if "auto_curriculum" not in metadata:
            snapshot = controller.state_dict()
            learned = max(0, int(runner.current_learning_iteration) - start + 1)
            snapshot["progress"] = min(1., initial_progress + (1 - initial_progress) * learned / total)
            metadata["auto_curriculum"] = snapshot
        original_save(path, infos=metadata)
        if on_save is not None:on_save(path)

    runner.save = save_with_curriculum


class AutoJumpController:
    def __init__(self, env, report, log_dir):
        self.env, self.report, self.log_dir = env, report, Path(log_dir)
        self.duration = float(report["duration_s"])
        self.bins = max(1, int(math.ceil(self.duration)))
        self.device = env.device
        self.difficulty = torch.ones(self.bins, device=self.device)
        self.prior = torch.zeros_like(self.difficulty)
        for event in report["events"]:
            lo, hi = event["phases"][0][0], event["phases"][-1][1]
            self.prior[max(0, int(lo - 1)):min(self.bins, int(hi) + 1)] += (
                1. if event["flight_reward"] else .3)
        self.prior.clamp_(max=3.)
        self.sums = torch.zeros_like(self.difficulty)
        self.counts = torch.zeros_like(self.difficulty)
        self.failures = torch.zeros_like(self.difficulty)
        self.visits = torch.zeros_like(self.difficulty)
        self.stage, self.progress, self.probe = "short", 0., None
        self.last_step = -1
        self.history = []
        self.quality_profile = None
        self.selection = None
        self.record_validation=False
        self.validation_scene=None
        self.render_validation=False
        self.validation_source="actual_training_validation"
        self.motion_feedback=None
        self.motion_analyzer=None
        if getattr(getattr(env,"cfg",None),"auto_motion_feedback",False):
            from source.train_workflow.utils.motion_feedback import MotionFeedback
            self.motion_feedback=MotionFeedback(self.bins)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def configure(self, progress):
        self.progress = float(progress)
        self.stage = "full" if self.report.get("tail_only") else curriculum_stage(progress)
        # Flight objectives present from the beginning, gradually strengthened.
        self.env._auto_jump_strength = 1.0 if self.report.get("tail_only") else .1 + .9 * min(1., progress / .5)

    def probabilities(self):
        uniform = torch.ones_like(self.difficulty) / self.bins
        prior = self.prior / self.prior.sum().clamp_min(1e-8)
        if not bool(self.prior.sum() > 0):
            prior = uniform
        difficulty = self.difficulty.clamp(.2, 4.)
        difficulty = difficulty / difficulty.sum()
        if self.motion_feedback is not None and bool(self.motion_feedback.priority.sum()>0):
            movement=torch.as_tensor(self.motion_feedback.priority,device=self.device,dtype=difficulty.dtype)
            movement=movement/movement.sum()
            return .4*uniform+.3*prior+.2*difficulty+.1*movement
        return .4 * uniform + .3 * prior + .3 * difficulty

    def reset(self, ids, reference_steps):
        env = self.env
        if self.probe is not None:
            probe = self.probe
            ended = probe["active"][ids] & (env.episode_length_buf[ids] > 0)
            ended_ids = ids[ended]
            if len(ended_ids):
                probe["completed"][ended_ids] = env.termination_manager.time_outs[ended_ids] & ~env.termination_manager.terminated[ended_ids]
                probe["steps"][ended_ids] = env.episode_length_buf[ended_ids]
                bins = (reference_steps[ended].float() * env.step_dt).long().clamp(0, self.bins - 1)
                probe["failures"].scatter_add_(0, bins, (~probe["completed"][ended_ids]).float())
                probe["active"][ended_ids] = False
            return probe["starts"][ids], probe["targets"][ids]
        if self.report.get("tail_only"):
            start = int(round(float(self.report["tail_start_seconds"]) / env.step_dt))
            target = max(1, int(round(float(self.report["tail_duration_s"]) / env.step_dt)))
            return (torch.full((len(ids),), start, device=self.device, dtype=torch.long),
                    torch.full((len(ids),), target, device=self.device, dtype=torch.long))
        # Terminal states are still present before the reset event writes poses.
        failed = env.termination_manager.terminated[ids] & (env.episode_length_buf[ids] > 0)
        seconds = reference_steps.float() * env.step_dt
        bins = seconds.long().clamp(0, self.bins - 1)
        self.failures.scatter_add_(0, bins, failed.float())
        count = len(ids)
        chosen = torch.multinomial(self.probabilities(), count, replacement=True)
        # Bias start before the difficult bin; include setup, not just the apex.
        start = (chosen.float() - .8 + torch.rand(count, device=self.device) * .4).clamp_min(0.)
        # Full-stage rehearsal must remain local. A 12s clip clamps difficult
        # tail starts back to 3.5s on a 15.5s dance, defeating tail priorities.
        # The long stage still practices 12s transitions; full episodes retain
        # continuity while local episodes can start directly before a failure.
        length = min(self.duration, 12. if self.stage == "long" else 3.)
        start.clamp_(max=max(0., self.duration - length))
        target = torch.full_like(start, length)
        if self.report.get("repair_training"):
            category = torch.rand(count, device=self.device)
            starts = [max(0., w["start_s"] - .6) for w in self.report["support_windows"]]
            starts += [max(0., e["phases"][0][0] - .6) for e in self.report["events"]
                       if not e["flight_reward"] and e.get("flight_evidence") == "moderate"]
            jumps = [max(0., e["phases"][0][0] - .6) for e in self.report["events"] if e["flight_reward"]]
            if not starts or not jumps:
                raise ValueError("Repair needs support reviews and retained flight events")
            a = torch.tensor(starts, device=self.device)[torch.randint(len(starts), (count,), device=self.device)]
            b = torch.tensor(jumps, device=self.device)[torch.randint(len(jumps), (count,), device=self.device)]
            start = torch.where(category < .4, a, torch.where(category < .75, b, 0.))
            start.clamp_(max=max(0., self.duration - 3.))
            target = torch.where(category < .75, min(3., self.duration), self.duration)
            target = target + (start + target >= self.duration - 1e-6) * self.report.get("end_hold_seconds", 0.)
            return (start / env.step_dt).round().long(), (target / env.step_dt).round().long().clamp_min(1)
        if self.stage == "full":
            # Retain 25% local rehearsal; 75% learn frame-zero continuity.
            full = torch.rand(count, device=self.device) < .75
            start = torch.where(full, 0., start)
            target = torch.where(full, self.duration, target)
        target = target + (start + target >= self.duration - 1e-6) * self.report.get("end_hold_seconds", 0.)
        return (start / env.step_dt).round().long(), (target / env.step_dt).round().long().clamp_min(1)

    def observe(self):
        env = self.env
        key = int(env.common_step_counter)
        if self.last_step == key:
            return
        self.last_step = key
        from source.my_task.motion_reference import motion_steps
        from source.my_task.mdp import jump
        action = env.cfg.actions.joint_pos
        steps = motion_steps(env)
        from source.my_task.mdp.auto_jump import tracking_metrics
        metrics = tracking_metrics(env,action.motion_h5_path,action.motion_window_seconds)
        error = (metrics["leg_rms_rad"]/.2).clamp(max=3.)
        error = error + (metrics["height_error_m"]/.08).clamp(max=2.)
        error = error + (metrics["tilt_error_rad"]/.3).clamp(max=2.)
        # Physical contact history covers every substep; missing expected flight
        # adds difficulty, never creates a new jump label in a non-jump region.
        phase = jump.phases(env)
        flight = jump.flight_state(env)
        qualified = (flight["air"] >= 2) & flight["stance"] & flight["rising"]
        missed = ~qualified & phase[:, 2]
        error = error + missed.float()
        from source.my_task.mdp.auto_jump import support_metrics
        support = support_metrics(env)
        error = error + support["unexpected_air"].float() + support["missing_feet"].sum(-1).float() * .5
        from source.my_task.mdp.auto_jump import swing_metrics
        swing = swing_metrics(env)
        error = error + (swing["error_m"] / .08).clamp(max=2.) + swing["early_contact"].float()
        bins = (steps.float() * env.step_dt).long().clamp(0, self.bins - 1)
        active = torch.ones(env.num_envs, device=self.device, dtype=torch.bool)
        sums, counts = self.sums, self.counts
        if self.probe is not None:
            active = self.probe["active"]
            sums, counts = self.probe["sums"], self.probe["counts"]
            self.probe["quality"].update(metrics,active,steps.float()*env.step_dt)
        sums.scatter_add_(0, bins[active], error[active].detach())
        counts.scatter_add_(0, bins[active], torch.ones_like(error[active]))
        if self.probe is not None:
            captures=self.probe.get("pose_captures",{})
            if self.probe.get("pose_capture") is not None:captures={0:self.probe["pose_capture"]}
            for index,capture in captures.items():
                if not bool(active[index]):continue
                robot=env.scene["robot"]
                capture.append(robot.data.root_state_w.torch[index,:7],robot.data.joint_pos.torch[index])
                if "motion_commands" in self.probe:
                    self.probe["motion_commands"][index].append(robot.data.joint_pos_target.torch[index,self.motion_analyzer.joints].detach().clone())
            if "trace" in self.probe:
                from source.my_task.motion_reference import get_or_create_motion_buffer
                robot=env.scene["robot"]
                ids=[list(robot.body_names).index(f"{side}_ankle_roll_link") for side in ("left","right")]
                velocity=robot.data.body_link_lin_vel_w.torch[:,ids,:2].norm(dim=-1)
                supported_feet=support["expected"] | (support["required"] & ~support["expected"].any(-1))[:,None]
                speed=torch.where(supported_feet & support["contacts"],velocity,0.).amax(-1)
                feet=robot.data.body_link_pose_w.torch[:,ids,:3]
                distance=(feet[:,0,:2]-feet[:,1,:2]).norm(dim=-1)
                buf=get_or_create_motion_buffer(env,action.motion_h5_path,action.motion_window_seconds)
                ref_vz=(buf.root_pos_delta(steps+1)[:,2]-buf.root_pos_delta(steps)[:,2])/env.step_dt
                values=torch.stack((steps.float()*env.step_dt,metrics["leg_rms_rad"],metrics["height_error_m"],
                    metrics["tilt_error_rad"],metrics["normal_weight"],support["required"].float(),
                    support["unexpected_air"].float(),support["contacts"][:,0].float(),support["contacts"][:,1].float(),
                    qualified.float(),robot.data.root_lin_vel_w.torch[:,2],distance,speed,
                    metrics["height_delta_m"].clamp_min(0.),(robot.data.root_lin_vel_w.torch[:,2]-ref_vz).clamp_min(0.),
                    swing["required"].float(),swing["error_m"],swing["early_contact"].float()),dim=-1)
                active_ids=active.nonzero().flatten()
                self.probe["trace"].append(torch.cat((active_ids[:,None].float(),values[active_ids]),dim=-1))
            expected = active & jump.phases(env)[:, 2]
            self.probe["flight_requests"].scatter_add_(0, bins[expected], torch.ones_like(error[expected]))
            self.probe["flight_missed"].scatter_add_(0, bins[expected], missed[expected].float())
            required = active & support["required"]
            self.probe["support_samples"].scatter_add_(0, bins[required], torch.ones_like(error[required]))
            self.probe["unexpected_air"].scatter_add_(0, bins[required], support["unexpected_air"][required].float())
            self.probe["air_run"] = torch.where(active & ~support["contacts"].any(-1), self.probe["air_run"] + 1, 0)
            self.probe["air_max"] = torch.maximum(self.probe["air_max"], self.probe["air_run"])
            self.probe["air_samples"].scatter_add_(0, bins[active], (~support["contacts"].any(-1))[active].float())
            windows = self.report.get("metric_windows", self.report.get("support_windows", []))
            for i, window in enumerate(windows):
                inside = active & (steps.float() * env.step_dt >= window["start_s"]) & (steps.float() * env.step_dt < window["end_s"])
                self.probe["window_samples"][i] += inside.sum()
                self.probe["window_air"][i] += (inside & ~support["contacts"].any(-1)).sum()

    def update(self):
        visited = self.counts > 0
        self.visits += self.counts
        observed = (self.sums / self.counts.clamp_min(1) +
                    self.failures / self.counts.clamp_min(1) * 10).clamp(.2, 4.)
        self.difficulty[visited] = .8 * self.difficulty[visited] + .2 * observed[visited]
        self.sums.zero_()
        self.counts.zero_()
        self.failures.zero_()

    def state_dict(self):
        return dict(version=1, source_sha256=self.report["source_sha256"],
                    reward_contract=self.report.get("reward_contract"),
                    duration_s=self.duration, end_hold_seconds=self.report.get("end_hold_seconds", 0.), control_hz=self.report.get("control_hz"),
                    difficulty=self.difficulty.cpu().tolist(), progress=self.progress,
                    observed_steps=(self.visits + self.counts).cpu().tolist(),
                    history=self.history,
                    motion_feedback=self.motion_feedback.state_dict() if self.motion_feedback is not None else None,
                    quality_selection=self.selection.state_dict() if self.selection is not None else None)

    def load_state_dict(self, state):
        if state.get("reward_contract") != self.report.get("reward_contract"):
            raise ValueError("Reward contract changed; start a fresh run instead of reusing old optimizer/best state")
        if state["source_sha256"] != self.report["source_sha256"]:
            raise ValueError("Auto curriculum reference differs from resumed checkpoint")
        if state.get("duration_s") != self.duration or state.get("control_hz") != self.report.get("control_hz"):
            raise ValueError("Auto curriculum window/control frequency differs from checkpoint")
        value = torch.tensor(state["difficulty"], device=self.device)
        if value.shape != self.difficulty.shape or not torch.isfinite(value).all():
            raise ValueError("Invalid auto curriculum difficulty state")
        self.difficulty.copy_(value.clamp(.2, 4.))
        visits = torch.tensor(state.get("observed_steps", [0.] * self.bins), device=self.device)
        if visits.shape != self.visits.shape or not torch.isfinite(visits).all() or bool((visits < 0).any()):
            raise ValueError("Invalid auto curriculum coverage state")
        self.visits.copy_(visits)
        self.history = state.get("history", [])
        saved_motion=state.get("motion_feedback")
        if saved_motion:
            if self.motion_feedback is None:raise ValueError("Resume this checkpoint with --auto_motion_feedback")
            from source.train_workflow.utils.motion_feedback import MotionFeedback
            self.motion_feedback=MotionFeedback(self.bins,saved_motion)

    def update_motion_feedback(self,records,iteration):
        if self.motion_feedback is None:return
        from source.train_workflow.utils.deployment import atomic_write_json
        latest=self.motion_feedback.update(records)
        atomic_write_json(self.log_dir/"motion_feedback.json",dict(iteration=iteration,**self.motion_feedback.state_dict()))
        print(f"[运动反馈] iter={iteration} | 三种子重复候选秒桶={latest['consensus_seconds']} | "
              f"反馈后的局部采样秒桶={latest['boosted_seconds']} | 占采样分布至多10%；候选需人工复核",flush=True)

    def save_report(self, iteration):
        state = self.state_dict()
        state.update(iteration=iteration, stage=self.stage,
                     reward_strength=float(self.env._auto_jump_strength),
                     probabilities=self.probabilities().cpu().tolist())
        (self.log_dir / "auto_curriculum.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        import csv
        with (self.log_dir / "auto_priorities.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["start_s", "end_s", "difficulty", "sampling_probability", "observed_steps"])
            for index in sorted(range(self.bins), key=lambda i: state["probabilities"][i], reverse=True):
                writer.writerow([index, min(index + 1, self.duration), state["difficulty"][index],
                                 state["probabilities"][index], state["observed_steps"][index]])

    def validate(self, runner, wrapped, *, full=False, seed=42, starts_override=None, trace_dir=None, seeds=None):
        """Deterministic policy, fixed starts, first episode only, no PPO updates."""
        env = self.env
        replicas=list(seeds) if seeds is not None else None
        if replicas is not None and (not full or not replicas or len(set(replicas))!=len(replicas) or len(replicas)>env.num_envs):
            raise ValueError("Parallel validation requires full rollouts, unique seeds and enough environments")
        policy = runner.get_inference_policy(device=env.device)
        torch_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(seed)
        tail_only = bool(self.report.get("tail_only"))
        validation_start = float(self.report.get("tail_start_seconds", 0.0)) if tail_only else 0.0
        starts = ([validation_start] if full and tail_only else
                  ([0.] if full else sorted(set(
                      [float(x) for x in range(self.bins)] +
                      [max(0., e["phases"][0][0] - .6) for e in self.report["events"]]))))
        if starts_override is not None:
            starts = list(starts_override)
        length = ((float(self.report.get("episode_duration_s", self.duration)) - validation_start)
                  if full and tail_only else self.report.get("episode_duration_s", self.duration) if full
                  else min(3., self.duration))
        starts = ([validation_start] * len(replicas) if replicas is not None and tail_only else
                  [0.] * len(replicas) if replicas is not None else
                  sorted(set(min(s, max(0., self.duration - length)) for s in starts)))
        progress=getattr(env,"_auto_training_progress",None)
        if progress is not None:
            progress.validation_started("full" if full else "coverage",replicas or seed,len(starts),length)
        results = []
        replica_traces=[]
        from source.train_workflow.utils.quality_selection import assess_trace, resolve_profile
        profile=self.quality_profile or resolve_profile(self.report)
        last_notice=time.monotonic()
        random_scope=None
        try:
            if full:
                from source.my_task.mdp.validation_random import ValidationRandom
                random_scope=ValidationRandom(env,replicas or [seed])
                random_scope.__enter__()
            for offset in range(0, len(starts), env.num_envs):
                batch = starts[offset:offset + env.num_envs]
                count = len(batch)
                self.probe = dict(starts=torch.zeros(env.num_envs, device=self.device, dtype=torch.long),
                    targets=torch.full((env.num_envs,), max(1, round(length / env.step_dt)), device=self.device, dtype=torch.long),
                    active=torch.zeros(env.num_envs, device=self.device, dtype=torch.bool),
                    completed=torch.zeros(env.num_envs, device=self.device, dtype=torch.bool),
                    steps=torch.zeros(env.num_envs, device=self.device, dtype=torch.long),
                    sums=torch.zeros_like(self.sums), counts=torch.zeros_like(self.counts),
                    failures=torch.zeros_like(self.sums), flight_requests=torch.zeros_like(self.sums),
                    flight_missed=torch.zeros_like(self.sums), support_samples=torch.zeros_like(self.sums),
                    unexpected_air=torch.zeros_like(self.sums),
                    air_samples=torch.zeros_like(self.sums),
                    air_run=torch.zeros(env.num_envs, device=self.device, dtype=torch.long),
                    air_max=torch.zeros(env.num_envs, device=self.device, dtype=torch.long),trace=[])
                from source.train_workflow.utils.motion_quality import MotionQualityMonitor
                self.probe["quality"] = MotionQualityMonitor(env.num_envs,self.device,env.step_dt)
                windows = self.report.get("metric_windows", self.report.get("support_windows", []))
                self.probe["window_samples"] = torch.zeros(len(windows), device=self.device)
                self.probe["window_air"] = torch.zeros(len(windows), device=self.device)
                self.probe["starts"][:count] = torch.tensor(batch, device=self.device).div(env.step_dt).round().long()
                env.reset()
                captures={}
                if full and (self.record_validation or self.motion_feedback is not None):
                    from source import sim_compat
                    from source.train_workflow.utils.validation_recording import ValidationPoseCapture
                    robot=env.scene["robot"]
                    origins=sim_compat.as_torch(env.scene.sim.get_clone_plan().positions)
                    for index in range(count):
                        capture=ValidationPoseCapture(origin=origins[index],joint_names=robot.joint_names,control_hz=1./env.step_dt)
                        capture.append(robot.data.root_state_w.torch[index,:7],robot.data.joint_pos.torch[index])
                        captures[index]=capture
                    self.probe["pose_captures"]=captures
                    if self.motion_feedback is not None:
                        if self.motion_analyzer is None:
                            from source.my_task.motion_reference import get_or_create_motion_buffer
                            from source.train_workflow.utils.motion_feedback import KinematicMotionAnalyzer
                            action=env.cfg.actions.joint_pos
                            buf=get_or_create_motion_buffer(env,action.motion_h5_path,action.motion_window_seconds)
                            self.motion_analyzer=KinematicMotionAnalyzer(env,self.report,buf)
                        self.probe["motion_commands"]={i:[] for i in captures}
                self.probe["active"][:count] = True
                obs = wrapped.get_observations()
                with torch.inference_mode():
                    for probe_step in range(max(1, round(length / env.step_dt)) + 1):
                        obs, _, _, _ = wrapped.step(policy(obs))
                        if not bool(self.probe["active"].any()):
                            break
                        if time.monotonic()-last_notice >= 30.:
                            if progress is not None:
                                progress.validation_progress(batch=offset//env.num_envs+1,
                                    batches=math.ceil(len(starts)/env.num_envs),
                                    elapsed=min(length,(probe_step+1)*env.step_dt),length=length,
                                    pending=int(self.probe["active"].sum()))
                            else:print(f"[验证进行中] seeds={replicas or [seed]} | {(probe_step+1)*env.step_dt:.2f}/{length:.2f}s",flush=True)
                            last_notice=time.monotonic()
                import numpy as np
                from source.train_workflow.utils.quality_selection import TRACE_FIELDS
                traces=torch.cat(self.probe["trace"]).detach().cpu().numpy() if self.probe["trace"] else np.empty((0,1+len(TRACE_FIELDS)))
                for index, start in enumerate(batch):
                    trial_seed=replicas[index] if replicas is not None else seed
                    trace=traces[traces[:,0]==index,1:]
                    if replicas is not None:replica_traces.append(trace)
                    acceptance=assess_trace(trace,self.report,profile,full=full)
                    if trace_dir is not None:
                        directory=Path(trace_dir);directory.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(directory/f"trace_{offset+index:04d}.npz",trace=trace,start_s=start,full=full,seed=trial_seed)
                    results.append(dict(start_s=start, completed=bool(self.probe["completed"][index]),
                                        reached_s=start + int(self.probe["steps"][index]) * env.step_dt,
                                        longest_air_s=float(self.probe["air_max"][index]) * env.step_dt,
                                        quality=self.probe["quality"].summary(index),acceptance=acceptance))
                    if index in captures and self.motion_feedback is not None:
                        from source.train_workflow.utils.deployment import atomic_write_json
                        iteration=int(runner.current_learning_iteration)+1
                        try:
                            poses=torch.stack(captures[index].frames).detach().cpu().numpy()
                            commands=torch.stack(self.probe["motion_commands"][index]).detach().cpu().numpy()
                            if self.record_validation or trace_dir is not None:
                                raw=self.log_dir/"motion_diagnostics"/f"iter_{iteration:06d}"/f"seed_{trial_seed}.npz"
                                raw.parent.mkdir(parents=True,exist_ok=True)
                                temporary=raw.with_suffix(".tmp")
                                with temporary.open("wb") as stream:
                                    np.savez_compressed(stream,poses=poses,commands=commands,trace=trace,
                                        joint_names=np.asarray(env.scene["robot"].joint_names),control_hz=1./env.step_dt,
                                        source_sha256=self.report["source_sha256"])
                                temporary.replace(raw)
                            movement=self.motion_analyzer.assess(poses,commands,trace)
                        except (ValueError,RuntimeError,OSError) as error:
                            movement=dict(protocol="reference-motion-v1",complete=False,score=None,candidates=[],error=str(error))
                        results[-1]["motion_diagnostics"]=movement
                        atomic_write_json(self.log_dir/"motion_diagnostics"/f"iter_{iteration:06d}"/f"seed_{trial_seed}.json",movement)
                    if index in captures and self.record_validation:
                        iteration=int(runner.current_learning_iteration)+1
                        target=self.log_dir/"validation_media"/f"iter_{iteration:06d}"/f"seed_{trial_seed}"/"poses.npz"
                        try:
                            path=captures[index].save(target,result=results[-1],metadata=dict(
                                source=self.validation_source,completed_iterations=iteration,seed=trial_seed,
                                validation_protocol="parallel-seed-streams-v1",parallel_seeds=replicas or [seed],
                                num_envs=env.num_envs,simulation_device=str(env.device),physics_hz=1./env.physics_dt,
                                decimation=env.cfg.decimation,observation_corruption=env.cfg.observations.policy.enable_corruption,
                                source_sha256=self.report["source_sha256"],scene=str(self.validation_scene)))
                            results[-1]["pose_recording"]=path
                            print(f"[验收轨迹已录制] iter={iteration} seed={trial_seed} | {path}",flush=True)
                        except (OSError,ValueError) as error:
                            results[-1]["pose_recording_error"]=str(error)
                            print(f"[验收轨迹保存失败，不改变验收判定] {error}",flush=True)
                visited = self.probe["counts"] > 0
                self.visits += self.probe["counts"]
                measured = ((self.probe["sums"] + 10 * self.probe["failures"]) /
                            self.probe["counts"].clamp_min(1)).clamp(.2, 4.)
                self.difficulty[visited] = .5 * self.difficulty[visited] + .5 * measured[visited]
                if offset == 0:
                    probe_sums = {key: self.probe[key].clone() for key in
                                  ("sums", "counts", "flight_requests", "flight_missed", "support_samples", "unexpected_air", "air_samples", "window_samples", "window_air")}
                else:
                    for key in probe_sums:
                        probe_sums[key] += self.probe[key]
        finally:
            self.probe = None
            try:
                if random_scope is not None:random_scope.__exit__(None,None,None)
                env.reset()
            finally:
                try:runner.alg.train_mode()
                finally:
                    torch.random.set_rng_state(torch_rng)
                    if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        if replicas is not None:
            records=[self._finish_validation(runner,True,s,[trial],None,windows,profile,progress,trace)
                     for s,trial,trace in zip(replicas,results,replica_traces)]
            self._render_validations(records,progress)
            return records
        record=self._finish_validation(runner,full,seed,results,probe_sums,windows,profile,progress)
        if full:self._render_validations([record],progress)
        return record

    def validate_seeds(self,runner,wrapped,seeds,**kwargs):
        return self.validate(runner,wrapped,full=True,seeds=seeds,**kwargs)

    def _render_validations(self,records,progress):
        if not self.render_validation:return
        from source.train_workflow.utils.validation_recording import render_recordings
        trials=[r for record in records for r in record["results"]]
        paths=[r.get("pose_recording") for r in trials]
        if not all(paths):
            print("[验收录像跳过] 部分seed轨迹缺失；原验收判定保留",flush=True)
            if progress is not None:progress.record_overhead("video",0.)
            return
        if progress is not None:
            progress.begin_overhead("video")
            progress.status("validation_video","并行物理验收已结束，正在生成同场景对比MP4；暂不进行PPO学习")
        try:
            video=render_recordings(paths)
        finally:
            if progress is not None:progress.end_overhead("video")
        for trial in trials:trial["video_recording"]=video
        if video.get("error"):print(f"[验收录像失败，原始轨迹已保留] {video['error']}",flush=True)

    def _finish_validation(self,runner,full,seed,results,probe_sums,windows,profile,progress,trace=None):
        record = dict(kind="full" if full else "coverage", seed=seed, results=results,
                      iteration=int(runner.current_learning_iteration), stage=self.stage,
                      completed=sum(r["completed"] for r in results), tested=len(results))
        record["error_metric_version"] = 2
        if full and self.motion_feedback is not None:
            record["motion_feedback_required"]=True
            summary=results[0].get("motion_diagnostics",{}) if results else {}
            print(f"[参考动作对照] seed={seed} | 完整记录={summary.get('complete')} | "
                  f"运动指标={summary.get('score')} | 候选窗口={len(summary.get('candidates',[]))}；仅作诊断候选",flush=True)
        if full:record["validation_protocol"]="parallel-seed-streams-v1"
        record["quality_completed"] = sum(r["completed"] and r["quality"]["passed"] for r in results)
        record["quality_passed"] = record["quality_completed"] == record["tested"] and bool(results)
        record["acceptance_version"]=1
        record["acceptance_scope"]="continuous_motion" if full else "zero_velocity_reset_diagnostic"
        record["acceptance_completed"]=sum(r["completed"] and r["quality"]["passed"] and r["acceptance"]["passed"] for r in results)
        record["acceptance_passed"]=record["acceptance_completed"]==record["tested"] and bool(results)
        record["acceptance_score"]=max((r["acceptance"].get("score") or 0.) for r in results) if record["acceptance_passed"] else None
        record["bins"] = [dict(start_s=i, samples=int(probe_sums["counts"][i]),
            no_contact_fraction=(float(probe_sums["air_samples"][i] / probe_sums["counts"][i])
                                 if probe_sums["counts"][i] > 0 else None),
            mean_error=(float(probe_sums["sums"][i] / probe_sums["counts"][i])
                        if probe_sums["counts"][i] > 0 else None),
            expected_flight_samples=int(probe_sums["flight_requests"][i]),
            support_samples=int(probe_sums["support_samples"][i]),
            unexpected_air_fraction=(float(probe_sums["unexpected_air"][i] / probe_sums["support_samples"][i])
                                     if probe_sums["support_samples"][i] > 0 else None),
            missed_flight_fraction=(float(probe_sums["flight_missed"][i] / probe_sums["flight_requests"][i])
                                   if probe_sums["flight_requests"][i] > 0 else None))
                          for i in range(self.bins)] if probe_sums is not None else []
        self.history.append(record)
        record["review_windows"] = [dict(start_s=w["start_s"], end_s=w["end_s"],
            samples=int(probe_sums["window_samples"][i]),
            no_contact_fraction=(float(probe_sums["window_air"][i] / probe_sums["window_samples"][i])
                                 if probe_sums["window_samples"][i] > 0 else None)) for i,w in enumerate(windows)] if probe_sums is not None else []
        if trace is not None:
            from source.train_workflow.utils.quality_selection import INDEX
            def trace_stats(mask):
                rows=trace[mask]
                if not len(rows):return dict(samples=0,no_contact_fraction=None)
                contacts=rows[:,[INDEX["contact_left"],INDEX["contact_right"]]].any(-1)
                return dict(samples=len(rows),no_contact_fraction=float((~contacts).mean()),
                    mean_leg_rms_rad=float(rows[:,INDEX["leg"]].mean()),
                    mean_height_error_m=float(rows[:,INDEX["height"]].mean()),
                    mean_tilt_error_rad=float(rows[:,INDEX["tilt"]].mean()))
            seconds=trace[:,INDEX["seconds"]]
            record["bins"]=[dict(start_s=i,**trace_stats((seconds>=i)&(seconds<i+1))) for i in range(self.bins)]
            record["review_windows"]=[dict(start_s=w["start_s"],end_s=w["end_s"],
                **trace_stats((seconds>=w["start_s"])&(seconds<w["end_s"]))) for w in windows]
            record["sampling_statistics_version"]=3
        print(f"[AUTO PROBE] {record['kind']} seed={seed}: {record['completed']}/{record['tested']} completed; "
              f"quality-qualified {record['quality_completed']}/{record['tested']}")
        failed_quality = [r for r in results if not r["quality"]["passed"]]
        if failed_quality:
            print(f"[质量未通过] {len(failed_quality)}个起点存在持续误差/未观测状态；"
                  f"首项起点={failed_quality[0]['start_s']:.2f}s, "
                  f"首次持续异常={failed_quality[0]['quality']['first_issue_s']}")
        failed_acceptance=[r for r in results if not r["acceptance"]["passed"]]
        print(f"[{'连续动作验收' if full else '中途重置诊断（仅作诊断）'}] "
              f"{record['acceptance_completed']}/{record['tested']} | 门槛={profile['kind']}")
        if failed_acceptance:
            print(f"[动作验收未通过] 起点={failed_acceptance[0]['start_s']:.2f}s | "
                  f"{failed_acceptance[0]['acceptance']['failures'][:3]}")
        if progress is not None:progress.validation_finished(record)
        return record


def train_auto(runner, wrapped, cfg, agent_cfg, report, log_dir, args, resume_infos=None):
    from source.train_workflow.utils.training_progress import TrainingProgress
    state=(resume_infos or {}).get("auto_curriculum") or {}
    progress=TrainingProgress(runner.logger,start=int(runner.current_learning_iteration)+(1 if args.resume else 0),
        total=int(agent_cfg.max_iterations),initial_progress=float(state.get("progress",0.)),
        motion_seconds=report.get("episode_duration_s",report["duration_s"]),log_dir=log_dir,
        update_interval=args.auto_jump_interval,probe_interval=args.auto_jump_interval)
    wrapped.unwrapped._auto_training_progress=progress
    try:
        with progress:
            _train_auto_impl(runner,wrapped,cfg,agent_cfg,report,log_dir,args,resume_infos,progress)
    except KeyboardInterrupt:
        # Best weights/bundle were already published at confirmation time.
        # Discard a potentially partial PPO update; keep the last complete
        # numbered checkpoint for resuming, and let main close the simulator.
        path=Path(log_dir)/"quality_selection.json"
        if path.exists():
            from source.train_workflow.utils.deployment import atomic_write_json
            try:
                saved=json.loads(path.read_text(encoding="utf-8"));saved["stop_reason"]="manual_stop"
                atomic_write_json(path,saved)
            except (OSError,ValueError) as error:progress.emit(f"[手动停止] 无法更新质量历史的停止标记：{error}")
    finally:
        del wrapped.unwrapped._auto_training_progress


def _train_auto_impl(runner, wrapped, cfg, agent_cfg, report, log_dir, args, resume_infos, progress):
    import shutil
    if getattr(wrapped.unwrapped,"num_envs",3)<3:
        raise ValueError("Auto-Jump parallel quality validation requires --num_envs >= 3")
    from source.my_task.motion_reference import get_or_create_motion_buffer
    from source.train_workflow.utils.deployment import save_bundle,atomic_copy_file,atomic_write_json
    from source.train_workflow.utils.quality_selection import QualitySelection,resolve_profile,continuous_coverage
    controller = AutoJumpController(wrapped.unwrapped, report, log_dir)
    wrapped.unwrapped._auto_jump_controller = controller
    state = (resume_infos or {}).get("auto_curriculum")
    if state:
        controller.load_state_dict(state)
    saved_selection=(state or {}).get("quality_selection")
    if state and state.get("end_hold_seconds", 0.) != report.get("end_hold_seconds", 0.):
        # Preserve policy, optimizer and sampler, but do not reuse acceptance
        # under a shorter/different hold requirement.
        saved_selection = None
        print(f"[AUTO HOLD] Hold changed {state.get('end_hold_seconds', 0.)} -> {report.get('end_hold_seconds', 0.)}s; previous best/acceptance reset, policy and sampler retained")
    profile_path=getattr(args,"auto_quality_profile",None)
    profile=(saved_selection["profile"] if saved_selection and not profile_path else resolve_profile(report,profile_path))
    # Legacy checkpoints did not freeze the formerly hard-coded swing limits.
    # Adopt today's explicit defaults and reset quality history on migration.
    import copy
    from source.train_workflow.utils.quality_selection import DEFAULT_SWING_LIMITS
    profile=copy.deepcopy(profile)
    profile.setdefault("swing_limits",dict(DEFAULT_SWING_LIMITS))
    if saved_selection and profile != saved_selection["profile"]:
        # Explicitly selecting another reference-bound acceptance profile starts
        # a fresh quality history. Keep the previous receipt for audit; model and
        # optimizer continuation are unaffected.
        atomic_write_json(Path(log_dir)/"quality_selection_before_profile_change.json", saved_selection)
        print("[验收标准变更] 新profile或旧版触地容差迁移；保留旧记录，重新累计验收和best。", flush=True)
        saved_selection=None
    # Frozen before learning. Resume preserves its saved profile unless another
    # profile is explicitly selected; it must still match this exact reference.
    from source.train_workflow.utils.quality_selection import validate_profile
    validate_profile(report,profile)
    selector=QualitySelection(profile,patience=getattr(args,"auto_quality_patience",3),
        interval=args.auto_jump_interval or 400,state=saved_selection)
    controller.quality_profile=profile;controller.selection=selector
    log_dir=Path(log_dir)
    atomic_write_json(log_dir/"quality_profile.json",profile)
    controller.record_validation=getattr(args,"auto_validation_record",True)
    controller.render_validation=controller.record_validation and getattr(args,"auto_validation_video",True)
    progress.expects_validation_video=bool(controller.render_validation)
    progress.expects_final_export=True
    if controller.record_validation:
        from source.train_workflow.utils.validation_recording import save_validation_scene
        action=cfg.actions.joint_pos
        buf=get_or_create_motion_buffer(wrapped.unwrapped,action.motion_h5_path,action.motion_window_seconds)
        controller.validation_scene=save_validation_scene(log_dir/"validation_media"/"scene.pt",
            env_cfg=cfg,reference=buf.export_reference(),task=args.task,report=report,
            reference_video=getattr(args,"validation_reference",None))
    # Quality is advisory. Even legacy callers requesting early stopping must
    # continue to the configured budget; only the user decides to stop sooner.
    early_enabled=False
    if selector.best is not None:
        source=Path(selector.best["checkpoint"])
        if not source.exists() or not source.with_suffix(".validation.json").exists():
            raise FileNotFoundError(f"Resume requires the saved best checkpoint and its validation report: {source}")
        target=log_dir/"best"/source.name;target.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve()!=target.resolve():
            shutil.copy2(source,target);shutil.copy2(source.with_suffix(".validation.json"),target.with_suffix(".validation.json"))
        selector.best["checkpoint"]=str(target.resolve())
        atomic_copy_file(target,target.parent/"model_best.pt")
    progress.quality_checked(selector,early_enabled)
    total = int(agent_cfg.max_iterations)
    done = 0
    if args.resume and not state:
        raise ValueError("Resume Auto-Jump with one of its own checkpoints; legacy specialists have a different observation layout")
    start = int(runner.current_learning_iteration) + (1 if args.resume else 0)
    # Resume progresses along the original curriculum rather than restarting it.
    initial_progress = float(state.get("progress", 0.)) if state else 0.
    if args.auto_jump_repair:
        with torch.no_grad():
            # Preserve the deterministic mean; limit noisy ankle exploration.
            runner.alg.get_policy().distribution.std_param.clamp_(min=.05, max=.35)
    if args.auto_jump_check_starts:
        baseline = [controller.validate(runner, wrapped, seed=args.seed + i,
                    starts_override=args.auto_jump_check_starts) for i in range(3)]
        (Path(log_dir) / "repair_before.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    install_checkpoint_metadata(runner, controller, start=start, total=total, initial_progress=initial_progress,
                                on_save=progress.checkpoint_saved)
    last_stage = None
    probe_start = float(getattr(args, "auto_jump_probe_start_fraction", 0.2))
    next_probe = first_probe_done(total, args.auto_jump_interval, start=start, start_fraction=probe_start)
    if next_probe is None:
        next_probe = args.auto_jump_interval
    elif args.auto_jump_interval > 0 and next_probe > args.auto_jump_interval:
        notify = getattr(progress, "emit", None)
        if next_probe <= total:
            message = (f"[训练计划] 累计完成达到总预算 {probe_start:.0%} 之前不进行整段验收和录像 | "
                       f"首次验收：累计完成 {start + next_probe} 次")
        else:
            message = (f"[训练计划] 累计完成达到总预算 {probe_start:.0%} 之前不进行整段验收和录像 | "
                       "中途验收均在该比例之前，仅在结束时验收")
        if notify is not None:
            notify(message)
        else:
            print(message, flush=True)
    last_screen=None
    seeds=[args.seed+i for i in range(3)]
    last_confirmation=None

    def persist_selection():
        atomic_write_json(log_dir/"quality_selection.json",selector.state_dict())
        progress.quality_checked(selector,early_enabled)

    def export_best_now():
        # Use a copy: online export must not change the live episode length,
        # reset sampling or curriculum while subsequent PPO updates continue.
        import copy
        best=selector.best
        if best is None:return
        checkpoint=Path(best["checkpoint"])
        validation=json.loads(checkpoint.with_suffix(".validation.json").read_text(encoding="utf-8"))
        action=cfg.actions.joint_pos
        buf=get_or_create_motion_buffer(wrapped.unwrapped,action.motion_h5_path,action.motion_window_seconds)
        export_cfg=copy.deepcopy(cfg)
        export_cfg.episode_length_s=buf.num_steps*wrapped.unwrapped.step_dt
        export_cfg.events.reset_robot_joints.params.update(segment_seconds=export_cfg.episode_length_s,
            random_start=False,motion_start_seconds=0.)
        if hasattr(export_cfg,"auto_jump_report"):
            export_cfg.auto_jump_report["last_validation"]=dict(acceptance_passed=True,
                full_rollouts_completed=True,tracking_quality_passed=True,export_reason="best_update",
                selected_checkpoint=str(checkpoint.resolve()),selected_completed_iterations=best["iteration"],seeds=best["seeds"],
                quality_profile_kind=profile["kind"])
        progress.begin_overhead("export")
        try:
            save_bundle(log_dir/"deploy/deployment.pt",checkpoint_path=checkpoint,
                reference=buf.export_reference(),env_cfg=export_cfg,agent_cfg=agent_cfg,task=args.task,dance=args.dance)
        finally:
            progress.end_overhead("export")
        atomic_write_json(log_dir/"auto_validation.json",validation)
        progress.status("best_exported",f"best部署包已更新，可人工验收：完成 {best['iteration']} 次 | "
                        f"{log_dir/'deploy/deployment.pt'}；继续训练")

    def confirm_candidate(iteration, first=None):
        nonlocal last_confirmation
        progress.status("quality_confirmation",f"质量候选 {iteration} 次，汇总三种子并行整段验收：{seeds}")
        validation=first if isinstance(first,list) else controller.validate_seeds(runner,wrapped,seeds)
        if getattr(controller,"motion_feedback",None) is not None:
            controller.update_motion_feedback(validation,iteration)
        target=log_dir/"best"/f"model_{iteration-1}.pt"
        passed,updated=selector.confirm(iteration,validation,seeds,str(target.resolve()))
        last_confirmation=dict(iteration=iteration,passed=passed,validation=validation)
        if updated:
            target.parent.mkdir(parents=True,exist_ok=True)
            runner.save(str(target),infos={"auto_curriculum":controller.state_dict()})
            atomic_write_json(target.with_suffix(".validation.json"),validation)
            atomic_copy_file(target,target.parent/"model_best.pt")
            export_best_now()
            progress.status("best_checkpoint",f"最佳合格检查点更新：完成 {iteration} 次，局部质量评分={selector.best['score']:.4f}")
        return passed,validation
    if selector.best is not None:export_best_now()
    while done < total:
        curriculum_progress = initial_progress + (1 - initial_progress) * done / max(1, total - 1)
        controller.configure(curriculum_progress)
        if controller.stage != last_stage:
            wrapped.unwrapped.reset()
            print(f"[AUTO CURRICULUM] stage={controller.stage}, progress={curriculum_progress:.3f}")
            last_stage = controller.stage
        # Bound chunks at course transitions, even for smoke tests.
        count = min(args.auto_jump_interval, total - done)
        for boundary in (.55, .85):
            if initial_progress < boundary and curriculum_progress < boundary:
                at = math.ceil((boundary - initial_progress) / (1 - initial_progress) * max(1, total - 1))
                count = min(count, max(1, at - done))
        if args.auto_jump_interval > 0:
            count = min(count, max(1, next_probe - done))
        runner.current_learning_iteration = start + done
        # Native KL adaptation is bounded inside every PPO mini-batch, not
        # merely once per 200-iteration chunk. Retain the current rate on entry.
        lr_policy = runner.alg.configure_learning_rate(controller.stage, repair=args.auto_jump_repair)
        report["learning_rate_policy"] = lr_policy
        if args.auto_jump_repair:
            runner.alg.entropy_coef = .001
        progress.set_block(done=done,count=count,stage=controller.stage,
            strength=wrapped.unwrapped._auto_jump_strength,learning_rate=runner.alg.learning_rate,
            next_probe=next_probe if args.auto_jump_interval>0 and next_probe<=total else None,
            lr_policy=lr_policy)
        runner.learn(num_learning_iterations=count, init_at_random_ep_len=False)
        done += count
        controller.update()
        controller.progress = initial_progress + (1 - initial_progress) * done / total
        if args.auto_jump_interval > 0 and done >= next_probe:
            reset_diagnostic=controller.validate(runner,wrapped,seed=args.seed)
            full_batch=controller.validate_seeds(runner,wrapped,seeds)
            full=full_batch[0]
            coverage=continuous_coverage(full)
            progress.status("continuous_coverage",f"连续执行窗口验收 {coverage['completed']}/{coverage['tested']}；"
                            "中途零速度重置检查仅作诊断")
            selector.screen(start+done,coverage,full)
            last_screen=dict(iteration=start+done,coverage=coverage,full=full,full_batch=full_batch,reset_diagnostic=reset_diagnostic)
            # All seeds have already run. Publish their actual confirmation even
            # if the first seed failed; never label the other runs as untested.
            confirm_candidate(start+done,full_batch)
            persist_selection()
            next_probe += args.auto_jump_interval
        checkpoint = Path(log_dir) / f"model_{start + done - 1}.pt"
        runner.save(str(checkpoint), infos={"auto_curriculum": controller.state_dict()})
        controller.save_report(start + done - 1)
    # At the ceiling, assess the last candidate even if an intermediate screen
    # failed. Qualification still needs BOTH local coverage and full rollouts.
    if done>=total:
        progress.status("final_validation",f"达到本次预算上限 {done}/{total}，执行最终验收")
        if last_screen is None or last_screen["iteration"]!=start+done:
            reset_diagnostic=controller.validate(runner,wrapped,seed=args.seed)
            full_batch=controller.validate_seeds(runner,wrapped,seeds)
            first=full_batch[0]
            coverage=continuous_coverage(first)
            selector.screen(start+done,coverage,first)
        else:
            coverage=last_screen["coverage"];first=last_screen["full"];reset_diagnostic=last_screen["reset_diagnostic"]
            full_batch=last_screen["full_batch"]
        if last_confirmation is not None and last_confirmation["iteration"]==start+done:
            validation=last_confirmation["validation"]
        else:_,validation=confirm_candidate(start+done,full_batch)
        (log_dir/"last_checkpoint_validation.json").write_text(json.dumps(dict(coverage=coverage,full=validation,
            reset_diagnostic=reset_diagnostic),indent=2),encoding="utf-8")
        selector.stop_reason="iteration_limit" if selector.best else "iteration_limit_unqualified"
    if args.auto_jump_check_starts:
        regression = [controller.validate(runner, wrapped, seed=args.seed + i,
                      starts_override=args.auto_jump_check_starts) for i in range(3)]
        (Path(log_dir) / "repair_after.json").write_text(json.dumps(regression, indent=2), encoding="utf-8")
    qualified=selector.best is not None
    selected=Path(selector.best["checkpoint"]) if qualified else checkpoint
    if qualified:
        validation=json.loads(selected.with_suffix(".validation.json").read_text(encoding="utf-8"))
    atomic_write_json(log_dir/"auto_validation.json",validation)
    action = cfg.actions.joint_pos
    buf = get_or_create_motion_buffer(wrapped.unwrapped, action.motion_h5_path, action.motion_window_seconds)
    # Freeze full-window evaluation defaults; the sampler is a training object only.
    cfg.episode_length_s = buf.num_steps * wrapped.unwrapped.step_dt
    cfg.events.reset_robot_joints.params.update(segment_seconds=cfg.episode_length_s,
        random_start=False, motion_start_seconds=0.)
    report["last_validation"] = dict(full_rollouts_completed=all(v["completed"] == v["tested"] for v in validation),
        tracking_quality_passed=qualified,
        acceptance_passed=qualified,selected_checkpoint=str(selected.resolve()),
        selected_completed_iterations=selector.best["iteration"] if qualified else start+done,
        actual_learning_iterations=done,stop_reason=selector.stop_reason,quality_profile_kind=profile["kind"],
        seeds=[v["seed"] for v in validation],
        note="Rollout completion alone does not certify jump naturalness or reference-intent accuracy")
    # Refresh the final checkpoint with independent validation history as well.
    runner.save(str(checkpoint), infos={"auto_curriculum": controller.state_dict()})
    deployment_path = Path(log_dir) / ("deploy/deployment.pt" if qualified else "deploy/deployment_unqualified.pt")
    progress.status("exporting",f"导出{'最佳合格模型' if qualified else '未合格调试模型'}："
                    f"完成 {selector.best['iteration'] if qualified else start+done} 次的检查点 → {deployment_path}")
    progress.begin_overhead("export")
    try:
        save_bundle(deployment_path, checkpoint_path=selected,
                    reference=buf.export_reference(), env_cfg=cfg, agent_cfg=agent_cfg,
                    task=args.task, dance=args.dance)
    finally:
        progress.end_overhead("export")
    controller.save_report(start + done - 1)
    persist_selection()
    print(f"[AUTO JUMP] Exported {deployment_path}; "
          f"completion and tracking quality passed: {qualified}")
    progress.finish(qualified)
