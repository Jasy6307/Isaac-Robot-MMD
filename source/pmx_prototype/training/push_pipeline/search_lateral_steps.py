"""Probe physically recoverable standing side steps off the acceptance grid."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
sys.path[:0] = [str(SRC), str(SRC / 'training/push_pipeline')]
import source_layout
import search_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-steps', action='store_true')
    parser.add_argument('--strengths', nargs='+', type=float, default=(.35, .45, .55, .65, .70))
    parser.add_argument('--onsets', type=int, default=5)
    args = parser.parse_args()
    if not 4 <= args.onsets <= 16 or not all(.20 <= value <= 1.20 for value in args.strengths):
        parser.error('Use 4–16 onsets and strengths from .20 to 1.20 m/s')
    import numpy as np

    reference_variants = search_steps.VARIANTS
    early = ((.17, .42, .11, .05, .08, 0.), (.21, .48, .18, .045, .12, 0.),
             (.24, .55, .24, .055, .16, 0.), (.27, .58, .19, .06, .16, 0.),
             (.29, .65, .27, .075, .20, 0.), (.35, .69, .29, .07, .18, .30))
    search_steps.VARIANTS = (*reference_variants, *early, *early)
    search_steps.GROUP_SIZE = len(search_steps.VARIANTS) + 2
    parent = search_steps.StepSearch

    class SideStepSearch(parent):
        def begin(self, ids, direction):
            super().begin(ids, direction)
            if not len(ids):
                return
            _, left = horizontal_axes(self.env.root[ids, 3:])
            foot_toward_push = ((left * direction).sum(-1) < 0.).long()
            variants = self.variant_ids[ids]
            first_foot = self.torch.where(variants >= len(reference_variants) + len(early),
                                          1 - foot_toward_push, foot_toward_push)
            selected = variants >= len(reference_variants)
            self.first_foot[ids] = self.torch.where(selected, first_foot, self.first_foot[ids])

    from anon_push_waveforms import horizontal_axes

    if args.reference_steps:
        from unified_walk_run_reference import sample_vectorized

        variants = [(speed, phase, cadence, duration, strength, 0.)
                    for speed, strength in ((.28, .9), (.42, .8), (.58, .7))
                    for phase in (.0, .25, .5, .75)
                    for cadence, duration in ((1.45, .68), (1.95, .54))]
        search_steps.VARIANTS = tuple(variants)
        search_steps.GROUP_SIZE = len(variants) + 2

        class ReferenceSideStepSearch(parent):
            def action(self, frame, onset, observation, base):
                torch, env = self.torch, self.env
                age = (frame - onset) * env.dt
                ids = ((onset >= 0) & (age > 0.) & (age < .85) & (self.variant_ids >= 0)).nonzero().flatten()
                if not len(ids):
                    return base
                age = age[ids]
                speed, phase, cadence, duration, strength, _ = self.params[ids].unbind(-1)
                _, left = horizontal_axes(env.root[ids, 3:])
                sign = (left * self.push_xy[ids]).sum(-1).sign()
                command = torch.zeros((len(ids), 3), device=env.device)
                command[:, 1] = sign * speed
                blend_in = (age / .10).clamp(0., 1.)
                blend_out = ((duration - age) / .25).clamp(0., 1.)
                weight = (blend_in * blend_out) * strength
                staged_pose = sample_vectorized(env.motion, phase + cadence * age, command, env.cfg,
                                                 joint_positions_only=True)['q'][:, env.q_order]
                still_pose = env.sample_reference(env.phase[ids], ids, joint_positions_only=True)['q'][:, env.q_order]
                difference = (staged_pose - still_pose) / env.cfg['imitation_residual_scale']
                recovery = base[ids][:, self.lower] + weight[:, None] * difference[:, self.lower]
                answer = base.clone()
                answer[ids[:, None], self.lower] = recovery.clamp(-2., 2.)
                return answer

        search_steps.StepSearch = ReferenceSideStepSearch
    else:
        search_steps.StepSearch = SideStepSearch
    rng = np.random.default_rng(args.seed)
    args.designs = [dict(case='stand', phase=(phase + rng.uniform(.03, .68)) / args.onsets,
                         direction=float(heading + rng.uniform(-9, 9)), strength=strength)
                    for strength in args.strengths
                    for heading in (90., 270.) for phase in range(args.onsets)]
    args.output.mkdir(parents=True, exist_ok=False)
    search_steps.run(args)


if __name__ == '__main__':
    main()
