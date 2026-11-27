"""Replay identical strong-push trials for round six and the corrective-step continuation."""

LARGE_PUSHES = (.25, .30, .35, .40, .525)
TRIAL_KEYS = ('trial_id', 'case', 'phase_bin', 'actual_phase', 'onset_seconds', 'direction',
              'impulse_m_s', 'impulse_Ns', 'peak_force_N', 'control_id')


def visibly_unsettled(row, control):
    tilt = float(row['post_max_projected_gravity_z'])
    control_tilt = float(control['post_max_projected_gravity_z'])
    height_loss = float(control['post_min_pelvis_z_m']) - float(row['post_min_pelvis_z_m'])
    return (row['fell'] or height_loss >= .07 or float(row['max_lateral_deviation_m']) >= .35
            or (tilt >= -.906307787 and tilt - control_tilt >= .035))


CASE_LABEL = {'walk_backward': '后退', 'walk_turn_left': '前走左转',
              'walk_forward': '前走', 'walk_left': '左移', 'walk_right': '右移',
              'stand': '站立', 'run_forward': '前跑'}
DIRECTION_LABEL = {'front': '前', 'front_left': '左前', 'left': '左', 'back_left': '左后',
                   'back': '后', 'back_right': '右后', 'right': '右', 'front_right': '右前'}


def shot_plan(source, candidate):
    pairs = []
    for old, new in zip(source, candidate, strict=True):
        if any(old[key] != new[key] for key in TRIAL_KEYS):
            raise ValueError('Mismatched acceptance trials in video')
        if old['impulse_m_s'] in LARGE_PUSHES and not old['precondition_fell'] and not new['precondition_fell']:
            severe = visibly_unsettled(new, candidate[int(new['control_id'])])
            pairs.append((old, new, severe))

    picks = []
    used = set()

    def select(matcher, title, strength, case=None, *, order=None):
        eligible = [pair for pair in pairs if pair[0]['trial_id'] not in used
                    and (case is None or pair[0]['case'] == case) and matcher(*pair)]
        if not eligible:
            return False
        old, new, severe = min(eligible, key=lambda pair: (
            abs(pair[0]['impulse_m_s'] - strength), not pair[2],
            order(*pair[:2]) if order else -float(pair[1]['max_lateral_deviation_m']),
            int(pair[0]['trial_id'])))
        used.add(old['trial_id'])
        picks.append(dict(trial_id=int(old['trial_id']), title=f'{CASE_LABEL[old["case"]]} · {title}',
                          direction_cn=DIRECTION_LABEL[old['direction']], impulse_m_s=old['impulse_m_s'],
                          expected_recovery=[old['recovered'], new['recovered']],
                          candidate_disturbance_visible=severe,
                          candidate_recovery_seconds=new['recovery_seconds']))
        return True

    for case in ('walk_backward', 'walk_turn_left'):
        if not select(lambda a,b,s:a['fell'] and b['recovered'], '续训版救回', .35, case):
            select(lambda a,b,s:s and b['recovered'], '受推恢复', .35, case)
    larger_tilt = lambda a,b: -float(b['post_max_projected_gravity_z'])
    closer_motion = lambda a,b: max(float(row['max_lateral_deviation_m']) for row in (a,b))
    if not select(lambda a,b,s:a['impulse_m_s'] == .525 and a['fell'] and b['recovered'],
                  '强推后救回', .525, order=larger_tilt):
        select(lambda a,b,s:a['impulse_m_s'] == .525, '强推对照', .525, order=closer_motion)
    select(lambda a,b,s:a['fell'] and b['fell'], '两版均失稳', .525, order=closer_motion)
    select(lambda a,b,s:a['recovered'] and b['fell'], '续训版失稳', .40, order=closer_motion)
    while len(picks) < 5 and select(lambda a,b,s:True, '同条件复测', .35):
        pass
    if len(picks) != 5:
        raise ValueError('Expected five distinct matched trials')
    return picks

