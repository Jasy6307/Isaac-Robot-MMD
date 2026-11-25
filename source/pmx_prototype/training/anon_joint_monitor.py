"""Validated display data and simulation-time history; no control writes."""
from collections import deque
import numpy as np

GROUPS = ('Torso / Neck', 'Left leg', 'Right leg', 'Left arm', 'Right arm', 'Other')
MOTION_LEVELS = ('Still', 'Gentle', 'Moving', 'Fast')
MOTION_BAR_SPEED = 180.  # Shared display scale in degrees/second, not motor effort.
PARTS = dict(waist_pitch='Waist pitch', waist_roll='Waist roll', waist_yaw='Waist yaw',
             neck_pitch='Neck pitch', neck_yaw='Neck yaw', shoulder_pitch='shldr pitch',
             shoulder_roll='shldr roll', shoulder_yaw='shldr yaw', elbow_flex='elbow flex',
             forearm_twist='forearm rot', wrist_pitch='wrist pitch', wrist_deviation='wrist dev',
             hip_yaw='hip yaw', hip_roll='hip roll', hip_pitch='hip pitch',
             knee_pitch='knee flex', ankle_pitch='ankle pitch', ankle_roll='ankle roll')


def joint_label(name):
    part = name.removesuffix('_joint')
    side = ''
    for prefix, label in (('left_', 'L '), ('right_', 'R ')):
        if part.startswith(prefix):
            part, side = part[len(prefix):], label
            break
    return side + PARTS.get(part, part)


def joint_group(name):
    if name.startswith(('waist_', 'neck_')):return GROUPS[0]
    if name.startswith(('left_', 'right_')):
        side = 'Left' if name.startswith('left_') else 'Right'
        return side + (' leg' if any(p in name for p in ('_hip_', '_knee_', '_ankle_')) else ' arm')
    return GROUPS[-1]


def joint_rows(names):
    """Pair matching left/right axes without assuming PhysX joint order."""
    groups = {key: {} for key in ('Legs', 'Arms', 'Torso / Neck', 'Other')}
    for name in names:
        group = joint_group(name)
        side = 0 if name.startswith('left_') else 1 if name.startswith('right_') else None
        category = 'Legs' if group.endswith('leg') else 'Arms' if group.endswith('arm') else group
        key = name.split('_', 1)[1] if side is not None else name
        rows = groups[category]
        if key not in rows:rows[key] = [joint_label(name)[2:] if side is not None else joint_label(name), None, None]
        rows[key][1 if side is None else side+1] = name
    return [(group, list(rows.values())) for group, rows in groups.items() if rows]


PART_AXES = dict(hip=('roll','pitch','yaw'), knee=('pitch',), ankle=('roll','pitch'),
                 shoulder=('roll','pitch','yaw'), elbow=('flex',), forearm=('twist',),
                 wrist=('pitch','deviation'), waist=('roll','pitch','yaw'), neck=('pitch','yaw'))


def joint_part(name):
    """Only combine recognized axes of the same anatomical part and side."""
    base = name.removesuffix('_joint')
    if base.startswith(('left_', 'right_')):base = base.split('_', 1)[1]
    part, _, axis = base.rpartition('_')
    if part in PART_AXES and axis in PART_AXES[part]:
        return part, part.title(), axis.title()
    label = joint_label(name)
    if name.startswith(('left_', 'right_')):label = label[2:]
    return base, label, 'Axis'


def joint_part_rows(names):
    groups = {}
    for group, rows in joint_rows(names):
        parts = {}
        for _, left, right in rows:
            name = left or right
            key, label, _ = joint_part(name)
            if key not in parts:parts[key] = [label, [], []]
            if left:parts[key][1].append(left)
            if right:parts[key][2].append(right)
        order = ('Roll','Pitch','Yaw','Flex','Twist','Deviation','Axis')
        groups[group] = [(label, tuple(sorted(left, key=lambda n:order.index(joint_part(n)[2]))),
                         tuple(sorted(right, key=lambda n:order.index(joint_part(n)[2]))))
                        for label,left,right in parts.values()]
    return list(groups.items())


class JointMonitorState:
    def __init__(self, seconds=5.):
        self.seconds = seconds
        self.names = ()
        self.latest = None
        self.history = deque(maxlen=2048)
        self.error = 'Waiting for joint data'
        self.identity = None

    def ingest(self, state):
        if 'joint_state_valid' not in state:return False
        try:
            if not bool(state['joint_state_valid']):
                raise ValueError(str(state.get('joint_state_error', 'Joint data unavailable')))
            names = tuple(str(n) for n in state['joint_names'])
            n = len(names)
            if not n or len(set(names)) != n:raise ValueError('Empty or duplicate joint names')
            columns = [np.array(state[key], dtype=np.float64, copy=True) for key in (
                'joint_positions', 'joint_velocities', 'joint_position_targets',
                'joint_velocity_targets', 'joint_actions')]
            limits = np.array(state['joint_limits'], dtype=np.float64, copy=True)
            mask = np.array(state['joint_action_mask'], dtype=bool, copy=True)
            timestamp = float(state['joint_sample_time'])
            if (any(x.shape != (n,) or not np.isfinite(x).all() for x in columns)
                    or limits.shape != (n, 2) or not np.isfinite(limits).all()
                    or np.any(limits[:, 1] <= limits[:, 0]) or mask.shape != (n,)
                    or not np.isfinite(timestamp)):
                raise ValueError('Invalid joint data shape, limits or values')
            identity = (str(state.get('checkpoint', '')), str(state.get('generation', '')), names)
            if (identity != self.identity or bool(state.get('joint_state_reset', False))
                    or (self.history and timestamp < self.history[-1][0])):
                self.history.clear()
            self.names, self.identity = names, identity
            q, qd, target, target_velocity, action = columns
            span = limits[:, 1] - limits[:, 0]
            error = target - q
            margin = np.minimum(q - limits[:, 0], limits[:, 1] - q)
            self.latest = dict(q=np.rad2deg(q), qd=np.rad2deg(qd), target=np.rad2deg(target),
                error=np.rad2deg(error), action=action, action_mask=mask,
                limits=np.rad2deg(limits), limit_position=np.clip((q-limits[:, 0])/span, 0., 1.),
                near_limit=margin <= .05*span, outside_limit=margin < 0.,
                error_fraction=np.abs(error)/span, margin=np.rad2deg(margin))
            if not self.history or timestamp > self.history[-1][0]:
                self.history.append((timestamp, self.latest['target'].copy(), self.latest['q'].copy(),
                                     self.latest['error'].copy(), self.latest['qd'].copy()))
            while len(self.history)>1 and timestamp-self.history[0][0]>self.seconds:
                self.history.popleft()
            self.error = ''
            return True
        except (KeyError, TypeError, ValueError) as error:
            self.latest = None
            self.history.clear()
            self.error = str(error)
            return False

    def motion(self):
        """Actual speed averaged over 0.3 s, and actual angle range over 1 s."""
        if self.latest is None or not self.history:return None
        now = self.history[-1][0]
        speed = np.mean([np.abs(row[4]) for row in self.history if now-row[0] <= .3+1e-9], axis=0)
        swing = np.ptp([row[2] for row in self.history if now-row[0] <= 1.+1e-9], axis=0)
        return dict(speed=speed, swing=swing, level=np.digitize(speed, [3., 30., 90.]),
                    fill=np.clip(speed/MOTION_BAR_SPEED, 0., 1.))

    def motion_summary(self):
        motion = self.motion()
        if motion is None:return self.error
        index = int(motion['speed'].argmax())
        parts = [side for _,rows in joint_part_rows(self.names) for _,left,right in rows for side in (left,right) if side]
        moving = sum(any(motion['level'][self.names.index(n)]>0 for n in side) for side in parts)
        return (f"Moving {moving}/{len(parts)} joints ({len(self.names)} axes)\n"
                f"Most active axis: {joint_label(self.names[index])}  {motion['speed'][index]:.0f}°/s")

    def part_motion(self, names, motion=None):
        """Fastest measured axis; not a vector norm or anatomical angular speed."""
        motion = self.motion() if motion is None else motion
        if motion is None:return None
        indices = [self.names.index(name) for name in names]
        index = max(indices, key=lambda i:motion['speed'][i])
        return dict(dominant=self.names[index], speed=float(motion['speed'][index]),
                    fill=float(motion['fill'][index]), level=int(motion['level'][index]))

    def summary(self):
        if self.latest is None:return self.error
        index = int(np.abs(self.latest['error']).argmax())
        return (f"Max error {abs(self.latest['error'][index]):.1f}°  {joint_label(self.names[index])}"
                f"   |   Near limit {int(self.latest['near_limit'].sum())}/{len(self.names)}")

    def series(self, index):
        if not self.history:return [], [], [], 0.
        values = [[float(row[column][index]) for row in self.history] for column in (1, 2, 3)]
        if len(self.history)==1:values=[v*2 for v in values]
        return (*values, self.history[-1][0]-self.history[0][0])
