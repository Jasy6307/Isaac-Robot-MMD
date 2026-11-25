"""Names shared by G1 dance training and evaluation."""

G1_TRAIN_TASK_ID = "Isaac-G1-Dance-Tracking-v0"
G1_TRAIN_EXPERIMENT_NAME = "g1_dance_tracking"
_FORMER_G1_TRAIN_TASK_ID = "Isaac-G1-Vmd-Auto-Jump-Conditional-v0"


def normalize_g1_train_task_id(task_id: str) -> str:
    """Accept the former CLI name for the same registered task."""
    return G1_TRAIN_TASK_ID if task_id == _FORMER_G1_TRAIN_TASK_ID else task_id
