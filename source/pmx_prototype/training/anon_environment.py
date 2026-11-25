"""Construct the Anon environment after Kit starts."""


def create_environment(num_envs=2048,seed=42,app=None,evaluation=False,config=None):
    from anon_lab_direct_env import AnonDirectEnv, AnonLabRunnerEnv
    return AnonLabRunnerEnv(AnonDirectEnv(num_envs,seed,app,evaluation=evaluation,config=config or {}))
