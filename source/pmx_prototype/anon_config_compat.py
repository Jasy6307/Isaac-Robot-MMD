"""Read historical null settings, but reject removed active compensation."""

def clean_legacy_config(config):
    result=dict(config)
    if result.pop('runtime_arm_clearance',None):
        raise ValueError('Runtime arm compensation has been removed. Remove the legacy runtime_arm_clearance override; use a policy trained with spatial clearance.')
    return result
