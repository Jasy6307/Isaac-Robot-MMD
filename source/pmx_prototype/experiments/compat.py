"""Load legacy controller namespaces, keeping recipe mutation isolated."""
import importlib
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'source/pmx_prototype'


def controller_module(name):
    for directory in (SOURCE, SOURCE/'training', SOURCE/'diagnostics'):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    if name == 'run_transition_retention':
        # The old entry mutated the imported default workflow globally. Give
        # this recipe its own module so importing it cannot change another run.
        path = Path(__file__).parent/'workflows/continue_sampling_pivot.py'
        spec = importlib.util.spec_from_file_location(__package__+'.transition_recipe', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.OUT = ROOT/'output/anon_locomotion/transition_retention_20261116'
        module.CONFIG = module.OUT/'prepared_config.json'
        module.STAGE_ITERATIONS = [100, 100, 100]
        module.CUMULATIVE_BASE = 48054.58603930008
        module.EXACT_PREFIX_CONFIG = ROOT/'output/anon_locomotion/reverse_turn_diagnosis_20261116/prefix_config.json'
        return module
    return importlib.import_module(__package__+'.workflows.'+name)


def legacy_main(module):
    if '--help' in sys.argv or '-h' in sys.argv:
        print(module.__doc__ or '')
        print('Historical recipe. List/preview with: python -m source.pmx_prototype.experiments list')
        return 0
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    try:
        module.main()
    except Exception as error:
        output = getattr(module, 'OUT', None)
        if output is not None:
            status = 'controller_status.json' if module.__name__.endswith(('run_split_scratch', 'resume_split_scratch', 'limit_split_scratch', 'compare_anon_parallelism')) else 'status.json'
            module.write(output/status, dict(stage='error', error=str(error)))
        raise
    return 0
