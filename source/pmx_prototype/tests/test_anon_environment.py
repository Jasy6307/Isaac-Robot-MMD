"""Lab setup stays lazy and accepts saved Anon configurations."""
from pathlib import Path
import runpy
import sys
from types import ModuleType
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock,patch

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import source_layout
from anon_environment import create_environment


class EnvironmentSelectionTests(unittest.TestCase):
    def test_all_saved_configs_use_lab_without_mutating_inputs(self):
        module=ModuleType('anon_lab_direct_env')
        module.AnonDirectEnv=Mock(return_value='environment')
        module.AnonLabRunnerEnv=Mock(return_value='wrapper')
        for metadata in ({}, {'environment_backend':'native'}, {'environment_backend':'lab-direct'}):
            with self.subTest(metadata=metadata), patch.dict(sys.modules,{'anon_lab_direct_env':module}):
                config=dict(metadata,full_motion_path='reference.npz')
                self.assertEqual(create_environment(8,77,'app',True,config),'wrapper')
                module.AnonDirectEnv.assert_called_with(8,77,'app',evaluation=True,config=config)
                module.AnonLabRunnerEnv.assert_called_with('environment')
                self.assertEqual(config,dict(metadata,full_motion_path='reference.npz'))

    def test_lab_entry_forwards_training_and_validation_options(self):
        run=runpy.run_path
        entry=SOURCE/'train_anon_lab.py'
        with patch.object(sys,'argv',[str(entry),'--mode','eval','--checkpoint','policy.pt']), \
             patch('runpy.run_path') as player:
            observed=[]
            player.side_effect=lambda *a,**k:observed.extend(sys.argv)
            run(str(entry),run_name='__main__')
        self.assertEqual(observed[1:],['--mode','eval','--checkpoint','policy.pt'])

    def test_lab_training_refuses_existing_run_before_starting_kit(self):
        entry=SOURCE/'train_anon_locomotion.py'
        with TemporaryDirectory() as directory:
            config=Path(directory)/'env_config.json'
            config.write_bytes(b'preserve existing experiment')
            with patch.object(sys,'argv',[str(entry),'--run-dir',directory]), \
                 patch('argparse.ArgumentParser.error',side_effect=SystemExit(2)) as error:
                with self.assertRaises(SystemExit):runpy.run_path(str(entry),run_name='__main__')
            self.assertIn('fresh --run-dir',error.call_args.args[0])
            self.assertEqual(config.read_bytes(),b'preserve existing experiment')


if __name__=='__main__':unittest.main()
