"""The reduced preview is explicit and never changes another actor's defaults."""
import json, os, subprocess, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
ENTRY=ROOT/'source/pmx_prototype/tools/preview_dance.py'

class PhysicsDetailTest(unittest.TestCase):
    def run_preview(self,*args):
        return subprocess.run([sys.executable,str(ENTRY),'--dry-run',*args],cwd=ROOT,
                              env=dict(os.environ,PYTHONIOENCODING='utf-8'),capture_output=True,text=True)
    def test_full_default_and_reduced_opt_in(self):
        full=self.run_preview('--model','anon','--dance','girls')
        reduced=self.run_preview('--model','anon','--dance','girls','--physics-detail','reduced')
        self.assertEqual(full.returncode,0,full.stderr);self.assertEqual(reduced.returncode,0,reduced.stderr)
        a=json.loads(full.stdout)['arguments'];b=json.loads(reduced.stdout)['arguments']
        self.assertNotIn('--skirt-edge-stride',a)
        self.assertEqual(b[b.index('--skirt-edge-stride')+1],'2')
        for option in ('--usd','--pmx','--secondary-profile','--look','--quality'):
            self.assertEqual(a[a.index(option)+1],b[b.index(option)+1])
    def test_reduced_rejected_for_other_models(self):
        result=self.run_preview('--model','mutsumi','--physics-detail','reduced')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('only configured for Anon',result.stderr)

    def test_interpolated_sparse_is_explicit(self):
        result=self.run_preview('--model','anon','--dance','girls','--physics-detail','sparse','--render-fps','60','--interpolate-60')
        self.assertEqual(result.returncode,0,result.stderr)
        args=json.loads(result.stdout)['arguments']
        self.assertEqual(args[args.index('--skirt-edge-stride')+1],'3')
        self.assertIn('--interpolate-60',args)
        invalid=self.run_preview('--model','anon','--interpolate-60')
        self.assertNotEqual(invalid.returncode,0)

if __name__=='__main__':unittest.main()
