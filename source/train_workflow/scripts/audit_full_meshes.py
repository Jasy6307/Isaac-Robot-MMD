"""Compatible FCL audit entry; shared implementation in utils/motion/mesh_audit.py."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from source.train_workflow.utils.motion.mesh_audit import main

if __name__ == '__main__':
    raise SystemExit(main(default_scope='all'))
