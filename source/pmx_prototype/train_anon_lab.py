"""Compatibility entry point for Anon's Isaac Lab training workflow."""
from source_layout import SOURCE_DIR
import runpy
import sys

if __name__=='__main__':
    entry=SOURCE_DIR/'train_anon_locomotion.py'
    sys.argv[0]=str(entry)
    runpy.run_path(str(entry),run_name='__main__')
