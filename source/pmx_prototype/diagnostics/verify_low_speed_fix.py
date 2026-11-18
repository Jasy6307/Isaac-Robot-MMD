"""Historical diagnostic entry; grouped implementation under diagnostics/analyses/."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from source.pmx_prototype.diagnostics.locomotion_tools import legacy_exports, main

if __name__ == '__main__':
    raise SystemExit(main(profile='verify_low_speed_fix'))
else:
    globals().update(legacy_exports('verify_low_speed_fix'))
