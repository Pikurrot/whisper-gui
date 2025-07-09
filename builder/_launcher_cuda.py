from pathlib import Path
import json
import sys
import runpy

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # running from a PyInstaller bundle
    BASE = Path(sys._MEIPASS)          # unpacked temp dir
else:
    # running from source checkout
    BASE = Path(__file__).resolve().parent.parent  # repo root

CFG = BASE / "configs" / "config.json"

if not CFG.exists():
    CFG.parent.mkdir(parents=True, exist_ok=True)
    CFG.write_text(json.dumps({"gpu_support": "cuda"}, indent=2))

# 'main' is bundled, but add BASE to sys.path for the non-frozen case
sys.path.insert(0, str(BASE))

sys.argv = ["main.py", "--autolaunch"]
runpy.run_module("main", run_name="__main__")
