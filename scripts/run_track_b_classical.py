"""MSX (ours) and classical Hawkes baselines on the public EasyTPP datasets (CPU).
    python scripts/run_track_b_classical.py taxi,taobao,...  -> results/json/track_b_classical_<ds>.json"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from infodiff.experiments.track_c import run
for ds in sys.argv[1].split(","):
    run(ds, Path(f"results/json/track_b_classical_{ds}.json"))
