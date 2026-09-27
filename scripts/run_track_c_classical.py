import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from infodiff.experiments.track_c import run
for tk in sys.argv[1].split(","):
    run(f"lob_{tk}", Path(f"results/json/track_c_classical_lob_{tk}.json"))
