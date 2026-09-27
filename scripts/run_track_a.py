"""Track A synthetic benchmark (5 scenarios x 5 seeds x 9 methods)."""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from infodiff.experiments.track_a import run

if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1] / "results" / "json")
