# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Build the anonymised supplementary-code archive for double-blind review.

    python scripts/make_supplement.py   ->  dist/supplement/  and  dist/supplementary_code.zip

The package is renamed (infodiff -> ptpp) because the original name matches a public
repository; author names, e-mail addresses, local paths and repository links are removed, and
outputs that the paper writes into manuscript/ go to results/ instead. A final scan fails the
build if any identifying string survives.
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
STAGE = DIST / "supplement"
PKG = "ptpp"

SCRIPTS = [  # what a reviewer needs to reproduce the paper (data, experiments, tables, figures)
    "fetch_easytpp.py", "download_ticks.py", "build_panel.py", "select_l1.py", "run_main.py", "run_track_d.py",
    "run_track_a.py", "summarize_track_a.py", "run_track_b_classical.py", "run_track_c_classical.py",
    "run_ept.py", "run_ept_seed.py", "run_ept_ablation.py", "run_ept_hpo.py", "make_tables.py", "make_figures.py",
    "gpu_queue.py", "wait_for_job.py", "bench_parallel.py", "channel_attribution.py",
]
FORBIDDEN = [r"sayak", r"dutta", r"itz-sayak", r"E87321", r"airamatrix", r"aiit\.com", r"infodiff",
             r"D:[/\\]+Quant", r"C:[/\\]+Users", r"gmail", r"github\.com/itz"]


def scrub(text: str) -> str:
    text = text.replace("# Copyright 2026 Sayak Dutta", "# Copyright 2026 Anonymous Authors")
    text = text.replace("Mozilla/5.0 (research; infodiff)", "Mozilla/5.0 (research)")
    text = text.replace("infodiff-research/0.1", "research/0.1")
    text = re.sub(r"\binfodiff\b", PKG, text)
    # paper outputs: keep them inside the archive
    text = text.replace('ROOT / "manuscript" / "tables"', 'ROOT / "results" / "tables"')
    text = text.replace('(ROOT / "manuscript" / "numbers.tex")', '(ROOT / "results" / "tables" / "numbers.tex")')
    text = text.replace('ROOT / "manuscript" / "figures"', 'ROOT / "results" / "figures"')
    return text


def copy_text(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(scrub(src.read_text(encoding="utf-8")), encoding="utf-8")


README = """# Exact phase-type point processes: supplementary code

Anonymous supplementary material for the TMLR submission *Exact Phase-Type Point Processes for
Certified Measurement and Prediction of Market Event Streams*.

## Contents
| Path | What |
|---|---|
| `src/ptpp/models/` | MSX (`msx.py`: certified EM/Newton estimator), state-space analytics (`statespace.py`), phase-type dictionary, EPT/EPT-X (`neural_pt.py`) |
| `src/ptpp/experiments/` | synthetic recovery, public benchmarks (`track_b.py`), order books (`track_c.py`), macro study, forecasting |
| `src/ptpp/baselines/` | classical Hawkes baselines (tick), NPHC, EasyTPP runner |
| `src/ptpp/sources/`, `src/ptpp/events/` | data pipelines (release calendar, ticks, LOBSTER, delta-crossing events, windows) |
| `scripts/` | data download, experiments, table and figure generation |
| `tests/` | unit tests (exact compensators vs quadrature, parallel scan vs event loop, causality, nesting, warm start, certificate) |
| `results/json/` | logged results of every run reported in the paper |
| `patches/` | EasyTPP patches: equal-length batching fix and the configurable IntensityFree clamp |
| `configs/` | selected configurations |

## Install
```bash
python -m venv .venv && source .venv/bin/activate        # Python >= 3.11
pip install -e .[dev]                                     # torch, numpy, scipy, numba, ...
pip install tick                                          # optional: classical Hawkes baselines
pytest                                                    # all tests should pass
```

## Reproduce the tables from the logged results (no data needed)
```bash
python scripts/make_tables.py        # writes results/tables/*.tex and results/tables/numbers.tex
```

## Re-run the experiments
**Public benchmarks (EasyTPP).**
```bash
python scripts/fetch_easytpp.py                                   # Hugging Face: easytpp/<name>
python scripts/run_ept_seed.py taxi 0 eptx '{"n_rates":8,"phases":4,"epochs":300,"renewal":true}'
```
The three components added in the final version are switched on per run (configurations selected
on validation are listed in the paper's appendix):
```bash
# residual mark law (Taxi: with dropout 0.1)
python scripts/run_ept_seed.py taxi 0 eptx_resdo '{"n_rates":8,"phases":4,"epochs":300,"patience":40,"renewal":true,"mark_head":"residual","dropout":0.1}'
# certified start from the MSX fit (reads results/json/track_b_classical_<ds>.json for the dictionary)
python scripts/run_ept_seed.py taobao 0 eptx_msx '{"n_rates":8,"phases":4,"epochs":300,"patience":40,"renewal":true,"init_from_msx":true}'
# data-adaptive (quantile) atoms + residual marks + certified start
python scripts/run_ept_seed.py amazon 0 eptx_qrm '{"n_rates":8,"phases":4,"epochs":300,"patience":40,"renewal":true,"rn_shift":true,"rn_orders":[1,4,16,64,256,1024],"rn_scales":96,"rn_quantile":96,"mark_head":"residual","init_from_msx":true}'
```
Training uses the parallel scan by default (`EPTConfig.parallel=True`); the event loop is kept as
the reference and `tests/test_ept.py` checks that both give the same likelihood and gradients.
```bash
python scripts/bench_parallel.py lob_aapl '{"n_rates":8,"phases":2,"renewal":true,"rn_shift":true,"rn_orders":[1,4,16,64,256,1024]}' cuda
python scripts/channel_attribution.py taxi:eptx_resdo taobao:eptx_msx      # exact channel shares
```
**Order books (LOBSTER).** Download the free level-1 sample files for AAPL, AMZN, GOOG, INTC and
MSFT (21 June 2012) from https://lobsterdata.com/info/DataSamples.php and save the message files as
`data/raw/lobster/<TICKER>_message_1.csv` (e.g. `AAPL_message_1.csv`). Then
```bash
python -c "from ptpp.sources.lobster import build; build()"     # data/processed/tpp/lob_*.jsonl
python scripts/run_track_c_classical.py aapl,amzn,goog,intc,msft
python scripts/run_ept_seed.py lob_aapl 0 eptx_sharp '{"n_rates":8,"phases":2,"renewal":true,"rn_shift":true,"rn_orders":[1,4,16,64,256,1024]}'
```
Neural baselines run in EasyTPP after applying the patches in `patches/`
(`python -m ptpp.baselines.easytpp_runner <Model> <dataset> <n_types> <seed> <epochs> <batch>`; set
`IFTPP_MIN_DT=1e-12` to score IntensityFree on the unclamped gaps).

**Synthetic recovery.** `python scripts/run_track_a.py` (simulated; no data needed).

**Macro-news application.** Requires a FRED API key in `.env` (`FRED_API_KEY=...`) and tick data
from HistData and Binance: `scripts/download_ticks.py`, `scripts/build_panel.py`,
`scripts/select_l1.py`, `scripts/run_main.py`, `scripts/run_track_d.py`. Market data are licensed
from their providers and are not redistributed.

Runs are resumable (per-epoch checkpoints in `results/ckpt/`). `scripts/gpu_queue.py` runs a job
file one job at a time.

## License
Code: Apache-2.0 (see `LICENSE`).
"""

PYPROJECT = """[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "ptpp"
version = "0.1.0"
description = "Exact phase-type point processes: certified Hawkes estimation, echo-corrected responses and EPT-X"
requires-python = ">=3.11"
license = "Apache-2.0"
dependencies = ["numpy", "scipy", "numba", "pandas", "polars", "pyarrow", "torch", "requests", "python-dotenv",
                "statsmodels", "matplotlib", "scikit-learn"]

[project.optional-dependencies]
dev = ["pytest"]
baselines = ["tick"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for f in (ROOT / "src" / "infodiff").rglob("*.py"):
        copy_text(f, STAGE / "src" / PKG / f.relative_to(ROOT / "src" / "infodiff"))
    for f in (ROOT / "tests").glob("*.py"):
        copy_text(f, STAGE / "tests" / f.name)
    for name in SCRIPTS:
        copy_text(ROOT / "scripts" / name, STAGE / "scripts" / name)
    for f in (ROOT / "configs").glob("*"):
        copy_text(f, STAGE / "configs" / f.name)
    for f in (ROOT / "docker").glob("*.patch"):
        copy_text(f, STAGE / "patches" / f.name)
    for f in (ROOT / "results" / "json").rglob("*.json"):
        copy_text(f, STAGE / "results" / "json" / f.relative_to(ROOT / "results" / "json"))
    shutil.copy(ROOT / "LICENSE", STAGE / "LICENSE")
    (STAGE / "README.md").write_text(README, encoding="utf-8")
    (STAGE / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (STAGE / ".gitignore").write_text("data/\nresults/ckpt/\n__pycache__/\n*.egg-info/\n", encoding="utf-8")

    # anonymity scan: every text file, every forbidden pattern
    bad = []
    for f in STAGE.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".md", ".toml", ".json", ".patch", ".txt", ".yaml", ".sh", ""}:
            t = f.read_text(encoding="utf-8", errors="ignore")
            for pat in FORBIDDEN:
                if re.search(pat, t, re.I):
                    bad.append(f"{f.relative_to(STAGE)}: {pat}")
    if bad:
        print("ANONYMITY CHECK FAILED:\n  " + "\n  ".join(bad))
        sys.exit(1)

    zpath = DIST / "supplementary_code.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file():
                z.write(f, Path("supplementary_code") / f.relative_to(STAGE))
    print(f"anonymity check passed; {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
