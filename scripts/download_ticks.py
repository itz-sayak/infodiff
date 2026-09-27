"""Download all tick data for the study period (resumable).

    python scripts/download_ticks.py histdata
    python scripts/download_ticks.py binance
"""
import sys
from pathlib import Path

from infodiff.sources.ticks import binance_all, histdata_all

ROOT = Path(__file__).resolve().parents[1] / "data"
START, END = "2022-01-01", "2026-08-31"

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "histdata":
        histdata_all(["EURUSD", "USDJPY", "XAUUSD", "SPXUSD"], START[:7], END[:7], ROOT)
    elif what == "binance":
        binance_all(["BTCUSDT", "ETHUSDT"], START, END, ROOT)
    else:
        raise SystemExit(__doc__)
