"""Build per-year windowed event panels (news + placebo windows, 6 assets)."""
import json, time, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
from infodiff.events.windows import WindowSpec, build_windows, save_eventdata
from infodiff.sources.expectations import mark_vector

KINDS = ["CPI", "NFP", "PPI", "RETAIL", "GDP", "CLAIMS", "FOMC", "FOMCPC", "ECB", "ECBPC", "BOJ"]
ASSETS = ["EURUSD", "USDJPY", "XAUUSD", "SPX", "BTC", "ETH"]
cal = pd.read_parquet("data/processed/calendar_surprises.parquet")
deltas = json.load(open("configs/deltas_bps.json"))
card = {}
for year in range(2022, 2027):
    t = time.time()
    sub = cal[cal.t_utc.dt.year == year]
    spec = WindowSpec(kinds=KINDS, assets=ASSETS, deltas_bps=deltas)
    data = build_windows(sub, spec, marks_fn=mark_vector, verbose=False)
    save_eventdata(data, Path(f"data/processed/panel_{year}.pkl"))
    pl = sum(m["placebo"] for m in data.meta["windows"])
    card[year] = dict(windows=int(data.n_windows), placebo=int(pl), news=int(data.n_windows - pl),
                      dropped=int(data.meta["dropped"]), events=int(len(data.times)),
                      per_dim=data.counts().tolist())
    print(year, card[year], f"{time.time() - t:.0f}s", flush=True)
Path("results/json").mkdir(parents=True, exist_ok=True)
Path("results/json/data_card.json").write_text(json.dumps(dict(assets=ASSETS, kinds=KINDS, deltas_bps=deltas, years=card), indent=1))
