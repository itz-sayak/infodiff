# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Download reference papers into paper/ and print arXiv titles so IDs can be verified."""
from __future__ import annotations

import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"

# (filename stem, arXiv id); titles are printed after download for verification.
ARXIV = {
    "rambaldi2015_fx_macro_news_hawkes": "1405.6047",
    "bacry2015_hawkes_in_finance": "1502.04592",
    "filimonov2012_quantifying_reflexivity": "1201.3572",
    "filimonov2015_apparent_criticality": "1308.6756",
    "hardiman2013_critical_reflexivity": "1302.1405",
    "bacry2016_slowly_decreasing_kernels": "1412.7096",
    "achab2017_nphc_cumulants": "1607.06333",
    "bacry2016_first_second_order_nonparam": "1401.0903",
    "bacry2012_nonparam_symmetric_hawkes": "1112.1838",
    "xu2016_granger_causality_hawkes": "1602.04511",
    "kirchner2017_inar_estimation": "1509.02017",
    "embrechts2018_hawkes_graphs": "1601.01879",
    "jaisson2015_nearly_unstable": "1310.2033",
    "omi2017_time_dependent_background": "1702.04443",
    "laub2015_hawkes_tutorial": "1507.02822",
    "huth2014_lead_lag": "1111.7103",
    "mei2017_neural_hawkes": "1612.09328",
    "zuo2020_transformer_hawkes": "2002.09291",
    "zhang2020_self_attentive_hawkes": "1907.07561",
    "shchur2020_intensity_free": "1909.12127",
    "omi2019_fullynn": "1905.09690",
    "yang2022_attnhp": "2201.00044",
    "chen2021_neural_stpp_odetpp": "2011.04583",
    "xue2023_easytpp": "2307.08097",
    "chang2024_s2p2_state_space_pp": "2412.19634",
    "gao2024_mamba_hawkes": "2407.05302",
    "boyd2025_hyper_hawkes": "2511.01096",
    "bosser2023_predictive_accuracy_neural_tpp": "2306.17066",
    "tpp_survey2025_bayes_neural_llm": "2501.14291",
    "hotpp2024_long_horizon": "2406.14341",
    "tsang2026_blockchain_prediction_market": "2603.03136",
    "cheng2026_polymarket_nba_arbitrage": "2605.00864",
}

DIRECT = {
    "zhou2013_adm4_icml": "http://proceedings.mlr.press/v28/zhou13.pdf",
    "andersen2003_micro_effects_macro_announcements": "https://www.nber.org/system/files/working_papers/w8959/w8959.pdf",
    "aitsahalia2015_contagion_mutually_exciting": "https://www.nber.org/system/files/working_papers/w15850/w15850.pdf",
    "du2016_rmtpp_kdd": "https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf",
}

UA = {"User-Agent": "infodiff-research/0.1 (academic literature fetch)"}


def arxiv_titles(ids: list[str]) -> dict[str, str]:
    url = "http://export.arxiv.org/api/query"
    r = requests.get(url, params={"id_list": ",".join(ids), "max_results": len(ids)}, headers=UA, timeout=60)
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = {}
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        aid = e.find("a:id", ns).text.rsplit("/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        out[aid] = " ".join(e.find("a:title", ns).text.split())
    return out


def fetch(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 10_000:
        return "cached"
    r = requests.get(url, headers=UA, timeout=120)
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        return f"FAILED ({r.status_code})"
    dest.write_bytes(r.content)
    time.sleep(1.0)
    return f"ok {len(r.content) // 1024} KB"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    titles = arxiv_titles(list(ARXIV.values()))
    for stem, aid in ARXIV.items():
        status = fetch(f"https://arxiv.org/pdf/{aid}", OUT / f"{stem}.pdf")
        print(f"{aid:12s} {status:14s} {stem:48s} | {titles.get(aid, '??')}")
    for stem, url in DIRECT.items():
        print(f"{'direct':12s} {fetch(url, OUT / f'{stem}.pdf'):14s} {stem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
