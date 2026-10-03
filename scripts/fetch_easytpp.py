# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Download the EasyTPP benchmark splits from the Hugging Face Hub (datasets easytpp/<name>)
into data/raw/easytpp/<name>_{train,validation,test}.jsonl, the layout read by load_split.

    python scripts/fetch_easytpp.py [taxi,taobao,stackoverflow,amazon,retweet]
"""
import json
import sys
from pathlib import Path

import requests

NAMES = ["taxi", "taobao", "stackoverflow", "amazon", "retweet"]
SPLITS = {"train": "train", "dev": "validation", "test": "test"}
OUT = Path("data/raw/easytpp")


def fetch(name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src, dst in SPLITS.items():
        target = OUT / f"{name}_{dst}.jsonl"
        if target.exists():
            print("exists", target)
            continue
        url = f"https://huggingface.co/datasets/easytpp/{name}/resolve/main/{src}.json"
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        seqs = r.json()
        with open(target, "w") as f:
            for s in seqs:
                f.write(json.dumps({k: s[k] for k in ("dim_process", "seq_len", "time_since_start",
                                                       "time_since_last_event", "type_event")}) + "\n")
        print(f"{target}: {len(seqs)} sequences")


if __name__ == "__main__":
    for n in (sys.argv[1].split(",") if len(sys.argv) > 1 else NAMES):
        fetch(n)
