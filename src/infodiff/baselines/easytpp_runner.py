# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Run EasyTPP neural baselines (RMTPP, NHP, SAHP, THP, AttNHP, IntensityFree, MHP, S2P2)
with the S2P2 authors' own per-model configuration (examples/configs/exp_config_taxi.yaml
of github.com/UCIDataLab/state_space_point_process), changing only the dataset, seed,
GPU and epoch budget.  The reported test metrics are those at the epoch with the best
validation log-likelihood, following the protocol of Chang et al. (2025).
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BASE_CFG = ROOT / "external" / "state_space_point_process" / "examples" / "configs" / "exp_config_taxi.yaml"
MODELS = ["RMTPP", "NHP", "SAHP", "THP", "AttNHP", "IntensityFree", "MHP", "S2P2"]


def dataset_files(name: str) -> dict:
    """Return .json paths (EasyTPP's loader keys on the extension) for a dataset name."""
    out = {}
    for split, tag in (("train", "train_dir"), ("validation", "valid_dir"), ("test", "test_dir")):
        for base in (ROOT / "data" / "processed" / "tpp", ROOT / "data" / "raw" / "easytpp"):
            src = base / f"{name}_{split}.jsonl"
            if src.exists():
                dst = src.with_suffix(".json")
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())
                out[tag] = str(dst)
                break
    if len(out) != 3:
        raise FileNotFoundError(name)
    return out


def run_model(model: str, dataset: str, n_types: int, seed: int = 2019, max_epoch: int | None = None,
              gpu: int = 0, workdir: Path = ROOT / "results" / "easytpp", batch_size: int | None = None) -> dict:
    """batch_size overrides the authors' 256 when needed: on a WDDM laptop GPU long kernels trip
    the Windows TDR watchdog ('unspecified launch failure'); smaller batches keep kernels short."""
    from easy_tpp.config_factory import Config
    from easy_tpp.runner import Runner

    base = yaml.safe_load(BASE_CFG.read_text())
    exp = copy.deepcopy(base[f"{model}_train"])
    exp["base_config"]["dataset_id"] = dataset
    exp["base_config"]["base_dir"] = str(workdir / "checkpoints") + "/"
    exp["trainer_config"]["seed"] = seed
    exp["trainer_config"]["gpu"] = gpu
    exp["trainer_config"]["metrics"] = []  # LL only; RMSE/acc via thinning are costly and optional
    if max_epoch:
        exp["trainer_config"]["max_epoch"] = max_epoch
    if batch_size:
        exp["trainer_config"]["batch_size"] = batch_size
    files = dataset_files(dataset)
    cfg = {"pipeline_config_id": "runner_config",
           "data": {dataset: dict(data_format="json", **files,
                                  data_specs=dict(num_event_types=n_types, pad_token_id=n_types,
                                                  padding_side="right", truncation_side="right"))},
           f"{model}_train": exp}
    workdir.mkdir(parents=True, exist_ok=True)
    cfg_path = workdir / f"cfg_{model}_{dataset}_{seed}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    config = Config.build_from_yaml_file(str(cfg_path), experiment_id=f"{model}_train")
    runner = Runner.build_from_config(config)
    t = time.time()
    log = runner.run()
    test = log["best_metrics"].get("test", {})
    import os
    if model == "IntensityFree" and "IFTPP_MIN_DT" in os.environ:
        model = f"IntensityFree (min dt {os.environ['IFTPP_MIN_DT']})"
    return dict(model=model, dataset=dataset, seed=seed, ll_per_event=test.get("loglike"),
                mark_ll=test.get("mark_ll"), time_ll=test.get("time_ll"),
                val_ll=log["best_valid_ll"], best_epoch=log["best_valid_epoch"],
                n_params=log["num_params"], seconds=time.time() - t,
                max_epoch=exp["trainer_config"]["max_epoch"], batch_size=exp["trainer_config"]["batch_size"])


if __name__ == "__main__":
    import sys
    model, dataset, n_types = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 2019
    max_epoch = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    bs = int(sys.argv[6]) if len(sys.argv) > 6 else None
    import torch
    r = run_model(model, dataset, n_types, seed, max_epoch, batch_size=bs, gpu=0 if torch.cuda.is_available() else -1)
    print("RESULT " + json.dumps(r), flush=True)
    out = ROOT / "results" / "json" / "easytpp_results.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    rows.append(r)
    out.write_text(json.dumps(rows, indent=1))
