# Copyright 2026 Sayak Dutta
# SPDX-License-Identifier: Apache-2.0
"""Track B: public marked-TPP benchmarks (EasyTPP splits, S2P2 protocol).

Metric definitions follow EasyTPP / S2P2 (Chang et al. 2025):
  * log-likelihood per event = sum over sequences of [sum_{n>=1} log lambda_{k_n}(t_n)
    - int_{t_0}^{t_N} lambda(t) dt] / number of scored events (first event conditions);
  * next-event time RMSE and next-mark accuracy over positions n >= 1.
Our model uses the exact compensator (no Monte-Carlo) and Bayes-optimal predictions.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..models.neural_pt import EPTConfig, EPTTPP

RAW = Path("data/raw/easytpp")
PROCESSED = Path("data/processed/tpp")


def load_split(name: str, split: str):
    f = RAW / f"{name}_{split}.jsonl"
    if not f.exists():
        f = PROCESSED / f"{name}_{split}.jsonl"
    df = pd.read_json(f, lines=True)
    seqs = [(np.asarray(d, float), np.asarray(k, int)) for d, k in zip(df.time_since_last_event, df.type_event)]
    return seqs, int(df.dim_process.iloc[0])


def batches(seqs, n_marks: int, bs: int, shuffle: bool, device, rng=None):
    idx = np.arange(len(seqs))
    if shuffle:
        rng.shuffle(idx)
    else:
        idx = np.argsort([len(s[0]) for s in seqs])  # length bucketing for eval
    for a in range(0, len(idx), bs):
        chunk = [seqs[i] for i in idx[a:a + bs]]
        L = max(len(c[0]) for c in chunk)
        dts = np.zeros((len(chunk), L))
        mk = np.full((len(chunk), L), n_marks)
        msk = np.zeros((len(chunk), L), bool)
        for r, (d, k) in enumerate(chunk):
            dts[r, :len(d)] = d
            mk[r, :len(k)] = k
            msk[r, :len(d)] = True
        dts[:, 0] = 0.0
        yield (torch.tensor(dts, dtype=torch.float32, device=device), torch.tensor(mk, device=device),
               torch.tensor(msk, device=device))


def evaluate(model: EPTTPP, seqs, n_marks, device, predict: bool = False, s_max: float = None):
    model.eval()
    tot_ll = tot_n = tot_time = 0.0
    se = []
    acc = []
    with torch.no_grad():
        for dts, mk, msk in batches(seqs, n_marks, 256, False, device):
            log_lam, comp, states, mus = model.forward(dts, mk, msk)
            m = msk[:, 1:].float()
            tot_ll += float(((log_lam - comp) * m).sum())
            tot_time += float(((model._last_log_tot - comp) * m).sum())
            tot_n += float(m.sum())
            if predict:
                # predict event n from state after event n-1
                st = states[:, :-1].reshape(-1, states.shape[-1])
                mu = mus[:, :-1].reshape(-1, mus.shape[-1])
                sel = msk[:, 1:].reshape(-1)
                hh = model._last_h[:, :-1].reshape(-1, model._last_h.shape[-1])[sel]
                st, mu = st[sel], mu[sel]
                tgt_dt = dts[:, 1:].reshape(-1)[sel]
                tgt_mk = mk[:, 1:].reshape(-1)[sel]
                # chunk so that the (chunk, grid, state) tensors stay ~<= 0.4 GB on an 8 GB GPU
                width = st.shape[-1] + n_marks * (1 + (model.J if model.cfg.renewal else 0))
                if model.cfg.renewal:
                    width += model.J * int(max(model.cfg.rn_orders))
                ch = int(max(16, min(4096, 1e8 / (400 * width))))
                for a in range(0, st.shape[0], ch):
                    e_dt, p_mk, _ = model.predict_next(st[a:a + ch], mu[a:a + ch],
                                                       torch.full((min(ch, st.shape[0] - a),), s_max, device=device),
                                                       h=hh[a:a + ch])
                    se.append(((e_dt - tgt_dt[a:a + ch]) ** 2).cpu().numpy())
                    acc.append((p_mk == tgt_mk[a:a + ch]).float().cpu().numpy())
    out = dict(ll_per_event=tot_ll / tot_n, n_events=tot_n, time_ll=tot_time / tot_n,
               mark_ll=(tot_ll - tot_time) / tot_n)
    if predict:
        out["rmse"] = float(np.sqrt(np.concatenate(se).mean()))
        out["acc"] = float(np.concatenate(acc).mean())
    return out


def train_one(name: str, seed: int, hidden=64, n_rates=8, n_channels=4, phases=2, lr=1e-2, epochs=300, patience=40,
              bs=64, device=None, verbose=False, weight_decay=0.0, dropout=0.0, gompertz=True, warmup=0.01,
              renewal=False, rn_scales=24, rn_orders=(1, 4, 16), input_v2=None, ept_channel=True, rn_shift=False,
              n_layers=1, layer_norm=False, ckpt: str | None = None, encoder="gru") -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, M = load_split(name, "train")
    va, _ = load_split(name, "validation")
    te, _ = load_split(name, "test")
    gaps = np.concatenate([d[1:] for d, _ in tr])
    gaps = gaps[gaps > 0]
    tau_min, tau_max = float(np.quantile(gaps, 0.02)), float(np.quantile(gaps, 0.995)) * 5
    input_v2 = renewal if input_v2 is None else input_v2
    q001, q01, q999 = (float(np.quantile(gaps, q)) for q in (0.001, 0.01, 0.999))
    eps = 0.1 * q01
    lg = np.log(gaps + eps)
    cfg = EPTConfig(n_marks=M, hidden=hidden, n_rates=n_rates, n_channels=n_channels, tau_min=tau_min,
                    tau_max=tau_max, dropout=dropout, phases=phases, gompertz=gompertz,
                    renewal=renewal, rn_scales=rn_scales, rn_orders=tuple(rn_orders),
                    rn_lo=0.5 * q001, rn_hi=5 * q999, input_v2=input_v2, gap_eps=eps,
                    gap_mu=float(lg.mean()), gap_sd=float(lg.std() + 1e-6), tie_thr=10 * q01,
                    ept_channel=ept_channel, rn_shift=rn_shift, n_layers=n_layers, layer_norm=layer_norm,
                    encoder=encoder)
    model = EPTTPP(cfg).to(device)
    if renewal:  # renewal mark law starts at the empirical mark frequencies
        freq = np.bincount(np.concatenate([k[1:] for _, k in tr]), minlength=M) + 1.0
        with torch.no_grad():
            model.rn_qb.copy_(torch.log(torch.tensor(freq / freq.sum(), dtype=torch.float32))[None].expand_as(model.rn_qb))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * math.ceil(len(tr) / bs)
    n_warm = max(1, int(warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / n_warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, k / steps))))
    best, best_state, bad, start = -np.inf, None, 0, 0
    t0 = time.time()
    ck = Path(ckpt) if ckpt else None
    if ck is not None and ck.exists():  # resume an interrupted run exactly where it stopped
        s = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        best, best_state, bad, start = s["best"], s["best_state"], s["bad"], s["ep"] + 1
        rng.bit_generator.state = s["rng"]
        torch.set_rng_state(s["torch_rng"])
        t0 -= s["seconds"]
        print(f"  resumed {name} seed={seed} from epoch {start}", flush=True)
    ep = start - 1
    for ep in range(start, epochs):
        if bad >= patience:
            break
        model.train()
        for dts, mk, msk in batches(tr, M, bs, True, device, rng):
            ll, n = model.loglik(dts, mk, msk)
            loss = -ll / n
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        v = evaluate(model, va, M, device)["ll_per_event"]
        if v > best + 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if verbose and ep % 5 == 0:
            print(f"  {name} seed={seed} ep={ep} val_ll={v:.4f} best={best:.4f}", flush=True)
        if ck is not None:
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), ep=ep,
                            best=best, best_state=best_state, bad=bad, rng=rng.bit_generator.state,
                            torch_rng=torch.get_rng_state(), seconds=time.time() - t0), str(ck) + ".tmp")
            Path(str(ck) + ".tmp").replace(ck)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    s_max = float(np.quantile(gaps, 0.999)) * 20
    res = evaluate(model, te, M, device, predict=True, s_max=s_max)
    res.update(dataset=name, seed=seed, val_ll=best, epochs=ep + 1, seconds=time.time() - t0,
               hidden=hidden, n_rates=n_rates, n_channels=n_channels, phases=phases, lr=lr, gompertz=gompertz,
               model=("EPT-X" if ept_channel else "EPT-X-renewal-only") if renewal else "EPT",
               renewal=renewal, input_v2=input_v2, ept_channel=ept_channel, rn_shift=rn_shift,
               rn_scales=rn_scales if renewal else None, rn_orders=list(rn_orders) if renewal else None,
               n_layers=n_layers, layer_norm=layer_norm, dropout=dropout, weight_decay=weight_decay, encoder=encoder,
               n_params=int(sum(p.numel() for p in model.parameters())))
    return res


def run(out: Path, datasets=("taxi", "taobao", "stackoverflow", "amazon", "retweet"), seeds=(0, 1, 2, 3, 4),
        grid=None, verbose=False):
    grid = grid or [dict(hidden=64, n_rates=8, n_channels=4)]
    rows = []
    for ds in datasets:
        # model selection on validation LL with seed 0, then 5 seeds with the chosen config
        sel = []
        for g in grid:
            r = train_one(ds, 0, verbose=verbose, **g)
            sel.append((r["val_ll"], g, r))
            print(json.dumps({k: v for k, v in r.items()}), flush=True)
        best_cfg = max(sel, key=lambda x: x[0])[1]
        for s in seeds:
            r = next((x[2] for x in sel if x[1] == best_cfg), None) if s == 0 else None
            r = r or train_one(ds, s, verbose=verbose, **best_cfg)
            r["selected"] = True
            rows.append(r)
            print(json.dumps(r), flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rows, indent=1))
    return rows


def train_one_safe(name: str, seed: int, bs=64, min_bs=8, **kw) -> dict:
    """train_one, halving the batch size on CUDA out-of-memory (8 GB GPU; many-mark datasets)."""
    while True:
        try:
            r = train_one(name, seed, bs=bs, **kw)
            r["batch_size"] = bs
            return r
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs // 2 < min_bs:
                raise
            bs //= 2
            print(f"OOM on {name}: retrying with batch size {bs}", flush=True)
