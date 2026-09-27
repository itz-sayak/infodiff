# Literature notes and reference numbers

PDFs in this folder were fetched by `scripts/fetch_papers.py`, which checks arXiv titles.

## 1. The primary reference: news → FX activity with Hawkes processes
**Rambaldi, Pennesi & Lillo (2015), PRE 91 012819** (`rambaldi2015_*.pdf`)
- Data: EBS best-quote changes, EUR/USD, EUR/JPY and USD/JPY, 2012, London hours, 100 ms stamps randomised within each tick.
- Endogenous kernel is either a double exponential or an approximate power law (a sum of 15 exponentials, with a cut-off φ(0)=0 at short lags).
- News kernel is a single exponential αN·e^(−βN·t), fitted separately on each isolated-news window. A non-causal (anticipation) kernel is added in Section V.
- Reported numbers:
  - Branching ratio n ≈ 0.84–0.95 (0.87 for EUR/USD before the tick-size change).
  - Power-law exponent p ≈ 1.2–1.5.
  - Uncertainties from the Hessian: about 10% on μ and about 2% on n.
  - The Excess-Dispersion test (Exp(1) residuals) rejects at 5% on **92.8% of days with the double exponential** and **42.2% with the power law**.
  - One NFP window: αN = 1.45 s⁻¹ and βN = 8.2e-4 s⁻¹ (power law), i.e. a very long news decay once echoes are *not* separated.
  - Surprise vs activity jump θ: Spearman 0.34 (0.62 for high-importance news).
- Gaps we address:
  - A single-exponential news kernel with no echo correction. The ln2/β "half-life" confounds direct absorption with endogenous echoes.
  - One window at a time, so no pooling.
  - Surprise enters only as a post-hoc correlation, not as a mark.
  - FX only, no cross-market structure.

## 2. Endogeneity, criticality, estimation
- **Filimonov & Sornette 2012, 2015.** Branching estimates drift toward 1 when the baseline is mis-specified (intraday seasonality, regime shifts). This motivates our per-window intercepts, placebo-identified time-of-day baseline, and a placebo news type.
- **Hardiman, Bercot & Bouchaud 2013.** Power-law kernels in E-mini futures give n ≈ 1 ("critical reflexivity").
- **Bacry, Jaisson & Muzy 2016.** Slowly decreasing kernels in order-book data, estimated non-parametrically (Wiener–Hopf).
- **Bacry & Muzy 2016.** The conditional-law / Wiener–Hopf estimator, i.e. tick's `HawkesConditionalLaw`.
- **Achab et al. 2017 (NPHC).** Integrated-cumulant matching recovers the branching matrix G without kernel shapes. tick 0.8 ships its PyTorch solver as a stub, so we reimplemented the objective (`baselines/nphc.py`).
- **Zhou, Zha & Song 2013 (ADM4)** and **Xu, Farajtabar & Zha 2016.** Sparse and low-rank Granger-causal Hawkes.
- **Kirchner 2017.** INAR (discretised) estimation.
- **Omi, Hirata & Aihara 2017.** A time-dependent background rate for high-frequency data.
- **Embrechts & Kirchner 2018 (Hawkes graphs)** and **Jaisson & Rosenbaum 2015 (nearly unstable Hawkes).**
- **Laub et al. 2015.** Tutorial, including the time-rescaling theorem.

## 3. Neural temporal point processes: current SOTA and protocol
**Chang et al. 2025, "Deep Continuous-Time State-Space Models for Marked Event Sequences" (S2P2)** is the current SOTA. Its code is in EasyTPP, and its fork is `external/state_space_point_process`.

Protocol:
- LL per event = total LL over scored events (the first event conditions) ÷ number of scored events.
- The integral runs from t₀ to t_N and is estimated by Monte Carlo (10 samples per interval).
- 5 seeds; hyperparameters are selected on validation LL, and test metrics are taken at the best validation epoch.

Test log-likelihood per event (↑). Standard deviation over 5 seeds in parentheses; from the paper's Table 2(a):

| Model | Amazon | Retweet | Taxi | Taobao | StackOverflow |
|---|---|---|---|---|---|
| RMTPP | −2.136 | −7.098 | 0.346 | 1.003 | −2.480 |
| SAHP | −2.074 | −6.708 | 0.298 | 1.168 | −2.341 |
| THP | −2.096 | −6.659 | 0.372 | 0.790 | −2.338 |
| IFTPP | 0.496 | −10.344 | 0.453 | 1.318 | −2.233 |
| MHP | −2.091 | −6.564 | 0.370 | 0.636 | −2.346 |
| NHP | 0.129 | **−6.348** | 0.514 | 1.157 | −2.241 |
| AttNHP | 0.484 | −6.499 | 0.493 | 1.259 | −2.194 |
| **S2P2** | **0.781** | −6.365 | **0.522** | **1.304** | **−2.163** |

Next-event time RMSE (↓) on Taxi: S2P2 0.281, NHP 0.282. Mark accuracy on Taxi: S2P2 93.1%, NHP 92.9%.

Other relevant work:
- **Hyper Hawkes Processes (Boyd et al. 2025).** Hawkes with hyper-network-driven dynamics that are linear piecewise; interpretable, reports SOTA. No public code at the time of writing.
- **Mamba Hawkes (Gao 2024).** Evaluated as MHP above.
- **Bosser & Ben Taieb 2023.** Standardised re-evaluation. Many reported neural-TPP gains shrink under consistent protocols, which is why we re-run baselines ourselves on the finance datasets.
- **HoTPP 2024.** A long-horizon evaluation benchmark.
- **EasyTPP (Xue et al. 2023).** The open benchmarking library.
- **"Financial Transactions" (Du et al. 2016, RMTPP):** buy/sell events in one stock-day, used by NHP and THP.
  - THP reports a Financial LL of −1.11 against −3.60 for NHP. Bosser & Ben Taieb flag THP's LL computation, so these numbers are not directly comparable.
  - The Google-Drive mirror now needs a resource key and returns 401. We substitute the public **LOBSTER** samples (5 stocks, 2012-06-21), which are richer: 6 order-book event types.

## 4. Macro-announcement microstructure
- **Andersen, Bollerslev, Diebold & Vega 2003.** The canonical FX response to US macro surprises. Surprises are standardised by their standard deviation, and news effects are fast (minutes) and asymmetric.
- **Aït-Sahalia, Cacho-Diaz & Laeven 2015.** Mutually exciting jumps across markets (contagion).
- **Huth & Abergel 2014.** High-frequency lead/lag, the Hayashi–Yoshida-style baseline in Track D.

## 5. Prediction markets
- **Tsang & Yang 2026.** Polymarket's microstructure in the 2024 US election; arbitrage deviations shrink from hours to under a minute.
- **Cheng, Yang & Zou 2026.** Polymarket NBA arbitrage: 75M snapshots, windows of about 3.6 s.

## 6. What we claim beyond the literature (checked in the benchmarks)
1. A concave, globally optimal multiscale phase-type Hawkes MLE, with a Fenchel duality-gap certificate.
2. Echo-corrected absorption times (t50/t90) and amplification in closed form, via the Markov embedding e^(At). Stability theorem: the state matrix is Hurwitz ⇔ ρ(G) < 1.
3. The price-discovery curve is exact under δ-crossing sampling.
4. Placebo-identified seasonality, plus a PLACEBO news type as a falsification test.
5. **EPT-TPP**: a neural TPP with an *exact* compensator (Erlang-R phase-type state plus a Gompertz head). It nests multivariate Hawkes and RMTPP.
