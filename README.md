<div align="center">

# DeepLOB + DLS Asset Allocation

### From limit-order-book prediction to portfolio allocation, execution, audit, and interactive replay

**DeepLOB signals → DLS portfolio weights → realistic execution → auditable trading-engine replay**

</div>

---

## Overview

This repository contains a complete **research-to-execution asset-allocation workflow** built on top of DeepLOB market signals.

The project combines four distinct layers:

1. **Market representation** — causal LOB/OFI and OHLCV feature engineering.
2. **Signal generation** — a pretrained DeepLOB model produces directional probabilities for each asset.
3. **Portfolio construction** — a Deep Learning Sharpe (DLS) network converts those signals into cross-sectional target weights.
4. **Execution and audit** — the target portfolio is translated into feasible orders, fills, holdings, metrics, CSV audit files, and an interactive Streamlit replay application.

The central idea is that **forecasting market direction and allocating capital are different problems**. DeepLOB answers:

> What does the model believe may happen to this asset?

DLS answers:

> Given all asset-level signals and their recent history, how should portfolio capital be distributed?

The execution engine then answers:

> Which desired portfolio changes can actually be implemented after trading constraints and costs?

This separation makes the full decision chain inspectable instead of treating the neural-network output as a direct trading instruction.

---

## Key Results

Latest documented executed run:

| Metric | DeepLOB + DLS |
|---|---:|
| Initial capital | **RMB 50,000,000** |
| OOS period | **D485–D726** |
| OOS trading days | **242** |
| Selected DLS seed | **33** |
| Final portfolio value | **≈ RMB 73.05M** |
| Total return | **46.11%** |
| Annualized Sharpe | **1.20** |
| Maximum drawdown | **−15.77%** |
| Score proxy | **20.41** |
| Average holdings/day | **235.3** |

### DeepLOB-only baseline vs. DeepLOB + DLS

| Metric | Base DeepLOB | DeepLOB + DLS |
|---|---:|---:|
| Total return | 40.88% | **46.11%** |
| Sharpe ratio | 0.96 | **1.20** |
| Maximum drawdown | −24.91% | **−15.77%** |
| Score proxy | 15.05 | **20.41** |
| Average holdings | 692.7 | **235.3** |
| Total trading costs | ≈ RMB 2.10M | ≈ RMB 3.08M |

In the documented run, the DLS layer therefore produced a **more selective portfolio**, higher return and Sharpe, and a less severe maximum drawdown, while also generating more active reallocation and therefore higher transaction costs.

---

# End-to-End Pipeline

<p align="center">
  <img src="assets/architecture-pipeline.svg" alt="End-to-end DeepLOB + DLS portfolio pipeline" width="900">
</p>

The pipeline is designed as a sequence of explicit state transitions:

```text
Raw LOB + Daily OHLCV
        ↓
Causal Feature Engineering
        ↓
DeepLOB Input Tensor
        ↓
DeepLOB Directional Probabilities
        ↓
DLS Signal Feature Panel
        ↓
50-Day Cross-Sectional Sequence
        ↓
ShifuDLSNet Portfolio Allocator
        ↓
Tradability + Quality Filtering
        ↓
Target Portfolio Weights
        ↓
Execution / Backtest Engine
        ↓
Submitted Orders
        ↓
Executed Trades
        ↓
End-of-Day Holdings + Cash + PnL
        ↓
Metrics + Audit CSVs + Streamlit Replay
```

## Stage 1 — Market Data and Feature Engineering

The notebook combines **limit-order-book information** with **daily OHLCV-derived variables** before DeepLOB inference.

### LOB / OFI block

The LOB representation uses:

- 10 order-book levels;
- 24 intraday time slots;
- order-flow imbalance (OFI) features;
- causal rolling normalization using historical information only.

The flattened intraday OFI grid contributes:

```text
24 time slots × 10 levels = 240 OFI features
```

### Daily market block

The daily block adds 19 variables, including:

- 1/5/10/20-day returns;
- 5-day and 20-day volatility;
- Amihud illiquidity;
- volume z-score;
- RSI;
- moving-average distance features;
- open-to-close return;
- high-low range;
- close-to-VWAP distance;
- raw open, close, volume, low, and high values.

The final feature width is therefore:

```text
240 LOB/OFI features + 19 daily features = 259 features
```

A rolling 50-day sequence is assembled into the DeepLOB tensor:

```text
B × 1 × 50 × 259
```

The feature preparation is deliberately causal so current or future observations are not silently introduced into historical normalization statistics.

---

## Stage 2 — DeepLOB Signal Generation

DeepLOB is used as the **predictive signal layer**, not as the portfolio allocator.

<p align="center">
  <img src="assets/deeplob-dls-network-architecture.svg" alt="DeepLOB and DLS network architecture" width="980">
</p>

The pretrained network contains convolutional feature-extraction blocks, an Inception-style module, a temporal LSTM, and a classification head.

For every asset `i` and signal day `t`, DeepLOB produces:

```text
P(down), P(flat), P(up)
```

with:

```text
P(down) + P(flat) + P(up) = 1
```

These probabilities are more informative than a single hard class label. For example, two assets can both be labeled as bullish while having very different probability distributions and therefore very different confidence levels.

---

## Stage 3 — DeepLOB Probabilities → DLS Features

The three DeepLOB outputs are transformed into a five-dimensional signal representation.

Two derived variables are added:

```text
signal_score = P(up) - P(down)

confidence = |signal_score| × (1 - P(flat))
```

Each asset/day is therefore represented as:

```text
[P(down), P(flat), P(up), signal_score, confidence]
```

| Feature | Interpretation |
|---|---|
| `P(down)` | bearish probability |
| `P(flat)` | neutral / low-direction probability |
| `P(up)` | bullish probability |
| `signal_score` | signed directional edge |
| `confidence` | directional strength after penalizing flatness |

This is the bridge between DeepLOB's classification output and DLS's portfolio-allocation problem.

---

## Stage 4 — DLS Temporal Input Construction

DLS does not make a portfolio decision using only one signal day. It receives a **history of cross-sectional probability panels**.

<p align="center">
  <img src="assets/dls-dataflow-pipeline.svg" alt="DLS probability-panel data flow" width="860">
</p>

The documented universe contains:

```text
N = 2,306 assets
```

Each day therefore contains:

```text
2,306 assets × 5 signal features = 11,530 values
```

The network uses a 50-day lookback:

```text
50 × 2,306 × 5
```

At each temporal step, the cross-sectional asset-feature panel is flattened to `5N = 11,530`, producing a sequence for the DLS LSTM.

As a result, the allocator can observe not only today's probabilities but also how the **market-wide signal state** has evolved over recent days.

---

## Stage 5 — ShifuDLSNet Portfolio Allocation

The portfolio network uses the following main architecture:

```text
Input size      : 11,530
Temporal lookback: 50 days
LSTM hidden size: 64
Projection      : 64 → 2,306 assets
Output          : asset-level allocation logits
```

The allocation sequence is:

```text
DeepLOB probability history
        ↓
DLS LSTM representation
        ↓
Asset-level logits
        ↓
Tradability mask
        ↓
Masked softmax
        ↓
Long-only raw target weights
```

The masked softmax ensures non-tradable names receive no allocation and generates a long-only cross-sectional weight vector.

This is substantially different from a rule such as "buy every asset whose `P(up)` exceeds a threshold." DLS learns how to distribute capital across the available opportunity set.

---

## Stage 6 — Portfolio-Level DLS Objective

The DLS model is trained with a **portfolio objective**, rather than a classification loss.

<p align="center">
  <img src="assets/dls-objective-components.svg" alt="DLS portfolio objective components" width="880">
</p>

The objective incorporates portfolio-level reward and penalty terms.

### Reward components

- Sharpe ratio;
- Sortino ratio.

### Risk and implementation penalties

- drawdown;
- turnover;
- portfolio concentration;
- inventory / gross-exposure deviation;
- commission exposure;
- optional sparsity diagnostics.

The documented final sparsity coefficient is zero. The current notebook also does not activate a quadratic market-impact term, so implementation-cost control is represented mainly through turnover and commission-related components.

The key point is that the model is optimized for the **economic behavior of the portfolio path**, not merely for per-asset classification accuracy.

---

## Stage 7 — Tradability, Quality Filtering, and Post-Processing

The raw DLS softmax output is not traded directly.

The portfolio-construction layer applies additional rules:

1. remove non-tradable assets;
2. require sufficient DeepLOB directional quality;
3. remove very small target weights;
4. cap the number of active names;
5. maintain minimum-holdings feasibility;
6. renormalize the surviving allocation to the desired gross exposure.

The documented signal-quality thresholds are:

```text
confidence >= 0.02
signal_score >= 0.00
```

Important portfolio parameters include:

| Parameter | Value |
|---|---:|
| Target gross exposure | **0.95** |
| Maximum DLS positions | **500** |
| Minimum target weight | **0.0005** |
| Rebalance band | **0.006** |
| Minimum holdings | **10** |

If too few assets survive the quality filter, the implementation can fall back to top-confidence tradable assets to avoid an infeasible state.

---

## Stage 8 — No-Lookahead Timing

A critical part of the workflow is the separation between **observation**, **execution**, and **evaluation**.

```text
Day t      : DeepLOB observes available information and creates signals
Day t + 1  : portfolio decision is traded
Day t + 2  : subsequent trade outcome is evaluated
```

The dashboard also exposes both `signal_day_id` and `trade_day_id` so this shift is visible during replay.

This design helps prevent the model from sizing a position with information that would not have been available at decision time.

---

## Stage 9 — Execution / Backtest Engine

The execution engine converts desired target weights into feasible orders and holdings.

<p align="center">
  <img src="assets/execution-engine-flowchart.svg" alt="Execution engine flowchart" width="920">
</p>

For each trade day, the engine approximately performs the following sequence:

1. value the current portfolio;
2. compute current portfolio weights;
3. compare current weights with DLS targets;
4. ignore changes inside the rebalance band;
5. process sells first to release cash;
6. process buys subject to available cash and tradability;
7. apply lot-size constraints;
8. reduce quantities if the desired trade is infeasible;
9. apply commissions and stamp duty;
10. update cash and shares;
11. create end-of-day holdings and detailed audit records.

Execution settings include:

| Parameter | Value |
|---|---:|
| Commission rate | **0.0001** |
| Stamp-duty rate | **0.0005** |
| Minimum commission | **5** |
| Lot size | **100 shares** |
| Cash buffer | **5%** |

This creates an important distinction:

```text
DLS target weight
        ≠
submitted order
        ≠
filled trade
        ≠
final EOD portfolio weight
```

That distinction is preserved in the exported audit files and is one of the main reasons the replay dashboard is useful.

---

# Notebook Guide

The repository currently contains two notebook artifacts:

| Notebook | Role |
|---|---|
| [`notebooks/deeplob_dls_asset_allocation.ipynb`](notebooks/deeplob_dls_asset_allocation.ipynb) | Full working DeepLOB + DLS research notebook containing the end-to-end asset-allocation workflow and generated outputs. |
| [`notebooks/deeplob_dls_asset_allocation-version1.ipynb`](notebooks/deeplob_dls_asset_allocation-version1.ipynb) | Versioned/executed experiment snapshot corresponding to the documented Kaggle-oriented run and reported metrics. |

The notebooks are more than training scripts: together they implement the entire experiment lifecycle.

## 1. Runtime and Input Discovery

The Kaggle-oriented version recursively discovers required files under:

```text
/kaggle/input
```

Expected inputs include equivalents of:

```text
daily_data_in_sample.parquet
lob_data_in_sample.parquet
daily_data_release_stage_out_of_sample.parquet
lob_data_release_stage_out_of_sample.parquet
best_model_alpha_0015.pt
```

Generated artifacts are written under:

```text
/kaggle/working/shifu_dls_oos/
```

and can be packaged into:

```text
/kaggle/working/shifu_dls_oos_results.zip
```

## 2. Data Preparation

The notebook loads and aligns in-sample and out-of-sample daily and LOB data, then constructs the causal OFI grid and daily feature block.

The documented run covers:

| Item | Value |
|---|---:|
| Assets | **2,306** |
| Total trading days | **726** |
| Warmup / in-sample period | **D001–D484** |
| OOS period | **D485–D726** |
| OOS days | **242** |
| Combined OHLCV rows | **1,606,720** |
| Processed LOB rows | **37,572,229** |
| DeepLOB signal days | **675** |

## 3. DeepLOB Inference

The pretrained DeepLOB checkpoint is loaded and used to generate three-class directional probabilities across the stock universe.

Those raw probabilities are retained so both the original DeepLOB strategy and the DLS-enhanced strategy can be reconstructed from the same signal source.

## 4. DLS Dataset Construction

The notebook builds:

- the five-feature DeepLOB-derived signal panel;
- 50-day temporal DLS sequences;
- tradability / entry masks;
- future-return labels;
- validity masks;
- volatility-related inputs;
- chronological train/validation samples.

## 5. DLS Training

Main training parameters:

| Hyperparameter | Value |
|---|---:|
| Lookback | **50** |
| Hidden size | **64** |
| Batch size | **64** |
| Maximum epochs | **100** |
| Learning rate | **0.001** |
| Early-stopping patience | **15** |

Training uses chronological data splits, Adam optimization, early stopping, and restoration of the selected model state.

## 6. Seed Search

Multiple neural-network initializations are evaluated:

```text
7, 11, 22, 33, 42, 55, 77, 88, 101, 123
```

The documented run selects seed **33**.

<p align="center">
  <img src="assets/dls-seed-search-ranking.svg" alt="DLS seed search ranking" width="880">
</p>

## 7. Dual Backtest

The notebook executes two strategies:

- **Original DeepLOB-only exact baseline**;
- **DeepLOB + DLS allocation strategy**.

This makes it possible to evaluate whether the allocation layer changes the economic use of the underlying DeepLOB signals instead of evaluating DLS in isolation.

<p align="center">
  <img src="assets/oos-performance-comparison.svg" alt="Out-of-sample performance comparison" width="860">
</p>

## 8. Audit Export

A major design feature is the export of detailed intermediate states rather than only final performance numbers.

Examples include:

- raw DeepLOB probabilities;
- daily DLS target weights;
- weight-step decision diagnostics;
- submitted orders;
- filled trades;
- end-of-day holdings;
- baseline daily portfolio logs;
- final strategy-comparison metrics;
- DLS training history;
- seed-search results.

These files power the Streamlit replay application.

## 9. Static Diagnostics

The research notebook also generates static diagnostics for quick inspection.

<p align="center">
  <img src="assets/static-backtest-report.svg" alt="Static DLS backtest report" width="960">
</p>

<p align="center">
  <img src="assets/weight-distribution-diagnostics.svg" alt="DLS weight-distribution diagnostics" width="900">
</p>

The plots cover portfolio value, returns, drawdown, rolling Sharpe, holdings, transaction costs, target-weight distributions, and the evolution of portfolio concentration.

---

# Streamlit Dashboard — Trading Engine Replay

The dashboard is an **interactive event-driven audit application**, not a static CSV viewer.

It reconstructs the same trading day as a state machine:

```text
DeepLOB probabilities
        ↓
Signal Score + Confidence
        ↓
Entry / Tradability Mask
        ↓
Quality Filter
        ↓
DLS Target Weights
        ↓
Submitted Orders
        ↓
Executed Trades
        ↓
End-of-Day Holdings
        ↓
Portfolio State + Model Comparison
```

The goal is to answer both:

> How did the strategy perform?

and:

> Why did this particular asset receive this particular action on this particular day?

---

# Dashboard Showcase

## 1. Engine Replay

<p align="center">
  <img src="assets/dashboard-engine-replay.webp" alt="Shifu DeepLOB + DLS Engine Replay dashboard" width="100%">
</p>

The **Engine Replay** view is the high-level trading-engine cockpit. The selected OOS day is reconstructed as a complete decision state rather than a single return observation.

The upper KPI strip shows information such as:

- engine clock / current trade day;
- corresponding signal day;
- DLS equity proxy;
- cumulative return proxy;
- target gross exposure;
- executed turnover;
- end-of-day holdings count.

### Six-stage decision funnel

The central stage line summarizes the full trading transformation:

```text
01 Signal Bus
      ↓
02 Entry Mask
      ↓
03 Quality Filter
      ↓
04 DLS Optimizer
      ↓
05 Execution
      ↓
06 Portfolio State
```

For the displayed first replay day in the screenshot, the flow is visually traceable from roughly **2,027 raw DeepLOB signals**, to **284 entry candidates**, **246 quality-passed names**, **237 DLS targets**, **236 executed names**, and **236 final holdings**.

Each stage has a different meaning:

- **Signal Bus** — all usable DeepLOB probability rows for the signal day.
- **Entry Mask** — assets eligible for trading on the corresponding execution day.
- **Quality Filter** — candidates satisfying directional confidence requirements.
- **DLS Optimizer** — names receiving a non-zero target allocation.
- **Execution** — names that actually generated filled trades after implementation constraints.
- **Portfolio State** — active end-of-day holdings after execution.

The view also provides a reconstructed DLS-vs-base return path, a compact engine-event log, and a visual decision funnel.

### Replay controls

The left control panel supports:

- Previous / Next day;
- replay-step slider;
- Play / Pause;
- Reset;
- playback speeds such as `0.5x`, `1x`, `2x`, `5x`, and `10x`;
- loop playback;
- optional auto-play refresh;
- fast rendering mode for smoother animated replay.

This makes the backtest behave more like a replayable simulation than a fixed report.

---

## 2. Model Comparison Cockpit

<p align="center">
  <img src="assets/dashboard-model-comparison.jpg" alt="DeepLOB + DLS model comparison dashboard" width="100%">
</p>

The **Model Comparison** view focuses on the practical question behind the project:

> Did the DLS allocation layer improve the economic use of the original DeepLOB signals?

The cockpit displays the final headline metrics side by side and visualizes differences in portfolio behavior.

It includes:

- total return comparison;
- Sharpe ratio comparison;
- maximum-drawdown comparison;
- final trading costs;
- DLS vs. base equity/return paths;
- drawdown paths;
- impact deltas such as return, Sharpe, drawdown, holdings and costs;
- the assets with the largest DLS-vs-base order differences for the selected day.

This last chart is especially useful because two models can generate similar headline returns while producing very different **orders and capital allocations** underneath.

---

## 3. Portfolio State

<p align="center">
  <img src="assets/dashboard-portfolio-state.webp" alt="End-of-day portfolio state dashboard" width="100%">
</p>

The **Portfolio State** view answers:

> What does the strategy actually hold at the end of the selected replay day?

The screen combines a ranked top-holdings chart with the exact holdings table.

For each position, the table can expose fields such as:

- asset identifier;
- shares held;
- closing price;
- market value;
- end-of-day portfolio weight.

This makes it possible to move from aggregate performance back to the exact position-level state of the portfolio.

---

# Dashboard Views and Capabilities

The sidebar exposes several specialized views.

## Engine Replay

Best for understanding the entire state transition of one OOS day. It answers:

- How many raw signals existed?
- How many were tradable?
- How many survived the quality filter?
- How many assets did DLS target?
- Which trades executed?
- What remained in the portfolio afterward?

## Decision Inspector

The **Decision Inspector** is the most detailed asset-level audit view.

It links the full decision chain for each asset, including fields such as:

```text
P(down)
P(flat)
P(up)
signal_score
confidence
entry_candidate
quality_candidate
previous_weight
target_weight
post_trade_weight
final_eod_weight
submitted_buy_%
submitted_sell_%
filled_buy_%
filled_sell_%
```

This allows the user to investigate why an individual stock was selected, ignored, increased, reduced, or left unchanged.

## Execution Tape

The **Execution Tape** focuses on actual implementation.

It reconciles:

```text
what DLS wanted
      ↓
what was submitted
      ↓
what actually filled
```

Typical fields include:

- order side;
- submitted buy/sell percentage;
- filled buy/sell percentage;
- shares;
- execution price;
- turnover;
- trading cost.

This view is important because optimizer targets and executable trades are not assumed to be identical.

## Portfolio State

Provides the exact current holdings snapshot, ranked exposures, market values, shares and EOD weights.

## Daily Metrics

Summarizes same-day behavior such as:

- daily return;
- turnover;
- transaction cost;
- number of holdings;
- gross exposure;
- cumulative replay statistics;
- base-vs-DLS day-level differences.

## Model Comparison

Provides the final benchmark comparison between **Base DeepLOB** and **DeepLOB + DLS**, including risk, return, cost and order-divergence views.

## Training Lab

Uses exported DLS training files to inspect:

- training and validation history;
- early-stopping behavior;
- seed-search results;
- selected model/seed diagnostics.

---

# Dashboard Replay State

At each selected day, the application constructs a unified engine state containing the current day plus its corresponding:

- raw DeepLOB signals;
- DLS weight-step rows;
- filled DLS trades;
- end-of-day DLS holdings;
- DLS optimizer/debug row;
- DLS submitted orders;
- base DeepLOB submitted orders.

Because all dashboard screens read from the same replay state, the user can switch from a high-level chart to an asset-level audit without losing the current day context.

---

## Dashboard Equity Reconstruction

The exported files contain detailed DLS holdings, trades and weights, while the base strategy includes a direct daily portfolio-value log.

For replay purposes, the dashboard reconstructs a DLS equity proxy approximately as:

```text
raw_dls_equity_proxy = EOD market value / target gross
```

The replay path is then normalized to the notebook's initial capital:

```text
RMB 50,000,000
```

This keeps the visual DLS and baseline paths on a comparable capital scale.

---

# Dashboard Data Files

The application is driven by notebook-generated audit exports.

| File | Purpose |
|---|---|
| `T001_original_deeplob_exact_raw_oos_signals.csv` | raw DeepLOB three-class probabilities and signals |
| `T001_dls_weight_step_audit.csv` | asset-level DLS decision chain |
| `T001_dls_weights_long_shifted_tplus2.csv` | long-format target portfolio weights |
| `T001_dls_trade_audit.csv` | filled DLS trades |
| `T001_dls_holding_snapshot_audit.csv` | end-of-day portfolio holdings |
| `T001_dls_weight_debug_shifted_tplus2_conf_filter.csv` | day-level filter/optimizer diagnostics |
| `T001_original_deeplob_exact_daily_log.csv` | base DeepLOB daily portfolio state |
| `T001_oos_original_deeplob_exact_sell_close.csv` | base submitted orders |
| `T001_oos_shifu_dls_kaggle_sell_open.csv` | DLS submitted orders |
| `T001_comparison_original_deeplob_exact_vs_dls.csv` | final model comparison metrics |
| `shifu_dls_seed_search_summary.csv` | seed-search diagnostics |
| `shifu_dls_training_history_oos.csv` | DLS training history |

---

# Repository Structure

```text
DeepLOB-DLS-Asset-Allocation/
├── notebooks/
│   ├── deeplob_dls_asset_allocation.ipynb
│   └── deeplob_dls_asset_allocation-version1.ipynb
│
├── dashboard/
│   ├── app.py
│   ├── README.md
│   ├── requirements.txt
│   └── .streamlit/
│       └── config.toml
│
├── docs/
│   ├── README.md
│   ├── DeepLOB_DLS_Report_documentation.pdf
│   └── DeepLOB_DLS_Asset_Allocation_Documentation.pdf
│
├── assets/
│   ├── architecture-pipeline.svg
│   ├── deeplob-dls-network-architecture.svg
│   ├── dls-dataflow-pipeline.svg
│   ├── dls-objective-components.svg
│   ├── execution-engine-flowchart.svg
│   ├── dls-seed-search-ranking.svg
│   ├── oos-performance-comparison.svg
│   ├── static-backtest-report.svg
│   ├── weight-distribution-diagnostics.svg
│   ├── dashboard-engine-replay.webp
│   ├── dashboard-model-comparison.jpg
│   └── dashboard-portfolio-state.webp
│
├── CONTRIBUTIONS.md
└── README.md
```

---

# Running the Dashboard

Install dependencies:

```bash
pip install -r dashboard/requirements.txt
```

The minimal dashboard stack includes:

```text
streamlit
pandas
numpy
plotly
streamlit-autorefresh
```

Place the notebook-generated CSV files in the data directory expected by the dashboard, then launch:

```bash
streamlit run dashboard/app.py
```

For dashboard-specific implementation details, see:

[`dashboard/README.md`](dashboard/README.md)

---

# Research Contributions

The contribution of this module is not just one neural network. It is the integration of a complete **research-to-execution workflow**:

```text
market data
    ↓
causal feature engineering
    ↓
DeepLOB probability inference
    ↓
probability-derived DLS features
    ↓
learned cross-sectional asset allocation
    ↓
quality and tradability filtering
    ↓
execution-aware portfolio construction
    ↓
realistic order simulation
    ↓
trade + holding + decision audit files
    ↓
interactive trading-engine replay
```

Notable implemented components include:

- using the full DeepLOB probability vector rather than only a hard class label;
- constructing directional score and confidence features;
- learning portfolio weights with a temporal LSTM allocator;
- optimizing a portfolio-level financial objective;
- explicitly controlling turnover and implementation cost;
- enforcing a shifted no-lookahead execution convention;
- exporting detailed state-transition audits;
- comparing DLS against the original DeepLOB baseline;
- replaying the complete decision funnel interactively in Streamlit.

For contribution provenance and original branch history, see:

[`CONTRIBUTIONS.md`](CONTRIBUTIONS.md)

---

# Limitations

This repository is a research implementation, not a production trading system.

Important limitations include:

- DLS is currently long-only due to masked-softmax allocation;
- the current loss does not activate a quadratic market-impact term;
- development-time seed selection is based on OOS total return in the documented experiment;
- the DLS input dimension is tied to a fixed asset universe;
- execution is simulated rather than routed to a live exchange;
- liquidity and market impact can differ materially in live conditions;
- the reported experiment represents one main OOS interval and should not be interpreted as evidence of future performance.

---

# Future Work

Natural next steps include:

- validation-only seed/model selection;
- nonlinear volatility-aware risk penalties;
- explicit quadratic market-impact modeling;
- dynamic transaction-cost calibration;
- linear / AR / statistical allocation baselines;
- equal-weight and top-confidence portfolio baselines;
- attention- or graph-based cross-asset architectures;
- sector/industry neutrality constraints;
- walk-forward evaluation across multiple market regimes;
- richer performance attribution;
- paper-trading or live-market-data integration.

---

## Author

**Mohammadreza Sheikholeslami**  
GitHub: `MohammadrezaSheikholeslami84`

---

## Disclaimer

This repository is intended for **research and educational purposes only**. Nothing in this repository constitutes financial or investment advice. Historical and backtested results do not guarantee future performance.
