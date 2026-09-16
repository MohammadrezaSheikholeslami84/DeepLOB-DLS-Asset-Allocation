# DeepLOB–DLS Asset Allocation Module

A complete **research-to-execution asset-allocation pipeline** built on top of DeepLOB signals.

The project combines:

- **limit-order-book and OHLCV feature engineering**;
- a pretrained **DeepLOB** directional signal model;
- a learned **Deep Learning Sharpe (DLS)** portfolio allocator;
- a transaction-cost-aware **execution/backtesting engine**;
- detailed audit outputs; and
- an interactive **Streamlit trading-engine replay dashboard**.

The key design idea is that **prediction and portfolio construction are different problems**. DeepLOB estimates what may happen to each asset; DLS decides how the available capital should be distributed across the cross-section; the execution layer then determines what can actually be traded under realistic constraints.

---

## Key Results

Latest documented executed run:

| Metric | Value |
|---|---:|
| Notebook | `deeplob_dls_asset_allocation-version1.ipynb` |
| Mode | Kaggle Edition |
| Initial Capital | RMB 50,000,000 |
| Out-of-sample period | D485 to D726 |
| Metric days | 242 |
| Selected DLS seed | 33 |
| Final portfolio value | RMB 73.05M |
| Total return | **46.11%** |
| CAGR | **46.11%** |
| Annualized Sharpe | **1.20** |
| Maximum Drawdown | **-15.77%** |
| Score proxy | **20.41** |
| Average holdings/day | **235.3** |

### DeepLOB-only baseline vs. DeepLOB + DLS

| Model | Return | Sharpe | MDD | Score | Avg Holdings |
|---|---:|---:|---:|---:|---:|
| Original DeepLOB-only exact | 40.88% | 0.96 | -24.91% | 15.05 | 692.7 |
| **DeepLOB + DLS** | **46.11%** | **1.20** | **-15.77%** | **20.41** | **235.3** |

The documented run therefore shows a more selective portfolio with higher return, higher Sharpe and a less severe drawdown, at the cost of greater turnover and transaction costs.

---

# End-to-End Pipeline

<p align="center">
  <img src="assets/architecture-pipeline.svg" alt="End-to-end DeepLOB + DLS pipeline" width="860">
</p>

The pipeline should be read as a sequence of **data transformation → prediction → allocation → execution → audit** stages.

```text
Raw LOB + daily OHLCV data
        ↓
Causal feature engineering
        ↓
DeepLOB input tensor
        ↓
DeepLOB directional probabilities
        ↓
DLS signal-feature panel
        ↓
50-day cross-sectional DLS sequence
        ↓
ShifuDLSNet portfolio allocator
        ↓
Tradability + signal-quality post-processing
        ↓
Target portfolio weights
        ↓
Execution / backtest engine
        ↓
Orders → fills → holdings → cash → PnL
        ↓
Metrics + audit CSVs + Streamlit replay
```

## Stage 1 — Market Data and Feature Engineering

The notebook combines **LOB snapshots** with **daily OHLCV information** before DeepLOB inference.

### LOB feature block

The LOB representation uses:

- 10 price/volume levels on both sides of the book;
- 24 intraday time slots;
- order-flow imbalance (OFI) calculations;
- causal rolling normalization based only on previous days.

The flattened OFI grid contributes:

```text
24 time slots × 10 levels = 240 OFI features
```

### Daily feature block

The daily block adds 19 market features, including:

- 1/5/10/20-day returns;
- 5-day and 20-day volatility;
- Amihud illiquidity;
- volume z-score;
- RSI;
- moving-average distance features;
- open-to-close return;
- high-low range;
- close-to-VWAP distance;
- raw open, close, volume, low and high values.

The final DeepLOB feature width is therefore:

```text
240 OFI features + 19 daily features = 259 features
```

A 50-day window is assembled into the DeepLOB input tensor:

```text
B × 1 × 50 × 259
```

The causality of feature normalization matters: current-day information is not allowed to leak backward into historical normalization statistics.

---

## Stage 2 — DeepLOB Signal Generation

DeepLOB is used strictly as the **signal-generation network**. It does not decide portfolio weights and it does not directly submit trades.

<p align="center">
  <img src="assets/deeplob-dls-network-architecture.svg" alt="DeepLOB and DLS network architecture" width="980">
</p>

The pretrained DeepLOB architecture contains:

1. convolutional feature-extraction blocks;
2. an Inception-style parallel convolution module;
3. a temporal LSTM;
4. a fully connected classification head; and
5. a softmax output layer.

For each asset `i` and signal day `t`, the network produces three probabilities:

```text
P(down), P(flat), P(up)
```

These probabilities form a complete directional state rather than a single hard label. This is important because DLS can distinguish, for example, between a high-confidence bullish signal and a weakly bullish signal dominated by `P(flat)`.

---

## Stage 3 — DeepLOB Probabilities → DLS Features

The three DeepLOB probabilities are converted into a five-dimensional per-asset feature vector.

Two additional quantities are constructed:

```text
signal_score = P(up) - P(down)

confidence = |signal_score| × (1 - P(flat))
```

So each asset/day is represented by:

```text
[P(down), P(flat), P(up), signal_score, confidence]
```

Interpretation:

| Feature | Role |
|---|---|
| `P(down)` | bearish probability |
| `P(flat)` | neutral / low-direction probability |
| `P(up)` | bullish probability |
| `signal_score` | signed directional edge |
| `confidence` | strength of that direction after penalizing flatness |

This converts DeepLOB from a classifier into a richer signal source that the portfolio network can consume.

---

## Stage 4 — Temporal DLS Input Construction

DLS does not look at only one signal day. It consumes the **history of probability panels**.

<p align="center">
  <img src="assets/dls-dataflow-pipeline.svg" alt="DLS data flow" width="860">
</p>

For `N = 2,306` assets, each day contains:

```text
N × 5 = 2,306 × 5 = 11,530 values
```

The notebook stacks a 50-day history:

```text
50 × N × 5
```

Each day's asset-feature panel is flattened to `5N`, producing a temporal sequence that is fed to the DLS LSTM.

This means the allocator receives information about:

- how bullish/bearish probabilities evolve over time;
- how confidence changes;
- how the signal distribution changes across the entire stock universe; and
- how today's opportunity set relates to recent signal history.

---

## Stage 5 — ShifuDLSNet Portfolio Allocation

The DLS network uses:

```text
Input size:   5N = 11,530
Lookback:     50 days
LSTM hidden:  64
Linear head:  64 → N assets
Output:       asset-level logits
```

Before softmax, non-tradable assets are suppressed using a decision-time entry mask. Masked softmax then produces a **long-only target-weight vector**.

Conceptually:

```text
DeepLOB probabilities
        ↓
50-day signal history
        ↓
LSTM representation of market-wide signal state
        ↓
asset-level logits
        ↓
tradability mask
        ↓
masked softmax
        ↓
raw portfolio weights
```

The allocator therefore learns **cross-sectional capital allocation**, rather than simply buying all stocks with `P(up)` above a fixed threshold.

---

## Stage 6 — Portfolio-Level DLS Objective

DLS is trained with a differentiable **portfolio objective**, not with classification accuracy.

<p align="center">
  <img src="assets/dls-objective-components.svg" alt="DLS objective components" width="880">
</p>

The objective combines reward and penalty terms:

### Reward terms

- Sharpe ratio;
- Sortino ratio.

### Risk / trading penalties

- maximum drawdown;
- turnover;
- concentration;
- inventory / gross-exposure deviation;
- explicit commission exposure;
- optional sparsity diagnostics.

The final sparsity coefficient is zero in the documented configuration. The latest notebook also does **not** use an active quadratic market-impact penalty; cost control is primarily represented through turnover/commission-style terms.

This objective lets the model optimize the economic quality of a portfolio path rather than only the accuracy of individual stock predictions.

---

## Stage 7 — Signal Quality and Target-Weight Post-Processing

Raw softmax weights are not immediately traded.

The notebook applies a second layer of portfolio construction rules:

1. remove non-tradable assets;
2. apply a DeepLOB-derived quality filter;
3. remove very small weights;
4. cap the maximum number of positions;
5. protect minimum-holdings feasibility;
6. normalize surviving positions to target gross exposure.

The documented signal-quality thresholds are:

```text
confidence >= 0.02
signal_score >= 0.00
```

If too few names pass, the workflow can fall back to top-confidence tradable assets rather than allowing an infeasible portfolio.

Important portfolio-construction settings include:

| Parameter | Value |
|---|---:|
| Target gross exposure | 0.95 |
| Maximum positions | 500 |
| Minimum target weight | 0.0005 |
| Rebalance band | 0.006 |
| Minimum holdings | 10 |

---

## Stage 8 — No-Lookahead Timing

The workflow explicitly separates signal generation, trading and outcome evaluation:

```text
Day t      : DeepLOB observes information and creates signals
Day t + 1  : portfolio decision is traded
Day t + 2  : future trade outcome is evaluated
```

The final DLS execution configuration uses open-price decision/execution conventions so same-day close information is reserved for end-of-day valuation and metric logging, rather than being used for same-day order sizing.

This timing discipline is one of the central safeguards against lookahead bias in the notebook.

---

## Stage 9 — Execution / Backtest Engine

The execution layer converts **desired portfolio weights** into **feasible orders and fills**.

<p align="center">
  <img src="assets/execution-engine-flowchart.svg" alt="Execution engine flowchart" width="920">
</p>

For every trading day, the engine:

1. computes portfolio value from cash and current holdings;
2. converts current positions into current weights;
3. calculates the gap between target and current weights;
4. ignores small gaps inside the rebalance band;
5. processes sells first to release cash;
6. processes buys subject to available cash and tradability;
7. enforces 100-share lot sizes;
8. reduces order sizes when the full order is not feasible;
9. applies commission and stamp-duty rules;
10. updates cash, shares, realized weights and EOD holdings;
11. writes detailed trade, holding and decision audit records.

Key execution parameters:

| Parameter | Value |
|---|---:|
| Commission rate | 0.0001 |
| Stamp duty rate | 0.0005 |
| Minimum commission | 5 |
| Lot size | 100 |
| Cash buffer | 0.05 |
| Minimum holdings | 10 |

The result is an explicit distinction between:

```text
model target weight
    ≠ submitted order
    ≠ filled trade
    ≠ final end-of-day weight
```

That distinction is what makes the audit layer and dashboard useful.

---

# Notebooks

The repository contains two notebook artifacts:

| Notebook | Purpose |
|---|---|
| [`notebooks/deeplob_dls_asset_allocation.ipynb`](notebooks/deeplob_dls_asset_allocation.ipynb) | Full working DeepLOB + DLS asset-allocation notebook containing the end-to-end research workflow and generated outputs. |
| [`notebooks/deeplob_dls_asset_allocation-version1.ipynb`](notebooks/deeplob_dls_asset_allocation-version1.ipynb) | Versioned/executed notebook snapshot corresponding to the documented Kaggle-style experiment and result reporting. |

## What the notebooks do

The notebook workflow is substantially more than a model-training script. It performs the complete experiment lifecycle:

### 1. Input discovery and runtime setup

The Kaggle-oriented workflow recursively searches `/kaggle/input` for:

- in-sample daily OHLCV parquet data;
- in-sample LOB parquet data;
- out-of-sample daily OHLCV data;
- out-of-sample LOB data;
- the pretrained `best_model_alpha_0015.pt` DeepLOB checkpoint.

Generated artifacts are written under:

```text
/kaggle/working/shifu_dls_oos/
```

and packaged into:

```text
/kaggle/working/shifu_dls_oos_results.zip
```

### 2. Data preparation

The notebook:

- loads in-sample and OOS daily data;
- loads tens of millions of LOB rows;
- constructs the fixed intraday OFI grid;
- performs causal feature normalization;
- builds daily market features;
- aligns asset/day identifiers across sources;
- constructs 50-day DeepLOB feature windows.

### 3. DeepLOB inference

It loads the pretrained checkpoint and produces the three-class probability panel across assets and days.

These raw probabilities are also exported so the original DeepLOB-only strategy and the DLS-enhanced strategy can be inspected from the same underlying signal source.

### 4. DLS dataset construction

The notebook builds the 5-feature probability panel and 50-day temporal DLS samples, together with:

- entry/tradability masks;
- future-return labels;
- return-validity masks;
- volatility scaling inputs;
- chronological train/validation sequences.

### 5. DLS training

The DLS model is trained chronologically using Adam, early stopping and checkpoint restoration. Important settings include:

| Hyperparameter | Value |
|---|---:|
| Lookback | 50 |
| Hidden size | 64 |
| Batch size | 64 |
| Max epochs | 100 |
| Learning rate | 0.001 |
| Early-stopping patience | 15 |

### 6. Seed search

The workflow evaluates multiple random seeds:

```text
7, 11, 22, 33, 42, 55, 77, 88, 101, 123
```

The documented experiment selects seed **33**.

<p align="center">
  <img src="assets/dls-seed-search-ranking.svg" alt="DLS seed search ranking" width="880">
</p>

### 7. Backtesting and baseline comparison

The notebook executes both:

- the original **DeepLOB-only exact baseline**; and
- the **DeepLOB + DLS** portfolio.

This enables direct comparison of return, Sharpe, drawdown, transaction costs, turnover and holdings rather than evaluating DLS in isolation.

<p align="center">
  <img src="assets/oos-performance-comparison.svg" alt="OOS performance comparison" width="860">
</p>

### 8. Audit-output generation

The notebook exports detailed CSVs used by the dashboard, including:

- raw DeepLOB probabilities;
- daily DLS target weights;
- DLS weight-step diagnostics;
- trade-level fills;
- holding snapshots;
- baseline daily logs;
- baseline/DLS submitted orders;
- model-comparison metrics;
- training history;
- seed-search summary.

### 9. Static diagnostics

The run also generates portfolio and weight diagnostics.

<p align="center">
  <img src="assets/static-backtest-report.svg" alt="Static DLS backtest report" width="960">
</p>

<p align="center">
  <img src="assets/weight-distribution-diagnostics.svg" alt="Weight distribution diagnostics" width="900">
</p>

These figures provide a quick view of equity, returns, drawdowns, rolling Sharpe, holdings, transaction costs, target-weight concentration and the evolution of active positions.

---

# Streamlit Dashboard — Trading Engine Replay

The dashboard is intentionally designed as an **audit and replay application**, not as a static chart viewer.

Its core state transition is:

```text
DeepLOB probabilities
    ↓
signal score + confidence
    ↓
entry / tradability mask
    ↓
quality filter
    ↓
DLS target weights
    ↓
submitted orders
    ↓
executed trades
    ↓
end-of-day holdings
    ↓
portfolio state + model comparison
```

The purpose is to answer not only **“How did the strategy perform?”** but also **“Why did this asset receive this action on this day?”**

## Dashboard data model

At every replay step, the app reconstructs a common daily `EngineState` containing the current day and the associated:

- raw DeepLOB signals;
- DLS weight-step rows;
- DLS filled trades;
- DLS holdings;
- DLS debug/optimizer state;
- DLS submitted orders;
- baseline DeepLOB submitted orders.

All views therefore inspect the same replay day and the same underlying engine state.

## Global status strip

The top-level status area shows the current state of the strategy through cards such as:

- **Engine Clock** — current trade day and corresponding signal day;
- **DLS Equity Proxy** — reconstructed DLS equity;
- **Return Proxy** — cumulative DLS return;
- **Target Gross** — optimizer gross-exposure target;
- **Executed Turnover** — actually filled buy/sell notional;
- **EOD Holdings** — active end-of-day positions.

## Pipeline stage line

One of the most useful dashboard components is the six-stage engine line:

```text
Signal Bus
   → Entry Mask
   → Quality Filter
   → DLS Optimizer
   → Execution
   → Portfolio State
```

Each stage has a different interpretation:

### Signal Bus
Shows the raw DeepLOB prediction universe and the distribution of `up`, `flat` and `down` signals. It represents **prediction availability**, not orders.

### Entry Mask
Shows which signaled assets are actually eligible for trading after checking valid execution information and tradability.

### Quality Filter
Shows which eligible assets have enough directional strength and confidence to continue to portfolio allocation.

### DLS Optimizer
Shows how many names receive non-zero DLS target weights and how those weights use the target gross exposure.

### Execution
Shows what was actually traded after rebalance bands, cash constraints, lot sizes, fees and feasibility checks.

### Portfolio State
Shows the resulting holdings after execution.

This visualization makes the transformation from **prediction → allocation → real position** directly inspectable.

---

## Dashboard Views

The app exposes several dedicated views from the sidebar.

### 1. Engine Replay

The main operational view for moving through the 242-day OOS test one day at a time.

It is useful for answering questions such as:

- How many DeepLOB signals existed today?
- How many were tradable?
- How many survived the confidence filter?
- How many names did DLS target?
- How many trades actually filled?
- What did the portfolio hold after execution?

### 2. Decision Inspector

This is the most detailed asset-level audit view.

For an individual asset, the decision chain can include:

- `P(down)`;
- `P(flat)`;
- `P(up)`;
- signal score;
- confidence;
- entry/quality-filter status;
- previous portfolio weight;
- DLS target weight;
- post-trade weight;
- final EOD weight;
- submitted buy/sell percentages;
- filled buy/sell percentages.

This allows a user to trace why a stock was selected, ignored, increased, reduced or left unchanged.

### 3. Execution Tape

Focuses on actual filled transactions and execution reconciliation.

It displays information such as:

- side;
- shares;
- execution price;
- turnover;
- trading cost;
- submitted order percentage;
- filled percentage.

The key purpose is to reveal the difference between what the optimizer wanted and what the execution engine could actually implement.

### 4. Portfolio State

Provides a portfolio-centric view of the current replay day:

- active holdings;
- market value;
- realized portfolio weights;
- number of names;
- cash/equity proxies;
- gross exposure;
- holdings concentration.

### 5. Daily Metrics

Compares same-day engine behavior and portfolio statistics, including:

- DLS vs baseline daily return;
- turnover;
- transaction costs;
- holdings count;
- execution activity;
- cumulative behavior up to the current replay day.

### 6. Model Comparison

A dedicated comparison cockpit for **Base DeepLOB** and **DeepLOB + DLS**.

The dashboard can compare:

- total return;
- Sharpe ratio;
- maximum drawdown;
- portfolio value paths;
- turnover;
- cumulative trading cost;
- number of holdings;
- order divergence / different trading behavior.

This view is especially useful because the goal of the project is not just to run DLS, but to measure whether the allocation layer improves how DeepLOB probabilities are economically used.

### 7. Training Lab

Provides DLS training diagnostics when the corresponding exported files are available, including:

- train/validation loss history;
- early-stopping behavior;
- seed-search information;
- selected-seed diagnostics.

---

## Replay Controls

The sidebar supports interactive navigation through the test period:

- Previous / Next day;
- direct replay-day slider;
- Play / Pause;
- playback speeds such as `0.5x`, `1x`, `2x`, `5x`, `10x`;
- optional loop playback;
- optional `streamlit-autorefresh`;
- fast-playback rendering mode to avoid heavy plotting while the replay is running.

The interface uses a single selected engine view rather than rendering all heavy Streamlit tabs simultaneously, which improves responsiveness for large audit tables and charts.

---

## Dashboard Equity Reconstruction

The DLS replay reconstructs an equity proxy from holdings and target gross exposure:

```text
raw_equity_proxy = EOD market value / engine target gross
```

The path is then normalized to begin at the same initial capital used by the notebook:

```text
RMB 50,000,000
```

This keeps visual comparisons between the notebook metrics and dashboard replay on the same capital scale.

---

## Dashboard Input Files

The dashboard is driven by notebook-generated audit outputs. Important files include:

| File | Dashboard role |
|---|---|
| `T001_original_deeplob_exact_raw_oos_signals.csv` | raw DeepLOB probabilities/signals |
| `T001_dls_weight_step_audit.csv` | asset-level decision chain |
| `T001_dls_weights_long_shifted_tplus2.csv` | long-format target weights |
| `T001_dls_trade_audit.csv` | actual DLS fills |
| `T001_dls_holding_snapshot_audit.csv` | EOD holdings |
| `T001_dls_weight_debug_shifted_tplus2_conf_filter.csv` | optimizer/filter diagnostics |
| `T001_original_deeplob_exact_daily_log.csv` | baseline portfolio state |
| `T001_oos_original_deeplob_exact_sell_close.csv` | baseline submitted orders |
| `T001_oos_shifu_dls_kaggle_sell_open.csv` | DLS submitted orders |
| `T001_comparison_original_deeplob_exact_vs_dls.csv` | final model comparison |
| `shifu_dls_seed_search_summary.csv` | seed-search diagnostics |
| `shifu_dls_training_history_oos.csv` | DLS training history |

---

# Repository Structure

```text
DeepLOB-DLS-Asset-Allocation/
├── notebooks/
│   ├── deeplob_dls_asset_allocation.ipynb
│   └── deeplob_dls_asset_allocation-version1.ipynb
├── dashboard/
│   ├── app.py
│   ├── README.md
│   ├── requirements.txt
│   └── .streamlit/
│       └── config.toml
├── docs/
│   ├── README.md
│   ├── DeepLOB_DLS_Report_documentation.pdf
│   └── DeepLOB_DLS_Asset_Allocation_Documentation.pdf
├── assets/
│   ├── architecture-pipeline.svg
│   ├── deeplob-dls-network-architecture.svg
│   ├── dls-dataflow-pipeline.svg
│   ├── dls-objective-components.svg
│   ├── execution-engine-flowchart.svg
│   ├── dls-seed-search-ranking.svg
│   ├── oos-performance-comparison.svg
│   ├── static-backtest-report.svg
│   └── weight-distribution-diagnostics.svg
├── CONTRIBUTIONS.md
└── README.md
```

---

# Running the Dashboard

Install dependencies:

```bash
pip install -r dashboard/requirements.txt
```

Place the notebook-exported CSV files in the dashboard data directory expected by the app, then run:

```bash
streamlit run dashboard/app.py
```

For full dashboard-specific documentation, see [`dashboard/README.md`](dashboard/README.md).

---

# Limitations and Future Work

Current documented limitations include:

- long-only DLS output due to masked softmax;
- no active quadratic market-impact term in the current loss;
- development-time seed selection using OOS total return;
- a DLS input size tied to the fixed asset universe;
- evaluation on one main OOS period.

Potential extensions include:

- explicit nonlinear / quadratic trading-impact penalties;
- validation-only model-selection protocols;
- attention or graph-based cross-asset architectures;
- sector / industry risk constraints;
- equal-weight, top-confidence and mean-variance allocation baselines;
- walk-forward evaluation across multiple OOS regimes;
- more realistic slippage and market-impact calibration.

---

# Contribution

The contribution of this module is a full **research-to-execution workflow**, not just a neural network:

```text
market data
→ causal feature engineering
→ DeepLOB prediction
→ DLS signal representation
→ learned asset allocation
→ execution-aware portfolio construction
→ realistic order simulation
→ detailed audit outputs
→ interactive replay and model comparison
```

For contribution provenance and original branch history, see [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).

---

## Author

**Mohammadreza Sheikholeslami**  
GitHub: `MohammadrezaSheikholeslami84`

---

## Disclaimer

This repository is intended for **research and educational purposes only**. Historical and backtested performance does not guarantee future results.