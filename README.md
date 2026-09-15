# DeepLOB–DLS Asset Allocation Module

A complete **asset-allocation extension** for the `LOB-Market-Making` project.
This module takes **DeepLOB directional probabilities** and turns them into an executable **portfolio strategy** using a **DLS portfolio-allocation network**, a realistic **execution/backtesting engine**, and a replayable **Streamlit dashboard**.

---

## Overview

The end-to-end workflow is:

**LOB + OHLCV features → DeepLOB signal model → class probabilities → DLS feature panel → DLS portfolio allocator → target weights → execution/backtest engine → metrics + dashboard replay**

Unlike a simple probability-threshold rule, this module learns how much capital should be assigned to each stock while accounting for turnover, drawdown, concentration, transaction cost, and execution constraints.

---

## Key Results

Latest executed notebook snapshot:

| Metric | Value |
|---|---:|
| Notebook | `deeplob-dls-version1.ipynb` |
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

### Improvement vs. the original DeepLOB-only exact baseline

| Model | Return | Sharpe | MDD | Score | Avg Holdings |
|---|---:|---:|---:|---:|---:|
| Original DeepLOB-only exact | 40.88% | 0.96 | -24.91% | 15.05 | 692.7 |
| **DeepLOB + DLS** | **46.11%** | **1.20** | **-15.77%** | **20.41** | **235.3** |

**DeepLOB + DLS** improves return, Sharpe ratio, drawdown, and score proxy while producing a more selective portfolio.

---

## System Architecture

<p align="center">
  <img src="assets/architecture-pipeline.png" alt="End-to-end DeepLOB + DLS pipeline" width="840">
</p>

<p align="center"><em>End-to-end DeepLOB + DLS portfolio pipeline.</em></p>

---

## Model Architecture

### 1) DeepLOB + DLS network overview

<p align="center">
  <img src="assets/deeplob-dls-network-architecture.png" alt="DeepLOB and DLS network architecture" width="960">
</p>

**DeepLOB** is used as the **signal generation model**. For each asset and day, it outputs `pdown`, `pflat`, and `pup`. These probabilities are then transformed into DLS features and fed into the portfolio-allocation model.

### 2) DLS data-flow pipeline

<p align="center">
  <img src="assets/dls-dataflow-pipeline.png" alt="DLS data flow" width="820">
</p>

The DLS feature vector for each asset/day is built from `pdown`, `pflat`, `pup`, `signal_score = pup - pdown`, and `confidence = |signal_score| × (1 - pflat)`. A 50-day temporal window is then passed through an LSTM-based allocator to produce **long-only target portfolio weights**.

---

## DLS Objective Function

The DLS model is not trained with a classification loss. It is trained with a **portfolio-level differentiable objective** designed to reward better risk-adjusted returns while penalizing undesirable trading behavior.

<p align="center">
  <img src="assets/dls-objective-components.png" alt="DLS objective components" width="860">
</p>

Active terms include **Sharpe**, **Sortino**, **drawdown**, **turnover**, **concentration**, **inventory/gross-exposure**, and **commission** terms. The optional sparsity term is configured as zero in the final notebook.

---

## Execution Engine

The execution engine converts theoretical target weights into feasible trades while respecting tradability masks, rebalance bands, lot-size rules, cash buffers, minimum holdings, commissions, stamp duty, sell-before-buy sequencing, and partial-fill feasibility checks.

<p align="center">
  <img src="assets/execution-engine-flowchart.png" alt="Execution engine flowchart" width="900">
</p>

This bridges the gap between **model output** and **executable trading behavior**.

---

## Seed Search and Model Selection

Multiple DLS seeds were trained and evaluated. The selected seed in the documented run is **33**.

<p align="center">
  <img src="assets/dls-seed-search-ranking.png" alt="DLS seed search ranking" width="880">
</p>

---

## Out-of-Sample Performance Comparison

<p align="center">
  <img src="assets/oos-performance-comparison.png" alt="OOS performance comparison" width="860">
</p>

The DLS allocation layer improves total return, risk-adjusted performance, drawdown control, and score proxy relative to the original DeepLOB-only exact baseline.

---

## Static Backtest Diagnostics

<p align="center">
  <img src="assets/static-backtest-report.png" alt="Static DLS backtest report" width="960">
</p>

The report includes portfolio value, daily-return distribution, drawdown, rolling Sharpe, holdings, and transaction-cost behavior.

---

## Weight Distribution Diagnostics

<p align="center">
  <img src="assets/weight-distribution-diagnostics.png" alt="Weight distribution diagnostics" width="900">
</p>

These charts inspect positive DLS target weights, log-weight behavior, active names over time, and max/median target weights.

---

## Repository Structure

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
│   └── README figures
├── CONTRIBUTIONS.md
└── README.md
```

---

## Dashboard

The Streamlit dashboard is an **interactive trading replay system**, not just a CSV viewer. It reconstructs:

**signals → filters → target weights → orders → executions → holdings → portfolio state**

It visualizes DeepLOB probabilities, DLS target weights, tradability masks, signal-quality filters, buy/sell orders, executed trades, holdings, transaction costs, portfolio value, drawdown, turnover, DeepLOB-only vs DeepLOB+DLS comparison, and asset-level audit tables.

### Run the dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

---

## Technical Notes

The final DeepLOB feature tensor combines a 10-level OFI grid with daily OHLCV features for a final width of **259 features**. The DLS allocator uses a **50-day lookback**, hidden size **64**, long-only masked-softmax output, target gross exposure **0.95**, maximum **500** positions, and rebalance band **0.006**.

Execution parameters include lot size **100**, commission rate **0.0001**, stamp duty **0.0005**, cash buffer **0.05**, and minimum holdings **10**.

---

## Limitations and Future Work

Current limitations include a long-only output, no active quadratic market-impact term in this notebook version, development-time seed selection using OOS return, a fixed asset-universe input dimension, and evaluation on one OOS period.

Potential extensions include quadratic impact penalties, validation-only model selection, attention/graph-based cross-asset modeling, additional risk constraints, stronger baseline comparisons, and walk-forward evaluation.

---

## Contribution

The contribution of this module is a full **research-to-execution workflow**: data processing, DeepLOB signal generation, DLS allocation learning, a code-verified portfolio objective, transaction-cost-aware execution, audit outputs, documentation, and an interactive replay dashboard.

For the original contribution history and provenance, see [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).

---

## Author

**Mohammadreza Sheikholeslami**  
GitHub: `MohammadrezaSheikholeslami84`

---

## Disclaimer

This repository is intended for **research and educational purposes only**. Historical and backtested performance does not guarantee future results.
