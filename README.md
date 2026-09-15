# DeepLOB-DLS-Asset-Allocation

DeepLOB signal generation combined with a Deep Learning Sharpe (DLS) portfolio-allocation layer, with an event-driven Streamlit replay dashboard and technical documentation.

This repository extracts the Asset Allocation work developed on the `asset-allocation` branch of `Alireza77n/LOB-Market-Making` and organizes it as a standalone project.

## Structure

```text
notebooks/
  deeplob_dls_asset_allocation.ipynb
  deeplob_dls_asset_allocation-version1.ipynb

dashboard/
  app.py
  README.md
  requirements.txt
  .streamlit/config.toml

docs/
  DeepLOB_DLS_Report_documentation.pdf
  DeepLOB_DLS_Asset_Allocation_Documentation.pdf
```

## Main components

- DeepLOB-based directional signal generation
- DLS portfolio allocation
- no-lookahead execution/backtesting workflow
- transaction-cost-aware portfolio simulation
- asset-level decision and execution audit
- Streamlit trading-engine replay dashboard
- DeepLOB-only vs DeepLOB + DLS comparison

## Original development context

The code and documentation in this repository were developed by Mohammadreza Sheikholeslami as contributions to the `asset-allocation` branch of the LOB-Market-Making project. This standalone repository is intended to collect those Asset Allocation components in one place.
