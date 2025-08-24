AI-Driven Optimization of 1-Day VaR (95%) for UK Banks

Reproducible research code for forecasting next-day 95% Value-at-Risk (VaR) on a portfolio of UK bank equities using classical risk models and deep quantile learning with macro & news-sentiment features.

🔎 Project summary

This repository implements an end-to-end pipeline to estimate 1-day VaR (95%) for an equal-weighted basket of BARC.L, HSBA.L, LLOY.L, NWG.L, STAN.L. We benchmark GARCH-t, Historical Simulation (HS) and Filtered Historical Simulation (FHS) against an LSTM Quantile Regression (LSTM-QR) and two hybrids:

Hybrid-Feat: LSTM-QR with engineered features including GARCH conditional volatility.

Hybrid-Ens: convex ensemble of GARCH/FHS/LSTM VaRs.

Exogenous drivers include realised volatility, macro state (BoE Bank Rate, UK 10Y yield) and FinBERT news sentiment (bank-specific & macro policy). Evaluation uses Kupiec & Christoffersen coverage tests plus pinball loss; explainability uses SHAP and permutation importance. The workflow and artifacts align with model-governance practices (PRA SS1/23).

🚀 Highlights

Fully scripted data → features → models → backtests → XAI → reports.

Reproducible splits, fixed seeds, saved configs and artifacts.

Works in Google Colab (GPU optional) or local Python 3.11.

📁 Notebook structure
.
├── notebooks/
│   ├── 01_environment_setup.ipynb
│   ├── 02_install_and_import_dependencies.ipynb
│   ├── 03_download_uk_bank_market_data.ipynb
│   ├── 04_get_macro_and_sentiment_factors.ipynb
│   ├── 05_collect_news_and_compute_finbert_sentiment.ipynb
│   ├── 06_data_preprocessing_and_feature_engineering.ipynb
│   ├── 07_exploratory_data_analysis.ipynb
│   ├── 08_build_baseline_var_models.ipynb
│   ├── 09_build_lstm_quantile_regression_model.ipynb
│   ├── 10_optional_hybrid_garch_lstm.ipynb
│   ├── 11_generate_var_and_backtesting.ipynb
│   ├── 12_visualize_and_compare_results.ipynb
│   ├── 13_explainability_analysis_xai.ipynb
│   ├── 14_save_models_and_results.ipynb
│   └── 15_report_and_reproducibility.ipynb


Outputs (figures, tables, models) are written under outputs/:

outputs/
  ├─ data/       # cached raw/processed
  ├─ features/   # engineered tables
  ├─ models/     # .pt/.pkl + VaR series
  ├─ eda/        # diagnostics
  └─ reports/    # comparison CSVs & figures

🧩 Data sources

Market prices: Yahoo Finance via yfinance (adjusted close).

Macro: FRED BOERUKM (BoE Bank Rate), IRLTLT01GBM156N (UK 10Y).

Sentiment: Kaggle news archives (e.g., News_Category_Dataset_v3.json, NYT articles) → filtered by UK-bank & macro keywords → FinBERT polarity → daily indices.

Kaggle setup (local or Colab):

Create a Kaggle API token (Account → Create New Token).

Place kaggle.json under ~/.kaggle/ (Linux/Colab) and set permission 600.

Example:

kaggle datasets download -d rmisra/news-category-dataset -p data/sentiment --unzip


Use only datasets that are public and compatible with your licence. Replace or add sources as needed; the sentiment module reads any CSV/JSON with date + text.

⚙️ Installation
Colab (recommended)

Open the notebooks in notebooks/ in order. Each notebook cell installs required packages and mounts Google Drive if desired.

Local
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

🔧 Configuration

All key settings live in configs/default_config.json:

{
  "tickers": ["BARC.L", "HSBA.L", "LLOY.L", "NWG.L", "STAN.L"],
  "alpha": 0.05,
  "windows": {"hs": 250, "rv": 30, "sent_ma": 7, "seq_len": 60},
  "train": {"epochs": 40, "batch_size": 64, "lr": 1e-3, "hidden": 64, "dropout": 0.2, "patience": 5},
  "paths": {"root": "outputs", "data": "outputs/data", "features": "outputs/features", "models": "outputs/models"}
}

📊 Reproducing paper figures & tables

Running notebooks 06 → 13 will generate:

VaR overlays vs returns (GARCH/HS/FHS/LSTM/Hybrids).

Breach heatmaps and coverage tables (Kupiec/Christoffersen, pinball, mean-excess).

Global SHAP bars, beeswarm, and permutation-importance bars.

A snapshot of out-of-sample results (example values from the study):

Model	#Breaches	BreachRate	Kupiec_p	Christoff_p	PinballLoss (×1e−4)	MeanExcess
GARCH	28	3.29%	0.015	0.042	7.982	0.623
HS	37	4.35%	0.377	0.017	7.082	0.706
FHS	44	5.18%	0.814	0.088	6.917	0.590
LSTM-QR	44	5.18%	0.814	0.249	6.531	0.587
Hybrid-Feat	39	4.59%	0.577	0.319	6.790	0.611
Hybrid-Ens	33	3.88%	0.120	0.125	7.241	0.642

Exact values will regenerate from your local data cut and configuration.

🧠 Explainability

SHAP: global bars & beeswarm on sequence features (abs_ret, rv_30d, sent_bank_7d, macro z-scores).

Permutation importance under pinball loss to quantify performance degradation when scrambling a feature.

Artifacts saved to outputs/reports/.

🧪 Testing & determinism

Fixed temporal split (train/val/test).

Seeds set for NumPy/PyTorch; early-stopping on validation pinball loss.

GARCH refit cadence and HS/FHS windows recorded in config.json.

🛡️ Governance, ethics & licence

SS1/23 alignment: challenger baselines retained; coverage & loss monitoring; config versioning; XAI records; change control via saved artifacts.

Ethics: only public/licensed data; no personal data; sentiment built from licensed archives; potential source bias mitigated via domain filtering and robustness checks.

Disclaimer: research code; not investment advice.

Licence: see LICENSE.

❓ Troubleshooting

Kaggle 403/404: dataset may be private/removed—switch to an alternative public dataset; the sentiment loader accepts any date+text CSV/JSON.

yfinance column Adj Close KeyError: use auto_adjust=True or access ['Close'] accordingly.

FRED gaps: series missing early years—forward-fill after converting to business-day frequency.

CUDA/CPU: SHAP for LSTMs may fall back to CPU; limit explanation sample size for speed.

🌱 Contributing

Issues and pull requests are welcome. Please keep changes modular (one feature/fix per PR) and include a short note on reproducibility impact.

📌 Citation

If you use this repository, please cite the dissertation/project and this codebase. A BibTeX entry template is included in CITATION.cff.

🙏 Acknowledgements

Thanks to the maintainers of arch, statsmodels, PyTorch, transformers, shap, yfinance, and FRED for open tooling and data access that made this project possible.