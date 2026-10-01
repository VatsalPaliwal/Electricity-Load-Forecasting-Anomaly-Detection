# Electricity-Load-Forecasting-Anomaly-Detection

An end-to-end MLOps pipeline for forecasting hourly electricity demand on
the PJM East interconnection, with a residual-based anomaly detection
layer evaluated as an additional forecasting input. A historical replay
simulation (sequential reveal over 2017–2018) stands in for live grid data
ingestion.

## Overview

The project forecasts hourly load (`PJME_MW`) using XGBoost, and
separately trains an Isolation Forest to detect anomalous demand periods.
The central question the project investigates: **does information about
recent anomalies improve forecast accuracy, particularly around anomalous
periods themselves?** Two XGBoost variants — a baseline model and an
anomaly-aware model — are trained under identical conditions and compared
on this basis.

The pipeline was originally scoped around live ingestion from PJM and
ERCOT grid APIs. After repeated upstream API failures (a broken signup
flow, failing `gridstatus` calls, and a 403 from the ERCOT fallback), the
design was revised to use a historical replay simulation instead —
sequentially revealing 2017–2018 data (predict → reveal actual → score →
update state → next) to preserve a live-feeling system without depending
on unreliable external APIs. The scope also expanded from standalone
anomaly detection into a forecasting pipeline with anomaly-aware features
as a tested hypothesis rather than an assumed improvement.

## Data

- **Source:** [PJME_hourly.csv](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
  (Kaggle — `robikscube/hourly-energy-consumption`)
- **Range:** 2002–2018, ~145k hourly rows
- **Columns:** `Datetime`, `PJME_MW` (hourly load in megawatts)

## Exploratory Data Analysis

Full EDA lives in `notebooks/eda.ipynb`. Key findings:

- **Structure/integrity:** 30 missing hours and 4 duplicate timestamps, both
  explained by daylight saving time transitions.
- **Trend:** no long-term trend, but high year-to-year volatility (annual
  peaks ranging 52,000–62,000 MW).
- **Seasonality at three levels:**
  - *Annual* — summer-dominant (AC load), winter-secondary (heating).
  - *Weekly* — weekday plateau vs. weekend drop; an unexplained Monday dip
    not accounted for by holidays.
  - *Daily* — single broad peak (~5–7 PM) in summer; dual peak (morning +
    evening) in winter. Winter load actually exceeds summer in early
    morning hours.
- **Holidays:** a real effect — roughly 5.8% load reduction vs. regular
  weekdays, once weekends are correctly excluded from the comparison.
- **ACF/PACF:** confirmed daily (lag 24) and weekly (lag 168) cycles; PACF
  showed lag 24 has a real but non-trivial, partly corrective (negative
  sign) direct effect.
- **Stationarity:** non-stationary (ADF), driven by seasonality rather than
  trend.
- **MSTL decomposition** (daily-resampled, period=[7, 365]) gave a clean
  trend/seasonal/residual split, independently validated against the real
  August 2006 heat wave — residual rose ~8,400–8,600 MW across the event.

## Feature Engineering

Eight EDA-justified features: `lag_1`, `lag_24`, `lag_168`, `hour`,
`dayofweek`, `is_weekend`, `month`, `is_holiday`. Season, year, and the MSTL
residual were deliberately excluded — the residual specifically because
MSTL uses future data to compute historical residuals, which would leak
into live scoring.

`preprocessing.py` exposes a single `preprocess(df)` function used
identically wherever features are needed, so training and scoring can never
drift apart.

## Pipeline Architecture

```
load_data (raw)
    → preprocess()
    → chronological split (train 2002–2014 / val 2015–2016 / test 2017–2018)
    → train_data / val_data / test_data (preprocessed, stored once — static dataset)

Isolation Forest
    → fit on train only
    → applied unchanged across train + val (test held out until final eval)
    → produces: anomaly_score, is_anomaly (current hour)
                anomaly_score_24h_mean, anomaly_score_24h_max (backward-looking, leakage-safe)

XGBoost Forecasting
    → baseline variant: 8 EDA features only
    → anomaly-aware variant: 8 features + anomaly_score_24h_mean + anomaly_score_24h_max
    → current-hour anomaly_score/is_anomaly excluded from model input (would leak —
      Isolation Forest is itself fit on PJME_MW, so the current-hour score
      can't be computed before that hour's actual load is known)

Replay Simulation (2017–2018)
    → sequential reveal: predict → reveal actual → score → update state → next
    → strict no-future-data boundary

Evidently drift monitoring → applied across the replay period
MLflow → experiment tracking for both Isolation Forest and XGBoost
FastAPI → serves the forecasting + anomaly pipeline
Streamlit → single-page dashboard: load chart with anomalies marked
```

A raw-storage philosophy runs through the whole project: `load_data` never
receives derived features back, and `preprocess()` is the single source of
truth for how raw data becomes model input — mirrored in how Isolation
Forest's output is kept in its own columns rather than silently recomputed
in multiple places.

## Why Forecasting, Not Just Anomaly Detection

The original Isolation Forest (fit on all 8 raw-feature columns across the
full date range) is still in the repo and was independently validated: it
cleanly flagged the real August 2006 heat wave, and its own
feature-comparison table made sense — `is_weekend` showed no anomaly
differentiation (a context feature, not a signal), `lag_168` showed a
smaller gap than `lag_1`/`lag_24` (anomalies deviate more from recent
history than from the weekly pattern), and `is_holiday` showed a large gap
(anomalies disproportionately land on holidays).

That version is superseded, not discarded — the final pipeline uses a
**train-only** Isolation Forest (to stay leakage-safe once its output feeds
into a forecasting model) and evaluates a specific question: does recent
anomaly information improve load forecasting, overall and specifically
around anomalous periods.

## Baselines

Two seasonal-naive baselines (previous-day, previous-week — i.e., `lag_24`
and `lag_168` used directly as predictions) establish the floor any real
model needs to beat:

| Baseline | MAE | RMSE |
|---|---|---|
| Previous-day (lag_24) | 2,136.77 | 2,935.75 |
| Previous-week (lag_168) | 3,313.02 | 4,525.32 |

Previous-day substantially outperforms previous-week (~55% lower error on
both metrics), with a near-identical RMSE/MAE ratio (~1.37) across both —
indicating lag_168's weakness is a general, evenly-distributed effect
rather than a few large tail errors. This confirms daily continuity is a
much stronger raw predictor than weekly continuity for this series.

## Forecasting Results

Both XGBoost variants were trained once on `train` and evaluated once on
`val` (full walk-forward retraining was scoped down to a single
train/evaluate pass with monthly error-stability reporting, given project
time constraints — noted here as a stated limitation rather than hidden).

| Model | MAE | RMSE |
|---|---|---|
| Baseline XGBoost (8 features) | 320.15 | 429.59 |
| Anomaly-aware XGBoost (+ 2 rolling anomaly features) | 327.53 | 436.75 |

Both variants are a dramatic improvement over the naive baselines (320–330
MAE vs. 2,136+ for the best naive approach), confirming XGBoost learns real
structure from the lag/calendar features well beyond simple lookup.

On the full validation set, the anomaly-aware variant was **consistently,
slightly worse** than baseline — not concentrated in one outlier month, but
distributed across most of 2015–2016, with a sharper divergence in
September 2016.

**[ ADD: anomalous-period-specific results here — MAE/RMSE for both
variants on val hours flagged by `is_anomaly`, and the final verdict on
whether anomaly information helps specifically around anomalies even
though it doesn't help on average. Also add final hyperparameters and
feature-importance findings once decided. ]**

**[ ADD: final untouched test-set (2017–2018) results for whichever
variant was chosen as production, once that evaluation is run. ]**

## Final Model Decision

**[ State here which variant (baseline or anomaly-aware) was chosen as the
production forecaster, and why — including an honest note if the result
was a negative finding (anomaly info didn't help), since that's still a
valid and defensible research outcome. ]**

## Leakage Safeguards

A few decisions were made specifically to keep the pipeline honest:

- `preprocess()` runs on the full continuous series *before* splitting, so
  lag features can correctly look back across split boundaries without
  truncation — this is not leakage, since lag features only use
  already-known past values.
- Isolation Forest is fit on `train` only and applied unchanged to `val`
  and (later) `test` — never refit on data it's meant to be evaluated
  against.
- The current-hour `anomaly_score`/`is_anomaly` are excluded from
  forecasting input, since Isolation Forest itself uses `PJME_MW` and so
  cannot be computed before the target hour's actual load is known. Only
  the 24-hour backward-looking rolling mean/max (shifted to exclude the
  current hour) are used as forecasting features.
- `test_data` (2017–2018) is not touched, scored, or inspected until a
  single final evaluation pass, to avoid iterative peeking during
  development.

## Repository Structure

```
project/
├── pjm_load.db
├── preprocessing.py
├── split.py
├── isolation_forest/
│   ├── train.py
│   └── generate_features.py
├── forecasting/
│   ├── train_xgboost.py
│   └── evaluate.py
├── replay/
│   └── simulate.py
├── monitoring/
│   └── drift.py
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── notebooks/
│   └── eda.ipynb
├── models/
└── mlruns/
```

## Tech Stack

Python, pandas, scikit-learn, XGBoost, MLflow, Evidently, FastAPI,
Streamlit, SQLite.

## Known Limitations

- Walk-forward validation was scoped down to a single train/evaluate split
  with monthly stability reporting, rather than full expanding-window
  retraining, due to project time constraints.
- Anomaly detection features were tested as forecasting inputs only in
  their backward-looking (24h rolling) form; a 1-hour-lagged score/flag
  variant was considered but not built.
- Isolation Forest includes the raw target (`PJME_MW`) among its own input
  features, so its current-hour output cannot be used directly as a
  forecasting feature — a version trained only on prediction-time-available
  features (lags + calendar) was discussed as a natural extension but not
  implemented.

## Project Status

### ✅ Completed
- [x] EDA (structure/integrity, seasonality at 3 levels, holiday effect, ACF/PACF, MSTL decomposition validated against the August 2006 heat wave)
- [x] `preprocessing.py` — 8 EDA-justified features
- [x] Chronological split (train 2002–2014 / val 2015–2016 / test 2017–2018), committed to `train_data` / `val_data` / `test_data`
- [x] Isolation Forest (original, full-data version) — validated against the August 2006 heat wave and its own feature-comparison table
- [x] Isolation Forest retrained on `train` only; `generate_features.py` produces `anomaly_score`, `is_anomaly`, `anomaly_score_24h_mean`, `anomaly_score_24h_max` for `train` + `val`
- [x] Seasonal-naive baselines (previous-day, previous-week) scored on `val`
- [x] Baseline XGBoost (8 features) trained on `train`, scored on `val`
- [x] Anomaly-aware XGBoost (8 features + 2 rolling anomaly features) trained on `train`, scored on `val`
- [x] Monthly MAE/RMSE stability comparison between the two XGBoost variants

### 🚧 Left to do
- [ ] Final modelling decisions (baseline vs. anomaly-aware, including the anomalous-period-specific comparison and feature importance)
- [ ] Final test run (untouched 2017–2018 evaluation)
- [ ] Wrap up the full pipeline (replay simulation, drift monitoring, MLflow/FastAPI/Docker)
- [ ] Streamlit frontend

## Acknowledgments

Dataset: [PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
by Rob Mulla (Kaggle).
