# APS Failure Prediction — Predictive Maintenance for Scania Trucks

![CI](https://github.com/dennisdang5/aps-failure-prediction/actions/workflows/ci.yml/badge.svg)

An end-to-end MLOps pipeline for predicting Air Pressure System failures in Scania trucks.

## Results

| Model | Test AUC | FP | FN | Cost |
|---|---|---|---|---|
| Random Forest (imbalanced) | 0.99 | 15 | 106 | $53,150 |
| Random Forest (balanced) | 0.99 | 14 | 160 | $80,140 |
| XGBoost Linear | 0.91 | 134 | 64 | $33,340 |
| XGBoost + SMOTE | 0.99 | 322 | 30 | $18,220 |

**Best model:** XGBoost with SMOTE achieved the lowest cost ($18,220) by catching 96% of failures, despite having a higher raw misclassification rate. This demonstrates why domain-specific cost metrics matter more than accuracy for imbalanced problems.

## Problem

The Air Pressure System (APS) generates pressurized air for critical truck functions like braking and gear changes. Predicting APS failures before they happen lets fleet operators schedule maintenance proactively, avoiding costly roadside breakdowns.

This is a cost-sensitive binary classification problem with a 59:1 class imbalance across 170 anonymized sensor features. A missed failure (false negative) costs `$500` thus a truck breaks down on the road. An unnecessary inspection (false positive) costs only `$10` where a mechanic checks a healthy truck. This 50x cost asymmetry makes standard accuracy a misleading metric and drives the entire modeling strategy.

## Key Findings

- **Class balancing doesn't always help.** Balanced Random Forest performed worse than the imbalanced baseline on the cost metric ($80,140 vs $53,150), thus producing more false negatives despite being designed to catch more failures.
- **Misclassification rate is misleading.** A naive "predict all negative" model achieves 98.3% accuracy but misses every single failure. The cost metric reveals the true picture.
- **SMOTE + XGBoost wins on cost.** By generating synthetic minority samples inside cross-validation (avoiding data leakage), the model catches 96% of failures at the expense of more false positives. This is exactly the tradeoff the cost structure rewards.
- **OOB error reliably estimates generalization.** Out-of-bag error and test error were consistently close across Random Forest experiments, confirming OOB as a dependable proxy for generalization performance.
- **The break-even inspection threshold is 2%.** Given the cost asymmetry (`$500` / `$10`), it's worth inspecting a truck even at a 2% failure probability. The API uses this threshold for its recommended action.

## Architecture

```
User/Dashboard ──→ FastAPI Prediction Service ──→ Trained Model (joblib)
                            │
                       MLflow Tracking
                            │
DVC Pipeline: Preprocess ──→ Train ──→ Evaluate ──→ Monitor (Evidently)
                            │
              GitHub Actions CI/CD ──→ GitHub Container Registry
```

The system separates concerns across layers: modular Python libraries (`src/`) handle the ML logic, a FastAPI service exposes predictions over HTTP, a Streamlit dashboard provides a user-facing front-end, and DVC orchestrates the reproducible training pipeline. MLflow tracks experiments during development, while the serving container loads a pinned model artifact for production stability.

## Project Structure

```
aps-failure-prediction/
├── .github/workflows/
│   └── workflows/
│       ├── ci.yml             # Lint (ruff) + tests (pytest) on every push
│       └── docker.yml         # Build and push serving image to GHCR
├── data/                      # Raw data (DVC tracked, not in git)
│   └── processed/             # Processed parquet files (generated)
├── docker/
│   ├── Dockerfile.train       # Training pipeline container
│   └── Dockerfile.serve       # Prediction API container
├── models/                    # Trained model artifacts
├── notebooks/
│   └── pipeline.ipynb         # Exploration and analysis notebook
├── results/                   # Model comparison CSV, drift reports
├── src/
│   ├── config.yml             # All hyperparameters and settings
│   ├── feature_names.json     # Expected feature names for API validation
│   ├── data_loader.py         # Data loading and config management
│   ├── preprocessing.py       # Imputation, encoding, feature splitting
│   ├── eda.py                 # Visualizations and feature analysis
│   ├── models.py              # Model training (RF, XGBoost, SMOTE)
│   ├── evaluate.py            # Metrics, cost computation, ROC curves
│   ├── serve.py               # FastAPI prediction service
│   ├── run_preprocessing.py   # DVC pipeline stage
│   ├── run_train.py           # DVC pipeline stage
│   ├── run_evaluate.py        # DVC pipeline stage
│   └── run_monitoring.py      # Data drift detection with Evidently
├── tests/
│   ├── test_evaluate.py       # Unit tests for evaluation metrics
│   ├── test_preprocessing.py  # Unit tests for data preprocessing
│   └── test_api.py            # Integration tests for API endpoints
├── dashboard.py               # Streamlit front-end
├── docker-compose.yml         # Orchestrate train + serve containers
├── dvc.yaml                   # Reproducible pipeline definition
├── pyproject.toml             # Package configuration and pytest settings
├── requirements.txt           # Python dependencies
└── environment.yml            # Conda environment specification
```

## Tech Stack

- **ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE), pandas, NumPy
- **MLOps:** MLflow (experiment tracking, model registry), DVC (data versioning, pipeline orchestration)
- **Serving:** FastAPI, Uvicorn, Pydantic (input validation), Streamlit (dashboard)
- **Containerization:** Docker, docker-compose
- **CI/CD:** GitHub Actions (linting, testing, Docker image builds), GitHub Container Registry
- **Monitoring:** Evidently (data drift detection)
- **Testing:** pytest, ruff (linting)
- **Visualization:** matplotlib, seaborn

## Quick Start

### Setup

```bash
git clone https://github.com/dennisdang5/aps-failure-prediction.git
cd aps-failure-prediction
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
pip install -e .
```

### Download the data

Download from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) and place the CSVs in `data/`.

### Run the DVC pipeline

```bash
dvc repro
```

This executes the full pipeline: preprocess → train → evaluate. DVC caches results and only reruns stages whose inputs changed.

### Start the prediction API

```bash
uvicorn src.serve:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### Launch the dashboard

```bash
streamlit run dashboard.py
```

Note: the API must be running for the dashboard to work.

### Run with Docker

```bash
docker compose build
docker compose up serve
```

### Run tests

```bash
pytest -v
```

### Run drift monitoring

```bash
python src/run_monitoring.py
```

Produces an HTML drift report and a JSON summary in `results/`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health check and model status |
| POST | `/predict` | Predict APS failure from sensor readings |
| GET | `/metrics` | Service metrics and prediction count |

### Example prediction request

```json
POST /predict
{
  "features": {
    "aa_000": 60.0,
    "ab_000": 0.0,
    "ac_000": 20.0,
    "ad_000": 12.0,
    "..."
  }
}
```

### Example response

```json
{
  "prediction": 0,
  "failure_probability": 0.037,
  "predicted_class": "no_failure",
  "recommended_action": "inspect"
}
```

The `recommended_action` uses a cost-aware threshold of 2% — even a low failure probability justifies an inspection because missing a failure costs 50x more than an unnecessary check.

## Monitoring

Data drift detection runs via Evidently, comparing the statistical distribution of new data against the training baseline using Kolmogorov-Smirnov tests across all 170 features.

```bash
python src/run_monitoring.py
```

Outputs:
- `results/drift_report.html` — interactive visual drift report
- `results/drift_summary.json` — machine-readable summary with drift status

The script exits with code 1 if significant drift is detected (>50% of features drifted), enabling integration with automated alerting systems.

## Data

[APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) — UCI Machine Learning Repository

- 60,000 training / 16,000 test samples
- 170 anonymized sensor features (numerical counters and histogram bins)
- Binary classification (APS failure vs. other component failure)
- Heavy class imbalance (59,000 negative : 1,000 positive)
- Missing values encoded as "na" in the raw data