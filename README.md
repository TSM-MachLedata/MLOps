# Football Match Prediction — MLOps Pipeline

This repository houses a production-ready MLOps pipeline for predicting football match outcomes across multiple leagues and seasons.  
It automates the entire lifecycle: data ingestion, preprocessing, model training, drift monitoring, conditional retraining, champion selection, API serving, and dashboard visualisation.

Everything is:
- versioned with DVC
- logged with MLflow
- containerised with Docker
- deployed on Google Cloud Run

---

## Useful Links

- MLflow Experiments (DagsHub): https://dagshub.com/LEBARMS/MLOps/experiments  
- Live API (Swagger / OpenAPI docs): https://football-mlops-api-1089778705681.europe-west4.run.app/docs#/default/predict_model2_predict_model2_post

---

## Table of Contents

- [Features](#features)
- [Pipeline Overview](#pipeline-overview)
- [Quick Start](#quick-start)
- [API Usage](#api-usage)
- [Monitoring and Retraining](#monitoring-and-retraining)
- [CI/CD and Deployment](#cicd-and-deployment)
- [Project Structure](#project-structure)
- [Acknowledgements and Citation](#acknowledgements-and-citation)

---

## Features

### Data Ingestion
Automated downloading and scraping of match fixtures and statistics via Python scripts  
(e.g. `fetch_data_universal.py`, `extract_matches_model2.py`).

### Preprocessing and Feature Engineering
Raw data is harmonised into three modeling modes:

- **Model 1** — Goal regression (XGBoost Regressor)
- **Model 2** — Match outcome classification (Home / Draw / Away) using team strength and xG (XGBoost Classifier)
- **Model 3** — Player mode using average player strength scores per team for custom lineups

### Training and Evaluation
- Models trained with XGBoost
- Metrics logged to MLflow
- Evaluation reports exported to JSON and stored in `reports/`

### Champion Selection
`select_champion.py` compares models using a priority of metrics:
1. `f1_macro`
2. `accuracy`

The selected model is written to:

app/champion_config.json

### Data Drift Monitoring

- monitor_drift.py applies Kolmogorov–Smirnov tests

- Drift computed on matches, player strengths, and team statistics

- If drift exceeds 30%, retraining is triggered automatically via retrain_if_drift.py

### API and Dashboard

- FastAPI service exposes prediction endpoints and champion info

- Streamlit dashboard visualises metrics, drift status, and allows interactive inference

### CI/CD

- GitHub Actions workflows for scheduled retraining and deployment

- Fully automated build and deploy to Google Cloud Run


## CI/CD and Deployment

Two GitHub Actions workflows orchestrate the pipeline end-to-end:

| Workflow    | Purpose | Trigger |
|------------|---------|---------|
| `retrain.yml` | Schedules drift monitoring and conditional retraining. Sets up DVC and MLflow and calls `retrain_if_drift.py`. | Daily at 03:00 (Europe/Zurich) or manual dispatch |
| `deploy.yml`  | Builds the Docker image, runs a local smoke test on `/docs`, pushes to Artifact Registry, and deploys to Cloud Run (`europe-west4`). | On push to `main` or manual dispatch |

---

## Project Structure

```text
MLOps/
├── .dvc/                      # DVC cache and config (partial)
├── app/
│   ├── main.py                # FastAPI application
│   └── champion_config.json   # Selected champion model (generated)
├── data/
│   ├── processed/             # Cleaned and feature-engineered datasets
│   ├── raw/                   # Raw datasets
│   └── team_name_mapping.csv
├── models/                    # Trained model artifacts (e.g. model2_xgb.json)
├── reports/                   # Metrics, drift reports and summary JSON
├── src/
│   ├── monitor_drift.py
│   ├── retrain_if_drift.py
│   ├── select_champion.py
│   ├── train.py               # Training Model 1 (regression)
│   ├── train_model2.py        # Training Model 2 (classification)
│   ├── eval_model3_player_mode.py
│   └── …
├── .github/workflows/
│   ├── retrain.yml            # Scheduled monitoring and conditional retrain
│   └── deploy.yml             # Build and deploy to Cloud Run
├── dvc.yaml                   # Pipeline definitions and dependencies
├── requirements.txt
├── Dockerfile
└── README.md                  # This file

```

---
## Citation
This project accompanies the report: "Pipeline MLOps de prédiction de matchs de football".
