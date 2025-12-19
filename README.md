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

- Features
- Pipeline Overview
- Quick Start
- API Usage
- Monitoring and Retraining
- CI/CD and Deployment
- Project Structure
- Acknowledgements and Citation

---

## Features

### Data Ingestion
Automated downloading and scraping of match fixtures and statistics via Python scripts  
(e.g. `fetch_data_universal.py`, `extract_matches_model2.py`).

### Preprocessing and Feature Engineering
Raw data is harmonised into three modeling modes:

- Model 1 — Goal regression (XGBoost Regressor)
- Model 2 — Match outcome classification (Home / Draw / Away) using team strength and xG (XGBoost Classifier)
- Model 3 — Player mode using average player strength scores per team for custom lineups

### Training and Evaluation
- Models trained with XGBoost
- Metrics logged to MLflow
- Evaluation reports exported to JSON and stored in `reports/`

### Champion Selection
`select_champion.py` compares models using a priority of metrics:
1. `f1_macro`
2. `accuracy`

The selected model is written to:
```text
app/champion_config.json
