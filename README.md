# Football Match Prediction — MLOps Pipeline

This repository houses a **production-ready MLOps pipeline** for predicting football match outcomes across multiple leagues and seasons.  
It automates the entire lifecycle: **data ingestion, preprocessing, model training, drift monitoring, conditional retraining, champion selection, API serving, and dashboard visualisation**.

Everything is:
- versioned with **DVC**
- logged with **MLflow**
- containerised with **Docker**
- deployed on **Google Cloud Run**

---

##  Useful Links

- ** MLflow Experiments (DagsHub)**  
  https://dagshub.com/LEBARMS/MLOps/experiments

- ** Live API (Swagger / OpenAPI docs)**  
  https://football-mlops-api-1089778705681.europe-west4.run.app/docs#/default/predict_model2_predict_model2_post

---

##  Table of Contents

- Features  
- Pipeline Overview  
- Quick Start  
- API Usage  
- Monitoring & Retraining  
- CI/CD & Deployment  
- Project Structure  
- Acknowledgements & Citation  

---

## ✨ Features

### Data Ingestion
Automated downloading and scraping of match fixtures and statistics via Python scripts  
(e.g. `fetch_data_universal.py`, `extract_matches_model2.py`).

### Preprocessing & Feature Engineering
Raw data is harmonised into **three modeling modes**:

- **Model 1** – Goal regression (XGBoost Regressor)
- **Model 2** – Match outcome classification (Home / Draw / Away) using team strength and xG
- **Model 3** – *Player mode* using average player strength scores for custom lineups

### Training & Evaluation
- Models trained with **XGBoost**
- Metrics logged to **MLflow**
- Evaluation reports exported to JSON and stored in `reports/`

### Champion Selection
`select_champion.py` compares models using:
1. `f1_macro`
2. `accuracy`

The selected model is written to:
```text
app/champion_config.json
Data Drift Monitoring
monitor_drift.py applies Kolmogorov–Smirnov tests

Drift computed on matches, player strengths, and team statistics

If drift exceeds 30%, retraining is triggered automatically via retrain_if_drift.py

API & Dashboard
FastAPI service exposes prediction endpoints and champion info

Streamlit dashboard visualises metrics, drift status, and allows interactive inference

CI/CD
GitHub Actions workflows for scheduled retraining and deployment

Fully automated build & deploy to Google Cloud Run

🔁 Pipeline Overview
haskell
Copy code
┌─────────────┐
│ Data fetch  │
└──────┬──────┘
       │
   Raw data (DVC)
       │
┌──────▼──────┐
│Preprocessing│
└──────┬──────┘
┌──────┼──────────────┐
│      │              │
▼      ▼              ▼
Model1 Model2        Model3
│      │              │
└──────┴──────┬───────┘
              ▼
     select_champion.py
              │
     champion_config.json
              │
         ┌────▼────┐
         │  API    │
         └────┬────┘
              │
        ┌─────▼─────┐
        │ Dashboard │
        └─────┬─────┘
              │
      monitor_drift.py
              │
      retrain_if_drift.py
🚀 Quick Start
Prerequisites
Python 3.11

DVC with GCS support

bash
Copy code
pip install "dvc[gs]"
MLflow (already in requirements.txt)

Google Cloud project + service account key stored as
GCP_SERVICE_ACCOUNT_KEY (GitHub secret)

Local Setup
Clone & install
bash
Copy code
git clone https://github.com/TSM-MachLedata/MLOps.git
cd MLOps
pip install -r requirements.txt
pip install "dvc[gs]"
Configure DVC & pull artifacts
bash
Copy code
echo "$GCP_SERVICE_ACCOUNT_KEY" > gcp-key.json
dvc remote modify --local gcsremote credentialpath gcp-key.json
dvc config cache.type copy --local
dvc pull
Reproduce the pipeline
bash
Copy code
dvc repro
Run API locally
bash
Copy code
uvicorn app.main:app --host 0.0.0.0 --port 8000
Swagger UI:

bash
Copy code
http://localhost:8000/docs
Launch dashboard
bash
Copy code
streamlit run dashboard_streamlit.py
🔌 API Usage
Predict match outcome (Model 2)
POST /predict/model2

json
Copy code
{
  "home_team": "Arsenal",
  "away_team": "Manchester City"
}
Response:

json
Copy code
{
  "model": "model2 (champion)",
  "home_team": "Arsenal",
  "away_team": "Manchester City",
  "prediction": "DRAW",
  "proba_away_win": 0.30,
  "proba_draw": 0.45,
  "proba_home_win": 0.25
}
Predict using players (Model 3)
POST /predict/model3

⚠️ Player names must exactly match
data/processed/player_strengths.csv

📊 Monitoring & Retraining
Drift computed per feature and dataset

Threshold: DRIFT_THRESHOLD = 0.30

If exceeded:

Models retrained

Champion re-selected

Artifacts pushed back to DVC

Metrics logged to MLflow (DagsHub)

This ensures long-term robustness against data distribution shifts.

🔄 CI/CD & Deployment
Workflow	Purpose	Trigger
retrain.yml	Drift monitoring + conditional retraining	Daily (03:00 CET) or manual
deploy.yml	Docker build + deploy to Cloud Run	Push to main

Secrets:

GCP_SERVICE_ACCOUNT_KEY

MLflow / DagsHub credentials

📁 Project Structure
stylus
Copy code
MLOps/
├── app/
│   ├── main.py
│   └── champion_config.json
├── data/
│   ├── raw/
│   ├── processed/
├── models/
├── reports/
├── src/
│   ├── monitor_drift.py
│   ├── retrain_if_drift.py
│   ├── select_champion.py
│   └── train*.py
├── .github/workflows/
│   ├── retrain.yml
│   └── deploy.yml
├── dvc.yaml
├── Dockerfile
├── requirements.txt
└── README.md
📚 Acknowledgements & Citation
This project accompanies the report
« Pipeline MLOps de prédiction de matchs de football ».

If you reuse this repository or its ideas, please consider citing or linking back.
Issues and pull requests are welcome — this repo is intended as a robust MLOps template for sports analytics.

livecodeserver
Copy code
