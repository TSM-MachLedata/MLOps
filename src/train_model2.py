import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from xgboost import XGBClassifier


# ---------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------
def log(msg: str):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")


# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
def load_data():
    log("Loading datasets...")

    pre = pd.read_csv("data/processed/model2_preprocessed.csv")
    team_stats = pd.read_csv("data/raw/team_stats_multi_leagues.csv")

    log(f"Preprocessed: {pre.shape}")
    log(f"Team stats: {team_stats.shape}")

    return pre, team_stats


# ---------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------
def create_features(pre, team_stats):
    log("Creating team long-term features...")

    selected_cols = [
        "team",
        "matches_played",
        "goals_for",
        "goals_against",
        "matches_home",
        "goals_for_home",
        "goals_against_home",
        "matches_away",
        "goals_for_away",
        "goals_against_away",
    ]
    team_stats = team_stats[selected_cols]

    ts_home = team_stats.rename(
        columns=lambda c: f"home_{c}" if c != "team" else "home_team_clean"
    )
    ts_away = team_stats.rename(
        columns=lambda c: f"away_{c}" if c != "team" else "away_team_clean"
    )

    pre = pre.merge(ts_home, on="home_team_clean", how="left")
    pre = pre.merge(ts_away, on="away_team_clean", how="left")

    pre["goals_for_diff"] = pre["home_goals_for"] - pre["away_goals_for"]
    pre["goals_against_diff"] = pre["home_goals_against"] - pre["away_goals_against"]
    pre["matches_played_diff"] = (
        pre["home_matches_played"] - pre["away_matches_played"]
    )

    return pre


# ---------------------------------------------------------------
# FINAL FEATURES
# ---------------------------------------------------------------
def select_model_features(df):
    log("Selecting features...")

    features = [
        "home_strength",
        "away_strength",
        "strength_diff",
        "home_goals_for",
        "away_goals_for",
        "home_goals_against",
        "away_goals_against",
        "goals_for_diff",
        "goals_against_diff",
        "matches_played_diff",
        "home_xg",
        "away_xg",
    ]

    features = [c for c in features if c in df.columns]

    log(f"Using {len(features)} features")

    return df[features], df["result"]


# ---------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------------
def main():
    pre, team_stats = load_data()
    df = create_features(pre, team_stats)

    df = df[df["result"].notna()]  # remove empty

    # -1,0,1  → 0,1,2
    df["result_xgb"] = df["result"].replace({-1: 0, 0: 1, 1: 2}).astype(int)

    X, y = select_model_features(df)
    y = df["result_xgb"]

    log("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log("Training XGBoost...")

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
    )

    model.fit(X_train, y_train)

    log("Evaluating...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # needed for metrics

    # === METRICS ===
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    y_true = np.zeros_like(y_proba)
    for i, cls in enumerate(y_test):
        y_true[i, cls] = 1.0

    mse_home = mean_squared_error(y_true[:, 0], y_proba[:, 0])
    mae_home = mean_absolute_error(y_true[:, 0], y_proba[:, 0])
    r2_home = r2_score(y_true[:, 0], y_proba[:, 0])

    mse_away = mean_squared_error(y_true[:, 2], y_proba[:, 2])
    mae_away = mean_absolute_error(y_true[:, 2], y_proba[:, 2])
    r2_away = r2_score(y_true[:, 2], y_proba[:, 2])

    print("📊 METRICS (Model 2) :")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  mse_home:  {mse_home:.4f}")
    print(f"  mae_home:  {mae_home:.4f}")
    print(f"  r2_home:   {r2_home:.4f}")
    print(f"  mse_away:  {mse_away:.4f}")
    print(f"  mae_away:  {mae_away:.4f}")
    print(f"  r2_away:   {r2_away:.4f}")

# === SAVE ARTIFACTS (model + dataset) ===
os.makedirs("models", exist_ok=True)

model_path = "models/model2_xgb.json"

# Workaround for XGBoost sklearn save_model metadata bug in CI
model._estimator_type = "classifier"

model.save_model(model_path)
log(f"Saved model → {model_path}")


    # 💾 Sauvegarde JSON pour comparaison de modèles
    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
        "mse_home": float(mse_home),
        "mae_home": float(mae_home),
        "r2_home": float(r2_home),
        "mse_away": float(mse_away),
        "mae_away": float(mae_away),
        "r2_away": float(r2_away),
    }

    os.makedirs("reports", exist_ok=True)
    metrics_path = os.path.join("reports", "model2_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log(f"Saved metrics JSON → {metrics_path}")

    # === LOG TO MLFLOW === ✅
    mlflow.set_experiment("football_prediction_mlops")
    with mlflow.start_run(run_name="model2_xgb_classifier"):
        mlflow.set_tag("model_name", "model2")
        mlflow.set_tag("stage", "train")

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.log_param("n_estimators", 400)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.05)

        # log artifacts
        mlflow.log_artifact("models/model2_xgb.json")
        mlflow.log_artifact("data/processed/model2_training_dataset.csv")
        mlflow.log_artifact(metrics_path)


if __name__ == "__main__":
    main()
