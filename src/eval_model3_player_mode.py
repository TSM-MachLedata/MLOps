import os
from datetime import datetime
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import mlflow


MODEL_PATH = "models/model2_xgb.json"
TRAIN_SET_PATH = "data/processed/model2_training_dataset.csv"
PLAYER_STRENGTH_PATH = "data/processed/player_strengths.csv"


def log(msg: str):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"[{now}] {msg}")


def build_xi_strengths(players_df: pd.DataFrame, top_n: int = 11) -> pd.DataFrame:
    """
    Calcule, pour chaque équipe, la force 'XI' = moyenne des top_n joueurs
    par player_score.
    """
    df = players_df.copy()

    if "team" not in df.columns or "player_score" not in df.columns:
        raise ValueError("player_strengths.csv doit contenir les colonnes 'team' et 'player_score'.")

    # Top N joueurs par équipe
    df = df.sort_values(["team", "player_score"], ascending=[True, False])
    df_top = df.groupby("team").head(top_n)

    xi = (
        df_top.groupby("team")["player_score"]
        .mean()
        .rename("xi_strength")
        .reset_index()
    )

    log(f"Computed XI strengths for {len(xi)} teams (top {top_n} players).")
    return xi


def main():
    # ----------------------------------------------------
    # 0. Sanity checks
    # ----------------------------------------------------
    log("Loading artifacts for MODEL 3 (player mode eval)...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(TRAIN_SET_PATH):
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_SET_PATH}")
    if not os.path.exists(PLAYER_STRENGTH_PATH):
        raise FileNotFoundError(f"Player strengths file not found: {PLAYER_STRENGTH_PATH}")

    # ----------------------------------------------------
    # 1. Load objects
    # ----------------------------------------------------
    log("Loading base model (model2_xgb)...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    log("Loading training dataset (model2_training_dataset.csv)...")
    df_base = pd.read_csv(TRAIN_SET_PATH)

    if "result_xgb" not in df_base.columns:
        raise ValueError("Column 'result_xgb' not found in training dataset.")

    if "home_team_clean" not in df_base.columns or "away_team_clean" not in df_base.columns:
        raise ValueError("Columns 'home_team_clean' / 'away_team_clean' missing in training dataset.")

    log("Loading player strengths...")
    players_df = pd.read_csv(PLAYER_STRENGTH_PATH)

    # ----------------------------------------------------
    # 2. Build XI-based team strengths
    # ----------------------------------------------------
    xi_df = build_xi_strengths(players_df, top_n=11)
    xi_map = xi_df.set_index("team")["xi_strength"].to_dict()

    df = df_base.copy()

    # Map XI strengths on home/away teams
    df["home_strength_xi"] = df["home_team_clean"].map(xi_map)
    df["away_strength_xi"] = df["away_team_clean"].map(xi_map)

    # Fallback: si pas de XI strength -> garder l'ancienne force d'équipe
    if "home_strength" not in df.columns or "away_strength" not in df.columns:
        raise ValueError("Columns 'home_strength' / 'away_strength' missing in training dataset.")

    df["home_strength_xi"] = df["home_strength_xi"].fillna(df["home_strength"])
    df["away_strength_xi"] = df["away_strength_xi"].fillna(df["away_strength"])

    # On remplace les colonnes utilisées par le modèle par la version "XI"
    df["home_strength"] = df["home_strength_xi"]
    df["away_strength"] = df["away_strength_xi"]
    df["strength_diff"] = df["home_strength"] - df["away_strength"]

    # ----------------------------------------------------
    # 3. Build feature matrix like model2
    # ----------------------------------------------------
    feature_cols = [
        "home_strength", "away_strength", "strength_diff",
        "home_goals_for", "away_goals_for",
        "home_goals_against", "away_goals_against",
        "goals_for_diff", "goals_against_diff",
        "matches_played_diff",
        "home_xg", "away_xg",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        raise ValueError("No feature columns found for evaluation.")

    X = df[feature_cols]
    y = df["result_xgb"].astype(int)

    log(f"Using {len(feature_cols)} features: {feature_cols}")
    log(f"Dataset size: {X.shape[0]} rows")

    # ----------------------------------------------------
    # 4. Train/test split IDENTIQUE (random_state=42)
    # ----------------------------------------------------
    # On ne réentraine pas le modèle, on recrée juste le même split pour
    # que la comparaison model2 vs player_mode soit cohérente.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # pour garder le ratio de classes
    )

    # ----------------------------------------------------
    # 5. Evaluate with XI-based features
    # ----------------------------------------------------
    log("Computing predictions on test set with XI-based strengths (player mode)...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # One-hot true labels (like in train_model2)
    y_true_onehot = np.zeros_like(y_proba)
    for i, cls in enumerate(y_test):
        y_true_onehot[i, cls] = 1.0

    # class 0 = away win, class 2 = home win (même convention que train_model2)
    mse_home = mean_squared_error(y_true_onehot[:, 2], y_proba[:, 2])
    mae_home = mean_absolute_error(y_true_onehot[:, 2], y_proba[:, 2])
    r2_home = r2_score(y_true_onehot[:, 2], y_proba[:, 2])

    mse_away = mean_squared_error(y_true_onehot[:, 0], y_proba[:, 0])
    mae_away = mean_absolute_error(y_true_onehot[:, 0], y_proba[:, 0])
    r2_away = r2_score(y_true_onehot[:, 0], y_proba[:, 0])

    print("\n📊 === MODEL 3 (PLAYER MODE) METRICS — TEST SET (XI-based) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"MSE Home : {mse_home:.4f}")
    print(f"MAE Home : {mae_home:.4f}")
    print(f"R2 Home  : {r2_home:.4f}")
    print(f"MSE Away : {mse_away:.4f}")
    print(f"MAE Away : {mae_away:.4f}")
    print(f"R2 Away  : {r2_away:.4f}")
    print("=============================================================\n")

        # ----------------------------------------------------
    # 6. Sauvegarde JSON pour comparaison de modèles
    # ----------------------------------------------------
    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "mse_home": float(mse_home),
        "mae_home": float(mae_home),
        "r2_home": float(r2_home),
        "mse_away": float(mse_away),
        "mae_away": float(mae_away),
        "r2_away": float(r2_away),
    }

    os.makedirs("reports", exist_ok=True)
    metrics_path = os.path.join("reports", "model3_player_mode_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ----------------------------------------------------
    # 7. Log in MLflow (pipeline = player_mode / logical_model = model3)
    # ----------------------------------------------------
    mlflow.set_experiment("football_prediction_mlops")
    with mlflow.start_run(run_name="model3_player_mode_eval_xi"):
        mlflow.set_tag("model_name", "model2")        # même modèle de base
        mlflow.set_tag("logical_model", "model3")     # alias logique
        mlflow.set_tag("pipeline", "player_mode")     # pipeline d'inférence
        mlflow.set_tag("stage", "eval")

        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("features", ",".join(feature_cols))
        mlflow.log_param("xi_top_n", 11)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("mse_home", mse_home)
        mlflow.log_metric("mae_home", mae_home)
        mlflow.log_metric("r2_home", r2_home)
        mlflow.log_metric("mse_away", mse_away)
        mlflow.log_metric("mae_away", mae_away)
        mlflow.log_metric("r2_away", r2_away)

        # utile pour tracer quels artefacts ont servi à cette éval
        mlflow.log_artifact(TRAIN_SET_PATH)
        mlflow.log_artifact(PLAYER_STRENGTH_PATH)
        mlflow.log_artifact(MODEL_PATH)

    log("MLflow logging done. Player-mode evaluation complete.")


if __name__ == "__main__":
    main()
