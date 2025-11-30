import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import os

import warnings
warnings.filterwarnings("ignore")


# ----------------------------------------------------------
# UTILS
# ----------------------------------------------------------

def log(msg):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")


# ----------------------------------------------------------
# LOAD MODEL + DATA
# ----------------------------------------------------------

def load_artifacts():
    log("Loading model & datasets...")

    model = XGBClassifier()
    model.load_model("models/model2_xgb.json")

    df_train = pd.read_csv("data/processed/model2_training_dataset.csv")

    return model, df_train


# ----------------------------------------------------------
# BUILD FEATURE ROW FOR PREDICTION
# ----------------------------------------------------------

def build_input_features(df_train, home, away):

    home_row = df_train[df_train["home_team_clean"] == home].tail(1)
    away_row = df_train[df_train["away_team_clean"] == away].tail(1)

    if home_row.empty:
        raise ValueError(f"No history found for HOME team: {home}")
    if away_row.empty:
        raise ValueError(f"No history found for AWAY team: {away}")

    # Build the row
    row = pd.DataFrame({
        "home_strength": home_row["home_strength"].values[0],
        "away_strength": away_row["away_strength"].values[0],
        "strength_diff": home_row["home_strength"].values[0] - away_row["away_strength"].values[0],

        "home_goals_for": home_row["home_goals_for"].values[0],
        "away_goals_for": away_row["away_goals_for"].values[0],

        "home_goals_against": home_row["home_goals_against"].values[0],
        "away_goals_against": away_row["away_goals_against"].values[0],

        "goals_for_diff": home_row["home_goals_for"].values[0] - away_row["away_goals_for"].values[0],
        "goals_against_diff": home_row["home_goals_against"].values[0] - away_row["away_goals_against"].values[0],

        "matches_played_diff": home_row["home_matches_played"].values[0] - away_row["away_matches_played"].values[0],

        "home_xg": home_row["home_xg"].values[0],
        "away_xg": away_row["away_xg"].values[0],
    }, index=[0])

    return row


# ----------------------------------------------------------
# PREDICT
# ----------------------------------------------------------

def predict(model, features, home, away):
    proba = model.predict_proba(features)[0]
    pred = np.argmax(proba)

    mapping = {
        0: f"{away} WIN",
        1: "DRAW",
        2: f"{home} WIN"
    }

    return mapping[pred], proba, pred


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    model, df_train = load_artifacts()

    print("\n=== FOOTBALL MATCH PREDICTION ===\n")
    home = input("Home team: ").strip()
    away = input("Away team: ").strip()

    features = build_input_features(df_train, home, away)

    outcome, proba, pred_class = predict(model, features, home, away)

    print("\n================= RESULT =================")
    print(f"Prediction: {outcome}")
    print(f"Probas:")
    print(f"  Away Win : {proba[0]:.3f}")
    print(f"  Draw     : {proba[1]:.3f}")
    print(f"  Home Win : {proba[2]:.3f}")
    print("==========================================\n")

    # ------------------------------------------------------
    # ADD METRICS LIKE MODEL 1
    # ------------------------------------------------------

    # Load true labels & preds to compute metrics
    y_true = df_train["result_xgb"]       # True labels 0/1/2
    features_all = df_train[features.columns]  # X for all dataset

    y_pred_all = model.predict(features_all)

    acc = accuracy_score(y_true, y_pred_all)
    f1 = f1_score(y_true, y_pred_all, average="macro")

    # For classification, we adapt regression metrics using mean of probabilities
    mse_home = mean_squared_error([1 if y==2 else 0 for y in y_true],
                                  [p[2] for p in model.predict_proba(features_all)])
    mae_home = mean_absolute_error([1 if y==2 else 0 for y in y_true],
                                   [p[2] for p in model.predict_proba(features_all)])
    r2_home = r2_score([1 if y==2 else 0 for y in y_true],
                       [p[2] for p in model.predict_proba(features_all)])

    mse_away = mean_squared_error([1 if y==0 else 0 for y in y_true],
                                  [p[0] for p in model.predict_proba(features_all)])
    mae_away = mean_absolute_error([1 if y==0 else 0 for y in y_true],
                                   [p[0] for p in model.predict_proba(features_all)])
    r2_away = r2_score([1 if y==0 else 0 for y in y_true],
                       [p[0] for p in model.predict_proba(features_all)])

    print("\n📊 === MODEL 2 METRICS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"MSE Home : {mse_home:.4f}")
    print(f"MAE Home : {mae_home:.4f}")
    print(f"R2 Home  : {r2_home:.4f}")
    print(f"MSE Away : {mse_away:.4f}")
    print(f"MAE Away : {mae_away:.4f}")
    print(f"R2 Away  : {r2_away:.4f}")
    print("==========================================\n")

    # ----------------------------------------------------------
    # DVC OUTPUT SAVE
    # ----------------------------------------------------------
    os.makedirs("data/predictions", exist_ok=True)
    path = "data/predictions/model2_predictions.csv"

    pd.DataFrame([{
        "home_team": home,
        "away_team": away,
        "prediction": outcome,
        "proba_away_win": proba[0],
        "proba_draw": proba[1],
        "proba_home_win": proba[2],
    }]).to_csv(path, index=False)

    print(f"📝 Saved to {path}")


if __name__ == "__main__":
    main()
