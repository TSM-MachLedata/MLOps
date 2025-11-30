import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import os
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split  # NEW


MODEL_PATH = "models/model2_xgb.json"
PLAYER_STRENGTH_PATH = "data/processed/player_strengths.csv"
TEAM_STATS_PATH = "data/raw/team_stats_multi_leagues.csv"
MATCH_STATS_PATH = "data/raw/team_match_stats_model2.csv"


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# --------------------------------------------------------------
# TEAM SELECTION
# --------------------------------------------------------------

def select_team(players_df, label):
    teams = sorted(players_df["team"].dropna().unique())
    print("\n=== AVAILABLE TEAMS ===")
    for t in teams:
        print(" -", t)

    team = input(f"\n{label} team: ").strip()
    while team not in teams:
        print("❌ Team not found. Try again.")
        team = input(f"{label} team: ").strip()

    return team


# --------------------------------------------------------------
# PLAYER SELECTION (11 players)
# --------------------------------------------------------------

def select_players(players_df, team):
    team_players = players_df[players_df["team"] == team]

    print(f"\n=== PLAYERS OF {team} ===")
    for p in team_players["player"].unique():
        print(" -", p)

    selected = []
    print("\nSelect 11 players:")

    while len(selected) < 11:
        p = input(f"Player {len(selected)+1}/11: ").strip()
        if p in team_players["player"].values and p not in selected:
            selected.append(p)
        else:
            print("❌ Invalid or duplicate player.")

    strength = team_players[team_players["player"].isin(selected)]["player_score"].mean()
    return strength


# --------------------------------------------------------------
# BUILD THE 12 FEATURES FOR THE MODEL
# --------------------------------------------------------------

def build_features(home_team, away_team):
    team_stats = pd.read_csv(TEAM_STATS_PATH)
    match_stats = pd.read_csv(MATCH_STATS_PATH)

    def get_team_stats(team):
        row = team_stats[team_stats["team"] == team]

        if row.empty:
            return 0, 0, 0, 0, 0, 0

        rf = row.iloc[0]
        return (
            rf["goals_for_home"],
            rf["goals_for_away"],
            rf["goals_against_home"],
            rf["goals_against_away"],
            rf["matches_played"],
            rf["goals_for"],
        )

    h_gf_home, h_gf_away, h_ga_home, h_ga_away, h_mp, h_gf_total = get_team_stats(home_team)
    a_gf_home, a_gf_away, a_ga_home, a_ga_away, a_mp, a_gf_total = get_team_stats(away_team)

    def get_team_xg(team):
        df = match_stats[match_stats["opponent"] == team]
        if df.empty:
            return 0
        return df["xG"].mean()

    home_xg = get_team_xg(home_team)
    away_xg = get_team_xg(away_team)

    return {
        "home_goals_for": h_gf_total,
        "away_goals_for": a_gf_total,
        "home_goals_against": h_ga_home + h_ga_away,
        "away_goals_against": a_ga_home + a_ga_away,
        "goals_for_diff": h_gf_total - a_gf_total,
        "goals_against_diff": (h_ga_home + h_ga_away) - (a_ga_home + a_ga_away),
        "matches_played_diff": h_mp - a_mp,
        "home_xg": home_xg,
        "away_xg": away_xg,
    }


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

def main():
    log("Loading model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    log("Loading player strengths...")
    players_df = pd.read_csv(PLAYER_STRENGTH_PATH)

    print("\n======= FOOTBALL MATCH PREDICTION (PLAYER MODE) ========\n")

    home_team = select_team(players_df, "Home")
    away_team = select_team(players_df, "Away")

    print("\n---- SELECT HOME PLAYERS ----")
    home_strength = select_players(players_df, home_team)

    print("\n---- SELECT AWAY PLAYERS ----")
    away_strength = select_players(players_df, away_team)

    strength_diff = home_strength - away_strength

    extra = build_features(home_team, away_team)

    X = pd.DataFrame([{
        "home_strength": home_strength,
        "away_strength": away_strength,
        "strength_diff": strength_diff,
        **extra
    }])

    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0]

    mapping = {0: "HOME WIN", 1: "DRAW", 2: "AWAY WIN"}

    print("\n=============== RESULT ===============")
    print("Prediction:", mapping[y_pred])
    print("\nProbabilities:")
    print("  Home Win :", round(y_proba[0], 3))
    print("  Draw     :", round(y_proba[1], 3))
    print("  Away Win :", round(y_proba[2], 3))

    # ----------------------------------------------------------
    # METRICS ON A TEST SET (like models 1 & 2)
    # ----------------------------------------------------------
    df_train = pd.read_csv("data/processed/model2_training_dataset.csv")

    feature_cols = [
        "home_strength", "away_strength", "strength_diff",
        "home_goals_for", "away_goals_for",
        "home_goals_against", "away_goals_against",
        "goals_for_diff", "goals_against_diff",
        "matches_played_diff",
        "home_xg", "away_xg",
    ]
    feature_cols = [c for c in feature_cols if c in df_train.columns]

    X_full = df_train[feature_cols]
    y_full = df_train["result_xgb"]

    # même split que pour l’idée générale des autres modèles
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)

    # accuracy / F1 sur le test set
    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average="macro")

    # one-hot des labels vrais (comme dans train_model2.py)
    y_true_onehot = np.zeros_like(y_proba_test)
    for i, cls in enumerate(y_test):
        y_true_onehot[i, cls] = 1.0

    # MSE / MAE / R2 pour HOME (classe 0) et AWAY (classe 2)
    mse_home = mean_squared_error(y_true_onehot[:, 0], y_proba_test[:, 0])
    mae_home = mean_absolute_error(y_true_onehot[:, 0], y_proba_test[:, 0])
    r2_home = r2_score(y_true_onehot[:, 0], y_proba_test[:, 0])

    mse_away = mean_squared_error(y_true_onehot[:, 2], y_proba_test[:, 2])
    mae_away = mean_absolute_error(y_true_onehot[:, 2], y_proba_test[:, 2])
    r2_away = r2_score(y_true_onehot[:, 2], y_proba_test[:, 2])

    print("\n📊 === MODEL 3 (PLAYER MODE) METRICS (TEST SET) ===")
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
    # SAVE OUTPUT FOR DVC (CSV)
    # ----------------------------------------------------------
    os.makedirs("data/predictions", exist_ok=True)
    output_path = "data/predictions/model3_players_output.csv"

    pd.DataFrame([{
        "home_team": home_team,
        "away_team": away_team,
        "prediction": mapping[y_pred],
        "proba_home_win": round(y_proba[0], 3),
        "proba_draw": round(y_proba[1], 3),
        "proba_away_win": round(y_proba[2], 3),
        "home_strength": home_strength,
        "away_strength": away_strength,
        "strength_diff": strength_diff
    }]).to_csv(output_path, index=False)

    print(f"\n📝 Output saved to: {output_path}")


if __name__ == "__main__":
    main()
