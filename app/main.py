from typing import List, Optional, Dict

import json
import os

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Football Prediction MLOps API",
    description="Serve multi-league regression (model1), team-level classifier (model2), and player-mode classifier (model3).",
    version="1.2.0",
)

# -------------------------------------------------
# Champion config
# -------------------------------------------------

CHAMPION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "champion_config.json")
CHAMPION_KEY = os.getenv("CHAMPION_KEY_OVERRIDE")  # optional override via env

champion_config = {}
if os.path.exists(CHAMPION_CONFIG_PATH):
    with open(CHAMPION_CONFIG_PATH, "r", encoding="utf-8") as f:
        champion_config = json.load(f)
    if not CHAMPION_KEY:
        CHAMPION_KEY = champion_config.get("champion_key", "model2")
else:
    CHAMPION_KEY = CHAMPION_KEY or "model2"

print(f"🏆 Loaded champion key: {CHAMPION_KEY}")

# Small helper to tag champion in the "model" field
def with_champion_tag(base_key: str) -> str:
    return base_key + (" (champion)" if CHAMPION_KEY == base_key else "")


# -------------------------------------------------
# Artifacts
# -------------------------------------------------

model2_clf: Optional[xgb.XGBClassifier] = None
df_model2_train: Optional[pd.DataFrame] = None
MODEL2_TEAMS: List[str] = []

player_strengths_df: Optional[pd.DataFrame] = None
team_stats_df: Optional[pd.DataFrame] = None
match_stats_df: Optional[pd.DataFrame] = None

home_reg: Optional[xgb.XGBRegressor] = None
away_reg: Optional[xgb.XGBRegressor] = None

PLAYER_TEAMS: List[str] = []
MODEL1_TEAMS: List[str] = []  # based on team_stats_multi_leagues.csv

# NEW: lookups for case-insensitive matching (lower -> canonical name)
MODEL1_TEAM_LOOKUP: Dict[str, str] = {}
MODEL2_TEAM_LOOKUP: Dict[str, str] = {}
PLAYER_TEAM_LOOKUP: Dict[str, str] = {}


def _build_lookup(names: List[str]) -> Dict[str, str]:
    """Map lower-cased name -> canonical name."""
    lookup: Dict[str, str] = {}
    for n in names:
        if n is None:
            continue
        canon = str(n).strip()
        if not canon:
            continue
        key = canon.lower()
        lookup.setdefault(key, canon)
    return lookup


# ---- Model 2 artifacts ------------------------------------------------
try:
    print("🔧 [model2] CWD:", os.getcwd())
    print("🔧 [model2] ls . :", os.listdir("."))
    print("🔧 [model2] models/ exists?", os.path.exists("models"))
    print("🔧 [model2] data/processed exists?", os.path.exists("data/processed"))

    model2_clf = xgb.XGBClassifier()
    model2_model_path = "models/model2_xgb.json"
    print(f"🔧 [model2] Loading classifier from: {model2_model_path} (exists? {os.path.exists(model2_model_path)})")
    model2_clf.load_model(model2_model_path)

    csv_path = "data/processed/model2_training_dataset.csv"
    print(f"🔧 [model2] Loading training dataset from: {csv_path} (exists? {os.path.exists(csv_path)})")
    df_model2_train = pd.read_csv(csv_path)
    print("🔧 [model2] df_model2_train shape:", df_model2_train.shape)

    if "home_team_clean" in df_model2_train.columns and "away_team_clean" in df_model2_train.columns:
        MODEL2_TEAMS = sorted(
            set(df_model2_train["home_team_clean"].dropna().unique())
            | set(df_model2_train["away_team_clean"].dropna().unique())
        )
    MODEL2_TEAM_LOOKUP = _build_lookup(MODEL2_TEAMS)
    print(f"✅ Loaded model2 classifier & training dataset. {len(MODEL2_TEAMS)} teams.")
except Exception as e:
    print(f"⚠️ Could not load model2 artifacts: {e}")

# ---- Player-mode + team stats artifacts (model3 + model1) ------------
try:
    player_strengths_path = "data/processed/player_strengths.csv"
    team_stats_path = "data/raw/team_stats_multi_leagues.csv"
    match_stats_path = "data/raw/team_match_stats_model2.csv"

    print(
        "🔧 [players/model1] Files existence:\n"
        f"  {player_strengths_path}: {os.path.exists(player_strengths_path)}\n"
        f"  {team_stats_path}: {os.path.exists(team_stats_path)}\n"
        f"  {match_stats_path}: {os.path.exists(match_stats_path)}"
    )

    player_strengths_df = pd.read_csv(player_strengths_path)
    team_stats_df = pd.read_csv(team_stats_path)
    match_stats_df = pd.read_csv(match_stats_path)

    # NEW: normalize columns for case-insensitive search
    if player_strengths_df is not None:
        player_strengths_df["team_norm"] = (
            player_strengths_df["team"].astype(str).str.strip().str.lower()
        )
        player_strengths_df["player_norm"] = (
            player_strengths_df["player"].astype(str).str.strip().str.lower()
        )

    if "team" in player_strengths_df.columns:
        PLAYER_TEAMS = sorted(player_strengths_df["team"].dropna().unique())
        PLAYER_TEAM_LOOKUP = _build_lookup(PLAYER_TEAMS)

    if "team" in team_stats_df.columns:
        MODEL1_TEAMS = sorted(team_stats_df["team"].dropna().unique())
        MODEL1_TEAM_LOOKUP = _build_lookup(MODEL1_TEAMS)

    # NEW: normalized opponent column for case-insensitive xG lookup
    if match_stats_df is not None and "opponent" in match_stats_df.columns:
        match_stats_df["opponent_norm"] = (
            match_stats_df["opponent"].astype(str).str.strip().str.lower()
        )

    print(
        f"✅ Loaded player-mode artifacts. "
        f"{len(PLAYER_TEAMS)} teams with player strengths, "
        f"{len(MODEL1_TEAMS)} teams with long-term stats."
    )
except Exception as e:
    print(f"⚠️ Could not load player-mode / team-stats artifacts: {e}")

# ---- Model 1 regression (home/away goals) -----------------------------
try:
    home_model_path = "app/models/home_model.json"
    away_model_path = "app/models/away_model.json"
    print(
        "🔧 [model1] Files existence:\n"
        f"  {home_model_path}: {os.path.exists(home_model_path)}\n"
        f"  {away_model_path}: {os.path.exists(away_model_path)}"
    )

    home_reg = xgb.XGBRegressor()
    away_reg = xgb.XGBRegressor()
    home_reg.load_model(home_model_path)
    away_reg.load_model(away_model_path)
    print("✅ Loaded model1 regression models.")
except Exception as e:
    print(f"⚠️ Could not load model1 regression models: {e}")


# -------------------------------------------------
# Pydantic schemas
# -------------------------------------------------

class TeamRequest(BaseModel):
    home_team: str
    away_team: str


class PlayerMatchRequest(BaseModel):
    home_team: str
    away_team: str
    home_players: List[str]
    away_players: List[str]


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def validate_team_exists(team: str, context: str = "model2") -> str:
    """
    Check that a team exists in the appropriate list, case-insensitive.
    Returns the canonical team name as in the training data.
    """
    raw = (team or "").strip()
    key = raw.lower()

    if context == "model2":
        canonical = MODEL2_TEAM_LOOKUP.get(key)
        where = "model2"
    elif context == "players":
        canonical = PLAYER_TEAM_LOOKUP.get(key)
        where = "player strengths"
    elif context == "model1":
        canonical = MODEL1_TEAM_LOOKUP.get(key)
        where = "model1"
    else:
        canonical = None
        where = context

    if canonical is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown team '{team}' for {where}. "
                "Team must exist in the training data."
            ),
        )

    return canonical


# ---------- model1 feature builder (regression) ------------------------

def build_features_model1(home_team: str, away_team: str) -> pd.DataFrame:
    """
    Build numeric features for model1 from team_stats_multi_leagues.csv
    based only on home_team / away_team (same spirit as training).
    """
    if team_stats_df is None:
        raise HTTPException(status_code=500, detail="Team statistics not loaded for model1.")

    if home_team == away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team must be different.")

    # canonical names (case-insensitive)
    home_team_canon = validate_team_exists(home_team, context="model1")
    away_team_canon = validate_team_exists(away_team, context="model1")

    def get_team_row(team: str):
        row = team_stats_df[team_stats_df["team"] == team]
        if row.empty:
            raise HTTPException(status_code=400, detail=f"No stats found for team: {team}")
        r = row.iloc[0]
        matches_played = float(r.get("matches_played", 0.0))
        goals_for = float(r.get("goals_for", 0.0))
        goals_against = float(r.get("goals_against", 0.0))
        goals_diff = goals_for - goals_against
        return matches_played, goals_for, goals_against, goals_diff

    h_mp, h_gf, h_ga, h_diff = get_team_row(home_team_canon)
    a_mp, a_gf, a_ga, a_diff = get_team_row(away_team_canon)

    return pd.DataFrame(
        [
            {
                "home_matches_played": h_mp,
                "home_goals_for": h_gf,
                "home_goals_against": h_ga,
                "home_goals_diff": h_diff,
                "away_matches_played": a_mp,
                "away_goals_for": a_gf,
                "away_goals_against": a_ga,
                "away_goals_diff": a_diff,
            }
        ]
    ), home_team_canon, away_team_canon


# ---------- model2 feature builder (team-level classifier) -------------

def build_features_model2(home_team: str, away_team: str) -> pd.DataFrame:
    if df_model2_train is None:
        raise HTTPException(status_code=500, detail="Model2 training dataset not loaded.")

    if home_team == away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team must be different.")

    # canonical names (case-insensitive)
    home_team_canon = validate_team_exists(home_team, context="model2")
    away_team_canon = validate_team_exists(away_team, context="model2")

    home_row = df_model2_train[df_model2_train["home_team_clean"] == home_team_canon].tail(1)
    away_row = df_model2_train[df_model2_train["away_team_clean"] == away_team_canon].tail(1)

    if home_row.empty:
        raise HTTPException(status_code=400, detail=f"No history found for HOME team: {home_team}")
    if away_row.empty:
        raise HTTPException(status_code=400, detail=f"No history found for AWAY team: {away_team}")

    row = pd.DataFrame(
        {
            "home_strength": home_row["home_strength"].values[0],
            "away_strength": away_row["away_strength"].values[0],
            "strength_diff": home_row["home_strength"].values[0] - away_row["away_strength"].values[0],
            "home_goals_for": home_row["home_goals_for"].values[0],
            "away_goals_for": away_row["away_goals_for"].values[0],
            "home_goals_against": home_row["home_goals_against"].values[0],
            "away_goals_against": away_row["away_goals_against"].values[0],
            "goals_for_diff": home_row["home_goals_for"].values[0] - away_row["away_goals_for"].values[0],
            "goals_against_diff": home_row["home_goals_against"].values[0] - away_row["away_goals_against"].values[0],
            "matches_played_diff": home_row["home_matches_played"].values[0]
            - away_row["away_matches_played"].values[0],
            "home_xg": home_row["home_xg"].values[0],
            "away_xg": away_row["away_xg"].values[0],
        },
        index=[0],
    )

    return row, home_team_canon, away_team_canon


# ---------- player-mode helpers ---------------------------------------

def compute_team_strength(team: str, players: List[str]) -> float:
    """Compute mean player_score for exactly 11 valid, unique players of a team (case-insensitive)."""
    if player_strengths_df is None:
        raise HTTPException(status_code=500, detail="Player strengths not loaded.")

    team_canon = validate_team_exists(team, context="players")

    pairs = [(p.strip(), p.strip().lower()) for p in players if p.strip()]
    cleaned = [p for p, _ in pairs]
    cleaned_norm = [pn for _, pn in pairs]

    if len(cleaned) != 11:
        raise HTTPException(
            status_code=400,
            detail=f"Team {team} must have exactly 11 players, got {len(cleaned)}.",
        )
    if len(set(cleaned_norm)) != 11:
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate player names detected for team {team}. Players must be unique.",
        )

    df_team = player_strengths_df[player_strengths_df["team_norm"] == team_canon.lower()]

    if df_team.empty:
        raise HTTPException(status_code=400, detail=f"No players found for team: {team_canon}")

    known_players_norm = set(df_team["player_norm"])
    missing_norm = [p for p in cleaned_norm if p not in known_players_norm]

    if missing_norm:
        missing_original = [
            cleaned[i] for i, norm in enumerate(cleaned_norm) if norm in missing_norm
        ]
        example_players = list(df_team["player"].astype(str))[:10]
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown players for team {team_canon}: {', '.join(missing_original)}. "
                f"Example valid players: {', '.join(example_players)}"
            ),
        )

    df_sel = df_team[df_team["player_norm"].isin(cleaned_norm)]
    if df_sel.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Could not match any provided players for team {team_canon}.",
        )

    return float(df_sel["player_score"].mean())


def build_features_player_mode(
    home_team: str,
    away_team: str,
    home_strength: float,
    away_strength: float,
) -> pd.DataFrame:
    if team_stats_df is None or match_stats_df is None:
        raise HTTPException(status_code=500, detail="Team stats / match stats not loaded.")

    if home_team == away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team must be different.")

    home_team_canon = validate_team_exists(home_team, context="players")
    away_team_canon = validate_team_exists(away_team, context="players")

    team_stats = team_stats_df
    match_stats = match_stats_df

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

    h_gf_home, h_gf_away, h_ga_home, h_ga_away, h_mp, h_gf_total = get_team_stats(home_team_canon)
    a_gf_home, a_gf_away, a_ga_home, a_ga_away, a_mp, a_gf_total = get_team_stats(away_team_canon)

    def get_team_xg(team):
        df = match_stats[match_stats["opponent_norm"] == team.lower()]
        if df.empty:
            return 0.0
        return float(df["xG"].mean())

    home_xg = get_team_xg(home_team_canon)
    away_xg = get_team_xg(away_team_canon)

    strength_diff = home_strength - away_strength

    return pd.DataFrame(
        [
            {
                "home_strength": home_strength,
                "away_strength": away_strength,
                "strength_diff": strength_diff,
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
        ]
    )


# -------------------------------------------------
# Endpoints
# -------------------------------------------------

@app.get("/info/champion")
def get_champion_info():
    return {
        "champion_key": CHAMPION_KEY,
        "champion_config": champion_config or {},
    }


# ---------- MODEL 1: regression on goals -------------------------------

@app.post("/predict/model1")
def predict_model1(req: TeamRequest):
    """
    Predict home & away goals with model1 (regression) + match result,
    using only home_team / away_team (features built from team_stats).
    """
    if home_reg is None or away_reg is None:
        raise HTTPException(status_code=500, detail="Model1 regression models not loaded.")

    X, home_team_canon, away_team_canon = build_features_model1(req.home_team, req.away_team)

    pred_home = float(home_reg.predict(X)[0])
    pred_away = float(away_reg.predict(X)[0])

    if pred_home > pred_away:
        result = f"{home_team_canon} WIN"
    elif pred_home < pred_away:
        result = f"{away_team_canon} WIN"
    else:
        result = "DRAW"

    return {
        "model": with_champion_tag("model1"),
        "is_champion": CHAMPION_KEY == "model1",
        "home_team": home_team_canon,
        "away_team": away_team_canon,
        "prediction": result,
        "predicted_home_goals": pred_home,
        "predicted_away_goals": pred_away,
    }


# ---------- MODEL 2: team-level classifier -----------------------------

@app.post("/predict/model2")
def predict_model2(req: TeamRequest):
    if model2_clf is None:
        raise HTTPException(status_code=500, detail="Model2 classifier not loaded.")

    X, home_team_canon, away_team_canon = build_features_model2(req.home_team, req.away_team)
    proba = model2_clf.predict_proba(X)[0]
    pred_class = int(proba.argmax())

    mapping = {
        0: f"{away_team_canon} WIN",
        1: "DRAW",
        2: f"{home_team_canon} WIN",
    }

    return {
        "model": with_champion_tag("model2"),
        "is_champion": CHAMPION_KEY == "model2",
        "home_team": home_team_canon,
        "away_team": away_team_canon,
        "prediction": mapping[pred_class],
        "proba_away_win": float(proba[0]),
        "proba_draw": float(proba[1]),
        "proba_home_win": float(proba[2]),
    }


# ---------- MODEL 3: player-mode classifier ----------------------------

@app.post("/predict/model3")
def predict_player_mode(req: PlayerMatchRequest):
    if model2_clf is None:
        raise HTTPException(status_code=500, detail="Base classifier not loaded.")

    home_strength = compute_team_strength(req.home_team, req.home_players)
    away_strength = compute_team_strength(req.away_team, req.away_players)

    X = build_features_player_mode(req.home_team, req.away_team, home_strength, away_strength)
    proba = model2_clf.predict_proba(X)[0]
    pred_class = int(proba.argmax())

    mapping = {
        0: f"{req.away_team} WIN",
        1: "DRAW",
        2: f"{req.home_team} WIN",
    }

    return {
        "model": with_champion_tag("model3_player_mode"),
        "is_champion": CHAMPION_KEY == "model3_player_mode",
        "home_team": req.home_team,
        "away_team": req.away_team,
        "prediction": mapping[pred_class],
        "proba_away_win": float(proba[0]),
        "proba_draw": float(proba[1]),
        "proba_home_win": float(proba[2]),
        "home_strength": home_strength,
        "away_strength": away_strength,
    }
