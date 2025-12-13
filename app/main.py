from typing import List, Optional, Dict

import csv
import json
import os
import traceback

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------------------------------------
# BOOT DEBUG (helps on Render / Cloud Run)
# -------------------------------------------------
print(">>> [app.main] import starting")
print(">>> [app.main] CWD:", os.getcwd())
print(">>> [app.main] ls .:", os.listdir("."))

critical_paths = [
    "data/processed/model2_training_dataset.csv",
    "data/processed/player_strengths.csv",
    "data/raw/team_stats_multi_leagues.csv",
    "data/raw/team_match_stats_model2.csv",
    "models/model2_xgb.json",
    "app/models/home_model.json",
    "app/models/away_model.json",
]
for p in critical_paths:
    print(f">>> [app.main] exists? {p} -> {os.path.exists(p)}")

app = FastAPI(
    title="Football Prediction MLOps API",
    description=(
        "Serve multi-league regression (model1), team-level classifier (model2), "
        "and player-mode classifier (model3)."
    ),
    version="1.2.0",
)

# -------------------------------------------------
# Champion config
# -------------------------------------------------

CHAMPION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "champion_config.json")
CHAMPION_KEY = os.getenv("CHAMPION_KEY_OVERRIDE")  # optional override via env

champion_config: Dict = {}
try:
    print(f">>> [app.main] Checking champion config at {CHAMPION_CONFIG_PATH}")
    if os.path.exists(CHAMPION_CONFIG_PATH):
        with open(CHAMPION_CONFIG_PATH, "r", encoding="utf-8") as f:
            champion_config = json.load(f)
        if not CHAMPION_KEY:
            CHAMPION_KEY = champion_config.get("champion_key", "model2")
    else:
        CHAMPION_KEY = CHAMPION_KEY or "model2"
except Exception as e:
    print("⚠️ [app.main] Error loading champion_config.json:", e)
    print(traceback.format_exc())
    CHAMPION_KEY = CHAMPION_KEY or "model2"

print(f"🏆 Loaded champion key: {CHAMPION_KEY}")


def with_champion_tag(base_key: str) -> str:
    return base_key + (" (champion)" if CHAMPION_KEY == base_key else "")


# -------------------------------------------------
# Artifacts (globals)
# -------------------------------------------------

# ✅ Use raw Booster objects in serving (more robust than sklearn wrappers)
model2_booster: Optional[xgb.Booster] = None
home_booster: Optional[xgb.Booster] = None
away_booster: Optional[xgb.Booster] = None

MODEL2_TEAMS: List[str] = []
MODEL2_TEAM_STATS: Dict[str, Dict[str, float]] = {}  # dict instead of df

player_strengths_df: Optional[pd.DataFrame] = None
team_stats_df: Optional[pd.DataFrame] = None
match_stats_df: Optional[pd.DataFrame] = None

PLAYER_TEAMS: List[str] = []
MODEL1_TEAMS: List[str] = []  # based on team_stats_multi_leagues.csv

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


# -------------------------------------------------
# MODEL 2: Load model (Booster) - separate from CSV parsing
# -------------------------------------------------
try:
    print("[BOOT] Loading model2 booster (models/model2_xgb.json) ...")
    model2_booster = xgb.Booster()
    model2_booster.load_model("models/model2_xgb.json")
    print("✅ Loaded model2 booster.")
except Exception as e:
    model2_booster = None
    print(f"⚠️ Could not load model2 booster: {e}")
    print(traceback.format_exc())


# -------------------------------------------------
# MODEL 2: Load team stats dict from training dataset (separate try)
# -------------------------------------------------
try:
    print("[BOOT] CWD:", os.getcwd())
    try:
        print("[BOOT] Listing data/processed & models ...")
        print("data/processed ->", os.listdir("data/processed"))
        print("models         ->", os.listdir("models"))
    except Exception as e:
        print("[BOOT] Could not list folders:", e)

    csv_path = "data/processed/model2_training_dataset.csv"
    print(f"[BOOT] Streaming team stats from {csv_path} ...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    MODEL2_TEAM_STATS = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # HOME TEAM
            home_team = (row.get("home_team_clean") or "").strip()
            if home_team:
                t = MODEL2_TEAM_STATS.setdefault(home_team, {})

                def _set_home(name: str, col: str) -> None:
                    v = row.get(col)
                    if v not in (None, ""):
                        try:
                            t[name] = float(v)
                        except ValueError:
                            pass

                _set_home("home_strength", "home_strength")
                _set_home("home_goals_for", "home_goals_for")
                _set_home("home_goals_against", "home_goals_against")
                _set_home("home_matches_played", "home_matches_played")
                _set_home("home_xg", "home_xg")

            # AWAY TEAM
            away_team = (row.get("away_team_clean") or "").strip()
            if away_team:
                t = MODEL2_TEAM_STATS.setdefault(away_team, {})

                def _set_away(name: str, col: str) -> None:
                    v = row.get(col)
                    if v not in (None, ""):
                        try:
                            t[name] = float(v)
                        except ValueError:
                            pass

                _set_away("away_strength", "away_strength")
                _set_away("away_goals_for", "away_goals_for")
                _set_away("away_goals_against", "away_goals_against")
                _set_away("away_matches_played", "away_matches_played")
                _set_away("away_xg", "away_xg")

    MODEL2_TEAMS = sorted(MODEL2_TEAM_STATS.keys())
    MODEL2_TEAM_LOOKUP = _build_lookup(MODEL2_TEAMS)
    print(f"✅ Built in-memory team stats for model2. {len(MODEL2_TEAMS)} teams.")
except Exception as e:
    MODEL2_TEAM_STATS = {}
    MODEL2_TEAMS = []
    MODEL2_TEAM_LOOKUP = {}
    print(f"⚠️ Could not load model2 team stats from CSV: {e}")
    print(traceback.format_exc())


# -------------------------------------------------
# Player-mode + team stats artifacts (model3 + model1 features)
# -------------------------------------------------
try:
    # player strengths → only what's needed
    player_strengths_df = pd.read_csv(
        "data/processed/player_strengths.csv",
        usecols=["team", "player", "player_score"],
    )

    # long-term team stats → only columns we actually use
    team_stats_df = pd.read_csv(
        "data/raw/team_stats_multi_leagues.csv",
        usecols=[
            "team",
            "matches_played",
            "goals_for",
            "goals_against",
            "goals_for_home",
            "goals_for_away",
            "goals_against_home",
            "goals_against_away",
        ],
    )

    # xG match stats → only opponent + xG
    match_stats_df = pd.read_csv(
        "data/raw/team_match_stats_model2.csv",
        usecols=["opponent", "xG"],
    )

    # normalize for case-insensitive search
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

    if match_stats_df is not None and "opponent" in match_stats_df.columns:
        match_stats_df["opponent_norm"] = (
            match_stats_df["opponent"].astype(str).str.strip().str.lower()
        )

    print(
        f"✅ Loaded player-mode artifacts (light). "
        f"{len(PLAYER_TEAMS)} teams with player strengths, "
        f"{len(MODEL1_TEAMS)} teams with long-term stats."
    )
except Exception as e:
    print(f"⚠️ Could not load player-mode / team-stats artifacts: {e}")
    print(traceback.format_exc())


# -------------------------------------------------
# MODEL 1 regression (home/away goals) — Booster load
# -------------------------------------------------
try:
    print("[BOOT] Loading model1 boosters (app/models/home_model.json, away_model.json) ...")
    home_booster = xgb.Booster()
    away_booster = xgb.Booster()
    home_booster.load_model("app/models/home_model.json")
    away_booster.load_model("app/models/away_model.json")
    print("✅ Loaded model1 boosters.")
except Exception as e:
    home_booster = None
    away_booster = None
    print(f"⚠️ Could not load model1 boosters: {e}")
    print(traceback.format_exc())


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


def _predict_proba_model2(X: pd.DataFrame) -> List[float]:
    """Predict class probabilities for model2 using Booster (softprob)."""
    if model2_booster is None:
        raise HTTPException(status_code=500, detail="Model2 booster not loaded.")

    dm = xgb.DMatrix(X)
    proba = model2_booster.predict(dm)

    # proba can be shape (1, 3) for single row
    if hasattr(proba, "shape") and len(proba.shape) == 2:
        return [float(p) for p in proba[0]]
    # fallback (shouldn't happen for softprob with 1 row, but just in case)
    return [float(p) for p in proba]


# -------------------------------------------------
# MODEL 1 feature builder (regression)
# -------------------------------------------------
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

    return (
        pd.DataFrame(
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
        ),
        home_team_canon,
        away_team_canon,
    )


# -------------------------------------------------
# MODEL 2 helpers (dict-based stats)
# -------------------------------------------------
def _get_team_stat(team: str, *keys: str) -> float:
    """Return first available stat for team among given keys, else 0.0."""
    stats = MODEL2_TEAM_STATS.get(team, {})
    for k in keys:
        if k in stats:
            return float(stats[k])
    return 0.0


def build_features_model2(home_team: str, away_team: str) -> pd.DataFrame:
    """
    Build features for model2 using the small in-memory dict MODEL2_TEAM_STATS
    instead of a giant DataFrame.
    """
    if not MODEL2_TEAM_STATS:
        raise HTTPException(status_code=500, detail="Model2 team stats not loaded.")

    if home_team == away_team:
        raise HTTPException(status_code=400, detail="home_team and away_team must be different.")

    # canonical names (case-insensitive)
    home_team_canon = validate_team_exists(home_team, context="model2")
    away_team_canon = validate_team_exists(away_team, context="model2")

    if home_team_canon not in MODEL2_TEAM_STATS:
        raise HTTPException(
            status_code=400, detail=f"No history found for HOME team: {home_team_canon}"
        )
    if away_team_canon not in MODEL2_TEAM_STATS:
        raise HTTPException(
            status_code=400, detail=f"No history found for AWAY team: {away_team_canon}"
        )

    # strength
    h_strength = _get_team_stat(home_team_canon, "home_strength", "away_strength")
    a_strength = _get_team_stat(away_team_canon, "home_strength", "away_strength")

    # goals for / against
    h_gf = _get_team_stat(home_team_canon, "home_goals_for", "away_goals_for")
    a_gf = _get_team_stat(away_team_canon, "away_goals_for", "home_goals_for")

    h_ga = _get_team_stat(home_team_canon, "home_goals_against", "away_goals_against")
    a_ga = _get_team_stat(away_team_canon, "away_goals_against", "home_goals_against")

    # matches played
    h_mp = _get_team_stat(home_team_canon, "home_matches_played", "away_matches_played")
    a_mp = _get_team_stat(away_team_canon, "away_matches_played", "home_matches_played")

    # expected goals
    h_xg = _get_team_stat(home_team_canon, "home_xg", "away_xg")
    a_xg = _get_team_stat(away_team_canon, "away_xg", "home_xg")

    strength_diff = h_strength - a_strength

    row = pd.DataFrame(
        [
            {
                "home_strength": h_strength,
                "away_strength": a_strength,
                "strength_diff": strength_diff,
                "home_goals_for": h_gf,
                "away_goals_for": a_gf,
                "home_goals_against": h_ga,
                "away_goals_against": a_ga,
                "goals_for_diff": h_gf - a_gf,
                "goals_against_diff": h_ga - a_ga,
                "matches_played_diff": h_mp - a_mp,
                "home_xg": h_xg,
                "away_xg": a_xg,
            }
        ]
    )

    return row, home_team_canon, away_team_canon


# -------------------------------------------------
# Player-mode helpers (MODEL 3)
# -------------------------------------------------
def compute_team_strength(team: str, players: List[str]) -> float:
    """Compute mean player_score for exactly 11 valid, unique players of a team (case-insensitive)."""
    if player_strengths_df is None:
        raise HTTPException(status_code=500, detail="Player strengths not loaded.")

    team_canon = validate_team_exists(team, context="players")

    # Clean + normalize player names
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

    def get_team_stats(team: str):
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

    def get_team_xg(team: str) -> float:
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
    if home_booster is None or away_booster is None:
        raise HTTPException(status_code=500, detail="Model1 boosters not loaded.")

    X, home_team_canon, away_team_canon = build_features_model1(req.home_team, req.away_team)

    dm = xgb.DMatrix(X)
    pred_home = float(home_booster.predict(dm)[0])
    pred_away = float(away_booster.predict(dm)[0])

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
    X, home_team_canon, away_team_canon = build_features_model2(req.home_team, req.away_team)
    proba = _predict_proba_model2(X)
    pred_class = int(max(range(len(proba)), key=lambda i: proba[i]))

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
    if model2_booster is None:
        raise HTTPException(status_code=500, detail="Base booster not loaded.")

    home_strength = compute_team_strength(req.home_team, req.home_players)
    away_strength = compute_team_strength(req.away_team, req.away_players)

    X = build_features_player_mode(req.home_team, req.away_team, home_strength, away_strength)
    proba = _predict_proba_model2(X)
    pred_class = int(max(range(len(proba)), key=lambda i: proba[i]))

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
